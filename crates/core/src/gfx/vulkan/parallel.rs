//! Splitting one frame's recording across the worker pool.

use std::ops::Range;

/// What a slot costs at the least, whatever its pass draws: a group of passes
/// that record nothing still costs a command buffer to begin and end, so a
/// frame of them spreads across the pool rather than collapsing onto one
/// worker.
const FLOOR: u32 = 1;

/// Divide `costs` into at most `groups` contiguous runs, minimising what the
/// heaviest run costs.
///
/// Contiguous because the runs are recorded into secondary command buffers the
/// primary executes in order, and the frame's order is the one the graph
/// compiled — a run that skipped a slot would have to hand it to someone else
/// to record into a buffer that lands between two of its own.
///
/// Minimising the heaviest run rather than equalising them: the wall clock is
/// the slowest worker, and a frame where one pass costs more than all the
/// others put together — which is the punctual shadow atlas, measured — has no
/// equal split to find.
pub(super) fn partition(costs: &[u32], groups: usize) -> Vec<Range<usize>> {
    if costs.is_empty() {
        return Vec::new();
    }
    if groups <= 1 {
        return std::iter::once(0..costs.len()).collect();
    }

    // Binary search the answer rather than solving the split directly: `runs`
    // is monotonic in the ceiling, so the smallest ceiling that still fits in
    // `groups` runs is the one the optimal split has.
    let mut low = costs
        .iter()
        .map(|cost| (*cost).max(FLOOR))
        .max()
        .unwrap_or(FLOOR);
    let mut high: u32 = costs.iter().map(|cost| (*cost).max(FLOOR)).sum();
    while low < high {
        let ceiling = low + (high - low) / 2;
        if runs(costs, ceiling).len() <= groups {
            high = ceiling;
        } else {
            low = ceiling + 1;
        }
    }
    runs(costs, low)
}

/// Fill runs from the front, closing one whenever the next slot would take it
/// past `ceiling`. Optimal for a given ceiling: a run that stopped earlier only
/// moves its own work into the next one.
fn runs(costs: &[u32], ceiling: u32) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let mut start = 0;
    let mut total = 0;
    for (slot, cost) in costs.iter().enumerate() {
        let cost = (*cost).max(FLOOR);
        if slot > start && total + cost > ceiling {
            runs.push(start..slot);
            start = slot;
            total = 0;
        }
        total += cost;
    }
    runs.push(start..costs.len());
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pass's own state is shared, not owned, while a frame records: the
    /// workers hold `&PassType` and record from it at the same time. Named as a
    /// bound of its own rather than left to the dispatch's inference, because
    /// the error a broken one produces there is a page of closure types, and
    /// the fix is always the same — something inside that pass gained a
    /// `SubbufferAllocator`, a `Cell` or a `RefCell` and wants a `Mutex` or an
    /// atomic instead.
    fn assert_shareable<T: Sync>() {}

    /// The bound above, exercised over the passes a frame records from and over
    /// what carries them to a worker.
    #[test]
    fn pass_state_is_shareable_across_workers() {
        assert_shareable::<super::super::FrameRecord<'static>>();
        assert_shareable::<super::super::bloom::BloomPass>();
        assert_shareable::<super::super::contact_shadows::ContactShadowPass>();
        assert_shareable::<super::super::dof::DofPass>();
        assert_shareable::<super::super::environment::EnvironmentPass>();
        assert_shareable::<super::super::exposure::ExposurePass>();
        assert_shareable::<super::super::fog::FogPass>();
        assert_shareable::<super::super::forward::ForwardPass>();
        assert_shareable::<super::super::hdr::HdrPass>();
        assert_shareable::<super::super::line::LinePass>();
        assert_shareable::<super::super::motion_blur::MotionBlurPass>();
        assert_shareable::<super::super::oit::OitPass>();
        assert_shareable::<super::super::prepass::GeometryPrepass>();
        assert_shareable::<super::super::refraction::RefractionPass>();
        assert_shareable::<super::super::shadow::ShadowPass>();
        assert_shareable::<super::super::ssao::SsaoPass>();
        assert_shareable::<super::super::ssr::SsrPass>();
        assert_shareable::<super::super::subsurface::SubsurfacePass>();
        assert_shareable::<super::super::taa::TaaPass>();
    }

    /// Every slot recorded exactly once, in order. A partition that drops or
    /// reorders one drops or reorders a pass.
    #[test]
    fn partition_covers_every_slot_in_order() {
        for len in 0..12usize {
            for groups in 1..6usize {
                let costs: Vec<u32> = (0..len).map(|slot| (slot as u32 * 7) % 5).collect();
                let runs = partition(&costs, groups);
                assert!(runs.len() <= groups, "len {len}, groups {groups}");
                assert!(runs.iter().all(|run| run.start < run.end));
                let mut next = 0;
                for run in &runs {
                    assert_eq!(run.start, next, "len {len}, groups {groups}");
                    next = run.end;
                }
                assert_eq!(next, len, "len {len}, groups {groups}");
            }
        }
    }

    /// The shape this exists for: one pass costs more than all the others put
    /// together, so the best split isolates it and the critical path is that
    /// pass alone.
    #[test]
    fn partition_isolates_the_expensive_pass() {
        let costs = [1, 1, 1, 40, 1, 1, 1, 1];
        let runs = partition(&costs, 3);
        assert_eq!(heaviest(&costs, &runs), 40);
        assert!(runs.contains(&(3..4)), "{runs:?}");
    }

    /// The property the search claims, checked against every contiguous split
    /// of a small frame.
    #[test]
    fn partition_minimises_the_heaviest_run() {
        let cases: [&[u32]; 4] = [
            &[3, 1, 4, 1, 5, 9, 2, 6],
            &[10, 1, 1, 1],
            &[1, 1, 1, 10],
            &[7, 7, 7, 7, 7],
        ];
        for costs in cases {
            for groups in 1..5usize {
                let runs = partition(costs, groups);
                assert_eq!(
                    heaviest(costs, &runs),
                    brute_force(costs, groups),
                    "costs {costs:?}, groups {groups}",
                );
            }
        }
    }

    /// One worker records the frame as it always did: one run over everything.
    #[test]
    fn partition_of_one_group_is_the_whole_frame() {
        assert_eq!(partition(&[4, 2, 9], 1), vec![0..3]);
        assert_eq!(partition(&[], 4), Vec::<Range<usize>>::new());
    }

    /// A pass that records nothing still occupies a slot, so a frame of them
    /// spreads rather than collapsing onto one worker.
    #[test]
    fn partition_spreads_passes_that_cost_nothing() {
        assert_eq!(partition(&[0, 0, 0, 0], 2).len(), 2);
    }

    fn heaviest(costs: &[u32], runs: &[Range<usize>]) -> u32 {
        runs.iter()
            .map(|run| costs[run.clone()].iter().map(|cost| (*cost).max(1)).sum())
            .max()
            .unwrap_or(0)
    }

    /// The smallest heaviest-run any contiguous split into at most `groups`
    /// parts can achieve, by trying all of them.
    fn brute_force(costs: &[u32], groups: usize) -> u32 {
        fn search(costs: &[u32], start: usize, groups: usize) -> u32 {
            if start == costs.len() {
                return 0;
            }
            if groups == 1 {
                return costs[start..].iter().map(|cost| (*cost).max(1)).sum();
            }
            (start + 1..=costs.len())
                .map(|end| {
                    let head: u32 = costs[start..end].iter().map(|cost| (*cost).max(1)).sum();
                    head.max(search(costs, end, groups - 1))
                })
                .min()
                .unwrap_or(0)
        }
        search(costs, 0, groups)
    }
}
