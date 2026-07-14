//! Broadphase: a bounding volume hierarchy over collider AABBs.
//!
//! Rebuilt from scratch every frame. That sounds wasteful, but a top-down
//! build is O(n log n), branchless-simple, and needs none of the incremental
//! machinery (fat AABBs, refitting, tree rotations) a persistent tree does —
//! Box2D-style dynamic trees only pay off once thousands of mostly-static
//! bodies exist. The seam is `build` + `query_pairs`; a persistent tree can
//! replace the internals later without touching `collision::run`.

use super::Aabb;

/// Index of a node inside [`Bvh::nodes`].
type NodeIndex = u32;

enum NodeKind {
    /// A single collider; `body` indexes the `bounds` slice passed to `build`
    /// (which `collision::run` keeps parallel to its body list).
    Leaf { body: u32 },
    Internal { left: NodeIndex, right: NodeIndex },
}

struct Node {
    /// Bound of everything below this node: its own AABB for a leaf, the
    /// union of both children for an internal node.
    aabb: Aabb,
    kind: NodeKind,
}

/// A static BVH over one frame's collider bounds.
pub struct Bvh {
    nodes: Vec<Node>,
    /// `None` when built over an empty slice.
    root: Option<NodeIndex>,
}

impl Bvh {
    /// Build a tree over `bounds`; leaf `body` values are indices into it.
    pub fn build(bounds: &[Aabb]) -> Self {
        // TODO(owner): top-down median split.
        //   1. Start with the full list of body indices [0, n).
        //   2. For a range of one index, emit a Leaf node.
        //   3. Otherwise: pick the axis where the range's AABB *centers* are
        //      most spread out (max extent of a bound over all centers),
        //      sort (or `select_nth_unstable_by`) the range by center on that
        //      axis, split at the middle, recurse into both halves, then emit
        //      an Internal node whose aabb is left.union(right).
        //   A recursive helper `fn build_range(&mut nodes, bounds, indices:
        //   &mut [u32]) -> NodeIndex` keeps this tidy. Don't chase SAH yet —
        //   median split is within ~15% of it for game scenes and far simpler.
        todo!("Bvh::build")
    }

    /// Append every overlapping body pair `(i, j)` with `i < j` to `out`.
    pub fn query_pairs(&self, out: &mut Vec<(u32, u32)>) {
        // TODO(owner): two workable strategies —
        //   a) Leaf-vs-tree (recommended first): for each leaf, walk the tree
        //      from the root, descending only into nodes whose aabb overlaps
        //      the leaf's; on reaching another leaf, emit the pair. Emit only
        //      when `self_body < other_body` so each pair appears once.
        //      O(n log n)-ish and easy to get right.
        //   b) Tree-vs-tree self-traversal: recurse on node pairs starting
        //      with (root, root); skip non-overlapping node pairs, handle the
        //      (node, same-node) case by recursing into (L,L), (R,R), (L,R).
        //      Faster (each subtree pair visited once) but the base cases are
        //      fiddly — a good second pass once (a) works and is tested.
        todo!("Bvh::query_pairs")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn aabb(min: [f32; 3], max: [f32; 3]) -> Aabb {
        Aabb { min: Vec3::from_array(min), max: Vec3::from_array(max) }
    }

    fn pairs_of(bounds: &[Aabb]) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        Bvh::build(bounds).query_pairs(&mut pairs);
        pairs.sort_unstable();
        pairs
    }

    // These run red until `build`/`query_pairs`/`Aabb::{overlaps,union}` are
    // implemented: `cargo test -p renderer-prototype collision`.

    #[test]
    fn empty_and_single_produce_no_pairs() {
        assert!(pairs_of(&[]).is_empty());
        assert!(pairs_of(&[aabb([0.0; 3], [1.0; 3])]).is_empty());
    }

    #[test]
    fn overlapping_pair_is_found_once() {
        let pairs = pairs_of(&[
            aabb([0.0; 3], [1.0; 3]),
            aabb([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]),
            aabb([10.0; 3], [11.0; 3]), // far away, matches nothing
        ]);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn matches_brute_force_on_a_grid() {
        // 4x4 grid of unit boxes spaced 0.75 apart: lots of overlap, so a
        // traversal bug (missed subtree, double-count) can't hide.
        let mut bounds = Vec::new();
        for x in 0..4 {
            for z in 0..4 {
                let min = Vec3::new(x as f32 * 0.75, 0.0, z as f32 * 0.75);
                bounds.push(Aabb { min, max: min + Vec3::ONE });
            }
        }
        let mut expected = Vec::new();
        for i in 0..bounds.len() {
            for j in (i + 1)..bounds.len() {
                if bounds[i].overlaps(&bounds[j]) {
                    expected.push((i as u32, j as u32));
                }
            }
        }
        assert_eq!(pairs_of(&bounds), expected);
    }
}
