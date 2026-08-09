//! The synchronisation tripwire the render graph exists to make possible (#7).
//!
//! A wrong barrier is the worst bug the renderer can have: it produces correct
//! pixels on the machine that wrote it and a race on someone else's scheduler,
//! and it survives review because nobody reads a barrier and sees it is one flag
//! too loose. The graph derives them instead, which means the derivation itself
//! can be pinned — this file renders the reference frame's plan to text and
//! compares it against a checked-in baseline.
//!
//! **What a failure means.** The plan changed. That is legitimate whenever the
//! frame's structure changed on purpose, and a regression whenever it didn't:
//! a barrier that quietly loosens, a transition that disappears, a pass that
//! moves. Read the diff before regenerating — the baseline is the review, so
//! updating it without reading it spends the whole mechanism.
//!
//! Regenerate with `ORRIN_UPDATE_GOLDEN=1 cargo test -p orrin-core --test
//! render_graph`, and put the diff in the commit.
//!
//! It runs on a GPU-less CI runner because compiling a graph takes no `Device`,
//! which is the property that made the derivation worth doing this way.

use std::fs;
use std::path::PathBuf;

use orrin_core::gfx::vulkan::frame::{FrameConfig, declare};
use vulkano::format::Format;

/// The swapchain format the baseline is written against. Real ones vary by
/// surface; the plan does not depend on it, and pinning it keeps the file from
/// depending on whoever regenerated it.
const COLOR_FORMAT: Format = Format::B8G8R8A8_SRGB;

/// The cascade resolution the baseline is written against. It changes what the
/// images are sized to, not what the plan says, so pinning it keeps the file
/// from depending on whoever regenerated it.
const SHADOW_RESOLUTION: u32 = 2048;

/// The bloom chain length the baseline is written against — what
/// `bloom::mip_count` yields for any frame from 1080p up. Pinned for the same
/// reason as the cascade resolution: the plan should not depend on the window
/// whoever regenerated it happened to have open.
const BLOOM_MIPS: u8 = 6;

fn configs() -> Vec<(&'static str, FrameConfig)> {
    vec![
        (
            "editor frame, TAA and SSAO on",
            FrameConfig {
                color_format: COLOR_FORMAT,
                ssao: true,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
            },
        ),
        // Four cascades write four layers of one image, which the graph tracks
        // as one resource — so consecutive cascades are separated by a
        // write-after-write barrier with no layout transition. That
        // serialisation is the cost of not tracking subresources in v1, and it
        // is in the baseline so that removing it later is a visible diff rather
        // than a silent one.
        (
            "editor frame, four cascades",
            FrameConfig {
                color_format: COLOR_FORMAT,
                ssao: true,
                taa: false,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
            },
        ),
        // SSAO off is a different graph, not a flag read at record time: the
        // three passes are never registered and the forward pass never declares
        // the read. Both shapes are baselined because both ship. Bloom is off
        // here too, which is the shape where the tonemap pass declares no bloom
        // input and samples a 1x1 black view instead.
        (
            "editor frame, SSAO off",
            FrameConfig {
                color_format: COLOR_FORMAT,
                ssao: false,
                taa: false,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                bloom_mips: 0,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
            },
        ),
        // The shape that proves the prepass belongs to the frame rather than to
        // SSAO: nothing reads its normal target here, and it still runs, because
        // TAA needs the motion vectors that come off the same rasterisation.
        (
            "editor frame, TAA without SSAO",
            FrameConfig {
                color_format: COLOR_FORMAT,
                ssao: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
            },
        ),
        // Metering off is the other shape that ships. Worth baselining for one
        // thing in particular: the tonemap pass still declares its read of
        // `exposure`, an import nothing writes in this configuration. That is
        // legal where reading an unwritten *transient* is not, and it is what
        // lets one tonemap pipeline serve both modes.
        (
            "editor frame, auto exposure off",
            FrameConfig {
                color_format: COLOR_FORMAT,
                ssao: true,
                taa: false,
                auto_exposure: false,
                motion_blur: false,
                dof: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
            },
        ),
        // A window too small for a real chain still gets one level, and that
        // shape has no upsample pass in it at all — so the tonemap pass
        // composites the down chain directly. It is the case where `result()`
        // takes its other branch, and nothing else in the suite reaches it.
        (
            "editor frame, one bloom level",
            FrameConfig {
                color_format: COLOR_FORMAT,
                ssao: true,
                taa: false,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                bloom_mips: 1,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
            },
        ),
        // The whole optical chain at once: lens, then shutter, then sensor.
        // Worth baselining as one shape rather than two because the passes are
        // chained through `scene_color` — each rebinds it for the next — so the
        // barriers between them are the thing that would break if the order
        // were ever rearranged.
        (
            "editor frame, depth of field and motion blur",
            FrameConfig {
                color_format: COLOR_FORMAT,
                ssao: true,
                taa: true,
                auto_exposure: true,
                motion_blur: true,
                dof: true,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
            },
        ),
        // The other shape that proves the prepass belongs to the frame rather
        // than to any one consumer: SSAO and TAA are both off here, and it still
        // runs, because motion blur needs the velocities and depth of field
        // needs the depth. It is also the only configuration where the chain
        // reads the forward pass's resolve directly instead of a TAA output.
        (
            "editor frame, depth of field and motion blur without TAA",
            FrameConfig {
                color_format: COLOR_FORMAT,
                ssao: false,
                taa: false,
                auto_exposure: true,
                motion_blur: true,
                dof: true,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
            },
        ),
        (
            "headless frame, no overlay",
            FrameConfig {
                color_format: COLOR_FORMAT,
                ssao: true,
                taa: false,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                bloom_mips: BLOOM_MIPS,
                overlay: false,
                shadow_cascades: 2,
                shadow_resolution: SHADOW_RESOLUTION,
            },
        ),
    ]
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/frame_graph.txt")
}

fn render() -> String {
    let mut out = String::new();
    for (label, config) in configs() {
        let frame = declare(config).expect("the reference frame must compile");
        out.push_str(&format!("=== {label} ===\n{}\n", frame.graph));
    }
    out
}

#[test]
fn the_derived_barrier_sequence_matches_the_baseline() {
    let actual = render();
    let path = golden_path();

    if std::env::var_os("ORRIN_UPDATE_GOLDEN").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, &actual).unwrap();
        return;
    }

    let expected = fs::read_to_string(&path).unwrap_or_default();
    assert_eq!(
        actual,
        expected,
        "the frame's derived barrier plan no longer matches {}.\n\
         If the frame's structure changed on purpose, regenerate with \
         ORRIN_UPDATE_GOLDEN=1 and commit the diff. If it did not, this is a \
         synchronisation regression: a barrier has loosened or a transition has \
         gone missing.",
        path.display(),
    );
}

/// A pass's declarations and the engine code that runs it are looked up by the
/// same index, so a registration that forgets its body would silently run the
/// wrong pass.
#[test]
fn every_declared_pass_has_a_body() {
    for (label, config) in configs() {
        let frame = declare(config).unwrap();
        assert_eq!(
            frame.bodies.len(),
            frame.graph.pass_count(),
            "{label}: {} passes declared but {} bodies registered",
            frame.graph.pass_count(),
            frame.bodies.len(),
        );
    }
}

/// Nothing in the shipped frame is dead. A culled pass is a pass that recorded
/// work no one reads, which is worth noticing the moment it appears rather than
/// when someone profiles it.
#[test]
fn the_reference_frame_culls_nothing() {
    for (label, config) in configs() {
        let frame = declare(config).unwrap();
        let culled: Vec<&str> = frame
            .graph
            .culled()
            .iter()
            .map(|&id| frame.graph.pass_name(id))
            .collect();
        assert!(culled.is_empty(), "{label}: culled {culled:?}");
    }
}

/// A one-cascade frame is the case that catches a view type derived from the
/// layer count rather than from the declaration: an image of exactly one array
/// layer looks like a plain 2D image to that heuristic, while the forward
/// pipeline's sampler is compiled as a `texture2DArray` regardless. It renders
/// on four cascades and fails on one, so the setting that provokes it is the
/// one nobody drags to on purpose.
#[test]
fn a_single_cascade_map_is_still_declared_as_an_array() {
    for count in 1..=4u8 {
        let frame = declare(FrameConfig {
            color_format: COLOR_FORMAT,
            ssao: true,
            taa: false,
            auto_exposure: true,
            motion_blur: false,
            dof: false,
            bloom_mips: BLOOM_MIPS,
            overlay: true,
            shadow_cascades: count,
            shadow_resolution: SHADOW_RESOLUTION,
        })
        .unwrap();

        let (_, image) = frame
            .graph
            .transient_images()
            .find(|(id, _)| frame.graph.resource_name(*id) == "shadow_cascades")
            .unwrap_or_else(|| panic!("{count} cascades declared no shadow map"));

        assert_eq!(
            image.desc.array_layers,
            Some(u32::from(count)),
            "{count} cascades must declare a {count}-layer array, not a plain image",
        );
    }
}

/// TAA ping-pongs two allocations, so the image this frame writes as `taa_color`
/// is the one next frame reads as `taa_history`. That only works if the frame
/// leaves `taa_color` in exactly the layout `taa_history` is declared to enter
/// in — otherwise every frame after the first samples an image the plan says is
/// in some other layout, and the resulting read is undefined on hardware that
/// takes the declaration seriously.
///
/// Nothing in the compiler enforces the pairing; it is a property of how the
/// executor binds the two, so it is asserted here rather than derived.
#[test]
fn the_taa_history_leaves_the_frame_where_the_next_one_expects_it() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        ssao: true,
        taa: true,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
    })
    .unwrap();

    let plan = format!("{}", frame.graph);
    assert!(
        plan.contains("taa_color General->ShaderReadOnlyOptimal"),
        "the resolve's output must end the frame sampled, not left in General:\n{plan}",
    );
    // An import already in its exit layout needs no closing transition, so the
    // absence of one is the assertion: a `taa_color` line among the final
    // barriers would mean the frame is handing the next one a layout it does not
    // expect.
    let closing: Vec<_> = frame
        .graph
        .final_barriers()
        .iter()
        .filter(|barrier| {
            let name = frame.graph.resource_name(barrier.resource);
            name == "taa_color" || name == "taa_history"
        })
        .collect();
    assert!(closing.is_empty(), "{closing:?}");
}

/// Light meets a lens, then a shutter, then a sensor, and the frame has to run
/// them in that order — it is what Unity and Unreal both settled on, and it is
/// not arbitrary: defocus before the shutter means the streak smears an image
/// the lens already formed, and both before the sensor means a defocused
/// highlight blooms as the wide soft thing it has become rather than as the
/// point it was.
///
/// Nothing in `compile` knows any of that. The order is a consequence of how
/// `declare` chains `scene_color` from one stage to the next, so reordering
/// those blocks would silently reorder the optics — which is why it is asserted
/// against the derived schedule rather than against the source.
#[test]
fn the_optical_chain_runs_lens_then_shutter_then_sensor() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        ssao: true,
        taa: true,
        auto_exposure: true,
        motion_blur: true,
        dof: true,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
    })
    .unwrap();

    let order: Vec<&str> = frame
        .graph
        .order()
        .iter()
        .map(|&id| frame.graph.pass_name(id))
        .collect();
    let at = |name: &str| {
        order
            .iter()
            .position(|&pass| pass == name)
            .unwrap_or_else(|| panic!("{name} is not in the frame: {order:?}"))
    };

    assert!(at("taa_resolve") < at("dof_prefilter"), "{order:?}");
    assert!(at("dof_composite") < at("motion_blur_gather"), "{order:?}");
    assert!(
        at("motion_blur_gather") < at("bloom_prefilter"),
        "{order:?}"
    );
    assert!(
        at("motion_blur_gather") < at("luminance_histogram"),
        "{order:?}"
    );
    assert!(at("motion_blur_gather") < at("tonemap"), "{order:?}");

    // The dilation has to sit between the tiles and the gather that reads them,
    // or a fast object's blur stops at the tile boundary it started in.
    assert!(
        at("motion_blur_tile_max") < at("motion_blur_neighbour_max"),
        "{order:?}"
    );
    assert!(
        at("motion_blur_neighbour_max") < at("motion_blur_gather"),
        "{order:?}"
    );
}

/// The geometry prepass exists for whoever needs what it writes, and after this
/// change that is four separate consumers. Each has to be able to keep it alive
/// on its own — a prepass gated on any subset of them would leave motion blur
/// gathering along velocities nobody rasterised, or depth of field focusing on
/// a depth buffer that was never allocated.
#[test]
fn any_single_consumer_keeps_the_geometry_prepass() {
    let base = FrameConfig {
        color_format: COLOR_FORMAT,
        ssao: false,
        taa: false,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
    };

    let consumers: [(&str, fn(&mut FrameConfig)); 4] = [
        ("ssao", |c| c.ssao = true),
        ("taa", |c| c.taa = true),
        ("motion blur", |c| c.motion_blur = true),
        ("depth of field", |c| c.dof = true),
    ];
    for (label, enable) in consumers {
        let mut config = base;
        enable(&mut config);
        let frame = declare(config).unwrap();
        assert!(
            frame.ids.prepass.is_some(),
            "{label} alone left the frame with no geometry prepass",
        );
    }

    // And nothing wants it when none of them do, so it is genuinely gated
    // rather than always present.
    assert!(declare(base).unwrap().ids.prepass.is_none());
}

/// The swapchain image is acquired undefined and handed back to the presentation
/// engine, so the frame owes a closing transition to `PresentSrc` whatever else
/// it does.
#[test]
fn every_configuration_leaves_the_swapchain_presentable() {
    for (label, config) in configs() {
        let frame = declare(config).unwrap();
        let closing: Vec<_> = frame
            .graph
            .final_barriers()
            .iter()
            .filter(|barrier| {
                frame.graph.resource_name(barrier.resource) == "swapchain_color"
                    && barrier.new_layout == vulkano::image::ImageLayout::PresentSrc
            })
            .collect();
        assert_eq!(
            closing.len(),
            1,
            "{label}: {:?}",
            frame.graph.final_barriers()
        );
    }
}
