//! Where a frame's time goes, per phase and per pass, with no window.
//!
//! `#[ignore]`d for the reason `offscreen.rs` is: it needs a GPU and CI has
//! none. This is the measuring half of that pair — `offscreen` says the frame
//! still looks right, this says what it costs — and it exists so an
//! optimisation is chosen against a table rather than against an intuition
//! about which pass is expensive.
//!
//! Run it with:
//!
//! ```text
//! cargo test --release -p orrin-core --test perf -- --ignored --nocapture
//! ```
//!
//! and read the two tables it prints: the CPU lane is the phases this harness
//! opens (extract, record, submit), the GPU lane is every node the compiled
//! graph ran. Environment:
//!
//! - `ORRIN_SCENE` picks the scene, exactly as the engine does.
//! - `ORRIN_STRESS` adds load, exactly as the engine does — `ORRIN_STRESS=4000`
//!   is what turns the CPU lane into a readable signal, since the built-in
//!   scenes have too few entities for extraction to show up at all.
//! - `ORRIN_PERF_FRAMES` is how many frames to average over (default 200).
//! - `ORRIN_PERF_EXTENT` is `WIDTHxHEIGHT` (default 1920x1080).
//!
//! Per-pass GPU timing serialises adjacent nodes that would otherwise overlap
//! (see `profile::gpu_passes_enabled`), so the GPU rows are a breakdown, not a
//! frame time. The whole-frame row is the number to compare across runs.

use std::sync::Arc;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

use orrin_core::gfx::punctual::{MAX_SHADOW_LIGHTS, ShadowAtlas};
use orrin_core::gfx::shadows::{CascadeSet, MAX_CASCADES, cascades};
use orrin_core::gfx::vulkan::{ShadowFrame, VulkanRenderer};
use orrin_core::gfx::{DecalInstance, DrawList, SceneLighting};
use orrin_core::profile::{self, Lane, Profiler};
use orrin_core::scene::entities::{SceneChoice, StressSpec, spawn_stress_scene};
use orrin_core::scene::{
    BloomSettings, Camera, ContactShadowSettings, Culling, DecalSettings, DofSettings,
    EnvironmentSettings, FogSettings, HdrSettings, MotionBlurSettings, RefractionSettings,
    ShadowSettings, SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings,
    TransparencySettings,
};
use orrin_core::systems::{self, FrameGeometry};
use orrin_ecs::World;

/// Frames whose numbers are thrown away: the temporal resolve has no history on
/// its first, auto-exposure adapts over several, and the graph compiles on the
/// first one it sees.
const WARMUP: u64 = 30;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(fallback)
}

fn extent() -> [u32; 2] {
    match std::env::var("ORRIN_PERF_EXTENT") {
        Ok(raw) => {
            let (w, h) = raw
                .split_once(['x', 'X'])
                .expect("ORRIN_PERF_EXTENT is WIDTHxHEIGHT");
            [
                w.trim().parse().expect("width"),
                h.trim().parse().expect("height"),
            ]
        }
        Err(_) => [1920, 1080],
    }
}

fn headless_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("failed to load vulkan library");
    Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .expect("failed to create a headless instance")
}

#[test]
#[ignore = "needs a GPU"]
fn frame_cost() {
    let extent = extent();
    let frames = env_usize("ORRIN_PERF_FRAMES", 200) as u64;
    let instance = headless_instance();
    let mut renderer = VulkanRenderer::offscreen(&instance, extent);

    let mut world = World::new();
    world.insert_resource(MotionBlurSettings::default());
    world.insert_resource(DofSettings::default());
    world.insert_resource(BloomSettings::default());
    world.insert_resource(HdrSettings::default());
    world.insert_resource(EnvironmentSettings::default());
    world.insert_resource(FogSettings::default());

    let scene = SceneChoice::from_env();
    scene.build(&mut world, &mut renderer);
    if let Some(spec) = StressSpec::from_env() {
        spawn_stress_scene(&mut world, &spec);
    }

    world.insert_resource(SsaoSettings::default());
    world.insert_resource(ContactShadowSettings::default());
    world.insert_resource(SsrSettings::default());
    world.insert_resource(SubsurfaceSettings::default());
    world.insert_resource(DecalSettings::default());
    world.insert_resource(TransparencySettings::default());
    world.insert_resource(RefractionSettings::default());
    world.insert_resource(TaaSettings::default());
    world.insert_resource(Culling::default());
    world.insert_resource(ShadowSettings {
        enabled: env_usize("ORRIN_PERF_SHADOWS", 1) != 0,
        ..ShadowSettings::default()
    });

    let mut lighting = SceneLighting::default();
    let mut geometry = FrameGeometry::default();
    let mut decals: Vec<DecalInstance> = Vec::new();
    let mut cascade_set;
    let mut atlas;
    let aspect = extent[0] as f32 / extent[1] as f32;

    // Big enough that a GPU readback landing several frames late still finds the
    // frame it belongs to; the aggregate below reads the whole ring.
    let mut profiler = Profiler::new((frames + WARMUP) as usize + 8);
    profile::set_enabled(true);
    profile::set_gpu_passes_enabled(true);

    let mut entities = 0usize;
    for frame in 0..frames + WARMUP {
        // The engine throws its numbers away for the first few frames; so does
        // this, by restarting the profiler once the transient is over.
        if frame == WARMUP {
            profiler = Profiler::new(frames as usize + 8);
        }
        {
            let _phase = profile::scope("extract");
            {
                let _s = profile::scope("propagate");
                orrin_core::scene::propagate_transforms(&mut world);
            }
            {
                let _s = profile::scope("lighting");
                systems::extract_lighting(&world, &mut lighting);
            }
            {
                let _s = profile::scope("decals");
                systems::extract_decals(&world, aspect, &mut decals);
            }
            let shadow_settings = *world.resource::<ShadowSettings>();
            let camera = *world.resource::<Camera>();
            cascade_set = if shadow_settings.enabled {
                cascades(
                    &camera,
                    aspect,
                    lighting.sun.direction,
                    &shadow_settings.cascade_config(),
                )
            } else {
                CascadeSet::default()
            };
            atlas = match shadow_settings.atlas_config() {
                Some(config) => orrin_core::gfx::punctual::fit(&lighting, camera.position, &config),
                None => ShadowAtlas::default(),
            };
            {
                let _s = profile::scope("geometry");
                systems::extract_geometry(&world, aspect, &cascade_set, &atlas, &mut geometry);
            }
            entities = geometry.visible().len();
        }

        let camera = *world.resource::<Camera>();
        let shadow_settings = *world.resource::<ShadowSettings>();
        {
            let _phase = profile::scope("render submit");
            let caster_lists: [DrawList<'_>; MAX_CASCADES] =
                std::array::from_fn(|i| geometry.cascade(i));
            let punctual_lists: [DrawList<'_>; MAX_SHADOW_LIGHTS] =
                std::array::from_fn(|i| geometry.punctual(i));
            renderer.render_with_overlay(
                geometry.visible(),
                geometry.transparent(),
                geometry.refractive(),
                &decals,
                &lighting,
                &camera,
                &world.resource::<SsaoSettings>().clone(),
                &world.resource::<ContactShadowSettings>().clone(),
                &world.resource::<SsrSettings>().clone(),
                &world.resource::<SubsurfaceSettings>().clone(),
                &world.resource::<TransparencySettings>().clone(),
                &world.resource::<RefractionSettings>().clone(),
                &world.resource::<TaaSettings>().clone(),
                &world.resource::<MotionBlurSettings>().clone(),
                &world.resource::<DofSettings>().clone(),
                &world.resource::<BloomSettings>().clone(),
                &world.resource::<HdrSettings>().clone(),
                &world.resource::<EnvironmentSettings>().clone(),
                &world.resource::<FogSettings>().clone(),
                1.0 / 60.0,
                &[],
                profiler.frame_index(),
                (cascade_set.count > 0).then(|| ShadowFrame {
                    cascades: &cascade_set,
                    casters: &caster_lists,
                    atlas: &atlas,
                    punctual_casters: &punctual_lists,
                    settings: &shadow_settings,
                }),
                None,
            );
        }
        renderer.drain_gpu_spans(&mut profiler);
        profiler.end_frame();
    }

    println!(
        "\nscene {scene:?} at {}x{}, {frames} frames, {entities} visible items",
        extent[0], extent[1]
    );
    for lane in [Lane::Cpu, Lane::Gpu] {
        let mut rows = profiler.aggregate(lane);
        rows.sort_by(|a, b| b.avg_ms.total_cmp(&a.avg_ms));
        println!(
            "\n{lane:?}: {:<28} {:>9} {:>9} {:>7}",
            "pass", "avg ms", "max ms", "calls"
        );
        for row in rows {
            println!(
                "     {:<28} {:>9.3} {:>9.3} {:>7}",
                row.name, row.avg_ms, row.max_ms, row.calls
            );
        }
    }
}
