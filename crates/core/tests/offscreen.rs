//! Renders the default scene to a PNG with no window.
//!
//! `#[ignore]`d because it needs a GPU and CI has none — the render graph's
//! golden is the part of the renderer CI can assert, and it asserts
//! synchronisation rather than pixels. This is the other half: a lighting term
//! that silently goes dark, a material that stops binding, a pass that
//! composites the wrong way round, none of which move the barrier plan at all.
//!
//! Run it with:
//!
//! ```text
//! cargo test -p orrin-core --test offscreen -- --ignored --nocapture
//! ```
//!
//! and look at what lands in `target/capture/`. The captures are an A/B set on
//! purpose: the same scene with each non-opaque queue switched off, and with the
//! subsurface diffusion switched off, which is what turns "this looks wrong" into
//! "this looks wrong because of that pass".

use std::path::PathBuf;

use glam::Vec3;
use orrin_core::capture::{CaptureSettings, capture_default_scene};
use orrin_core::scene::Camera;

fn output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/capture");
    std::fs::create_dir_all(&dir).expect("failed to create the capture directory");
    dir
}

#[test]
#[ignore = "needs a GPU"]
fn captures_the_default_scene() {
    let dir = output_dir();

    for (name, settings) in [
        ("scene", CaptureSettings::default()),
        (
            "no-refraction",
            CaptureSettings {
                refraction: false,
                ..CaptureSettings::default()
            },
        ),
        (
            "opaque-only",
            CaptureSettings {
                transparency: false,
                refraction: false,
                ..CaptureSettings::default()
            },
        ),
        // The diffusion off, everything else as it ships. Scattering is still on
        // in both — `subsurface_wrap` widens to cover the missing passes — so the
        // difference between this and `scene.png` is what the two blurs and the
        // composite actually buy, on the marble and wax spheres in the lobe row.
        // It is also the shape that exercises the *other* forward render pass, so
        // a break in either variant shows up as one of the two files being wrong.
        (
            "no-subsurface-diffusion",
            CaptureSettings {
                subsurface: false,
                ..CaptureSettings::default()
            },
        ),
        // The same frame with nothing stamped on it. A decal is applied before
        // shading, so in `scene.png` it is indistinguishable from a surface that
        // was always that colour — which is the point of the technique and the
        // reason it cannot be checked by looking at one image. The difference
        // between this file and that one is every decal in the scene, and
        // nothing else: no pass is added or removed, no target is reallocated,
        // and the barrier plan is byte for byte the same.
        (
            "no-decals",
            CaptureSettings {
                decals: false,
                ..CaptureSettings::default()
            },
        ),
        // Close on the ground decals and the foliage behind them, because both
        // features live in the pixels and neither survives the demo camera's
        // distance. Three things are legible only from here. The crater's rim is
        // shaded by the sun and the point lights rather than painted, so it has
        // a lit side and a dark one. The decal stops where the ground turns up
        // into the cubes standing in it, which is the angle fade. And the leaf
        // cards' edges are resolved across the four MSAA samples the frame was
        // already paying for — against the sky they are a gradient, where a hard
        // alpha test would give the same silhouette in four steps.
        (
            "zoom-decals-foliage",
            CaptureSettings {
                camera: Some(Camera {
                    position: Vec3::new(-16.5, 3.2, 15.5),
                    target: Vec3::new(-10.5, 0.4, 5.5),
                    ..Camera::default()
                }),
                ..CaptureSettings::default()
            },
        ),
        // The only place the cutout's *caster* pipeline is visible. A leaf card
        // is a quad, so with the plain depth-only pipeline it casts the shadow of
        // a rectangle — the single most obvious way foliage goes wrong, and one
        // that no other capture here can show, because every other capture has
        // the cascades switched off. Look at the ground beneath the stand: leaves
        // and stems, with sky between them.
        (
            "zoom-foliage-shadow",
            CaptureSettings {
                shadows: true,
                camera: Some(Camera {
                    // Steeply down onto the open ground the stand throws its
                    // shadow across, with the cards themselves still in frame:
                    // the point is the pair, not either alone.
                    position: Vec3::new(-13.5, 7.5, 11.0),
                    target: Vec3::new(-12.0, 0.0, 1.5),
                    ..Camera::default()
                }),
                ..CaptureSettings::default()
            },
        ),
        // The two masonry walls, close and at a glancing angle, which is the one
        // vantage point where parallax occlusion mapping is legible: the left
        // wall marches the height field and the right one has the same albedo,
        // the same normals and no field to march. So this is an A/B inside a
        // single frame — same light, same exposure, same mip level — and the
        // difference between the two halves is the whole feature. Mortar that
        // sits behind its blocks on the left and level with them on the right,
        // and courses that stay parallel on the right while the left's shift
        // against each other with depth.
        (
            "zoom-parallax",
            CaptureSettings {
                // The scene's third decal is on this wall, deliberately — a
                // decal projected onto a marched height field is the sharpest
                // check that the prepass and the forward pass agree about where
                // a surface *is*. It is taken back out here, because this
                // capture's whole claim is that the only difference between the
                // upper panel and the lower one is the height map, and a mark
                // straddling the seam between them is a second variable. Each
                // capture isolates one feature; that is what the switches are
                // for.
                decals: false,
                camera: Some(Camera {
                    position: Vec3::new(3.76, 1.5, -12.99),
                    target: Vec3::new(0.0, 1.5, -13.0),
                    ..Camera::default()
                }),
                ..CaptureSettings::default()
            },
        ),
    ] {
        let path = dir.join(format!("{name}.png"));
        capture_default_scene(&path, &settings);

        let written = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("{} was not written: {e}", path.display()))
            .len();
        // A PNG of a frame that rendered nothing still has a header, so the
        // assertion is on something a blank image could not produce.
        assert!(
            written > 4096,
            "{} is {written} bytes, which is too small to be a rendered frame",
            path.display(),
        );
        println!("wrote {} ({written} bytes)", path.display());
    }
}
