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
