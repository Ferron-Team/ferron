use orrin_ecs::World;

use crate::gfx::shadows::MAX_CASCADES;

use super::{color_row, vec3_row};
use crate::scene::{
    AmbientLight, BloomSettings, Camera, ContactShadowSettings, DecalSettings, DofSettings,
    EnvironmentSettings, FogSettings, HdrSettings, MotionBlurSettings, RefractionSettings,
    ShadowSettings, SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings,
    TransparencySettings,
};

type Column = fn(&mut egui::Ui, &World);

const COLUMNS: [Column; 4] = [
    screen_space_column,
    shadow_column,
    lighting_column,
    camera_column,
];

/// A slider plus the label that sits after it. Narrower than this and a column's
/// contents run into its neighbour instead of wrapping — `Ui::columns` divides
/// the width but does nothing to make the contents fit it.
///
/// This was once a full-window strip along the bottom, where four cramped
/// columns were the intended look and folding was only a guard against a
/// negative width. Docked, the same panel is a 300px tab, and four columns there
/// is four overlapping ones. The number is now what it says it is.
const MIN_COLUMN_WIDTH: f32 = 190.0;

/// How many columns `available` can carry. Never zero: `Ui::columns` divides by
/// the count, and it asserts on the negative width a shortfall would produce.
fn column_count(available: f32) -> usize {
    ((available / MIN_COLUMN_WIDTH) as usize).clamp(1, COLUMNS.len())
}

pub fn body(ui: &mut egui::Ui, world: &mut World) {
    // A docked tab is as wide as its split, which can be nothing at all.
    let available = ui.available_width();
    if available > 0.0 {
        let columns = column_count(available);
        ui.columns(columns, |cols| {
            let mut previous = usize::MAX;
            for (index, column) in COLUMNS.iter().enumerate() {
                let target = index * columns / COLUMNS.len();
                if target == previous {
                    cols[target].add_space(10.0);
                    cols[target].separator();
                }
                column(&mut cols[target], world);
                previous = target;
            }
        });
    }
    ui.add_space(4.0);
}

fn screen_space_column(ui: &mut egui::Ui, world: &World) {
    ui.strong("SSAO");
    {
        let mut s = world.resource_mut::<SsaoSettings>();
        ui.checkbox(&mut s.enabled, "Enabled");
        ui.add(egui::Slider::new(&mut s.radius, 0.0..=4.0).text("Radius"));
        ui.add(egui::Slider::new(&mut s.bias, 0.0..=0.1).text("Bias"));
        ui.add(egui::Slider::new(&mut s.power, 0.1..=4.0).text("Power"));
    }
    ui.add_space(6.0);
    ui.strong("Temporal AA")
        .on_hover_text("Jitter the camera each frame and accumulate the results");
    {
        let mut taa = world.resource_mut::<TaaSettings>();
        // Frame structure, like SSAO and bloom: it registers the resolve node
        // and it is what keeps the geometry prepass in a frame with SSAO off.
        ui.checkbox(&mut taa.enabled, "Enabled");
        ui.add(egui::Slider::new(&mut taa.feedback, 0.5..=0.98).text("Feedback"))
            .on_hover_text("How much of the reprojected history each frame keeps");
        ui.add(egui::Slider::new(&mut taa.jitter_scale, 0.0..=1.5).text("Jitter"))
            .on_hover_text("Fraction of a pixel the camera samples across");
    }
    ui.add_space(6.0);
    ui.strong("Reflections").on_hover_text(
        "Reflect the frame off itself, where the frame contains what a surface faces",
    );
    {
        let mut ssr = world.resource_mut::<SsrSettings>();
        // Frame structure like the rest: it registers the pyramid, the trace and
        // the composite, and it is another reason the prepass exists.
        ui.checkbox(&mut ssr.enabled, "Screen-space reflections");
        ui.add(egui::Slider::new(&mut ssr.intensity, 0.0..=1.0).text("Intensity"))
            .on_hover_text("1.0 is the physical answer; lower hides the screen-space failures");
        ui.add(egui::Slider::new(&mut ssr.max_roughness, 0.05..=1.0).text("Max roughness"))
            .on_hover_text("Rougher than this is left to the environment cubemap");
        ui.add(
            egui::Slider::new(&mut ssr.thickness, 0.01..=2.0)
                .logarithmic(true)
                .text("Thickness")
                .suffix(" m"),
        )
        .on_hover_text(
            "How deep a surface is assumed to be — the depth buffer records only its front",
        );
        ui.add(
            egui::Slider::new(&mut ssr.max_distance, 1.0..=200.0)
                .logarithmic(true)
                .text("Range")
                .suffix(" m"),
        );
        ui.add(egui::Slider::new(&mut ssr.max_steps, 8..=128).text("Steps"))
            .on_hover_text("A step crosses a whole cell of the depth pyramid, not a texel");
    }
    ui.add_space(6.0);
    ui.strong("Subsurface scattering")
        .on_hover_text("Light that enters a surface, scatters inside it and leaves somewhere else");
    {
        let mut subsurface = world.resource_mut::<SubsurfaceSettings>();
        // Frame structure, and more of it than the rest: this switch decides which
        // of the two forward render passes the frame opens, so it registers three
        // passes *and* a second colour attachment. It is also one more reason the
        // prepass exists — the blur weights its taps by depth.
        ui.checkbox(&mut subsurface.enabled, "Screen-space diffusion")
            .on_hover_text(
                "Off keeps scattering: the analytic wrap widens to stand in for these passes",
            );
        ui.add(
            egui::Slider::new(&mut subsurface.max_radius, 4.0..=128.0)
                .logarithmic(true)
                .text("Max radius")
                .suffix(" px"),
        )
        .on_hover_text(
            "A performance clamp. The physical width comes from the material's mean free path",
        );
        // Three sliders rather than a colour picker: these are distances, not a
        // colour, and the widest is pinned at one by construction — only the
        // ratios reach the shader.
        ui.add(egui::Slider::new(&mut subsurface.profile.x, 0.0..=1.0).text("Red reach"));
        ui.add(egui::Slider::new(&mut subsurface.profile.y, 0.0..=1.0).text("Green reach"));
        ui.add(egui::Slider::new(&mut subsurface.profile.z, 0.0..=1.0).text("Blue reach"))
            .on_hover_text("How far each channel travels, relative to the furthest");
    }
    ui.add_space(6.0);
    ui.strong("Transparency")
        .on_hover_text("Weighted-blended order-independent transparency (McGuire & Bavoil)");
    {
        let mut transparency = world.resource_mut::<TransparencySettings>();
        // Frame structure like the rest: it registers the accumulation and
        // composite nodes, and it is one more thing that keeps the prepass
        // alive — the accumulation depth-tests against what that pass wrote.
        ui.checkbox(&mut transparency.enabled, "Enabled")
            .on_hover_text("Off draws nothing at all where a blended material is used");
    }
    ui.add_space(6.0);
    ui.strong("Refraction").on_hover_text(
        "Screen-space refraction: a transmissive material samples the frame behind it",
    );
    {
        let mut refraction = world.resource_mut::<RefractionSettings>();
        // Beside transparency rather than under it: the two queues answer to
        // opposite rules about ordering, and each is its own A/B when a
        // non-opaque surface looks wrong.
        ui.checkbox(&mut refraction.enabled, "Enabled")
            .on_hover_text("Off draws nothing at all where a transmissive material is used");
    }
    ui.add_space(6.0);
    ui.strong("Decals")
        .on_hover_text("Boxes that stamp maps onto the surfaces inside them, before shading");
    {
        let mut decals = world.resource_mut::<DecalSettings>();
        // Unlike everything above it in this column, this is *not* frame
        // structure: decals add no pass and no target, so the graph is the same
        // either way and nothing is reallocated when it is clicked. It is here
        // as the A/B — decals land under the lighting, so "is this a decal or a
        // light?" is otherwise a hard question to ask of a still frame.
        ui.checkbox(&mut decals.enabled, "Enabled")
            .on_hover_text("Off leaves every surface as its own material describes it");
    }
}

fn shadow_column(ui: &mut egui::Ui, world: &World) {
    ui.strong("Shadows");
    let mut s = world.resource_mut::<ShadowSettings>();
    ui.checkbox(&mut s.enabled, "Enabled");
    // Cascade count and resolution are frame *structure*: changing either
    // recompiles the graph and reallocates the maps, so they are steppers
    // rather than sliders — a drag would do that once per frame.
    ui.horizontal(|ui| {
        ui.label("Cascades");
        ui.add(egui::DragValue::new(&mut s.cascade_count).range(1..=MAX_CASCADES));
    });
    ui.horizontal(|ui| {
        ui.label("Resolution");
        egui::ComboBox::from_id_salt("shadow_resolution")
            .selected_text(format!("{}", s.resolution))
            .show_ui(ui, |ui| {
                for size in [512u32, 1024, 2048, 4096] {
                    ui.selectable_value(&mut s.resolution, size, format!("{size}"));
                }
            });
    });
    ui.add(
        egui::Slider::new(&mut s.max_distance, 10.0..=500.0)
            .logarithmic(true)
            .text("Distance"),
    );
    ui.add(egui::Slider::new(&mut s.lambda, 0.0..=1.0).text("Split blend"));
    ui.add(egui::Slider::new(&mut s.pullback, 0.0..=200.0).text("Pullback"));
    ui.add(egui::Slider::new(&mut s.constant_bias, 0.0..=8.0).text("Bias"));
    ui.add(egui::Slider::new(&mut s.slope_bias, 0.0..=8.0).text("Slope bias"));
    ui.add(egui::Slider::new(&mut s.strength, 0.0..=1.0).text("Strength"));
    ui.checkbox(&mut s.debug_cascades, "Tint cascades");

    ui.add_space(6.0);
    ui.strong("Point & spot shadows")
        .on_hover_text("One atlas, a tile per cube face, drawn in a single pass");
    // Whether a *given* light casts is on the light, in the inspector; these are
    // the shape of the atlas it competes for.
    ui.checkbox(&mut s.punctual_enabled, "Enabled");
    ui.horizontal(|ui| {
        ui.label("Atlas");
        // 4096 is the ceiling on purpose, twice over. Vulkan guarantees
        // `maxImageDimension2D` of only 4096, so offering 8192 is offering a
        // startup panic on a conformant device that Metal happens to let this
        // one get away with. And it would buy nothing if it worked: the budget
        // is `MAX_SHADOW_LIGHTS` casters, so the most faces anything can ask for
        // is 48, and 4096 at 512 already holds 64.
        egui::ComboBox::from_id_salt("shadow_atlas_resolution")
            .selected_text(format!("{}", s.atlas_resolution))
            .show_ui(ui, |ui| {
                for size in [1024u32, 2048, 4096] {
                    ui.selectable_value(&mut s.atlas_resolution, size, format!("{size}"));
                }
            });
        egui::ComboBox::from_id_salt("shadow_atlas_tile")
            .selected_text(format!("{}", s.atlas_tile_size))
            .show_ui(ui, |ui| {
                for size in [128u32, 256, 512, 1024] {
                    ui.selectable_value(&mut s.atlas_tile_size, size, format!("{size}"));
                }
            });
    });
    // Both are structure, so the reader deserves to know what they bought: a
    // point light spends six of these and a spot one.
    {
        let tiles = s.atlas_config().map_or(0, |config| config.capacity());
        ui.weak(format!("{tiles} tiles — {} point lights", tiles / 6));
    }
    ui.add(
        egui::Slider::new(&mut s.punctual_near, 0.005..=1.0)
            .logarithmic(true)
            .text("Near")
            .suffix(" m"),
    )
    .on_hover_text("Depth precision near the light, which is where its contacts are");
    ui.add(egui::Slider::new(&mut s.punctual_constant_bias, 0.0..=8.0).text("Bias"));
    ui.add(egui::Slider::new(&mut s.punctual_slope_bias, 0.0..=8.0).text("Slope bias"));
    drop(s);

    ui.add_space(6.0);
    ui.strong("Contact shadows")
        .on_hover_text("March the depth buffer for the band a cascade texel is too coarse to hold");
    {
        let mut s = world.resource_mut::<ContactShadowSettings>();
        // Frame structure like the rest: it registers the march and it is one
        // more thing that keeps the geometry prepass in the frame.
        ui.checkbox(&mut s.enabled, "Enabled");
        ui.add(
            egui::Slider::new(&mut s.ray_length, 0.01..=2.0)
                .logarithmic(true)
                .text("Length")
                .suffix(" m"),
        )
        .on_hover_text("Longer than a near cascade's texel spends steps on shadows the maps have");
        ui.add(egui::Slider::new(&mut s.steps, 4..=64).text("Steps"))
            .on_hover_text("The shortest occluder the march can find is one step of the ray");
        ui.add(
            egui::Slider::new(&mut s.bias, 0.0..=0.2)
                .logarithmic(true)
                .text("Bias")
                .suffix(" m"),
        )
        .on_hover_text("How far in front of the ray a surface must be to be something else");
        ui.add(
            egui::Slider::new(&mut s.thickness, 0.01..=2.0)
                .logarithmic(true)
                .text("Thickness")
                .suffix(" m"),
        )
        .on_hover_text(
            "How deep a surface is assumed to be past the bias — the buffer records only its front",
        );
        ui.add(egui::Slider::new(&mut s.intensity, 0.0..=1.0).text("Strength"));
        ui.add(
            egui::Slider::new(&mut s.fade_distance, 1.0..=200.0)
                .logarithmic(true)
                .text("Range")
                .suffix(" m"),
        )
        .on_hover_text("Where the band is thinner than a pixel and the march is only noise");
    }
}

fn lighting_column(ui: &mut egui::Ui, world: &World) {
    ui.strong("Tonemap");
    {
        let mut hdr = world.resource_mut::<HdrSettings>();
        // Metering on or off is frame *structure*: it registers or drops the two
        // compute passes, so it recompiles the graph. A checkbox, not something
        // draggable.
        ui.checkbox(&mut hdr.auto_exposure, "Auto exposure")
            .on_hover_text("Meter the frame's luminance and adapt to it");
        ui.add(
            egui::Slider::new(&mut hdr.exposure_compensation, -4.0..=4.0)
                .text("Compensation")
                .suffix(" EV"),
        );
        if hdr.auto_exposure {
            ui.add(
                egui::Slider::new(&mut hdr.adaptation_brighten, 0.0..=4.0)
                    .text("Brighten")
                    .suffix(" s"),
            );
            ui.add(
                egui::Slider::new(&mut hdr.adaptation_darken, 0.0..=4.0)
                    .text("Darken")
                    .suffix(" s"),
            );
            ui.add(egui::Slider::new(&mut hdr.min_log_luminance, -16.0..=0.0).text("Meter floor"));
            ui.add(egui::Slider::new(&mut hdr.max_log_luminance, 0.0..=20.0).text("Meter ceiling"));
        } else {
            ui.add(egui::Slider::new(&mut hdr.manual_ev100, -6.0..=16.0).text("EV100"));
        }
    }
    ui.add_space(6.0);
    ui.strong("Bloom").on_hover_text(
        "Filtered on the exposed image, so the strength holds across lighting changes",
    );
    {
        let mut bloom = world.resource_mut::<BloomSettings>();
        // Like metering, on/off is frame structure: it registers or drops the
        // whole chain, so it recompiles the graph.
        ui.checkbox(&mut bloom.enabled, "Enabled");
        ui.add(egui::Slider::new(&mut bloom.strength, 0.0..=0.3).text("Strength"));
        ui.add(egui::Slider::new(&mut bloom.radius, 0.5..=3.0).text("Radius"));
        ui.add(egui::Slider::new(&mut bloom.scatter, 0.0..=0.95).text("Scatter"))
            .on_hover_text("Weight toward the blurrier levels — how far the glow reaches");
    }
    ui.add_space(6.0);
    ui.strong("Ambient").on_hover_text(
        "Fallback only: an environment's irradiance replaces this when one is loaded",
    );
    {
        let mut ambient = world.resource_mut::<AmbientLight>();
        color_row(ui, "Color", &mut ambient.color);
        ui.add(
            egui::Slider::new(&mut ambient.nits, 0.0..=10_000.0)
                .logarithmic(true)
                .suffix(" cd/m²")
                .text("Luminance"),
        )
        .on_hover_text("Clear zenith a few thousand, overcast under a thousand, dusk tens");
    }
    ui.add_space(6.0);
    ui.strong("Sky");
    {
        let mut env = world.resource_mut::<EnvironmentSettings>();
        ui.horizontal(|ui| {
            ui.label("HDRI");
            // A button rather than applying as you type: the bake blocks on the
            // GPU for every keystroke otherwise, and every prefix of a filename
            // is an error worth reporting exactly once.
            if ui.button("Load").clicked() {
                env.reload_requested = true;
            }
        });
        ui.add(
            egui::TextEdit::singleline(&mut env.hdri)
                .hint_text("relative to assets/")
                .desired_width(f32::INFINITY),
        );
        ui.checkbox(&mut env.show_skybox, "Show skybox");
        // Logarithmic over five orders of magnitude, because that is the range
        // real skies cover and because a downloaded `.hdr` needs a factor in the
        // thousands to reach any of it. The old control was a raw multiplier
        // capped at 4.0, which could not express the calibration a real capture
        // wants at all.
        ui.add(
            egui::Slider::new(&mut env.sky_luminance, 1.0..=50_000.0)
                .logarithmic(true)
                .suffix(" cd/m²")
                .text("Sky"),
        )
        .on_hover_text(
            "How bright this sky is, whatever the file's own numbers are: clear \
             zenith ~8 000, overcast 1 000–2 000, dusk in the tens. Set the sun \
             beside it to match — a daylight capture wants tens of thousands of lux.",
        );
        ui.add(
            egui::Slider::new(&mut env.exposure_offset, -4.0..=4.0)
                .suffix(" EV")
                .text("Offset"),
        )
        .on_hover_text("Stops on top of the calibration. Zero is what the sky above says it is.");
        // Rotates the sampling direction, so it needs no rebake and can be
        // dragged.
        ui.add(egui::Slider::new(&mut env.yaw, -180.0..=180.0).text("Rotation"));
    }
    ui.add_space(6.0);
    ui.strong("Fog");
    let mut fog = world.resource_mut::<FogSettings>();
    color_row(ui, "Albedo", &mut fog.albedo);
    ui.add(
        egui::Slider::new(&mut fog.density, 0.0..=0.1)
            .logarithmic(true)
            .text("Density"),
    )
    .on_hover_text(
        "Extinction per metre at the reference height. 0.005 is a light haze, \
         0.05 is thick enough to lose a building at fifty metres.",
    );
    ui.add(egui::Slider::new(&mut fog.height_falloff, 0.0..=1.0).text("Falloff"));
    ui.add(egui::Slider::new(&mut fog.height, -20.0..=20.0).text("Height"));
    ui.checkbox(&mut fog.volumetric, "Volumetric")
        .on_hover_text(
            "March the first `Distance` metres as a froxel volume, so the shadow \
             maps reach the air and the sun casts shafts. Off, the same medium is \
             integrated analytically along each view ray — smooth, and unshadowed.",
        );
    ui.add_enabled_ui(fog.volumetric, |ui| {
        ui.add(
            egui::Slider::new(&mut fog.distance, 8.0..=256.0)
                .suffix(" m")
                .text("Distance"),
        )
        .on_hover_text(
            "How far the volume reaches. The froxel count is fixed, so this trades \
             reach against resolution and not against cost.",
        );
        ui.add(egui::Slider::new(&mut fog.anisotropy, -0.9..=0.9).text("Anisotropy"))
            .on_hover_text(
                "Henyey-Greenstein g. Positive scatters forward, which is what makes \
                 looking toward the sun through haze so much brighter than looking away.",
            );
        ui.add(egui::Slider::new(&mut fog.feedback, 0.0..=0.98).text("Feedback"))
            .on_hover_text(
                "How much reprojected history each froxel keeps. Low values make the \
                 depth jitter visible as crawling slices.",
            );
    });
}

fn camera_column(ui: &mut egui::Ui, world: &World) {
    ui.strong("Camera");
    let fov_y = {
        let mut cam = world.resource_mut::<Camera>();
        vec3_row(ui, "Position", &mut cam.position, 0.1);
        vec3_row(ui, "Target", &mut cam.target, 0.1);

        let mut fov = cam.fov_y.to_degrees();
        if ui
            .add(egui::Slider::new(&mut fov, 20.0..=110.0).text("FOV"))
            .changed()
        {
            cam.fov_y = fov.to_radians();
        }
        cam.fov_y
    };

    // Under the camera rather than beside the other post-process effects,
    // because that is where their inputs are: the lens takes its focal length
    // from the field of view above, so widening the shot deepens the image
    // exactly as it would on a real one.
    ui.add_space(6.0);
    ui.strong("Lens")
        .on_hover_text("Defocus everything the lens is not focused on");
    {
        let mut dof = world.resource_mut::<DofSettings>();
        // Frame structure, like bloom and TAA: it registers or drops all four
        // passes, so it recompiles the graph.
        ui.checkbox(&mut dof.enabled, "Depth of field");
        ui.add(
            egui::Slider::new(&mut dof.focus_distance, 0.1..=100.0)
                .logarithmic(true)
                .text("Focus")
                .suffix(" m"),
        );
        ui.add(
            egui::Slider::new(&mut dof.f_number, 1.0..=22.0)
                .logarithmic(true)
                .text("Aperture")
                .prefix("f/"),
        )
        .on_hover_text("Larger is a smaller opening, and a deeper image");
        ui.add(
            egui::Slider::new(&mut dof.sensor_height, 5.0..=36.0)
                .text("Sensor")
                .suffix(" mm"),
        )
        .on_hover_text("24mm is full frame, 14.2mm is Super 35");
        // Derived, never set — showing it is what makes the coupling to the FOV
        // slider above legible rather than something to be discovered.
        ui.label(format!("≈{:.0}mm lens", dof.focal_length(fov_y) * 1e3))
            .on_hover_text("Derived from the field of view and the sensor height");
    }

    ui.add_space(6.0);
    ui.strong("Shutter")
        .on_hover_text("Reconstruct the exposure from the frame's motion vectors");
    {
        let mut motion_blur = world.resource_mut::<MotionBlurSettings>();
        ui.checkbox(&mut motion_blur.enabled, "Motion blur");
        ui.add(
            egui::Slider::new(&mut motion_blur.shutter_angle, 0.0..=360.0)
                .text("Angle")
                .suffix("°"),
        )
        .on_hover_text("180° is the film standard: half a frame of travel");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The width this panel is docked at. One column, because four at 75px each
    /// overlap into an unreadable mess — which is exactly what shipped when this
    /// threshold was still tuned for a full-window strip.
    #[test]
    fn the_docked_width_folds_to_one_column() {
        assert_eq!(column_count(300.0), 1);
    }

    /// Widen it — float it, or drag the split out — and the columns come back.
    #[test]
    fn a_wide_tool_spreads_back_out() {
        assert_eq!(column_count(400.0), 2);
        assert_eq!(column_count(600.0), 3);
        assert_eq!(column_count(800.0), 4);
        assert_eq!(column_count(2000.0), 4);
    }

    #[test]
    fn a_sliver_still_yields_a_usable_count() {
        assert_eq!(column_count(20.0), 1);
        assert_eq!(column_count(0.0), 1);
    }
}
