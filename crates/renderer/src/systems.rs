//! Systems: free functions that read and mutate the ECS [`World`].
//!
//! Each system takes the world (plus any frame inputs it needs) and runs one
//! query. They're plain functions called in order from the app's frame loop —
//! there's no scheduler yet, which keeps the data flow explicit.

use glam::Vec3;

use ferron_ecs::World;

use crate::gfx::{PointLight, RenderItem, SceneLighting, MAX_POINT_LIGHTS};
use crate::scene::{AmbientLight, Light, LocalTransform, MaterialHandle, MeshHandle, Spin};

/// Advance every entity that has a [`Spin`] by one frame of `dt` seconds.
pub fn spin(world: &World, dt: f32) {
    world
        .query::<(&mut LocalTransform, &Spin)>()
        .for_each(|_entity, (transform, spin)| spin.apply(transform, dt));
}

/// Build this frame's draw list from every entity that has both a
/// [`LocalTransform`] and a [`MeshHandle`].
///
/// This is the bridge from ECS data to the renderer: it produces plain
/// [`RenderItem`]s so the backend never has to know about the world. An
/// entity's [`MaterialHandle`] is optional — meshes without one fall back to
/// `MaterialHandle(0)`, the default material the backend seeds at startup.
pub fn extract_renderables(world: &World, out: &mut Vec<RenderItem>) {
    // Reuse the caller's buffer: `clear` keeps the capacity, so a steady scene
    // does no per-frame heap allocation.
    out.clear();
    world
        .query::<(&LocalTransform, &MeshHandle)>()
        .for_each(|entity, (transform, mesh)| {
            // Distinct component type → distinct storage, so this borrow doesn't
            // clash with the query's borrows.
            let material = world
                .get::<MaterialHandle>(entity)
                .map(|m| *m)
                .unwrap_or(MaterialHandle(0));
            out.push(RenderItem {
                model: transform.matrix(),
                mesh: *mesh,
                material,
            });
        });
}

/// Build this frame's [`SceneLighting`] from the world.
///
/// Every entity that has both a [`LocalTransform`] and a [`Light`] contributes:
/// directional lights take their direction from the transform's forward (`-Z`)
/// axis, point lights take their position from the transform's translation. The
/// shader supports one directional "sun", so the first directional light wins;
/// any directional lights and the ambient term fall back to sensible defaults
/// when not supplied.
pub fn extract_lighting(world: &World, out: &mut SceneLighting) {
    // Reset the scalar fields to defaults each frame so a scene with no light
    // entities still renders, but reuse `point_lights`' allocation via `clear`.
    let defaults = SceneLighting::default();
    out.ambient_color = defaults.ambient_color;
    out.ambient_intensity = defaults.ambient_intensity;
    out.sun = defaults.sun;
    out.shininess = defaults.shininess;
    out.specular_strength = defaults.specular_strength;
    out.point_lights.clear();

    // Ambient is a world-global, so it lives in a resource rather than on an
    // entity. Use it if present, otherwise keep the default fill.
    if let Some(ambient) = world.get_resource::<AmbientLight>() {
        out.ambient_color = ambient.color;
        out.ambient_intensity = ambient.intensity;
    }

    let mut has_sun = false;
    world
        .query::<(&LocalTransform, &Light)>()
        .for_each(|_entity, (transform, light)| match *light {
            Light::Directional { color, intensity } => {
                // First directional light becomes the sun; the rest are ignored
                // until the shader grows support for more.
                if !has_sun {
                    let direction = (transform.rotation * Vec3::NEG_Z).normalize_or_zero();
                    out.sun.direction = direction;
                    out.sun.color = color;
                    out.sun.intensity = intensity;
                    has_sun = true;
                }
            }
            Light::Point {
                color,
                intensity,
                range,
            } => {
                if out.point_lights.len() < MAX_POINT_LIGHTS {
                    out.point_lights.push(PointLight {
                        position: transform.translation,
                        color,
                        intensity,
                        range,
                    });
                }
            }
        });
}
