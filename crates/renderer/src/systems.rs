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
pub fn extract_renderables(world: &World) -> Vec<RenderItem> {
    let mut items = Vec::new();
    world
        .query::<(&LocalTransform, &MeshHandle)>()
        .for_each(|entity, (transform, mesh)| {
            // Distinct component type → distinct storage, so this borrow doesn't
            // clash with the query's borrows.
            let material = world
                .get::<MaterialHandle>(entity)
                .map(|m| *m)
                .unwrap_or(MaterialHandle(0));
            items.push(RenderItem {
                model: transform.matrix(),
                mesh: *mesh,
                material,
            });
        });
    items
}

/// Build this frame's [`SceneLighting`] from the world.
///
/// Every entity that has both a [`LocalTransform`] and a [`Light`] contributes:
/// directional lights take their direction from the transform's forward (`-Z`)
/// axis, point lights take their position from the transform's translation. The
/// shader supports one directional "sun", so the first directional light wins;
/// any directional lights and the ambient term fall back to sensible defaults
/// when not supplied.
pub fn extract_lighting(world: &World) -> SceneLighting {
    // Start from the defaults so a scene with no light entities still renders.
    let mut lighting = SceneLighting::default();

    // Ambient is a world-global, so it lives in a resource rather than on an
    // entity. Use it if present, otherwise keep the default fill.
    if let Some(ambient) = world.get_resource::<AmbientLight>() {
        lighting.ambient_color = ambient.color;
        lighting.ambient_intensity = ambient.intensity;
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
                    lighting.sun.direction = direction;
                    lighting.sun.color = color;
                    lighting.sun.intensity = intensity;
                    has_sun = true;
                }
            }
            Light::Point {
                color,
                intensity,
                range,
            } => {
                if lighting.point_lights.len() < MAX_POINT_LIGHTS {
                    lighting.point_lights.push(PointLight {
                        position: transform.translation,
                        color,
                        intensity,
                        range,
                    });
                }
            }
        });

    lighting
}
