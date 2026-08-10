mod scene;
mod stress;
mod textures;

pub use scene::build_default_scene;
pub use stress::{StressSpec, spawn_stress_scene};

use glam::{Quat, Vec3};

use orrin_ecs::{Entity, World};

use crate::scene::{Light, LocalTransform, MaterialHandle, MeshHandle, Name, Transform};

pub fn spawn_mesh(
    world: &mut World,
    name: impl Into<String>,
    transform: Transform,
    mesh: MeshHandle,
    material: MaterialHandle,
) -> Entity {
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(transform))
        .with(mesh)
        .with(material)
        .id()
}

/// `lumens` is the fixture's total luminous power, the number a bulb's box
/// prints: 800 for a domestic bulb, a few thousand for a shop light.
pub fn spawn_point_light(
    world: &mut World,
    name: impl Into<String>,
    position: Vec3,
    color: Vec3,
    lumens: f32,
    range: f32,
) -> Entity {
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(Transform::from_translation(position)))
        .with(Light::point(color, lumens, range))
        .id()
}

/// Like a point light, but coned. The axis is the entity's forward, `-Z`, the
/// same convention `spawn_directional_light` uses, and the angles are half
/// angles from that axis in degrees.
///
/// `lumens` is the same quantity a point light takes, and the spot gets a
/// reflector: the cone concentrates that power rather than masking it, so
/// narrowing `outer_angle` brightens the beam.
pub fn spawn_spot_light(
    world: &mut World,
    name: impl Into<String>,
    position: Vec3,
    direction: Vec3,
    color: Vec3,
    lumens: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
) -> Entity {
    let rotation = Quat::from_rotation_arc(Vec3::NEG_Z, direction.normalize_or_zero());
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(Transform {
            translation: position,
            rotation,
            ..Default::default()
        }))
        .with(Light::spot(color, lumens, range, inner_angle, outer_angle))
        .id()
}

/// The direction is stored as the entity's rotation (forward = `-Z`), so it can
/// be reoriented like any other transform.
///
/// `lux` is the illuminance the light lays on a surface facing it: about
/// 100 000 for noon sun, 20 000 for an overcast day, under 1 000 near sunset.
pub fn spawn_directional_light(
    world: &mut World,
    name: impl Into<String>,
    direction: Vec3,
    color: Vec3,
    lux: f32,
) -> Entity {
    let rotation = Quat::from_rotation_arc(Vec3::NEG_Z, direction.normalize_or_zero());
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(Transform {
            rotation,
            ..Default::default()
        }))
        .with(Light::directional(color, lux))
        .id()
}
