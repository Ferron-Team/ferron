//! Prefab spawn helpers and default-scene assembly.
//!
//! These are the ergonomic, scene-level way to put things in the world: one call
//! per object instead of a `spawn` followed by a run of `insert`s. They're thin
//! wrappers over [`World::spawn_entity`](ferron_ecs::World::spawn_entity), so an
//! editor's "Add" menu can reuse the exact same helpers it sees here.

mod scene;
mod textures;

pub use scene::build_default_scene;

use glam::{Quat, Vec3};

use ferron_ecs::{Entity, World};

use crate::scene::{Light, LocalTransform, MaterialHandle, MeshHandle, Name, Transform};

/// Spawn a renderable mesh entity: a name, a transform, and a mesh + material.
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

/// Spawn a point light at `position` that fades to nothing by `range`.
pub fn spawn_point_light(
    world: &mut World,
    name: impl Into<String>,
    position: Vec3,
    color: Vec3,
    intensity: f32,
    range: f32,
) -> Entity {
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(Transform::from_translation(position)))
        .with(Light::point(color, intensity, range))
        .id()
}

/// Spawn a directional ("sun") light shining along `direction`.
///
/// The light's direction is stored as the entity's rotation (forward = `-Z`), so
/// it can be reoriented like any other transform.
pub fn spawn_directional_light(
    world: &mut World,
    name: impl Into<String>,
    direction: Vec3,
    color: Vec3,
    intensity: f32,
) -> Entity {
    let rotation = Quat::from_rotation_arc(Vec3::NEG_Z, direction.normalize_or_zero());
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(Transform {
            rotation,
            ..Default::default()
        }))
        .with(Light::directional(color, intensity))
        .id()
}
