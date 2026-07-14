//! Collision volume attached to an entity, in the entity's local space.

use glam::Vec3;

/// The shape of a [`Collider`]. Dimensions are local-space and get scaled by
/// the entity's transform when the collision system computes world bounds.
#[derive(Clone, Copy, Debug)]
pub enum ColliderShape {
    /// A box centered on the entity. Narrowphase treats boxes as world-space
    /// AABBs, so a rotated entity gets a conservatively enlarged volume rather
    /// than a true OBB test.
    Box { half_extents: Vec3 },
    Sphere { radius: f32 },
}

/// Makes an entity participate in collision detection (`collision::run`).
///
/// `is_trigger` colliders fire the same enter/exit events as solid ones but
/// are exempt from overlap resolution — nothing gets pushed out of a trigger,
/// and a trigger is never pushed.
#[derive(Clone, Copy, Debug)]
pub struct Collider {
    pub shape: ColliderShape,
    pub is_trigger: bool,
}
