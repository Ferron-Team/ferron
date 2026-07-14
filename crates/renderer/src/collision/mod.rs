//! Collision detection: BVH broadphase → shape narrowphase → enter/exit
//! events, with positional (MTV) overlap resolution for solid pairs.
//!
//! [`run`] executes once per frame, after the transform-mutating systems and
//! before the script tick (see `app.rs`), so scripts observe this frame's
//! contacts. It only *writes* events into the [`CollisionState`] resource;
//! delivering them to C# callbacks is the script tick's job — this module has
//! no scripting dependency and compiles without the feature.
//!
//! Conventions (shared with the C# `Collision` struct):
//! - A [`Contact`] normal is unit length and points **from `a` toward `b`**,
//!   where `(a, b)` is the canonical pair order from [`pair_key`].
//! - `depth` is the penetration distance along that normal, `>= 0`.

mod bvh;
mod narrowphase;

use std::collections::HashMap;

use glam::Vec3;

use ferron_ecs::{Entity, World};

use crate::scene::{Collider, ColliderShape, LocalTransform};

pub use bvh::Bvh;

/// A single overlapping contact between two shapes.
#[derive(Clone, Copy, Debug)]
pub struct Contact {
    /// World-space point of contact (center of the overlap region).
    pub point: Vec3,
    /// Unit normal pointing from shape `a` toward shape `b`.
    pub normal: Vec3,
    /// Penetration depth along `normal`, `>= 0`.
    pub depth: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollisionEventKind {
    Enter,
    Exit,
}

/// One enter/exit transition for the pair `(a, b)`; `point`/`normal` follow
/// the [`Contact`] convention (exit events carry the last known contact).
#[derive(Clone, Copy, Debug)]
pub struct CollisionEvent {
    pub kind: CollisionEventKind,
    pub a: Entity,
    pub b: Entity,
    pub point: Vec3,
    pub normal: Vec3,
}

/// World resource: which pairs touched last frame, and this frame's events.
///
/// `events` is refilled by [`run`] each frame and drained by the script tick;
/// without scripting they're simply overwritten next frame.
#[derive(Default)]
pub struct CollisionState {
    touching: HashMap<(Entity, Entity), Contact>,
    pub events: Vec<CollisionEvent>,
}

/// A world-axis-aligned bounding box. The broadphase currency: every collider
/// reduces to one of these, whatever its shape.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// True when the boxes overlap (touching counts as overlapping).
    pub fn overlaps(&self, other: &Aabb) -> bool {
        // TODO(owner): two AABBs overlap iff they overlap on *every* axis —
        // a.min <= b.max && b.min <= a.max, per component. This is the single
        // most load-bearing predicate in the whole system.
        todo!("Aabb::overlaps")
    }

    /// The smallest AABB containing both boxes (the BVH's node bound).
    pub fn union(&self, other: &Aabb) -> Aabb {
        // TODO(owner): component-wise min of mins, max of maxes.
        todo!("Aabb::union")
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}

/// A collider brought into world space, ready for narrowphase.
#[derive(Clone, Copy, Debug)]
pub enum WorldShape {
    Box(Aabb),
    Sphere { center: Vec3, radius: f32 },
}

impl WorldShape {
    /// Conservative world AABB, the broadphase view of this shape.
    pub fn bounds(&self) -> Aabb {
        match *self {
            WorldShape::Box(aabb) => aabb,
            WorldShape::Sphere { center, radius } => Aabb {
                min: center - Vec3::splat(radius),
                max: center + Vec3::splat(radius),
            },
        }
    }
}

/// Bring a collider into world space using its entity's transform.
fn world_shape(transform: &LocalTransform, collider: &Collider) -> WorldShape {
    match collider.shape {
        ColliderShape::Sphere { radius } => {
            // TODO(owner): center is the translation; the radius scales by the
            // *largest* scale component (non-uniform scale can't shrink a
            // sphere without turning it into an ellipsoid — be conservative).
            todo!("world_shape: sphere")
        }
        ColliderShape::Box { half_extents } => {
            // TODO(owner): scale the half extents, then account for rotation.
            // The classic trick for the AABB of a rotated box: world half
            // extents = abs(R) * local_half_extents, where abs(R) is the
            // rotation matrix with every element made non-negative
            // (glam: Mat3::from_quat(rotation), then abs() each column or use
            // Mat3::abs). Center is the translation. Identity rotation should
            // give back exactly translation ± half_extents * scale.
            todo!("world_shape: box")
        }
    }
}

/// Canonical ordering for an unordered entity pair, so `(a, b)` and `(b, a)`
/// hash to the same key. All stored contacts are oriented a → b in this order.
fn pair_key(a: Entity, b: Entity) -> (Entity, Entity) {
    if (a.index, a.generation) <= (b.index, b.generation) {
        (a, b)
    } else {
        (b, a)
    }
}

/// Diff last frame's touching pairs against this frame's, appending Enter
/// events for new pairs and Exit events for vanished ones.
fn diff_pairs(
    previous: &HashMap<(Entity, Entity), Contact>,
    current: &HashMap<(Entity, Entity), Contact>,
    events: &mut Vec<CollisionEvent>,
) {
    // TODO(owner):
    // - Enter: every key in `current` that is not in `previous`, with the
    //   current contact's point/normal.
    // - Exit: every key in `previous` that is not in `current`. There is no
    //   contact this frame, so reuse the *previous* frame's contact — same
    //   trade-off Unity makes for OnCollisionExit. (This also covers pairs
    //   that vanished because an entity despawned; the script tick already
    //   skips dead entities when routing.)
    todo!("diff_pairs")
}

/// Detect collisions and produce this frame's events + positional corrections.
///
/// Call order per frame: transforms settle → `run` → script tick (which drains
/// `CollisionState::events`). Solid–solid pairs are pushed apart immediately;
/// the pair still counts as touching this frame, so a clean separation shows
/// up as an Exit event next frame — matching how impulse engines report it.
pub fn run(world: &mut World) {
    world.resource_mut::<CollisionState>().events.clear();

    struct Body {
        entity: Entity,
        shape: WorldShape,
        is_trigger: bool,
    }

    let mut bodies: Vec<Body> = Vec::new();
    world
        .query::<(&LocalTransform, &Collider)>()
        .for_each(|entity, (transform, collider)| {
            bodies.push(Body {
                entity,
                shape: world_shape(transform, collider),
                is_trigger: collider.is_trigger,
            });
        });

    // Broadphase: candidate index pairs whose AABBs overlap.
    let mut candidates: Vec<(u32, u32)> = Vec::new();
    if bodies.len() >= 2 {
        let bounds: Vec<Aabb> = bodies.iter().map(|body| body.shape.bounds()).collect();
        Bvh::build(&bounds).query_pairs(&mut candidates);
    }

    // Narrowphase: exact shape tests on the survivors. Contacts are stored
    // re-oriented to the canonical pair order so the diff and the resolver
    // never have to guess which way the normal points.
    let mut current: HashMap<(Entity, Entity), Contact> = HashMap::new();
    let mut corrections: Vec<(Entity, Vec3)> = Vec::new();
    for &(i, j) in &candidates {
        let (a, b) = (&bodies[i as usize], &bodies[j as usize]);
        let Some(contact) = narrowphase::test(&a.shape, &b.shape) else {
            continue;
        };

        let key = pair_key(a.entity, b.entity);
        let contact = if key.0 == a.entity {
            contact
        } else {
            Contact { normal: -contact.normal, ..contact }
        };
        current.insert(key, contact);

        if !a.is_trigger && !b.is_trigger {
            let (offset_a, offset_b) = narrowphase::resolve_offsets(&contact);
            corrections.push((key.0, offset_a));
            corrections.push((key.1, offset_b));
        }
    }

    // Apply MTV corrections one entity at a time: `get_mut` borrows the whole
    // LocalTransform storage, so holding two at once would panic the RefCell.
    for (entity, offset) in corrections {
        if let Some(mut transform) = world.get_mut::<LocalTransform>(entity) {
            transform.translation += offset;
        }
    }

    let mut state = world.resource_mut::<CollisionState>();
    let CollisionState { touching, events } = &mut *state;
    if !(touching.is_empty() && current.is_empty()) {
        diff_pairs(touching, &current, events);
    }
    *touching = current;
}
