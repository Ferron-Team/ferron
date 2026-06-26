//! An entity's transform relative to its parent (or the world, if unparented).

use std::ops::{Deref, DerefMut};

use crate::scene::Transform;

/// Position, rotation, and scale of an entity in its local space.
///
/// This is the ECS component form of [`Transform`]. It derefs to the inner
/// `Transform`, so all of its helpers (`matrix`, `translation`, ...) are
/// available directly on a `LocalTransform`. A companion `GlobalTransform`
/// component can be added later once a parent/child hierarchy exists.
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalTransform(pub Transform);

impl LocalTransform {
    #[inline]
    pub fn new(transform: Transform) -> Self {
        Self(transform)
    }
}

impl From<Transform> for LocalTransform {
    #[inline]
    fn from(transform: Transform) -> Self {
        Self(transform)
    }
}

impl Deref for LocalTransform {
    type Target = Transform;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LocalTransform {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
