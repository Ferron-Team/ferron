//! Light sources attached to scene entities.

use glam::Vec3;

/// A light source on an entity.
///
/// Its placement comes from the entity's
/// [`LocalTransform`](crate::scene::LocalTransform), so a light is just an
/// entity you move and rotate like anything else:
/// - a [`Light::Point`] radiates from the transform's translation;
/// - a [`Light::Directional`] shines along the transform's forward (`-Z`) axis.
///
/// The [`extract_lighting`](crate::systems::extract_lighting) system turns these
/// into the renderer's [`SceneLighting`](crate::gfx::SceneLighting) each frame.
/// The current shader supports a single directional "sun", so the first
/// directional light wins; point lights fill up to
/// [`MAX_POINT_LIGHTS`](crate::gfx::MAX_POINT_LIGHTS).
#[derive(Clone, Copy, Debug)]
pub enum Light {
    /// Parallel rays like the sun. Direction is taken from the entity's rotation.
    Directional {
        /// Linear RGB color.
        color: Vec3,
        /// Brightness multiplier.
        intensity: f32,
    },
    /// Radiates from the entity's position, fading to nothing by `range`.
    Point {
        /// Linear RGB color.
        color: Vec3,
        /// Brightness multiplier.
        intensity: f32,
        /// Distance at which the contribution reaches zero.
        range: f32,
    },
}

impl Light {
    /// A directional (sun-like) light.
    #[inline]
    pub fn directional(color: Vec3, intensity: f32) -> Self {
        Self::Directional { color, intensity }
    }

    /// A point light that fades out by `range` world units.
    #[inline]
    pub fn point(color: Vec3, intensity: f32, range: f32) -> Self {
        Self::Point {
            color,
            intensity,
            range,
        }
    }
}

/// Global ambient fill light, stored as a world resource (like `Camera`).
///
/// Approximates bounced/sky light so faces turned away from every light aren't
/// pure black. Insert one with
/// [`World::insert_resource`](ferron_ecs::World::insert_resource); if absent,
/// [`extract_lighting`](crate::systems::extract_lighting) falls back to the
/// [`SceneLighting`](crate::gfx::SceneLighting) default.
#[derive(Clone, Copy, Debug)]
pub struct AmbientLight {
    /// Linear RGB color.
    pub color: Vec3,
    /// Brightness multiplier.
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.15,
        }
    }
}
