//! Screen-space ambient occlusion settings, stored as a world resource.

/// Tunables for the SSAO passes.
///
/// Like [`AmbientLight`](crate::scene::AmbientLight), this is a world-global
/// stored as a resource rather than on an entity. Insert one with
/// [`World::insert_resource`](ferron_ecs::World::insert_resource); if absent, the
/// renderer falls back to these defaults.
#[derive(Clone, Copy, Debug)]
pub struct SsaoSettings {
    /// When false the SSAO passes are skipped entirely and ambient occlusion is
    /// 1.0 (no darkening) everywhere — the scene renders as if SSAO were absent.
    pub enabled: bool,
    /// Hemisphere sample radius in world units. Larger reaches further between
    /// surfaces, spreading the contact darkening.
    pub radius: f32,
    /// Depth bias that fights self-occlusion acne on flat surfaces.
    pub bias: f32,
    /// Contrast applied to the result (`ao = pow(ao, power)`). Higher = darker,
    /// punchier contact shadows.
    pub power: f32,
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 1.0,
            bias: 0.025,
            power: 1.0,
        }
    }
}
