//! HDR / tonemapping settings, stored as a world resource.

/// Tunables for the HDR tonemap pass.
///
/// Like [`SsaoSettings`](crate::scene::SsaoSettings) and
/// [`AmbientLight`](crate::scene::AmbientLight), this is a world-global stored as
/// a resource. Insert one with
/// [`World::insert_resource`](ferron_ecs::World::insert_resource); if absent, the
/// renderer falls back to these defaults.
#[derive(Clone, Copy, Debug)]
pub struct HdrSettings {
    /// Linear exposure multiplier applied to scene radiance before the ACES
    /// filmic curve. Greater than 1 brightens, less than 1 darkens.
    pub exposure: f32,
}

impl Default for HdrSettings {
    fn default() -> Self {
        Self { exposure: 1.0 }
    }
}
