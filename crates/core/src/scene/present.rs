/// How a finished frame is handed to the display.
///
/// Named behaviour rather than a `PresentMode`, because nothing in `scene` may
/// name a graphics API — `gfx::vulkan::swapchain` does the mapping, the same
/// rule that keeps [`RenderBackend`](crate::gfx::RenderBackend) implementable
/// twice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VsyncMode {
    /// Capped to the display's refresh, no tearing. The only mode a driver has
    /// to support, so it is also what the other two fall back to.
    #[default]
    Fifo,
    /// Uncapped and tear-free: the presentation engine keeps the newest queued
    /// frame and discards the rest.
    Mailbox,
    /// Uncapped, may tear — and the mode to measure in. It never withholds an
    /// image, so a frame time taken under it is the renderer's own cost rather
    /// than a queue depth.
    Immediate,
}

/// What the swapchain is built with.
///
/// A setting rather than a constant because it is unmeasurable otherwise: every
/// frame-time figure this engine produces is only meaningful next to the present
/// mode it was taken under, and recompiling to change one is how a baseline goes
/// unrecorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentSettings {
    pub vsync: VsyncMode,
    /// How many images the swapchain holds.
    ///
    /// Two is double buffering. Three is what an uncapped mode wants: with two,
    /// the CPU blocks in the acquire as soon as one image is queued for
    /// presentation and the other is being drawn, so the measurement becomes the
    /// presentation engine's turnaround rather than the frame's.
    ///
    /// Clamped to what the surface advertises, so a value it cannot honour falls
    /// back instead of failing swapchain creation.
    pub images: u32,
}

impl Default for PresentSettings {
    fn default() -> Self {
        // Exactly what the engine did when these were constants, so a baseline
        // taken before this became a setting and one taken after are the same
        // measurement. Changing either default is a change to every number
        // anybody has already written down.
        Self {
            vsync: VsyncMode::Fifo,
            images: 2,
        }
    }
}
