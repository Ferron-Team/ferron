/// What the frame measures about itself.
///
/// Every field here moves a *number* and never a pixel of the scene, which is
/// why they are grouped: a performance figure is comparable only to one taken
/// with the same values, so this is the block to state alongside it. The
/// startup banner prints exactly these, for that reason.
#[derive(Clone, Copy, Debug)]
pub struct Diagnostics {
    /// Whether the editor's egui overlay is built and drawn at all.
    ///
    /// Off takes the whole editor out of the frame: the UI is not run, window
    /// events are not fed to egui, and the graph drops its `overlay` pass. That
    /// is the only honest way to ask what the editor costs — an overlay that
    /// draws an empty window still pays a tessellation, a second queue
    /// submission, and a full-resolution pass over the swapchain image.
    ///
    /// Toggled with **F1**, because a UI that is switched off cannot offer a
    /// checkbox to switch it back on.
    pub overlay: bool,
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self { overlay: true }
    }
}
