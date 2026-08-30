/// Frustum-culling switch, and the counts the last extraction produced.
///
/// The counts live with the switch rather than in `FrameStats` because they only
/// mean anything together: "412 of 1000" is a measurement of this pass, and the
/// switch is how you check that the 588 it dropped really were off screen.
#[derive(Clone, Copy, Debug)]
pub struct Culling {
    /// When false every renderable is drawn, whatever the frustum says. The A/B
    /// for "is this missing object a culling bug?".
    pub enabled: bool,
    /// `None` when the frustum test ran on the GPU, which is where the answer
    /// then is — the CPU offered every opaque renderable to the dispatch and
    /// was never told which survived. A number that would otherwise read as
    /// "nothing was culled", which is the one wrong thing this panel could say.
    visible: Option<usize>,
    total: usize,
}

impl Default for Culling {
    fn default() -> Self {
        Self {
            enabled: true,
            visible: None,
            total: 0,
        }
    }
}

impl Culling {
    pub fn record(&mut self, visible: Option<usize>, total: usize) {
        self.visible = visible;
        self.total = total;
    }

    pub fn visible(&self) -> Option<usize> {
        self.visible
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn culled(&self) -> Option<usize> {
        self.visible
            .map(|visible| self.total.saturating_sub(visible))
    }
}
