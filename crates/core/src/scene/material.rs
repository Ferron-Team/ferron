use crate::gfx::BlendMode;
use crate::scene::MaterialHandle;

/// Which queue every uploaded material draws in, indexed by [`MaterialHandle`].
///
/// The world-side mirror of what the backend was handed at upload, for the
/// reason [`MeshBounds`](crate::scene::MeshBounds) is one: extraction has to
/// split the frame's draw order into the opaque, blended and refractive queues,
/// and it runs with a `World` and no renderer. Uploads only ever append, so a
/// handle's slot never changes meaning.
#[derive(Default)]
pub struct MaterialBlends {
    modes: Vec<BlendMode>,
}

impl MaterialBlends {
    pub fn insert(&mut self, handle: MaterialHandle, mode: BlendMode) {
        let index = handle.0 as usize;
        if self.modes.len() <= index {
            // A gap means a material was uploaded without registering here.
            // Opaque is the safe fill: it draws the surface, where the blended
            // queue would silently swallow one whose alpha nobody set.
            self.modes.resize(index + 1, BlendMode::Opaque);
        }
        self.modes[index] = mode;
    }

    /// [`BlendMode::Opaque`] for a handle that was never registered — a
    /// material nobody described is drawn rather than blended away.
    pub fn get(&self, handle: MaterialHandle) -> BlendMode {
        self.modes
            .get(handle.0 as usize)
            .copied()
            .unwrap_or_default()
    }
}
