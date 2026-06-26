//! Handle to a GPU mesh owned by the render backend.

/// A lightweight reference to a mesh that has been uploaded to the renderer.
///
/// The `u32` indexes the backend's mesh table; obtain one from
/// [`RenderBackend::load_mesh`](crate::gfx::RenderBackend::load_mesh).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u32);

