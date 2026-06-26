//! Handle to a material owned by the render backend.

/// A lightweight reference to the material/pipeline used to shade an entity.
///
/// The `u32` indexes the backend's material table. The forward pass currently
/// binds a single hard-coded pipeline, so this is a placeholder component until
/// materials are wired into the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialHandle(pub u32);
