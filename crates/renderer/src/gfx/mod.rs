pub mod vulkan;

use crate::scene::{Camera, CpuMesh, MeshHandle};
use glam::Mat4;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex as VertexTrait;

#[derive(BufferContents, VertexTrait, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}

/// One thing to draw this frame: a mesh and the model matrix to draw it with.
///
/// The app builds a slice of these from the ECS world each frame (see
/// [`systems::extract_renderables`](crate::systems::extract_renderables)) and
/// hands them to the backend. Keeping the hand-off a plain slice keeps the
/// backend free of any ECS or scene types.
#[derive(Clone, Copy, Debug)]
pub struct RenderItem {
    pub model: Mat4,
    pub mesh: MeshHandle,
}

// The seam between the engine and a concrete graphics API. Implement this for
// other backends (e.g. wgpu, D3D12) without touching scene/app code.
pub trait RenderBackend {
    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle;
    fn resize(&mut self, extent: [u32; 2]);
    fn render(&mut self, items: &[RenderItem], camera: &Camera);
}
