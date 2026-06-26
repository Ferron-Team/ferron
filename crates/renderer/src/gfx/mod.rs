pub mod vulkan;

use crate::scene::{Camera, CpuMesh, MeshHandle};
use glam::{Mat4, Vec3};
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

pub const MAX_POINT_LIGHTS: usize = 16;

/// A sun-like light: parallel rays with a single direction, no position.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    /// The direction the light *travels* (e.g. roughly downward for a sun).
    pub direction: Vec3,
    /// Linear RGB color of the light.
    pub color: Vec3,
    /// Brightness multiplier.
    pub intensity: f32,
}

/// A light that radiates from a point and falls off with distance.
#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    /// Linear RGB color of the light.
    pub color: Vec3,
    /// Brightness multiplier.
    pub intensity: f32,
    /// Distance at which the light's contribution reaches zero.
    pub range: f32,
}

/// This frame's lighting environment, handed to the backend alongside the draw
/// list. Like [`RenderItem`], it's plain data so the backend never touches ECS
/// or scene types — the app builds it via
/// [`systems::extract_lighting`](crate::systems::extract_lighting).
#[derive(Clone, Debug)]
pub struct SceneLighting {
    /// Flat fill light applied everywhere, approximating bounced/sky light.
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    /// The primary directional light.
    pub sun: DirectionalLight,
    /// Extra local lights. Anything past [`MAX_POINT_LIGHTS`] is ignored.
    pub point_lights: Vec<PointLight>,
    /// Blinn-Phong specular exponent. Higher = smaller, sharper highlight.
    pub shininess: f32,
    /// Global multiplier on every light's specular contribution.
    pub specular_strength: f32,
}

impl Default for SceneLighting {
    fn default() -> Self {
        Self {
            // Cool, dim sky fill so shadowed faces aren't pure black.
            ambient_color: Vec3::new(0.6, 0.7, 1.0),
            ambient_intensity: 0.15,
            sun: DirectionalLight {
                direction: Vec3::new(-0.4, -1.0, -0.6).normalize(),
                color: Vec3::new(1.0, 0.97, 0.92),
                intensity: 1.0,
            },
            point_lights: Vec::new(),
            shininess: 32.0,
            specular_strength: 0.4,
        }
    }
}

// The seam between the engine and a concrete graphics API. Implement this for
// other backends (e.g. wgpu, D3D12) without touching scene/app code.
pub trait RenderBackend {
    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle;
    fn resize(&mut self, extent: [u32; 2]);
    fn render(&mut self, items: &[RenderItem], lighting: &SceneLighting, camera: &Camera);
}
