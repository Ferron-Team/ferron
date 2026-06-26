mod camera;
mod components;
mod mesh;
mod transform;
//mod time;

pub use camera::Camera;
pub use components::{LocalTransform, MaterialHandle, MeshHandle};
pub use mesh::CpuMesh;
pub use transform::Transform;

pub struct RenderObject {
    pub mesh: MeshHandle,
    pub transform: Transform,
}

#[derive(Default)]
pub struct Scene {
    pub objects: Vec<RenderObject>,
}

impl Scene {
    #[inline]
    pub fn spawn(&mut self, mesh: MeshHandle, transform: Transform) -> &mut RenderObject {
        self.objects.push(RenderObject { mesh, transform });
        self.objects.last_mut().unwrap()
    }
}
