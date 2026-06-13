mod camera;
mod mesh;
mod transform;

pub use camera::Camera;
pub use mesh::CpuMesh;
pub use transform::Transform;

#[derive(Clone, Copy, Debug)]
pub struct MeshHandle(pub u32);

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
