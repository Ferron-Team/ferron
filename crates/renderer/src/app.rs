use std::sync::Arc;
use std::time::Instant;

use glam::{Quat, Vec3};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::gfx::vulkan::VulkanRenderer;
use crate::gfx::RenderBackend;
use crate::scene::{Camera, CpuMesh, LocalTransform, MeshHandle, Scene, Transform};
use ferron_ecs::World;

struct Active {
    window: Arc<Window>,
    renderer: VulkanRenderer,
    cube: MeshHandle,
}

pub struct App {
    instance: Arc<Instance>,
    active: Option<Active>,
    scene: Scene,
    camera: Camera,
    start: Instant,
    world: World,
}

impl App {
    pub fn run() {
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);

        let library = VulkanLibrary::new().expect("failed to load vulkan library");
        let required_extensions = Surface::required_extensions(&event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let mut app = App {
            instance,
            active: None,
            scene: Scene::default(),
            camera: Camera::default(),
            start: Instant::now(),
            world: World::default(),
        };

        app.world.insert_resource(Camera::default());
        event_loop.run_app(&mut app).unwrap();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.active.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("renderer-prototype"))
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let size = window.inner_size();
        let mut renderer =
            VulkanRenderer::new(&self.instance, surface, [size.width, size.height]);

        let cube = renderer.load_mesh(&CpuMesh::cube());

        let e = self.world.spawn();
        self.world.insert(e, LocalTransform::default());
        self.world.insert(e, cube);

        self.scene.objects.clear();
        self.scene.spawn(cube, Transform::default());

        self.active = Some(Active {
            window,
            renderer,
            cube,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(active) = self.active.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                active.renderer.resize([size.width, size.height]);
            }
            WindowEvent::RedrawRequested => {
                let t = self.start.elapsed().as_secs_f32();
                if let Some(object) = self.scene.objects.first_mut() {
                    object.transform.rotation =
                        Quat::from_axis_angle(Vec3::Y, t) * Quat::from_axis_angle(Vec3::X, t * 0.5);
                }
                active.renderer.render(&self.scene, &self.camera);
                let _ = active.cube;
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(active) = self.active.as_ref() {
            active.window.request_redraw();
        }
    }
}
