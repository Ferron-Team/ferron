use std::sync::Arc;
use std::time::Instant;

use glam::Vec3;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::gfx::vulkan::VulkanRenderer;
use crate::gfx::RenderBackend;
use crate::scene::{Camera, CpuMesh, LocalTransform, Spin, Time};
use crate::systems;
use ferron_ecs::World;

struct Active {
    window: Arc<Window>,
    renderer: VulkanRenderer,
}

pub struct App {
    instance: Arc<Instance>,
    active: Option<Active>,
    world: World,
    start: Instant,
    last_frame: f32,
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
            world: World::default(),
            start: Instant::now(),
            last_frame: 0.0,
        };

        // World-global state lives in resources, not on `App`.
        app.world.insert_resource(Camera::default());
        app.world.insert_resource(Time::new());

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

        // The mesh must be uploaded before we can hand entities a `MeshHandle`.
        let cube = renderer.load_mesh(&CpuMesh::cube());

        let entity = self.world.spawn();
        self.world.insert(entity, LocalTransform::default());
        self.world.insert(entity, cube);
        self.world.insert(entity, Spin::new(Vec3::Y, 1.0));

        self.active = Some(Active { window, renderer });
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
                let elapsed = self.start.elapsed().as_secs_f32();
                let delta = elapsed - self.last_frame;
                self.last_frame = elapsed;
                self.world.resource_mut::<Time>().update(delta);

                // Simulation systems run, then we extract a draw list for the
                // backend — which never sees the ECS world directly.
                systems::spin(&self.world, delta);

                let items = systems::extract_renderables(&self.world);
                let camera = *self.world.resource::<Camera>();
                active.renderer.render(&items, &camera);
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
