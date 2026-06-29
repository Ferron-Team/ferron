use std::sync::Arc;
use std::time::Instant;

use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::camera_controller::CameraController;
use crate::editor::Editor;
use crate::gfx::vulkan::VulkanRenderer;
use crate::gfx::{RenderBackend, RenderItem, SceneLighting};
use crate::scene::entities::build_default_scene;
use crate::scene::{AmbientLight, Camera, HdrSettings, SsaoSettings, Time};
use crate::stats::FrameStats;
use crate::systems;
use ferron_ecs::World;

struct Active {
    window: Arc<Window>,
    renderer: VulkanRenderer,
    editor: Editor,
}

pub struct App {
    instance: Arc<Instance>,
    active: Option<Active>,
    world: World,
    camera_controller: CameraController,
    start: Instant,
    last_frame: f32,
    // Reused each frame so a steady scene does no per-frame allocation.
    render_items: Vec<RenderItem>,
    lighting: SceneLighting,
    #[cfg(feature = "scripting")]
    scripting: Option<crate::scripting::Scripting>,
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
            camera_controller: CameraController::new(),
            start: Instant::now(),
            last_frame: 0.0,
            render_items: Vec::new(),
            lighting: SceneLighting::default(),
            #[cfg(feature = "scripting")]
            scripting: None,
        };

        // World-global state lives in resources, not on `App`.
        app.world.insert_resource(Camera::default());
        app.world.insert_resource(Time::new());
        app.world.insert_resource(AmbientLight::default());
        app.world.insert_resource(SsaoSettings::default());
        app.world.insert_resource(HdrSettings::default());
        app.world.insert_resource(FrameStats::new());

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
            VulkanRenderer::new(&self.instance, surface.clone(), [size.width, size.height]);

        build_default_scene(&mut self.world, &mut renderer);
        self.camera_controller
            .sync_from(&self.world.resource::<Camera>());

        // Boot C# scripting and attach the demo Behaviour to the first renderable.
        #[cfg(feature = "scripting")]
        {
            let scripting = crate::scripting::Scripting::boot(std::path::Path::new(
                "scripting/Ferron/bin/Debug/net10.0",
            ));
            if let Some(scripting) = &scripting {
                let mut target = None;
                self.world
                    .query::<&crate::scene::MeshHandle>()
                    .for_each(|entity, _| {
                        if target.is_none() {
                            target = Some(entity);
                        }
                    });
                if let Some(entity) = target {
                    scripting.attach(&mut self.world, entity, "Ferron.Demo.Hover, Ferron");
                }
            }
            self.scripting = scripting;
        }

        let editor = Editor::new(event_loop, surface, renderer.queue(), renderer.color_format());

        self.active = Some(Active {
            window,
            renderer,
            editor,
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

        // The editor sees events first; when it doesn't want one, the camera
        // controller may. Toggling look mode grabs/hides the cursor.
        let egui_wants = active.editor.on_window_event(&event);
        let was_looking = self.camera_controller.looking();
        self.camera_controller.process_window_event(&event, egui_wants);
        if self.camera_controller.looking() != was_looking {
            let looking = self.camera_controller.looking();
            active.window.set_cursor_visible(!looking);
            let grab = if looking {
                CursorGrabMode::Locked
            } else {
                CursorGrabMode::None
            };
            // Locked is unsupported on some platforms; fall back to Confined.
            let _ = active.window.set_cursor_grab(grab).or_else(|_| {
                active.window.set_cursor_grab(if looking {
                    CursorGrabMode::Confined
                } else {
                    CursorGrabMode::None
                })
            });
        }

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
                self.world.resource_mut::<FrameStats>().record(delta);

                // Simulation systems run, then we extract a draw list for the
                // backend — which never sees the ECS world directly.
                systems::spin(&self.world, delta);

                // Tick C# scripts (their OnUpdate may edit components this frame).
                #[cfg(feature = "scripting")]
                if let Some(scripting) = &self.scripting {
                    scripting.tick(&mut self.world, delta);
                }

                // Build the editor UI; it may spawn/despawn/edit entities, so it
                // runs before we extract this frame's draw data.
                active.editor.run(&mut self.world);

                // Apply this frame's camera input (after the UI, so the editor's
                // own camera edits are the baseline the controller builds on).
                self.camera_controller
                    .update(&mut self.world.resource_mut::<Camera>(), delta);

                systems::extract_renderables(&self.world, &mut self.render_items);
                systems::extract_lighting(&self.world, &mut self.lighting);
                let camera = *self.world.resource::<Camera>();
                let ssao = *self.world.resource::<SsaoSettings>();
                let hdr = *self.world.resource::<HdrSettings>();

                // Render the scene, then composite the editor onto the final
                // image before present.
                let Active {
                    renderer, editor, ..
                } = active;
                let mut overlay = |before, image| editor.draw(before, image);
                renderer.render_with_overlay(
                    &self.render_items,
                    &self.lighting,
                    &camera,
                    &ssao,
                    &hdr,
                    &mut overlay,
                );

                // Pull this frame's GPU timing and VRAM from the backend into the
                // stats resource the overlay reads (one frame of latency is fine).
                let gpu_ms = renderer.gpu_frame_ms();
                let (vram_used, vram_total) = renderer.gpu_memory();
                self.world
                    .resource_mut::<FrameStats>()
                    .set_gpu_stats(gpu_ms, vram_used, vram_total);
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // Raw mouse motion drives look mode; the controller ignores it unless the
        // right button is held.
        self.camera_controller.process_device_event(&event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(active) = self.active.as_ref() {
            active.window.request_redraw();
        }
    }
}
