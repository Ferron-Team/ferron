use std::io::Write;
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
use crate::gfx::{Material, RenderBackend};
use crate::scene::{
    AmbientLight, Camera, CpuMesh, HdrSettings, Light, LocalTransform, Spin, SsaoSettings, Time,
    Transform,
};
use crate::systems;
use ferron_ecs::World;

/// The scene spawns `GRID * GRID` cubes. Set to 1 for a single cube.
const GRID: i32 = 10;
/// World-space spacing between adjacent cubes.
const SPACING: f32 = 2.0;

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
    // FPS counter: frames and elapsed time accumulated over the current window.
    fps_accum: f32,
    fps_frames: u32,
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
            fps_accum: 0.0,
            fps_frames: 0,
        };

        // World-global state lives in resources, not on `App`.
        app.world.insert_resource(Camera::default());
        app.world.insert_resource(Time::new());
        app.world.insert_resource(AmbientLight::default());
        app.world.insert_resource(SsaoSettings::default());
        app.world.insert_resource(HdrSettings::default());

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

        // A small palette spanning the metallic-roughness range so the PBR BRDF
        // is visible across the field. `load_material` returns a `MaterialHandle`
        // that doubles as the shader's index into the material table.
        let materials = [
            // Polished gold: full metal, tight highlight.
            Material {
                base_color: Vec3::new(1.0, 0.84, 0.40),
                metallic: 1.0,
                roughness: 0.18,
                ..Material::default()
            },
            // Brushed copper: metal, broader highlight.
            Material {
                base_color: Vec3::new(0.95, 0.64, 0.54),
                metallic: 1.0,
                roughness: 0.45,
                ..Material::default()
            },
            // Glossy dielectric: no metal, sharp specular over a diffuse base.
            Material {
                base_color: Vec3::new(0.9, 0.9, 0.95),
                metallic: 0.0,
                roughness: 0.12,
                reflectance: 0.7,
                ..Material::default()
            },
            // Matte clay: rough dielectric, mostly diffuse.
            Material {
                base_color: Vec3::splat(0.8),
                metallic: 0.0,
                roughness: 0.85,
                ..Material::default()
            },
            // Neon
            Material {
                base_color: Vec3::splat(0.4),
                metallic: 0.0,
                roughness: 0.0,
                reflectance: 0.0,
                emissive: Vec3::new(8.0, 8.0, 8.0),
                ..Material::default()
            },
        ]
        .map(|material| renderer.load_material(&material));

        // Procedural textures (raw RGBA, generated below) wired into one material.
        // Swapping these for files only changes how the bytes are produced.
        let tex = 256;
        let albedo = renderer.load_texture(
            &checkerboard(tex, 8, [220, 60, 50], [240, 240, 245]),
            tex,
            tex,
            true, // color map: sRGB
        );
        let normal = renderer.load_texture(&bump_normals(tex, 6.0, 1.5), tex, tex, false);
        let metal_rough = renderer.load_texture(&metallic_roughness(tex), tex, tex, false);
        let textured = renderer.load_material(&Material {
            base_color: Vec3::ONE,
            metallic: 1.0,  // scaled by the metallic-roughness map
            roughness: 1.0, // scaled by the metallic-roughness map
            albedo_texture: Some(albedo),
            normal_texture: Some(normal),
            metallic_roughness_texture: Some(metal_rough),
            ..Material::default()
        });

        let (px, w, h) = load_rgba(include_bytes!("assets/Rocks016_1K-JPG_Color.jpg"));
        let rock_albedo = renderer.load_texture(&px, w, h, true);
        let (px, w, h) = load_rgba(include_bytes!("assets/Rocks016_1K-JPG_NormalDX.jpg"));
        let rock_normal = renderer.load_texture(&px, w, h, false);
        let (px, w, h) = load_rgba(include_bytes!("assets/Rocks016_1K-JPG_Roughness.jpg"));
        let rock_rough = renderer.load_texture(&px, w, h, false);
        let rock = renderer.load_material(&Material {
            base_color: Vec3::ONE,
            metallic: 0.0,  // no metallic map; rock is a dielectric
            roughness: 1.0, // driven by the roughness map (green channel)
            albedo_texture: Some(rock_albedo),
            normal_texture: Some(rock_normal),
            metallic_roughness_texture: Some(rock_rough),
            ..Material::default()
        });

        let half = (GRID - 1) as f32 * SPACING * 0.5;
        for x in 0..GRID {
            for z in 0..GRID {
                let pos = Vec3::new(x as f32 * SPACING - half, 0.0, z as f32 * SPACING - half);
                let transform = LocalTransform::from(Transform::from_translation(pos));

                // Vary spin speed a little so the field isn't perfectly uniform.
                let speed = 0.5 + ((x + z) % 5) as f32 * 0.4;

                // Cycle the rock-textured, procedurally-textured, and solid
                // palette materials across the grid.
                let material = match (x + z) % 3 {
                    0 => rock,
                    1 => textured,
                    _ => materials[(x + z) as usize % materials.len()],
                };

                let entity = self.world.spawn();
                self.world.insert(entity, transform);
                self.world.insert(entity, cube);
                self.world.insert(entity, material);
                self.world.insert(entity, Spin::new(Vec3::Y, speed));
            }
        }

        // Ground plane: a flattened cube the grid sits on, giving SSAO real
        // contact surfaces to darken (the floating grid alone barely shows it).
        let ground_material = renderer.load_material(&Material {
            base_color: Vec3::splat(0.7),
            metallic: 0.0,
            roughness: 0.9,
            ..Material::default()
        });
        let ground = self.world.spawn();
        self.world.insert(
            ground,
            LocalTransform::from(Transform {
                translation: Vec3::new(0.0, -0.75, 0.0),
                scale: Vec3::new(GRID as f32 * SPACING * 1.5, 0.5, GRID as f32 * SPACING * 1.5),
                ..Default::default()
            }),
        );
        self.world.insert(ground, cube);
        self.world.insert(ground, ground_material);

        // Lights are ordinary entities: a transform plus a `Light`. The sun's
        // direction comes from its rotation (forward = -Z); point lights sit at
        // their transform's translation.
        let sun_dir = Vec3::new(-0.4, -1.0, -0.6).normalize();
        let sun = self.world.spawn();
        self.world.insert(
            sun,
            LocalTransform::from(Transform {
                rotation: Quat::from_rotation_arc(Vec3::NEG_Z, sun_dir),
                ..Default::default()
            }),
        );
        self.world
            .insert(sun, Light::directional(Vec3::new(1.0, 0.97, 0.92), 1.0));

        // A few colored point lights hovering over the field to show off falloff.
        for (pos, color) in [
            (Vec3::new(-4.0, 3.0, -4.0), Vec3::new(1.0, 0.35, 0.1)), // warm
            (Vec3::new(4.0, 3.0, 4.0), Vec3::new(0.2, 0.5, 1.0)),    // cool
            (Vec3::new(4.0, 3.0, -4.0), Vec3::new(0.2, 1.0, 0.4)),   // green
        ] {
            let light = self.world.spawn();
            self.world
                .insert(light, LocalTransform::from(Transform::from_translation(pos)));
            self.world.insert(light, Light::point(color, 8.0, 10.0));
        }

        // Pull the camera back so the whole field is in frame.
        let span = GRID as f32 * SPACING;
        *self.world.resource_mut::<Camera>() = Camera {
            position: Vec3::new(0.0, span * 0.6, span * 1.1),
            target: Vec3::ZERO,
            ..Camera::default()
        };

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
                let lighting = systems::extract_lighting(&self.world);
                let camera = *self.world.resource::<Camera>();
                let ssao = *self.world.resource::<SsaoSettings>();
                let hdr = *self.world.resource::<HdrSettings>();
                active.renderer.render(&items, &lighting, &camera, &ssao, &hdr);

                // Average FPS over ~1s windows
                self.fps_accum += delta;
                self.fps_frames += 1;
                if self.fps_accum >= 1.0 {
                    let fps = self.fps_frames as f32 / self.fps_accum;
                    print!("\rFPS: {fps:6.1}  ({:5.2} ms/frame)", 1000.0 / fps);
                    let _ = std::io::stdout().flush();
                    self.fps_accum = 0.0;
                    self.fps_frames = 0;
                }
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

/// Decode an encoded image (PNG/JPG/…) into row-major RGBA8 + its dimensions,
/// the form `RenderBackend::load_texture` expects.
fn load_rgba(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(bytes)
        .expect("failed to decode texture")
        .to_rgba8();
    let (width, height) = img.dimensions();
    (img.into_raw(), width, height)
}

// --- Procedural textures -----------------------------------------------------

/// Two-color checkerboard with `checks` cells per axis (a color/albedo map).
fn checkerboard(size: u32, checks: u32, a: [u8; 3], b: [u8; 3]) -> Vec<u8> {
    let cell = (size / checks).max(1);
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let c = if ((x / cell) + (y / cell)).is_multiple_of(2) { a } else { b };
            data.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    data
}

/// Tangent-space normal map of a grid of rounded bumps, encoded as (n*0.5+0.5).
fn bump_normals(size: u32, freq: f32, strength: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32 * std::f32::consts::TAU * freq;
            let v = y as f32 / size as f32 * std::f32::consts::TAU * freq;
            // Slope is the gradient of the height field h = sin(u) * sin(v).
            let dx = strength * u.cos() * v.sin();
            let dy = strength * u.sin() * v.cos();
            let n = Vec3::new(-dx, -dy, 1.0).normalize();
            let e = (n * 0.5 + 0.5) * 255.0;
            data.extend_from_slice(&[e.x as u8, e.y as u8, e.z as u8, 255]);
        }
    }
    data
}

/// Metallic-roughness map (glTF convention: G = roughness, B = metallic).
/// Roughness ramps left→right; metallic alternates in vertical bands.
fn metallic_roughness(size: u32) -> Vec<u8> {
    let band = (size / 8).max(1);
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for _ in 0..size {
        for x in 0..size {
            let roughness = (x * 255 / size.max(1)) as u8;
            let metallic = if (x / band).is_multiple_of(2) { 255 } else { 0 };
            data.extend_from_slice(&[0, roughness, metallic, 255]);
        }
    }
    data
}
