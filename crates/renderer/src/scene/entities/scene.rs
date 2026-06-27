//! Builds the default demo scene: assets are loaded through the backend and
//! registered by name in [`Assets`], then entities are spawned via the prefab
//! helpers. This is the one place that knows the concrete starting scene; swap
//! it (or call the `spawn_*` helpers yourself) to build a different world.

use glam::Vec3;

use ferron_ecs::World;

use super::textures::{bump_normals, checkerboard, load_rgba, metallic_roughness};
use super::{spawn_directional_light, spawn_mesh, spawn_point_light};
use crate::gfx::{Material, RenderBackend};
use crate::scene::{Assets, Camera, CpuMesh, Spin, Transform};

/// The scene spawns `GRID * GRID` cubes. Set to 1 for a single cube.
const GRID: i32 = 10;
/// World-space spacing between adjacent cubes.
const SPACING: f32 = 2.0;

/// Load assets into `backend`, register them in an [`Assets`] resource, and
/// populate `world` with the default scene (a spinning textured grid, a ground
/// plane, a sun, and a few point lights) framed by the camera.
pub fn build_default_scene(world: &mut World, backend: &mut impl RenderBackend) {
    let assets = load_assets(backend);

    // Handles to spawn with; also stored in the `Assets` resource below.
    let cube = assets.mesh("cube").unwrap();
    let plane = assets.mesh("plane").unwrap();
    let textured = assets.material("textured").unwrap();
    let rock = assets.material("rock").unwrap();
    let ground_material = assets.material("ground").unwrap();
    let palette: Vec<_> = ["gold", "copper", "glossy", "clay", "neon"]
        .into_iter()
        .map(|name| assets.material(name).unwrap())
        .collect();

    world.insert_resource(assets);

    let half = (GRID - 1) as f32 * SPACING * 0.5;
    let mut index = 0;
    for x in 0..GRID {
        for z in 0..GRID {
            let pos = Vec3::new(x as f32 * SPACING - half, 0.0, z as f32 * SPACING - half);

            let material = match (x + z) % 3 {
                0 => rock,
                1 => textured,
                _ => palette[(x + z) as usize % palette.len()],
            };

            let entity = spawn_mesh(
                world,
                format!("Cube {index}"),
                Transform::from_translation(pos),
                cube,
                material,
            );
            // Vary spin speed a little so the field isn't perfectly uniform.
            let speed = 0.5 + ((x + z) % 5) as f32 * 0.4;
            world.insert(entity, Spin::new(Vec3::Y, speed));
            index += 1;
        }
    }

    // A flattened cube the grid sits on, giving SSAO real contact surfaces to
    // darken (the floating grid alone barely shows it). The real `plane` mesh is
    // registered for the editor, but the demo floor stays a cube so output is
    // unchanged.
    let _ = plane;
    spawn_mesh(
        world,
        "Ground",
        Transform {
            translation: Vec3::new(0.0, -0.75, 0.0),
            scale: Vec3::new(GRID as f32 * SPACING * 1.5, 0.5, GRID as f32 * SPACING * 1.5),
            ..Default::default()
        },
        cube,
        ground_material,
    );

    let sun_dir = Vec3::new(-0.4, -1.0, -0.6).normalize();
    spawn_directional_light(world, "Sun", sun_dir, Vec3::new(1.0, 0.97, 0.92), 1.0);

    for (i, (pos, color)) in [
        (Vec3::new(-4.0, 3.0, -4.0), Vec3::new(1.0, 0.35, 0.1)), // warm
        (Vec3::new(4.0, 3.0, 4.0), Vec3::new(0.2, 0.5, 1.0)),    // cool
        (Vec3::new(4.0, 3.0, -4.0), Vec3::new(0.2, 1.0, 0.4)),   // green
    ]
    .into_iter()
    .enumerate()
    {
        spawn_point_light(world, format!("Point Light {i}"), pos, color, 8.0, 10.0);
    }

    // Pull the camera back so the whole field is in frame.
    let span = GRID as f32 * SPACING;
    world.insert_resource(Camera {
        position: Vec3::new(0.0, span * 0.6, span * 1.1),
        target: Vec3::ZERO,
        ..Camera::default()
    });
}

/// Upload every mesh, material, and texture the scene needs and return a name →
/// handle [`Assets`] registry. The `plane`/`sphere` primitives are registered
/// even though the demo doesn't place them, so an editor can spawn them.
fn load_assets(backend: &mut impl RenderBackend) -> Assets {
    let mut assets = Assets::new();

    // --- meshes -----------------------------------------------------------
    assets.insert_mesh("cube", backend.load_mesh(&CpuMesh::cube()));
    assets.insert_mesh("plane", backend.load_mesh(&CpuMesh::plane()));
    assets.insert_mesh("sphere", backend.load_mesh(&CpuMesh::sphere(32, 16)));

    // --- solid material palette ------------------------------------------
    // A spread across the metallic-roughness range so the PBR BRDF is visible.
    let palette = [
        // Polished gold: full metal, tight highlight.
        ("gold", Material {
            base_color: Vec3::new(1.0, 0.84, 0.40),
            metallic: 1.0,
            roughness: 0.18,
            ..Material::default()
        }),
        // Brushed copper: metal, broader highlight.
        ("copper", Material {
            base_color: Vec3::new(0.95, 0.64, 0.54),
            metallic: 1.0,
            roughness: 0.45,
            ..Material::default()
        }),
        // Glossy dielectric: no metal, sharp specular over a diffuse base.
        ("glossy", Material {
            base_color: Vec3::new(0.9, 0.9, 0.95),
            metallic: 0.0,
            roughness: 0.12,
            reflectance: 0.7,
            ..Material::default()
        }),
        // Matte clay: rough dielectric, mostly diffuse.
        ("clay", Material {
            base_color: Vec3::splat(0.8),
            metallic: 0.0,
            roughness: 0.85,
            ..Material::default()
        }),
        // Neon: emissive.
        ("neon", Material {
            base_color: Vec3::splat(0.4),
            metallic: 0.0,
            roughness: 0.0,
            reflectance: 0.0,
            emissive: Vec3::new(8.0, 8.0, 8.0),
            ..Material::default()
        }),
    ];
    for (name, material) in palette {
        assets.insert_material(name, backend.load_material(&material));
    }

    // --- procedural textures ---------------------------------------------
    // Raw RGBA generated on the CPU, wired into one material. Swapping these for
    // files only changes how the bytes are produced.
    let tex = 256;
    let albedo = backend.load_texture(
        &checkerboard(tex, 8, [220, 60, 50], [240, 240, 245]),
        tex,
        tex,
        true, // color map: sRGB
    );
    let normal = backend.load_texture(&bump_normals(tex, 6.0, 1.5), tex, tex, false);
    let metal_rough = backend.load_texture(&metallic_roughness(tex), tex, tex, false);
    assets.insert_texture("proc_albedo", albedo);
    assets.insert_texture("proc_normal", normal);
    assets.insert_texture("proc_metal_rough", metal_rough);
    assets.insert_material(
        "textured",
        backend.load_material(&Material {
            base_color: Vec3::ONE,
            metallic: 1.0,  // scaled by the metallic-roughness map
            roughness: 1.0, // scaled by the metallic-roughness map
            albedo_texture: Some(albedo),
            normal_texture: Some(normal),
            metallic_roughness_texture: Some(metal_rough),
            ..Material::default()
        }),
    );

    // --- file-backed rock material ---------------------------------------
    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_Color.jpg"));
    let rock_albedo = backend.load_texture(&px, w, h, true);
    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_NormalDX.jpg"));
    let rock_normal = backend.load_texture(&px, w, h, false);
    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_Roughness.jpg"));
    let rock_rough = backend.load_texture(&px, w, h, false);
    assets.insert_texture("rock_albedo", rock_albedo);
    assets.insert_texture("rock_normal", rock_normal);
    assets.insert_texture("rock_rough", rock_rough);
    assets.insert_material(
        "rock",
        backend.load_material(&Material {
            base_color: Vec3::ONE,
            metallic: 0.0,  // no metallic map; rock is a dielectric
            roughness: 1.0, // driven by the roughness map (green channel)
            albedo_texture: Some(rock_albedo),
            normal_texture: Some(rock_normal),
            metallic_roughness_texture: Some(rock_rough),
            ..Material::default()
        }),
    );

    // --- ground material --------------------------------------------------
    assets.insert_material(
        "ground",
        backend.load_material(&Material {
            base_color: Vec3::splat(0.7),
            metallic: 0.0,
            roughness: 0.9,
            ..Material::default()
        }),
    );

    assets
}
