use glam::Vec3;

use orrin_ecs::World;

use super::textures::{bump_normals, checkerboard, load_rgba, metallic_roughness, sky_equirect};
use super::{spawn_directional_light, spawn_mesh, spawn_point_light, spawn_spot_light};
use crate::gfx::{BlendMode, Material, RenderBackend};
use crate::scene::{Assets, Camera, CpuMesh, MaterialBlends, MeshBounds, Spin, Transform};

const GRID: i32 = 10;
const SPACING: f32 = 2.0;

/// Spheres per roughness row. Five puts a sample at 0, 0.25, 0.5, 0.75 and 1,
/// which covers the prefiltered specular chain's six levels closely enough that
/// a bad level shows up as one sphere that does not belong in the row.
const SWEEP: usize = 5;

pub fn build_default_scene(world: &mut World, backend: &mut impl RenderBackend) {
    let (assets, mesh_bounds, material_blends) = load_assets(backend);

    let cube = assets.mesh("cube").unwrap();
    let plane = assets.mesh("plane").unwrap();
    let textured = assets.material("textured").unwrap();
    let rock = assets.material("rock").unwrap();
    let ground_material = assets.material("ground").unwrap();
    let glass = assets.material("glass").unwrap();
    let palette: Vec<_> = ["gold", "copper", "glossy", "clay", "neon"]
        .into_iter()
        .map(|name| assets.material(name).unwrap())
        .collect();
    let sphere = assets.mesh("sphere").unwrap();
    let sweep: Vec<_> = (0..SWEEP)
        .map(|step| {
            let percent = step * 100 / (SWEEP - 1);
            (
                assets.material(&format!("metal_{percent:03}")).unwrap(),
                assets
                    .material(&format!("dielectric_{percent:03}"))
                    .unwrap(),
            )
        })
        .collect();

    world.insert_resource(assets);
    world.insert_resource(mesh_bounds);
    world.insert_resource(material_blends);

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
            let speed = 0.5 + ((x + z) % 5) as f32 * 0.4;
            world.insert(entity, Spin::new(Vec3::Y, speed));
            index += 1;
        }
    }

    // A flattened cube gives SSAO real contact surfaces to darken; the floating
    // grid alone barely shows it. The real `plane` mesh is registered for the
    // editor, but the demo floor stays a cube so output is unchanged.
    let _ = plane;
    spawn_mesh(
        world,
        "Ground",
        Transform {
            translation: Vec3::new(0.0, -0.75, 0.0),
            scale: Vec3::new(
                GRID as f32 * SPACING * 1.5,
                0.5,
                GRID as f32 * SPACING * 1.5,
            ),
            ..Default::default()
        },
        cube,
        ground_material,
    );

    // In front of the grid and clear of the ground, so each sphere sees sky,
    // ground and cubes at once — the three things a reflection has to get
    // right, and the fastest way to spot one that does not.
    let stride = 3.0;
    let offset = (SWEEP - 1) as f32 * stride * 0.5;
    for (step, (metal, dielectric)) in sweep.iter().enumerate() {
        let x = step as f32 * stride - offset;
        let percent = step * 100 / (SWEEP - 1);
        for (name, material, z) in [
            (format!("Metal {percent}%"), *metal, 11.0),
            (format!("Dielectric {percent}%"), *dielectric, 14.0),
        ] {
            spawn_mesh(
                world,
                name,
                Transform {
                    translation: Vec3::new(x, 2.5, z),
                    scale: Vec3::splat(2.0),
                    ..Default::default()
                },
                sphere,
                material,
            );
        }
    }

    // Three panes at different depths, deliberately intersecting. Two things to
    // look for: the crossings have no seam, because nothing sorted them; and
    // nothing pops as the camera orbits, because there is no order to flip.
    for (index, (x, z, yaw)) in [
        (-3.0f32, 6.0f32, 0.0f32),
        (0.0, 7.5, 25.0),
        (3.0, 6.5, -20.0),
    ]
    .into_iter()
    .enumerate()
    {
        spawn_mesh(
            world,
            format!("Glass {index}"),
            Transform {
                translation: Vec3::new(x, 2.5, z),
                rotation: glam::Quat::from_rotation_y(yaw.to_radians()),
                scale: Vec3::new(4.0, 4.0, 0.1),
            },
            cube,
            glass,
        );
    }

    // A row of the four extra lobes. Spheres, for the reason the roughness sweep
    // uses them: a lobe is a shape in angle, and a flat face samples one angle.
    //
    // Floating above the sweep rather than beside them, and that is not
    // decoration. At ground level this row sat behind two rows of
    // radius-2 spheres and was almost entirely hidden — a readout nobody can see
    // is not a readout. Up here each one is against the cube grid, which is what
    // the two refractive ones need behind them to refract at all, and clear of
    // the sweep's spheres by more than the two radii so nothing intersects.
    let stride = 2.9;
    let offset = 2.0 * stride;
    for (index, name) in ["clearcoat", "velvet", "brushed", "crystal", "frosted"]
        .into_iter()
        .enumerate()
    {
        let material = world.resource::<Assets>().material(name).unwrap();
        spawn_mesh(
            world,
            name,
            Transform {
                translation: Vec3::new(index as f32 * stride - offset, 6.0, 13.0),
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::splat(1.2),
            },
            sphere,
            material,
        );
    }

    let sun_dir = Vec3::new(-0.4, -1.0, -0.6).normalize();
    // Deep twilight, and chosen for that rather than inherited. Physical units put
    // the sun, the sky and every fixture on one scale, and under a 100 000 lux
    // noon sun a real lamp lands a fraction of a percent above the ambient — which
    // is correct, and is also a demo where the punctual lights, their shadows and
    // the emissive material are all invisible. Lighting the scene at the hour when
    // a sun, a lamp and a neon tube are within a few stops of each other is what
    // keeps every feature on screen without any of them lying about its units.
    spawn_directional_light(world, "Sun", sun_dir, Vec3::new(1.0, 0.97, 0.92), 250.0);

    // Fed the direction *toward* the sun, so the disc in the sky and the
    // directional light that casts the shadows agree.
    const SKY: [u32; 2] = [1024, 512];
    backend.load_environment(&sky_equirect(SKY[0], SKY[1], -sun_dir), SKY[0], SKY[1]);

    for (i, (pos, color)) in [
        (Vec3::new(-4.0, 3.0, -4.0), Vec3::new(1.0, 0.35, 0.1)),
        (Vec3::new(4.0, 3.0, 4.0), Vec3::new(0.2, 0.5, 1.0)),
        (Vec3::new(4.0, 3.0, -4.0), Vec3::new(0.2, 1.0, 0.4)),
    ]
    .into_iter()
    .enumerate()
    {
        // 8 000 lumens is a large architectural fixture rather than a bulb,
        // because these sit three metres from what they light: through 4pi
        // steradian and an inverse square, that is about 70 lux where it lands,
        // a quarter of the sun. Inverse-square is unforgiving about distance in
        // a way the old unitless 3.0 hid.
        spawn_point_light(world, format!("Point Light {i}"), pos, color, 8000.0, 10.0);
    }

    // Aimed down at the grid from one corner, which is where a cone's shadow
    // reads: the cubes are far enough apart that each throws its own onto the
    // floor rather than into its neighbour.
    spawn_spot_light(
        world,
        "Spot Light",
        Vec3::new(-6.0, 9.0, 6.0),
        Vec3::new(0.55, -1.0, -0.55).normalize(),
        Vec3::new(1.0, 0.9, 0.75),
        // Nine metres up, so the beam arrives at roughly the sun's own
        // illuminance and reads as a lit cone rather than a clipped highlight — a
        // shadow inside a blown-out region is a shadow nobody can see. The
        // reflector is what makes 10 000 lumens reach that far: the 32 degree cone
        // below concentrates them into about 10 000 candela, where the same power
        // from a bare bulb would be 800.
        10_000.0,
        30.0,
        22.0,
        32.0,
    );

    let span = GRID as f32 * SPACING;
    world.insert_resource(Camera {
        position: Vec3::new(0.0, span * 0.6, span * 1.1),
        target: Vec3::ZERO,
        ..Camera::default()
    });
}

fn load_assets(backend: &mut impl RenderBackend) -> (Assets, MeshBounds, MaterialBlends) {
    let mut assets = Assets::new();
    let mut bounds = MeshBounds::default();
    let mut blends = MaterialBlends::default();

    load_mesh(backend, &mut assets, &mut bounds, "cube", &CpuMesh::cube());
    load_mesh(
        backend,
        &mut assets,
        &mut bounds,
        "plane",
        &CpuMesh::plane(),
    );
    load_mesh(
        backend,
        &mut assets,
        &mut bounds,
        "sphere",
        &CpuMesh::sphere(32, 16),
    );

    // A spread across the metallic-roughness range so the PBR BRDF is visible.
    let palette = [
        (
            "gold",
            Material {
                base_color: Vec3::new(1.0, 0.84, 0.40),
                metallic: 1.0,
                roughness: 0.18,
                ..Material::default()
            },
        ),
        (
            "copper",
            Material {
                base_color: Vec3::new(0.95, 0.64, 0.54),
                metallic: 1.0,
                roughness: 0.45,
                ..Material::default()
            },
        ),
        (
            "glossy",
            Material {
                base_color: Vec3::new(0.9, 0.9, 0.95),
                metallic: 0.0,
                roughness: 0.12,
                reflectance: 0.7,
                ..Material::default()
            },
        ),
        (
            "clay",
            Material {
                base_color: Vec3::splat(0.8),
                metallic: 0.0,
                roughness: 0.85,
                ..Material::default()
            },
        ),
        (
            "neon",
            Material {
                base_color: Vec3::splat(0.4),
                metallic: 0.0,
                roughness: 0.0,
                reflectance: 0.0,
                // Nits, like the frame, because emission is a luminance and the
                // frame is measured in luminance. The lit floor sits near 80, so
                // this is three stops above white: bright enough to feed the bloom
                // chain, not so bright that it is the only thing metering sees.
                emissive: Vec3::splat(600.0),
                ..Material::default()
            },
        ),
    ];
    for (name, material) in palette {
        load_material(backend, &mut assets, &mut blends, name, &material);
    }

    // A roughness sweep at both ends of the metallic range: the readout for
    // image-based lighting. A cube face samples essentially one direction and
    // says almost nothing about a reflection; a sphere shows the whole
    // environment at once, and a row of them shows every level of the
    // prefiltered chain side by side. Neutral base colours on purpose — a
    // tinted metal hides in its own colour what the environment is doing.
    for step in 0..SWEEP {
        let roughness = step as f32 / (SWEEP - 1) as f32;
        let percent = step * 100 / (SWEEP - 1);

        load_material(
            backend,
            &mut assets,
            &mut blends,
            format!("metal_{percent:03}"),
            &Material {
                base_color: Vec3::new(0.95, 0.93, 0.88),
                metallic: 1.0,
                roughness,
                ..Material::default()
            },
        );
        load_material(
            backend,
            &mut assets,
            &mut blends,
            format!("dielectric_{percent:03}"),
            &Material {
                base_color: Vec3::splat(0.5),
                metallic: 0.0,
                roughness,
                reflectance: 0.5,
                ..Material::default()
            },
        );
    }

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
    load_material(
        backend,
        &mut assets,
        &mut blends,
        "textured",
        &Material {
            base_color: Vec3::ONE,
            metallic: 1.0,
            roughness: 1.0, // both scaled by the metallic-roughness map
            albedo_texture: Some(albedo),
            normal_texture: Some(normal),
            metallic_roughness_texture: Some(metal_rough),
            ..Material::default()
        },
    );

    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_Color.jpg"));
    let rock_albedo = backend.load_texture(&px, w, h, true);
    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_NormalDX.jpg"));
    let rock_normal = backend.load_texture(&px, w, h, false);
    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_Roughness.jpg"));
    let rock_rough = backend.load_texture(&px, w, h, false);
    assets.insert_texture("rock_albedo", rock_albedo);
    assets.insert_texture("rock_normal", rock_normal);
    assets.insert_texture("rock_rough", rock_rough);
    load_material(
        backend,
        &mut assets,
        &mut blends,
        "rock",
        &Material {
            base_color: Vec3::ONE,
            metallic: 0.0,  // no metallic map; rock is a dielectric
            roughness: 1.0, // driven by the roughness map (green channel)
            albedo_texture: Some(rock_albedo),
            normal_texture: Some(rock_normal),
            metallic_roughness_texture: Some(rock_rough),
            ..Material::default()
        },
    );

    load_material(
        backend,
        &mut assets,
        &mut blends,
        "ground",
        &Material {
            base_color: Vec3::splat(0.7),
            metallic: 0.0,
            roughness: 0.9,
            ..Material::default()
        },
    );

    // Three panes of the same glass, stacked in front of the camera below, so
    // the overlap is the readout: weighted-blended transparency is
    // order-independent, and where they cross there must be no seam and no
    // flicker as the camera moves around them.
    load_material(
        backend,
        &mut assets,
        &mut blends,
        "glass",
        &Material {
            base_color: Vec3::new(0.6, 0.85, 0.9),
            alpha: 0.35,
            blend: BlendMode::Blend,
            metallic: 0.0,
            roughness: 0.05,
            reflectance: 0.9,
            ..Material::default()
        },
    );

    // One material per extra lobe, each authored so the lobe is the only thing
    // it shows. A material that switched on two of them at once would be a
    // prettier sphere and a worse test — when it looked wrong there would be
    // nothing to say which lobe was wrong.
    for (name, material) in [
        // Car paint: a rough, dark base under a mirror film. The readout is that
        // the highlight is *two* highlights of different widths — a coat over a
        // base of the same roughness is just a brighter base.
        (
            "clearcoat",
            Material {
                base_color: Vec3::new(0.55, 0.05, 0.08),
                metallic: 0.0,
                roughness: 0.6,
                clearcoat: 1.0,
                clearcoat_roughness: 0.03,
                ..Material::default()
            },
        ),
        // Velvet. The readout is the rim: a sheen lobe peaks away from the
        // normal, so the silhouette is brighter than the centre — which no
        // amount of roughness on the base lobe can produce.
        (
            "velvet",
            Material {
                base_color: Vec3::new(0.14, 0.03, 0.10),
                metallic: 0.0,
                roughness: 0.95,
                sheen_color: Vec3::new(0.7, 0.25, 0.45),
                sheen_roughness: 0.25,
                ..Material::default()
            },
        ),
        // Brushed metal. The readout is that the highlight is a *streak* along
        // the tangent rather than a disc, and that it stays a streak in the
        // environment reflection — the bent normal in `ibl_reflection` is what
        // that second half tests.
        (
            "brushed",
            Material {
                base_color: Vec3::new(0.91, 0.92, 0.94),
                metallic: 1.0,
                roughness: 0.35,
                anisotropy: 0.85,
                ..Material::default()
            },
        ),
        // Solid glass, with a volume rather than a pane: thickness is what bends
        // the lookup at all, so a zero-thickness version of this refracts
        // nothing and only proves the queue runs. The green tint arrives through
        // Beer-Lambert over that thickness rather than through the base colour,
        // which is what makes the edges of the sphere darker than its middle.
        (
            "crystal",
            Material {
                base_color: Vec3::ONE,
                blend: BlendMode::Transmissive,
                metallic: 0.0,
                roughness: 0.05,
                transmission: 1.0,
                ior: 1.52,
                thickness: 0.9,
                attenuation_color: Vec3::new(0.72, 0.94, 0.82),
                attenuation_distance: 1.5,
                ..Material::default()
            },
        ),
        // The same glass roughened. Its background comes from a coarser level of
        // the scene pyramid, so this is the material that says whether that
        // chain was built at all — a frosted sphere with a sharp world behind it
        // means the roughness never reached the lookup.
        (
            "frosted",
            Material {
                base_color: Vec3::ONE,
                blend: BlendMode::Transmissive,
                metallic: 0.0,
                roughness: 0.45,
                transmission: 1.0,
                ior: 1.45,
                thickness: 0.6,
                ..Material::default()
            },
        ),
    ] {
        load_material(backend, &mut assets, &mut blends, name, &material);
    }

    (assets, bounds, blends)
}

/// Upload a material and register it in both world-side tables at once, so a
/// material can never reach a draw list without the blend mode extraction sorts
/// it by — a blended one registered nowhere would draw as an opaque wall.
fn load_material(
    backend: &mut impl RenderBackend,
    assets: &mut Assets,
    blends: &mut MaterialBlends,
    name: impl Into<String>,
    material: &Material,
) {
    let handle = backend.load_material(material);
    assets.insert_material(name, handle);
    blends.insert(handle, material.blend);
}

/// Upload a mesh and register it in both world-side tables at once, so a mesh
/// can never reach a draw list without the bounds the culler needs.
fn load_mesh(
    backend: &mut impl RenderBackend,
    assets: &mut Assets,
    bounds: &mut MeshBounds,
    name: &str,
    mesh: &CpuMesh,
) {
    let handle = backend.load_mesh(mesh);
    assets.insert_mesh(name, handle);
    // Read back from the backend's own upload, so the box culling tests is the
    // box the drawn geometry occupies.
    if let Some(aabb) = backend.mesh_bounds(handle) {
        bounds.insert(handle, aabb);
    }
}
