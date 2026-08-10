pub mod graph;
pub mod headless;
pub mod punctual;
pub mod sh;
pub mod shadows;
pub mod vulkan;

pub use headless::HeadlessBackend;

use crate::geom::Aabb;
use crate::scene::{
    BloomSettings, Camera, ContactShadowSettings, CpuMesh, DofSettings, EnvironmentSettings,
    HdrSettings, MaterialHandle, MeshHandle, MotionBlurSettings, RefractionSettings, SsaoSettings,
    SsrSettings, TaaSettings, TransparencySettings,
};
use glam::{Mat3, Mat4, Vec3};
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
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    /// Object-space tangent (+U texture direction) in `xyz`; `w` is the
    /// bitangent handedness (±1) used to rebuild the TBN basis for normal maps.
    #[format(R32G32B32A32_SFLOAT)]
    pub tangent: [f32; 4],
}

/// One renderable instance, as extraction hands it to the passes. Everything a
/// pass needs is here: no pass reaches back into the world, and none recomputes
/// what extraction already knew.
///
/// Not necessarily *visible* — an object the camera culls still gets one if any
/// cascade wants it as a caster. What each pass draws is a [`DrawList`] naming a
/// subset of these, never the array itself.
#[derive(Clone, Copy, Debug)]
pub struct RenderItem {
    pub model: Mat4,
    /// What `model` was the last time this entity was extracted, and `model`
    /// itself the first time it ever was. The prepass reprojects each vertex
    /// through it to write a motion vector, so an object that appears this frame
    /// reads as stationary rather than as having flown in from wherever the
    /// slot's previous occupant stood.
    pub prev_model: Mat4,
    /// Inverse-transpose of `model`'s upper 3x3, so normals stay perpendicular
    /// under non-uniform scale. Derived from the transform's rotation and scale
    /// at extraction rather than inverted out of `model` per pass per frame.
    pub normal_matrix: Mat3,
    /// World-space bounds: what the camera frustum tested, and what a shadow
    /// cascade tests against its own frustum without re-deriving anything.
    pub bounds: Aabb,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}

/// One pass's draw order over a shared item array.
///
/// Extraction derives each renderable's matrices and bounds once, into a single
/// `items` array, and every list a frame draws — the camera's, each cascade's —
/// is an ordering of `u32` indices into it. A cascade that shares an object with
/// the camera shares the `RenderItem`, so widening a list costs four bytes
/// rather than the 144 a `RenderItem` occupies.
///
/// `order` is grouped into maximal (mesh, material) runs, which is what lets a
/// pass collapse each run into one instanced draw.
#[derive(Clone, Copy)]
pub struct DrawList<'a> {
    pub items: &'a [RenderItem],
    pub order: &'a [u32],
}

impl<'a> DrawList<'a> {
    pub fn new(items: &'a [RenderItem], order: &'a [u32]) -> Self {
        Self { items, order }
    }

    pub const EMPTY: DrawList<'static> = DrawList {
        items: &[],
        order: &[],
    };

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The item at position `i` in the draw order.
    pub fn item(&self, i: usize) -> &'a RenderItem {
        &self.items[self.order[i] as usize]
    }

    /// Split the order into maximal runs sharing a mesh and material.
    ///
    /// Correct only because extraction groups on the same key: an ungrouped
    /// order still yields valid runs, just short ones, so a missed grouping
    /// costs performance rather than producing wrong pixels.
    pub fn runs(&self) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
        let list = *self;
        let mut start = 0usize;
        std::iter::from_fn(move || {
            if start >= list.len() {
                return None;
            }
            let first = list.item(start);
            let key = (first.mesh.0, first.material.0);
            let mut end = start + 1;
            while end < list.len() && {
                let item = list.item(end);
                (item.mesh.0, item.material.0) == key
            } {
                end += 1;
            }
            let run = start..end;
            start = end;
            Some(run)
        })
    }
}

pub const MAX_POINT_LIGHTS: usize = 16;

/// Half the point-light cap, because a cone is a narrower thing to want a lot
/// of and every slot costs the same uniform-buffer space whether a scene uses it
/// or not. Keep in sync with `MAX_SPOT_LIGHTS` in `forward.frag`.
pub const MAX_SPOT_LIGHTS: usize = 8;

/// Size of the shader's bound texture array (set 2). Keep at or below the
/// device's `maxPerStageDescriptorSampledImages` (≥16 guaranteed; MoltenVK
/// allows far more).
pub const MAX_TEXTURES: usize = 64;

/// The `u32` is the texture's index in the shader's array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    /// The direction the light *travels* (e.g. roughly downward for a sun).
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl DirectionalLight {
    /// The direction *toward* the light, which is what every shader wants.
    ///
    /// One definition because two consumers march along it: the forward pass
    /// shades with it, and the contact-shadow pass traces the depth buffer
    /// toward it. A sign flipped in one of them would be a shadow cast on the
    /// lit side of everything, and nothing about either result would say which
    /// one was wrong.
    pub fn direction_to_light(&self) -> Vec3 {
        (-self.direction).normalize_or_zero()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    /// Whether this light asks for atlas tiles. Asking is not getting: the atlas
    /// has a fixed number of tiles and a point light spends six, so
    /// [`fit`](punctual::fit) spends them on the lights that matter most and
    /// leaves the rest shading unshadowed.
    pub casts_shadows: bool,
}

/// A cone of light. `inner_cos`/`outer_cos` are cosines of the half angles
/// rather than the angles, because that is what the shader compares against —
/// converting once here keeps a transcendental out of the per-light loop.
#[derive(Clone, Copy, Debug)]
pub struct SpotLight {
    pub position: Vec3,
    /// The direction the cone points, normalised.
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_cos: f32,
    pub outer_cos: f32,
    pub casts_shadows: bool,
}

#[derive(Clone, Debug)]
pub struct SceneLighting {
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    pub sun: DirectionalLight,
    /// Anything past [`MAX_POINT_LIGHTS`] is ignored.
    pub point_lights: Vec<PointLight>,
    /// Anything past [`MAX_SPOT_LIGHTS`] is ignored.
    pub spot_lights: Vec<SpotLight>,
    /// Blinn-Phong specular exponent. Higher = smaller, sharper highlight.
    pub shininess: f32,
    pub specular_strength: f32,
    pub fog_color: Vec3,
    /// Fog extinction at `fog_height`. Zero disables the effect.
    pub fog_density: f32,
    pub fog_height_falloff: f32,
    pub fog_height: f32,
}

/// Which queue a material draws in.
///
/// A property of the material rather than of the entity, as it is in glTF and in
/// every engine that reads glTF: opacity is authored with the base colour and
/// the maps, so it belongs with them. Extraction reads it through
/// [`MaterialBlends`](crate::scene::MaterialBlends) to split the frame's draw
/// order in two.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    /// Depth-tested, depth-writing, and shaded straight into the frame's colour.
    /// [`Material::alpha`] is ignored.
    #[default]
    Opaque,
    /// Accumulated by the weighted-blended pass instead: depth-tested against
    /// the opaque scene but writing no depth, and composited afterwards. Never
    /// written into the geometry prepass, so nothing screen-space — occlusion,
    /// reflections, defocus, the shutter — treats it as a surface.
    Blend,
    /// Refracted rather than blended: drawn by its own pass, which samples the
    /// already-composited frame through the surface instead of mixing with what
    /// the framebuffer happens to hold.
    ///
    /// A queue of its own rather than a flag on [`Blend`](BlendMode::Blend)
    /// because the two disagree about ordering. Weighted-blended transparency is
    /// commutative and must not be sorted; a surface that *fetches* its own
    /// background has to be drawn back to front, since the second one to sample
    /// the same pixel would otherwise double-count what is behind it. Kept out
    /// of the prepass and the caster lists exactly as `Blend` is.
    Transmissive,
}

impl BlendMode {
    /// Whether this mode leaves the opaque queue — out of the prepass, out of
    /// every caster list, and therefore invisible to everything screen-space.
    ///
    /// One predicate rather than two comparisons at each site, so adding a third
    /// non-opaque mode cannot half-land: extraction, `MaterialBlends` and the
    /// demo scene all ask this question and none of them cares *which* non-opaque
    /// queue the material ends up in.
    pub fn is_opaque(self) -> bool {
        matches!(self, BlendMode::Opaque)
    }
}

/// Named for the glTF extensions each block mirrors — `KHR_materials_clearcoat`,
/// `_sheen`, `_anisotropy`, `_transmission` and `_volume` — so a loader has one
/// obvious place to put what it read and no translation table to get wrong.
///
/// Every block is inert at its default: the shader gates each lobe on a flag
/// derived from whether the block was actually set, so a plain metallic-roughness
/// material costs exactly what it did before any of this existed.
#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub base_color: Vec3,
    /// Opacity, multiplied by the albedo map's alpha. Only read for
    /// [`BlendMode::Blend`].
    pub alpha: f32,
    pub blend: BlendMode,
    pub metallic: f32,
    pub roughness: f32,
    pub reflectance: f32,
    pub emissive: Vec3,
    pub albedo_texture: Option<TextureHandle>,
    pub normal_texture: Option<TextureHandle>,
    pub metallic_roughness_texture: Option<TextureHandle>,
    pub emissive_texture: Option<TextureHandle>,

    /// Strength of the second specular lobe, `0` for none. The coat is a thin
    /// dielectric film over everything else: it adds its own reflection and
    /// attenuates the layers beneath by what it reflected away.
    pub clearcoat: f32,
    /// Perceptual roughness of that lobe, independent of the base's — a scuffed
    /// coat over polished metal is the whole reason the two are separate.
    pub clearcoat_roughness: f32,
    /// `r` = strength, `g` = roughness, multiplying the two scalars above.
    pub clearcoat_texture: Option<TextureHandle>,
    /// Tangent-space normals for the coat alone. Absent, the coat uses the
    /// *geometric* normal rather than the base layer's normal-mapped one, which
    /// is glTF's rule and the physical one: an orange-peel coat and the grain
    /// under it are different surfaces.
    pub clearcoat_normal_texture: Option<TextureHandle>,

    /// Retroreflective rim lobe for cloth. Black for none; this is a colour
    /// rather than a scalar because velvet and satin owe their look to a sheen
    /// tinted away from the base.
    pub sheen_color: Vec3,
    /// Width of that lobe. Low is a tight satin edge, high a broad velvet bloom.
    pub sheen_roughness: f32,
    /// `rgb` = colour, `a` = roughness, multiplying the two above.
    pub sheen_texture: Option<TextureHandle>,

    /// How far the specular highlight is stretched, in `[-1, 1]`. Positive
    /// stretches along the tangent (brushed metal, hair), negative across it;
    /// `0` is the isotropic GGX everything else uses.
    pub anisotropy: f32,
    /// Rotation of the stretch within the tangent plane, in radians. What lets a
    /// brushed disc have circular grain under a UV set that does not.
    pub anisotropy_rotation: f32,
    /// `rg` = direction as a signed tangent-space vector, `b` = strength.
    pub anisotropy_texture: Option<TextureHandle>,

    /// How much light passes *through* rather than being diffusely reflected.
    /// Non-zero only means anything for [`BlendMode::Transmissive`], which is
    /// the queue that owns the pass able to fetch what is behind the surface.
    pub transmission: f32,
    /// Index of refraction, `1.5` for glass. Drives both the Fresnel term and
    /// how far the refraction pass bends its lookup.
    pub ior: f32,
    /// Thickness of the volume behind the surface, in local units. Zero makes it
    /// a *thin* surface — a window pane, refracting but with no interior to
    /// travel through — which is why it is the default.
    pub thickness: f32,
    /// What the volume absorbs, Beer-Lambert, over `attenuation_distance`. White
    /// is a clear medium.
    pub attenuation_color: Vec3,
    /// The distance at which `attenuation_color` is reached. Infinite for a
    /// medium that never absorbs.
    pub attenuation_distance: f32,
    /// `r` = transmission, `g` = thickness, multiplying the two above.
    pub transmission_texture: Option<TextureHandle>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Vec3::splat(0.8),
            alpha: 1.0,
            blend: BlendMode::Opaque,
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            emissive: Vec3::ZERO,
            albedo_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,

            clearcoat: 0.0,
            // Only read when `clearcoat` is non-zero, so this is what a coat
            // looks like the moment one is switched on rather than a value that
            // does nothing: a mirror-smooth film, which is what a coat is unless
            // it was authored otherwise.
            clearcoat_roughness: 0.03,
            clearcoat_texture: None,
            clearcoat_normal_texture: None,

            sheen_color: Vec3::ZERO,
            sheen_roughness: 0.3,
            sheen_texture: None,

            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            anisotropy_texture: None,

            transmission: 0.0,
            ior: 1.5,
            thickness: 0.0,
            attenuation_color: Vec3::ONE,
            attenuation_distance: f32::INFINITY,
            transmission_texture: None,
        }
    }
}

impl Default for SceneLighting {
    fn default() -> Self {
        Self {
            ambient_color: Vec3::new(0.6, 0.7, 1.0),
            ambient_intensity: 0.15,
            sun: DirectionalLight {
                direction: Vec3::new(-0.4, -1.0, -0.6).normalize(),
                color: Vec3::new(1.0, 0.97, 0.92),
                intensity: 1.0,
            },
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
            shininess: 32.0,
            specular_strength: 0.4,
            fog_color: Vec3::new(0.55, 0.62, 0.72),
            fog_density: 0.005,
            fog_height_falloff: 0.1,
            fog_height: 0.0,
        }
    }
}

// The seam between the engine and a concrete graphics API: implement for other
// backends (wgpu, D3D12) without touching scene/app code.
pub trait RenderBackend {
    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle;
    /// Object-space bounds derived at upload; `None` for a handle this backend
    /// never issued. Mirrored into [`MeshBounds`](crate::scene::MeshBounds) at
    /// load, since culling runs before any backend type is in reach.
    fn mesh_bounds(&self, mesh: MeshHandle) -> Option<Aabb>;
    fn load_material(&mut self, material: &Material) -> MaterialHandle;
    fn load_texture(&mut self, pixels: &[u8], width: u32, height: u32, srgb: bool)
    -> TextureHandle;
    /// Replace the environment with one baked from an equirectangular source:
    /// tightly packed RGBA f32, row-major from the top-left.
    ///
    /// Baking is synchronous — it happens once, at load, and the alternative is
    /// a half-written cubemap visible to the first frame.
    fn load_environment(&mut self, pixels: &[f32], width: u32, height: u32);
    fn resize(&mut self, extent: [u32; 2]);
    /// `dt` is the seconds elapsed since the last frame — what any temporal
    /// effect a backend runs needs, exposure adaptation being the first of them.
    /// Zero means "converge immediately", which is what a one-shot render wants.
    #[allow(clippy::too_many_arguments)]
    fn render(
        &mut self,
        draws: DrawList<'_>,
        // What the weighted-blended transparency pass accumulates, grouped by
        // (mesh, material) and deliberately not ordered by depth.
        transparent: DrawList<'_>,
        // What the refraction pass draws, grouped the same way and then ordered
        // back to front — which the list above must not be, and this one must.
        refractive: DrawList<'_>,
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        contact_shadows: &ContactShadowSettings,
        ssr: &SsrSettings,
        transparency: &TransparencySettings,
        refraction: &RefractionSettings,
        taa: &TaaSettings,
        motion_blur: &MotionBlurSettings,
        dof: &DofSettings,
        bloom: &BloomSettings,
        hdr: &HdrSettings,
        environment: &EnvironmentSettings,
        dt: f32,
    );
}

#[cfg(test)]
mod draw_list_tests {
    use super::{DrawList, MaterialHandle, MeshHandle, RenderItem};
    use crate::geom::Aabb;
    use glam::{Mat3, Mat4, Vec3};

    fn item(mesh: u32, material: u32) -> RenderItem {
        RenderItem {
            model: Mat4::IDENTITY,
            prev_model: Mat4::IDENTITY,
            normal_matrix: Mat3::IDENTITY,
            bounds: Aabb {
                min: Vec3::splat(-0.5),
                max: Vec3::splat(0.5),
            },
            mesh: MeshHandle(mesh),
            material: MaterialHandle(material),
        }
    }

    /// Runs over the identity order, which is what a list nothing reordered has.
    fn ranges(items: &[RenderItem]) -> Vec<(usize, usize)> {
        let order: Vec<u32> = (0..items.len() as u32).collect();
        DrawList::new(items, &order)
            .runs()
            .map(|run| (run.start, run.end))
            .collect()
    }

    #[test]
    fn a_grouped_order_collapses_into_one_run_per_mesh_material_pair() {
        let items = [
            item(0, 0),
            item(0, 0),
            item(0, 1),
            item(3, 1),
            item(3, 1),
            item(3, 1),
        ];
        assert_eq!(ranges(&items), vec![(0, 2), (2, 3), (3, 6)]);
    }

    /// Every run must be non-empty, contiguous, and cover the order exactly —
    /// `object_base` is the run's start, so a gap or overlap would draw an
    /// instance against another object's transform.
    #[test]
    fn runs_partition_the_order_without_gaps_or_overlaps() {
        let items = [item(1, 0), item(1, 0), item(2, 7), item(2, 7), item(9, 9)];
        let ranges = ranges(&items);
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, items.len());
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "runs must be contiguous: {ranges:?}");
        }
        assert!(ranges.iter().all(|(start, end)| end > start));
        assert_eq!(
            ranges.iter().map(|(s, e)| e - s).sum::<usize>(),
            items.len()
        );
    }

    /// Same mesh but a different material can't share an instanced draw: the
    /// material index is a per-run push constant.
    #[test]
    fn a_material_change_breaks_a_run() {
        let items = [item(4, 0), item(4, 1)];
        assert_eq!(ranges(&items), vec![(0, 1), (1, 2)]);
    }

    /// An ungrouped order still has to partition correctly — it just yields
    /// more, shorter runs. Wrong pixels are not an acceptable cost of a missed
    /// grouping.
    #[test]
    fn an_ungrouped_order_still_partitions_correctly() {
        let items = [item(0, 0), item(5, 0), item(0, 0)];
        assert_eq!(ranges(&items), vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn an_empty_order_has_no_runs() {
        assert!(ranges(&[]).is_empty());
    }

    /// The point of the indirection: runs are keyed by what the *order* points
    /// at, not by where the items happen to sit in the array.
    #[test]
    fn runs_follow_the_order_not_the_item_array() {
        let items = [item(0, 0), item(7, 7), item(0, 0)];
        let order = [0u32, 2, 1];
        let ranges: Vec<_> = DrawList::new(&items, &order)
            .runs()
            .map(|run| (run.start, run.end))
            .collect();
        assert_eq!(ranges, vec![(0, 2), (2, 3)]);
    }
}
