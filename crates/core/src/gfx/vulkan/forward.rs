use std::sync::Arc;

use glam::{Mat4, Vec3};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as _, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{RenderPass, Subpass};

use crate::geom::Aabb;
use crate::gfx::punctual::{LightKind, MAX_ATLAS_FACES, MAX_SHADOW_LIGHTS, ShadowAtlas};
use crate::gfx::sh::SH9;
use crate::gfx::shadows::MAX_CASCADES;
use crate::gfx::{
    BlendMode, DrawList, MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS, MAX_TEXTURES, Material, SceneLighting,
    Vertex,
};
use crate::scene::{Camera, EnvironmentSettings};

use super::context::VkContext;
use super::subsurface::SUBSURFACE_FORMAT;
use super::swapchain::DEPTH_FORMAT;
use super::taa::FrameView;
use super::{ShadowFrame, VulkanRenderer};

pub struct GpuMesh {
    pub vertex_buffer: Subbuffer<[Vertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub index_count: u32,
    /// Object-space bounds, derived here because upload is the last place the
    /// vertex data exists on the CPU. Culling reads them through
    /// [`RenderBackend::mesh_bounds`](crate::gfx::RenderBackend::mesh_bounds).
    pub bounds: Aabb,
}

/// Per-run push constants. Only the small, per-run-varying values live here; the
/// fat per-object matrices are in the set-4 storage buffer so this range stays
/// under the 128-byte guaranteed `maxPushConstantsSize` (it was 196).
///
/// `view_proj` replaced a pre-multiplied `mvp` when draws became instanced: a
/// run covers many models, so the model half has to be applied in the shader.
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct PushConstants {
    view_proj: [[f32; 4]; 4],
    material_index: u32,
    /// First set-4 object row of this run; the shader adds `gl_InstanceIndex`.
    object_base: u32,
}

impl PushConstants {
    /// Also what the transparency pass pushes: it draws the same geometry
    /// through the same pipeline layout, so it pushes the same range.
    pub(super) fn new(view_proj: Mat4, material_index: u32, object_base: u32) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            material_index,
            object_base,
        }
    }
}

/// Per-object transforms, indexed by [`PushConstants::object_index`] from a
/// storage buffer (set 4). std430 matches this `#[repr(C)]` layout exactly
/// because every field is a 64-byte `mat4` (a multiple of 16).
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct GpuObject {
    model: [[f32; 4]; 4],
    /// Inverse-transpose of `model`'s rotation/scale, for transforming normals
    /// correctly under non-uniform scaling. Stored as a mat4; only the upper-left
    /// 3x3 is used in the shader.
    normal_matrix: [[f32; 4]; 4],
    /// Last frame's `model`, for the motion vector the prepass writes. Uploaded
    /// for every pass rather than only the one that reads it: the buffer is
    /// shared, so the row's stride is shared too, and a second layout for the
    /// passes that ignore this field would be two ways for one object row to be
    /// wrong.
    prev_model: [[f32; 4]; 4],
}

/// Where `forward.frag` declares the cascade comparison sampler. It is bound
/// immutably at pipeline-layout construction, so these have to match the shader
/// by hand rather than being derived from it.
///
/// Shared with the refraction pass, which builds a layout of its own — the
/// immutable sampler is part of the set layout, so a pipeline that patched it
/// differently would no longer be set-compatible with this one and could not be
/// handed the same five descriptor sets.
pub(super) const SHADOW_SET: usize = 3;
pub(super) const SHADOW_SAMPLER_BINDING: u32 = 2;

/// Default texture indices, matching the order `VulkanRenderer::new` seeds them.
const WHITE_TEXTURE: u32 = 0;
const FLAT_NORMAL_TEXTURE: u32 = 1;

/// Feature bits in [`GpuMaterial::flags`], mirrored by `shading.glsl`.
///
/// The whole point of the word: `push.material_index` is dynamically uniform, so
/// a draw either takes a lobe's branch or does not, and the cost of a feature a
/// material never asked for is one coherent test. Set from whether the block was
/// actually authored rather than from a separate toggle, so a material cannot
/// claim a lobe and supply nothing to it.
pub(crate) mod material_flags {
    pub const CLEARCOAT: u32 = 1 << 0;
    pub const SHEEN: u32 = 1 << 1;
    pub const ANISOTROPY: u32 = 1 << 2;
    pub const TRANSMISSION: u32 = 1 << 3;
    pub const SUBSURFACE: u32 = 1 << 4;
    pub const PARALLAX: u32 = 1 << 5;
}

/// Ceiling on a material's parallax step counts. Keep in sync with
/// `PARALLAX_STEP_LIMIT` in `shaders/parallax.glsl`, which is what actually
/// bounds the loop: a material asking for more would pay for the extra
/// iterations in register pressure and get none of them.
const PARALLAX_STEP_LIMIT: u32 = 64;

#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub(crate) struct GpuMaterial {
    /// `rgb` = albedo, `a` = opacity.
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4], // metallic, roughness, reflectance, ior
    /// Indices into the set-2 texture array: [albedo, normal, metal-rough, emissive].
    tex_indices: [u32; 4],
    clearcoat: [f32; 4],    // strength, roughness
    sheen: [f32; 4],        // rgb = colour, a = roughness
    anisotropy: [f32; 4],   // strength, cos(rotation), sin(rotation)
    transmission: [f32; 4], // transmission, thickness, attenuation distance
    /// `rgb` = what the volume absorbs over `transmission[2]`.
    attenuation: [f32; 4],
    subsurface: [f32; 4], // rgb = scattering tint, a = forward-scatter power
    /// `rgb` = per-channel mean free path in metres. Separate from the block
    /// above because the shader wants the tint and the distances at different
    /// points — the tint weights what comes back, the distances decide how far
    /// across the image it is allowed to come back from.
    subsurface_radius: [f32; 4],
    /// Depth of the height field in metres, then the step counts the march is
    /// allowed head-on and edge-on.
    parallax: [f32; 4],
    /// [clearcoat, clearcoat normal, sheen, anisotropy].
    tex_indices_ext: [u32; 4],
    /// [transmission, feature flags, subsurface, height]. The flags ride here
    /// rather than in a float field so the shader can test them without
    /// `floatBitsToUint`.
    tex_flags: [u32; 4],
}

/// Pack the engine's [`SceneLighting`] into the std140 layout the shader expects.
#[allow(clippy::too_many_arguments)]
fn to_gpu_lighting(
    lighting: &SceneLighting,
    camera_pos: Vec3,
    extent: [u32; 2],
    shadows: Option<ShadowFrame<'_>>,
    atlas: &ShadowAtlas,
    irradiance: [Vec3; SH9],
    environment_yaw: f32,
    env_specular: Vec3,
) -> GpuLighting {
    let (w, h) = (extent[0] as f32, extent[1] as f32);
    let count = lighting.point_lights.len().min(MAX_POINT_LIGHTS);
    let mut point_lights = [GpuPointLight::ZERO; MAX_POINT_LIGHTS];
    for (index, (slot, light)) in point_lights
        .iter_mut()
        .zip(lighting.point_lights.iter().take(count))
        .enumerate()
    {
        slot.position = [
            light.position.x,
            light.position.y,
            light.position.z,
            light.range.max(1e-4),
        ];
        slot.color = [light.color.x, light.color.y, light.color.z, light.candela];
        // A light that asked for tiles and did not get them keeps its -1 and
        // shades unshadowed, which is the whole behaviour of the budget: too
        // many casters costs shadows, never correctness.
        slot.shadow = match atlas.caster(LightKind::Point, index) {
            Some(caster) => [caster.first_face as f32, caster.near, 0.0, 0.0],
            None => [-1.0, 0.0, 0.0, 0.0],
        };
    }

    let spot_count = lighting.spot_lights.len().min(MAX_SPOT_LIGHTS);
    let mut spot_lights = [GpuSpotLight::ZERO; MAX_SPOT_LIGHTS];
    for (index, (slot, light)) in spot_lights
        .iter_mut()
        .zip(lighting.spot_lights.iter().take(spot_count))
        .enumerate()
    {
        slot.position = [
            light.position.x,
            light.position.y,
            light.position.z,
            light.range.max(1e-4),
        ];
        slot.direction = [
            light.direction.x,
            light.direction.y,
            light.direction.z,
            light.outer_cos,
        ];
        slot.color = [light.color.x, light.color.y, light.color.z, light.candela];
        let face = atlas
            .caster(LightKind::Spot, index)
            .map_or([-1.0, 0.0], |caster| {
                [caster.first_face as f32, caster.near]
            });
        slot.params = [light.inner_cos, face[0], face[1], 0.0];
    }

    // The shader wants the direction *toward* the light, so negate.
    let to_sun = (-lighting.sun.direction).normalize_or_zero();

    let mut cascade_view_proj = [[[0.0f32; 4]; 4]; MAX_CASCADES];
    let mut cascade_splits = [0.0f32; MAX_CASCADES];
    let mut cascade_texel_sizes = [0.0f32; MAX_CASCADES];
    // A zero count is what makes every shadow lookup return "lit"; the arrays
    // above are then never indexed.
    let shadow_params = match shadows {
        Some(shadows) => {
            for (slot, cascade) in cascade_view_proj
                .iter_mut()
                .zip(&shadows.cascades.cascades[..shadows.cascades.count])
            {
                *slot = cascade.view_proj.to_cols_array_2d();
            }
            for (index, cascade) in shadows.cascades.cascades[..shadows.cascades.count]
                .iter()
                .enumerate()
            {
                cascade_splits[index] = cascade.split_distance;
                cascade_texel_sizes[index] = cascade.texel_world_size;
            }
            [
                shadows.cascades.count as f32,
                crate::gfx::shadows::OVERLAP,
                shadows.settings.strength,
                if shadows.settings.debug_cascades {
                    1.0
                } else {
                    0.0
                },
            ]
        }
        None => [0.0; 4],
    };

    GpuLighting {
        camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
        ambient: [
            lighting.ambient_color.x,
            lighting.ambient_color.y,
            lighting.ambient_color.z,
            lighting.ambient_nits,
        ],
        sun_direction: [to_sun.x, to_sun.y, to_sun.z, 0.0],
        sun_color: [
            lighting.sun.color.x,
            lighting.sun.color.y,
            lighting.sun.color.z,
            lighting.sun.illuminance,
        ],
        params: [
            count as f32,
            lighting.shininess,
            lighting.specular_strength,
            spot_count as f32,
        ],
        viewport: [w, h, 1.0 / w, 1.0 / h],
        fog_color: [
            lighting.fog_color.x,
            lighting.fog_color.y,
            lighting.fog_color.z,
            lighting.fog_density.max(0.0),
        ],
        fog_params: [lighting.fog_height_falloff, lighting.fog_height, 0.0, 0.0],
        cascade_view_proj,
        cascade_splits,
        cascade_texel_sizes,
        shadow_params,
        point_lights,
        spot_lights,
        environment: {
            let (sin, cos) = environment_yaw.to_radians().sin_cos();
            [sin, cos, 0.0, 0.0]
        },
        env_specular: [env_specular.x, env_specular.y, env_specular.z, 0.0],
        irradiance: irradiance.map(|c| [c.x, c.y, c.z, 0.0]),
    }
}

/// GPU mirror of a [`PointLight`](crate::gfx::PointLight), padded to std140
/// (three `vec4`s).
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuPointLight {
    /// xyz = world position, w = range.
    position: [f32; 4],
    /// rgb = color, w = luminous intensity in candela.
    color: [f32; 4],
    /// x = index of this light's first atlas face, negative for a light the
    /// atlas had no room for; y = the near plane its faces were rendered with,
    /// which the shader needs to bias the comparison against.
    shadow: [f32; 4],
}

impl GpuPointLight {
    const ZERO: Self = Self {
        position: [0.0; 4],
        color: [0.0; 4],
        shadow: [-1.0, 0.0, 0.0, 0.0],
    };
}

/// GPU mirror of a [`SpotLight`](crate::gfx::SpotLight), padded to std140.
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuSpotLight {
    /// xyz = world position, w = range.
    position: [f32; 4],
    /// xyz = cone axis, w = cosine of the outer half angle.
    direction: [f32; 4],
    /// rgb = color, w = luminous intensity in candela, the reflector already
    /// divided out.
    color: [f32; 4],
    /// x = cosine of the inner half angle, y = atlas face index (negative for
    /// none), z = near plane.
    params: [f32; 4],
}

impl GpuSpotLight {
    const ZERO: Self = Self {
        position: [0.0; 4],
        // A `w` of 1.0 is a cone of zero width, so an unused slot lights
        // nothing even if the count were ever wrong.
        direction: [0.0, 0.0, -1.0, 1.0],
        color: [0.0; 4],
        params: [1.0, -1.0, 0.0, 0.0],
    };
}

/// GPU layout of the per-frame lighting uniform buffer (set 0, binding 0).
/// Every field is a `vec4` so the Rust `#[repr(C)]` layout matches std140 with
/// no hidden padding.
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuLighting {
    /// xyz = camera world position.
    camera_pos: [f32; 4],
    /// rgb = ambient color, w = ambient luminance in cd/m².
    ambient: [f32; 4],
    /// xyz = normalized direction toward the sun.
    sun_direction: [f32; 4],
    /// rgb = sun color, w = illuminance in lux on a surface facing it.
    sun_color: [f32; 4],
    /// x = point light count, y = shininess, z = specular strength,
    /// w = spot light count.
    params: [f32; 4],
    /// x=w, y=h, z=1/w, w=1/h
    viewport: [f32; 4],
    /// rgb = fog color, w = density at the reference height.
    fog_color: [f32; 4],
    /// x = height falloff, y = reference height.
    fog_params: [f32; 4],
    /// Per-cascade light view-projection. std140 lays a `mat4` out as four
    /// `vec4`s with no padding between them, which is exactly what this is.
    cascade_view_proj: [[[f32; 4]; 4]; MAX_CASCADES],
    /// Split distances, as radial distance from the camera.
    cascade_splits: [f32; MAX_CASCADES],
    /// World size of one shadow texel in each cascade, for the normal-offset
    /// bias. It differs per cascade because each fits a different-sized box to
    /// the same number of texels.
    cascade_texel_sizes: [f32; MAX_CASCADES],
    /// x = cascade count, y = blend overlap fraction, z = strength,
    /// w = 1.0 to tint by cascade index.
    shadow_params: [f32; 4],
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
    spot_lights: [GpuSpotLight; MAX_SPOT_LIGHTS],
    /// x = sin(environment yaw), y = cos(environment yaw). The same rotation
    /// the skybox samples through, so the sky and what it lights agree.
    environment: [f32; 4],
    /// rgb = what sampled environment radiance is multiplied by. Carries the
    /// scene's flat ambient when no environment is loaded, which is what makes
    /// the 1x1 white fallback cube behave as a uniform environment.
    env_specular: [f32; 4],
    /// Diffuse irradiance as nine spherical-harmonic coefficients, already
    /// convolved with the cosine lobe and divided by pi — see `gfx::sh`. `vec4`
    /// rather than `vec3` because std140 pads an array element to 16 bytes
    /// either way, so the padding may as well be visible on both sides.
    irradiance: [[f32; 4]; SH9],
}

/// One atlas face as the shader reads it: the matrix that rendered it, and the
/// slice of the atlas it landed in.
///
/// The matrix is stored rather than rebuilt from the light's position and a face
/// convention, which the `vector_to_depth` trick would allow. Storing it means
/// the shader projects with *the same* matrix the tile was rasterised with, so
/// the two cannot disagree about a near plane, a border, or a handedness — the
/// only thing left to get right is which face, and that is one comparison.
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuShadowFace {
    view_proj: [[f32; 4]; 4],
    /// xy = atlas UV offset, zw = atlas UV scale.
    rect: [f32; 4],
}

impl GpuShadowFace {
    const ZERO: Self = Self {
        view_proj: [[0.0; 4]; 4],
        rect: [0.0; 4],
    };
}

/// This frame's per-object rows, and where each list's block begins in them.
///
/// One buffer for every geometry pass in the frame, which is what
/// `object_transforms` says in the graph. The opaque camera list indexes from
/// zero and needs no base.
pub(super) struct ObjectRows {
    pub buffer: Subbuffer<[GpuObject]>,
    /// Where the blended items' rows start.
    pub transparent_base: u32,
    /// Where the refractive items' rows start.
    pub refractive_base: u32,
    pub cascade_bases: [u32; MAX_CASCADES],
    pub punctual_bases: [u32; MAX_SHADOW_LIGHTS],
}

/// The five descriptor sets a pass shading into the lit frame binds, in bind
/// order.
///
/// Both such passes bind the same five — the transparency pass draws through
/// this pass's own pipeline layout, so it can and must. Bundling them is what
/// keeps that from being a comment somebody has to keep true.
pub(super) struct ForwardSets {
    sets: Vec<Arc<DescriptorSet>>,
}

impl ForwardSets {
    pub(super) fn as_vec(&self) -> Vec<Arc<DescriptorSet>> {
        self.sets.clone()
    }
}

pub struct ForwardPass {
    pub render_pass: Arc<RenderPass>,
    /// The same pass with a second colour target and a second resolve, for the
    /// frame that diffuses subsurface light.
    ///
    /// Built alongside the first rather than in place of it, and both kept for
    /// the whole session. Which one a frame uses is structural — it comes out of
    /// `FrameConfig` — but a render pass is not something the graph owns, and
    /// rebuilding this one on a toggle would mean rebuilding every pipeline that
    /// shares it: this pass's, the skybox's and the debug lines'. Two of each,
    /// made once, costs three extra pipelines at startup and nothing at all
    /// afterwards.
    pub subsurface_render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    subsurface_pipeline: Arc<GraphicsPipeline>,
    uniform_buffer_allocator: SubbufferAllocator,
    /// Per-frame storage for the atlas face table. A storage buffer rather than
    /// more of the lighting uniform: forty-eight matrices is three kilobytes,
    /// and the guaranteed uniform-buffer range is sixteen.
    shadow_face_allocator: SubbufferAllocator,
    /// Per-frame streaming allocator for the set-4 per-object transform buffer.
    object_buffer_allocator: SubbufferAllocator,
    sampler: Arc<Sampler>,
    ao_sampler: Arc<Sampler>,
}

impl ForwardPass {
    pub fn new(
        device: &Arc<Device>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color_format: Format,
    ) -> Self {
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                msaa_color: {
                    format: color_format,
                    samples: 4,
                    load_op: Clear,
                    store_op: DontCare,
                },
                depth: {
                    format: DEPTH_FORMAT,
                    samples: 4,
                    load_op: Clear,
                    store_op: DontCare,
                },

                color: {
                    format: color_format,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [msaa_color],
                color_resolve: [color],
                depth_stencil: {depth},
            },
        )
        .unwrap();

        // The second target is `DontCare`/resolve in exactly the shape the first
        // is, and cleared for one reason: every other pipeline that shares this
        // render pass — the skybox, the debug lines — masks the channel off rather
        // than writing to it, so the clear is what a pixel they covered is left
        // holding. Zero there reads as "nothing scattered here", which is the only
        // mask the diffusion passes need.
        let subsurface_render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                msaa_color: {
                    format: color_format,
                    samples: 4,
                    load_op: Clear,
                    store_op: DontCare,
                },
                msaa_subsurface: {
                    format: SUBSURFACE_FORMAT,
                    samples: 4,
                    load_op: Clear,
                    store_op: DontCare,
                },
                depth: {
                    format: DEPTH_FORMAT,
                    samples: 4,
                    load_op: Clear,
                    store_op: DontCare,
                },

                color: {
                    format: color_format,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
                subsurface: {
                    format: SUBSURFACE_FORMAT,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [msaa_color, msaa_subsurface],
                color_resolve: [color, subsurface],
                depth_stencil: {depth},
            },
        )
        .unwrap();

        // One sampler for both pipelines. See `build_pipeline`.
        let shadow_sampler = super::shadow::comparison_sampler(device);
        let pipeline = build_pipeline(
            device,
            &render_pass,
            fs::load(device.clone()).unwrap(),
            &shadow_sampler,
        );
        let subsurface_pipeline = build_pipeline(
            device,
            &subsurface_render_pass,
            fs_sss::load(device.clone()).unwrap(),
            &shadow_sampler,
        );

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let object_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let shadow_face_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        // Material textures are the only ones with a mip chain to sample; the
        // AO and tonemap inputs are screen-space targets read at 1:1.
        let anisotropy = device.enabled_features().sampler_anisotropy.then(|| {
            device
                .physical_device()
                .properties()
                .max_sampler_anisotropy
                .min(16.0)
        });

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                anisotropy,
                ..SamplerCreateInfo::simple_repeat_linear()
            },
        )
        .unwrap();

        let ao_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();

        Self {
            render_pass,
            subsurface_render_pass,
            pipeline,
            subsurface_pipeline,
            uniform_buffer_allocator,
            object_buffer_allocator,
            shadow_face_allocator,
            sampler,
            ao_sampler,
        }
    }

    /// The layout the transparency pass builds its own pipeline with. Shared
    /// rather than derived a second time, so the two pipelines cannot disagree
    /// about a binding — see `oit.rs`.
    pub(super) fn pipeline_layout(&self) -> &Arc<PipelineLayout> {
        self.pipeline.layout()
    }

    /// The set-4 per-object descriptor set for this frame's object buffer.
    ///
    /// Built once per frame by the executor rather than once per pass: the
    /// buffer changes every frame, so nothing here can be cached across frames,
    /// but nothing needs to be rebuilt within one either.
    pub(super) fn build_object_set(
        &self,
        ctx: &VkContext,
        objects: &Subbuffer<[GpuObject]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[4].clone(),
            [WriteDescriptorSet::buffer(0, objects.clone())],
            [],
        )
        .unwrap()
    }

    /// Build the set-1 material descriptor set over the shared table. Cached by
    /// the renderer and only rebuilt when the material table changes.
    pub fn build_material_set(
        &self,
        ctx: &VkContext,
        buffer: &Subbuffer<[GpuMaterial]>,
    ) -> Arc<DescriptorSet> {
        let buffer = buffer.clone();
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(0, buffer)],
            [],
        )
        .unwrap()
    }

    /// Build the set-2 texture array + sampler descriptor set. Cached by the
    /// renderer and only rebuilt when a texture is added.
    pub fn build_texture_set(
        &self,
        ctx: &VkContext,
        textures: &[Arc<ImageView>],
    ) -> Arc<DescriptorSet> {
        let default_view = textures[0].clone();
        let texture_array = (0..MAX_TEXTURES).map(|i| {
            textures
                .get(i)
                .cloned()
                .unwrap_or_else(|| default_view.clone())
        });
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[2].clone(),
            [
                WriteDescriptorSet::image_view_array(0, 0, texture_array),
                WriteDescriptorSet::sampler(1, self.sampler.clone()),
            ],
            [],
        )
        .unwrap()
    }

    /// Build this frame's per-object rows, written straight into the mapped
    /// subbuffer. Shared by every geometry pass in the frame: they need the same
    /// rows, and the allocator recycles the storage frame to frame.
    ///
    /// One row per item, including items whose mesh is missing, so a run's
    /// object rows stay contiguous and a run's base is just its start.
    ///
    /// The opaque `visible` items go first so the forward and prepass passes
    /// keep indexing from zero; the blended ones follow, then each cascade's
    /// casters, then each punctual light's, and the returned bases say where.
    /// One buffer rather than one per list is what keeps `object_transforms` a
    /// single resource in the graph rather than a convenient fiction.
    pub(super) fn upload_objects(
        &self,
        visible: DrawList<'_>,
        transparent: DrawList<'_>,
        refractive: DrawList<'_>,
        casters: &[DrawList<'_>],
        punctual: &[DrawList<'_>],
    ) -> ObjectRows {
        let total: usize = visible.len()
            + transparent.len()
            + refractive.len()
            + casters.iter().map(DrawList::len).sum::<usize>()
            + punctual.iter().map(DrawList::len).sum::<usize>();
        // allocate_slice rejects length 0; an empty scene still needs a bindable
        // buffer, so round up to one (unwritten, unread) slot.
        let buffer = self
            .object_buffer_allocator
            .allocate_slice::<GpuObject>(total.max(1) as u64)
            .unwrap();

        let transparent_base;
        let refractive_base;
        let mut cascade_bases = [0u32; MAX_CASCADES];
        let mut punctual_bases = [0u32; MAX_SHADOW_LIGHTS];
        {
            let mut rows = buffer.write().unwrap();
            let mut next = 0usize;
            let mut write = |list: &DrawList<'_>, next: &mut usize| {
                for i in 0..list.len() {
                    let item = list.item(i);
                    rows[*next] = GpuObject {
                        model: item.model.to_cols_array_2d(),
                        normal_matrix: Mat4::from_mat3(item.normal_matrix).to_cols_array_2d(),
                        prev_model: item.prev_model.to_cols_array_2d(),
                    };
                    *next += 1;
                }
            };
            write(&visible, &mut next);
            transparent_base = next as u32;
            write(&transparent, &mut next);
            refractive_base = next as u32;
            write(&refractive, &mut next);
            for (base, list) in cascade_bases.iter_mut().zip(casters) {
                *base = next as u32;
                write(list, &mut next);
            }
            for (base, list) in punctual_bases.iter_mut().zip(punctual) {
                *base = next as u32;
                write(list, &mut next);
            }
        }
        ObjectRows {
            buffer,
            transparent_base,
            refractive_base,
            cascade_bases,
            punctual_bases,
        }
    }

    /// Build the five descriptor sets both passes into the lit frame bind, and
    /// upload the per-frame blocks two of them point at.
    ///
    /// Once per frame rather than once per pass, and *before* the executor walks
    /// the schedule rather than inside the forward pass's body: the transparency
    /// pass binds the same five, and where the compiler chose to put it relative
    /// to this one is not something either pass may depend on.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_frame(
        &self,
        renderer: &VulkanRenderer,
        lighting: &SceneLighting,
        camera: &Camera,
        extent: [u32; 2],
        ao_view: Arc<ImageView>,
        contact_shadow_view: Arc<ImageView>,
        shadow_view: Arc<ImageView>,
        atlas_view: Arc<ImageView>,
        shadows: Option<ShadowFrame<'_>>,
        atlas: &ShadowAtlas,
        material_set: Arc<DescriptorSet>,
        texture_set: Arc<DescriptorSet>,
        object_set: Arc<DescriptorSet>,
        environment: &EnvironmentSettings,
    ) -> ForwardSets {
        let lighting_buffer = self
            .uniform_buffer_allocator
            .allocate_sized::<GpuLighting>()
            .unwrap();
        // Both halves fall back to the scene's flat ambient when nothing is
        // loaded — the diffuse as a band-0-only series, the specular as a tint
        // on a white cube. Two descriptions of the same uniform environment,
        // which is what keeps them from disagreeing.
        let ambient = lighting.ambient_color * lighting.ambient_nits;
        let irradiance = renderer.environment.irradiance(ambient, environment);
        let env_specular = renderer.environment.specular_tint(ambient, environment);
        *lighting_buffer.write().unwrap() = to_gpu_lighting(
            lighting,
            camera.position,
            extent,
            shadows,
            atlas,
            irradiance,
            environment.yaw,
            env_specular,
        );

        // Always at least one entry: `allocate_slice` rejects a length of zero,
        // and a frame where nothing punctual casts still has to bind something
        // for the descriptor. Every light's face index is -1 in that frame, so
        // the row is never read.
        let face_count = atlas.faces.len().min(MAX_ATLAS_FACES).max(1);
        let shadow_faces = self
            .shadow_face_allocator
            .allocate_slice::<GpuShadowFace>(face_count as u64)
            .unwrap();
        {
            let mut rows = shadow_faces.write().unwrap();
            rows.fill(GpuShadowFace::ZERO);
            for (row, face) in rows.iter_mut().zip(&atlas.faces) {
                *row = GpuShadowFace {
                    view_proj: face.view_proj.to_cols_array_2d(),
                    rect: face.tile.uv_rect(atlas.resolution),
                };
            }
        }

        let lighting_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, lighting_buffer)],
            [],
        )
        .unwrap();

        // Set 3 is the screen-space and shadow inputs. The cascades are kept as
        // a separate image and comparison sampler rather than a combined one,
        // for the same reason the texture array is: Metal caps samplers per
        // stage far lower than sampled images.
        let ao_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[3].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, ao_view, self.ao_sampler.clone()),
                WriteDescriptorSet::image_view(1, shadow_view),
                WriteDescriptorSet::image_view(3, renderer.environment.specular_view()),
                WriteDescriptorSet::sampler(4, renderer.environment.sampler()),
                WriteDescriptorSet::image_view_sampler(
                    5,
                    contact_shadow_view,
                    self.ao_sampler.clone(),
                ),
                // The atlas reads through the cascades' comparison sampler at
                // binding 2 — same conventions, same `Less` against a map
                // cleared to 1.0 — so it needs no sampler of its own. Which
                // matters: Metal caps samplers per stage far below sampled
                // images, and this shader is already at five.
                WriteDescriptorSet::image_view(6, atlas_view),
                WriteDescriptorSet::buffer(7, shadow_faces),
            ],
            [],
        )
        .unwrap();

        ForwardSets {
            sets: vec![lighting_set, material_set, texture_set, ao_set, object_set],
        }
    }

    /// Draw the opaque geometry. `sets` is what
    /// [`begin_frame`](Self::begin_frame) built.
    /// `subsurface` picks the variant that splits the diffusible radiance into a
    /// second target. It is the graph's answer, not this pass's — the executor
    /// reads it off the frame's ids, so the pipeline bound here and the
    /// framebuffer already bound around it cannot disagree about how many
    /// attachments there are.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        draws: DrawList<'_>,
        view: &FrameView,
        extent: [u32; 2],
        sets: &ForwardSets,
        subsurface: bool,
    ) {
        // The jittered one, from the frame's shared view: every pass that
        // rasterises geometry has to agree on it to a subpixel.
        let view_proj = view.view_proj;
        let pipeline = if subsurface {
            &self.subsurface_pipeline
        } else {
            &self.pipeline
        };

        builder
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [extent[0] as f32, extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                sets.as_vec(),
            )
            .unwrap();

        // `extract_geometry` groups the order by (mesh, material), so each run
        // is one instanced draw: the recording cost stops scaling with entity
        // count and starts scaling with distinct mesh/material pairs.
        for run in draws.runs() {
            let item = draws.item(run.start);
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            let push = PushConstants::new(view_proj, item.material.0, run.start as u32);

            builder
                .push_constants(pipeline.layout().clone(), 0, push)
                .unwrap()
                .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(mesh.index_buffer.clone())
                .unwrap();
            unsafe {
                builder
                    .draw_indexed(mesh.index_count, run.len() as u32, 0, 0, 0)
                    .unwrap();
            }
        }
    }
}

/// The material table both geometry passes read.
///
/// One buffer rather than one per pipeline: the prepass writes the `f0` and
/// roughness the forward pass shades with, and two uploads of the same table are
/// two ways for one material to be described differently.
pub(super) fn material_buffer(
    ctx: &VkContext,
    materials: &[GpuMaterial],
) -> Subbuffer<[GpuMaterial]> {
    Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        materials.iter().copied(),
    )
    .expect("failed to allocate material buffer")
}

pub fn upload_mesh(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    vertices: &[Vertex],
    indices: &[u32],
) -> GpuMesh {
    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices.iter().copied(),
    )
    .expect("failed to allocate vertex buffer");

    let index_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        indices.iter().copied(),
    )
    .expect("failed to allocate index buffer");

    GpuMesh {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        bounds: Aabb::from_points(vertices.iter().map(|v| Vec3::from(v.position))),
    }
}

pub(super) fn to_gpu_material(m: &Material) -> GpuMaterial {
    // Derived from what was authored, never from a separate switch: a lobe is on
    // exactly when it would change a pixel. `clearcoat_texture` alone turns the
    // coat on because a map that modulates a zero would otherwise be silently
    // dead, and `transmission` only counts on the queue that owns a pass able to
    // fetch what is behind the surface.
    let mut flags = 0u32;
    if m.clearcoat > 0.0 || m.clearcoat_texture.is_some() {
        flags |= material_flags::CLEARCOAT;
    }
    if m.sheen_color != Vec3::ZERO || m.sheen_texture.is_some() {
        flags |= material_flags::SHEEN;
    }
    if m.anisotropy != 0.0 || m.anisotropy_texture.is_some() {
        flags |= material_flags::ANISOTROPY;
    }
    if m.blend == BlendMode::Transmissive
        && (m.transmission > 0.0 || m.transmission_texture.is_some())
    {
        flags |= material_flags::TRANSMISSION;
    }
    // Deliberately *not* gated on the blend mode, unlike transmission above. The
    // screen-space diffusion only reaches the opaque queue, but the analytic half
    // of scattering is a lighting term like any other, so a blended leaf and an
    // opaque one scatter by the same rule — which is the whole reason
    // `shading.glsl` is one file.
    if m.subsurface_color != Vec3::ZERO || m.subsurface_texture.is_some() {
        flags |= material_flags::SUBSURFACE;
    }
    // The one lobe whose map is not optional, and the exception proves the rule
    // the others follow: a clear coat with no map is a coat of uniform strength,
    // while a height field with no heights is a plane. So this asks for both, and
    // a depth of zero is the same plane said the other way.
    if m.height_texture.is_some() && m.parallax_depth > 0.0 {
        flags |= material_flags::PARALLAX;
    }

    // Beer-Lambert wants an extinction coefficient per unit distance, and an
    // infinite attenuation distance is the "absorbs nothing" case the shader
    // would otherwise reach by dividing by infinity. Zero is that same case
    // expressed as a coefficient, so both collapse to one path there.
    // The march interpolates between these by how head-on the surface is, so a
    // maximum below the minimum would spend *more* samples the less they are
    // needed. Ordered here rather than in the shader, and the upper bound is
    // taken from the already-clamped minimum because `clamp` panics on a range
    // that runs backwards — a material asking for a thousand steps would
    // otherwise supply one.
    let parallax_min = m.parallax_min_steps.clamp(1, PARALLAX_STEP_LIMIT);
    let parallax_max = m
        .parallax_max_steps
        .clamp(parallax_min, PARALLAX_STEP_LIMIT);

    let attenuation_distance = if m.attenuation_distance.is_finite() {
        m.attenuation_distance.max(0.0)
    } else {
        0.0
    };

    // Missing maps fall back to the default textures, which make the sample a
    // no-op (white = ×1, flat normal = unchanged geometric normal).
    GpuMaterial {
        // Opacity rides in `w` because that is where a base colour's alpha
        // belongs and because it costs nothing: the field was already a `vec4`
        // for std430's sake. The opaque pipeline ignores it.
        base_color: [
            m.base_color.x,
            m.base_color.y,
            m.base_color.z,
            m.alpha.clamp(0.0, 1.0),
        ],
        emissive: [m.emissive.x, m.emissive.y, m.emissive.z, 0.0],
        params: [m.metallic, m.roughness, m.reflectance, m.ior.max(1.0)],
        tex_indices: [
            m.albedo_texture.map_or(WHITE_TEXTURE, |h| h.0),
            m.normal_texture.map_or(FLAT_NORMAL_TEXTURE, |h| h.0),
            m.metallic_roughness_texture.map_or(WHITE_TEXTURE, |h| h.0),
            m.emissive_texture.map_or(WHITE_TEXTURE, |h| h.0),
        ],
        clearcoat: [m.clearcoat, m.clearcoat_roughness, 0.0, 0.0],
        sheen: [
            m.sheen_color.x,
            m.sheen_color.y,
            m.sheen_color.z,
            m.sheen_roughness,
        ],
        // The rotation resolves to its sine and cosine here rather than in the
        // shader: it is per material, not per fragment, and a transcendental per
        // pixel to rotate a constant frame is the kind of cost that never shows
        // up in a profile as itself.
        anisotropy: [
            m.anisotropy.clamp(-1.0, 1.0),
            m.anisotropy_rotation.cos(),
            m.anisotropy_rotation.sin(),
            0.0,
        ],
        transmission: [
            m.transmission.clamp(0.0, 1.0),
            m.thickness.max(0.0),
            attenuation_distance,
            0.0,
        ],
        attenuation: [
            m.attenuation_color.x,
            m.attenuation_color.y,
            m.attenuation_color.z,
            0.0,
        ],
        subsurface: [
            m.subsurface_color.x,
            m.subsurface_color.y,
            m.subsurface_color.z,
            // Floored: the lobe is `exp2(power * (cos - 1))`, and a zero power is
            // a lobe that is one in every direction — forward scattering that
            // does not fall off is a uniform glow rather than a halo.
            m.subsurface_forward_scatter.max(1e-3),
        ],
        // Clamped away from zero rather than allowed to reach it: every consumer
        // divides by a mean free path, and a channel that scatters over no
        // distance at all is the "no scattering" case already expressed by a
        // black `subsurface_color`.
        subsurface_radius: [
            m.subsurface_radius.x.max(1e-6),
            m.subsurface_radius.y.max(1e-6),
            m.subsurface_radius.z.max(1e-6),
            0.0,
        ],
        parallax: [
            m.parallax_depth.max(0.0),
            parallax_min as f32,
            parallax_max as f32,
            0.0,
        ],
        tex_indices_ext: [
            m.clearcoat_texture.map_or(WHITE_TEXTURE, |h| h.0),
            m.clearcoat_normal_texture
                .map_or(FLAT_NORMAL_TEXTURE, |h| h.0),
            m.sheen_texture.map_or(WHITE_TEXTURE, |h| h.0),
            // The direction half of this map is signed and decoded as
            // `rg * 2 - 1`, so the neutral fill is the flat normal's (0.5, 0.5)
            // rather than white — which would read as a 45-degree rotation.
            m.anisotropy_texture.map_or(FLAT_NORMAL_TEXTURE, |h| h.0),
        ],
        tex_flags: [
            m.transmission_texture.map_or(WHITE_TEXTURE, |h| h.0),
            flags,
            m.subsurface_texture.map_or(WHITE_TEXTURE, |h| h.0),
            // White is the top of the field everywhere, so an unauthored map is a
            // flat surface rather than one displaced by its full depth — the
            // march reads the same no-op out of it that every other default
            // texture gives its lobe.
            m.height_texture.map_or(WHITE_TEXTURE, |h| h.0),
        ],
    }
}

/// Blend states for a pipeline sharing the forward render pass but writing only
/// its colour target — the skybox and the debug lines.
///
/// The subsurface variant of that render pass has a second colour attachment
/// those two shaders know nothing about, and Vulkan requires a blend state per
/// attachment regardless. Leaving the extra one at its default would let them
/// write whatever their fragment shader happened to leave in an output it never
/// declared, which is undefined — and the sky covers most of the frame, so the
/// undefined value would be sitting in the scatter target under every blur tap
/// near a silhouette. An empty write mask is the whole fix, and unlike disabling
/// the write it needs no device feature: the attachment keeps its clear, and a
/// clear of zero is exactly "nothing scattered here".
pub(super) fn co_tenant_blend_states(subpass: &Subpass) -> ColorBlendState {
    let mut state = ColorBlendState::with_attachment_states(
        subpass.num_color_attachments(),
        ColorBlendAttachmentState::default(),
    );
    for attachment in state.attachments.iter_mut().skip(1) {
        attachment.color_write_mask =
            vulkano::pipeline::graphics::color_blend::ColorComponents::empty();
    }
    state
}

/// The vertex shader both passes into the lit frame rasterise with. A blended
/// surface is the same geometry read from the same per-object rows, and
/// `shading.glsl` reads the same varyings out of it whichever fragment shader
/// includes it.
pub(super) fn vertex_shader(device: &Arc<Device>) -> vulkano::shader::EntryPoint {
    vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap()
}

/// Both opaque pipelines, differing only in the fragment shader and the render
/// pass it targets.
///
/// The layout is derived from the stages, so the two come out identical — which is
/// what has to be true: `oit.rs` and `refraction.rs` build their pipelines from
/// [`ForwardPass::pipeline_layout`], and the five descriptor sets the executor
/// binds are bound once for whichever opaque variant ran.
fn build_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPass>,
    fragment: Arc<vulkano::shader::ShaderModule>,
    // Passed in rather than made here, and that is the whole reason this parameter
    // exists: Vulkan compares immutable samplers by *identity*, so two calls to
    // `comparison_sampler` — identical in every field — produce two set-3 layouts
    // that are not compatible. The executor binds one set of five descriptor sets
    // for whichever opaque variant ran, so both pipelines have to have been built
    // around the same sampler object. The same rule `refraction.rs` obeys by
    // lifting this layout instead of deriving it.
    shadow_sampler: &Arc<Sampler>,
) -> Arc<GraphicsPipeline> {
    let vs = vertex_shader(device);
    let fs = fragment.entry_point("main").unwrap();

    let vertex_input_state = Vertex::per_vertex().definition(&vs).unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    // The shadow comparison sampler has to be immutable — part of the layout
    // rather than something written into a descriptor set — because MoltenVK
    // cannot accept a written one. Everything else in the layout is still
    // derived from the shaders' own interface.
    let mut layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
    layout_info.set_layouts[SHADOW_SET]
        .bindings
        .get_mut(&SHADOW_SAMPLER_BINDING)
        .expect("forward.frag must declare the shadow comparison sampler")
        .immutable_samplers = vec![shadow_sampler.clone()];

    let layout = PipelineLayout::new(
        device.clone(),
        layout_info
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState {
                rasterization_samples: vulkano::image::SampleCount::Sample4,
                ..Default::default()
            }),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/forward.vert",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/forward.frag",
        include: ["shaders"],
    }
}

mod fs_sss {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/forward_sss.frag",
        include: ["shaders"],
    }
}
