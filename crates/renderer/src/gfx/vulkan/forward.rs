use std::sync::Arc;

use glam::{Mat3, Mat4, Vec3};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as _, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{RenderPass, Subpass};

use crate::gfx::{Material, RenderItem, SceneLighting, Vertex, MAX_POINT_LIGHTS, MAX_TEXTURES};
use crate::scene::Camera;

use super::context::VkContext;
use super::swapchain::DEPTH_FORMAT;
use super::VulkanRenderer;

pub struct GpuMesh {
    pub vertex_buffer: Subbuffer<[Vertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub index_count: u32,
}

/// Per-draw data. Stays in push constants because it changes for every object.
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    /// Inverse-transpose of `model`'s rotation/scale, for transforming normals
    /// correctly under non-uniform scaling. Stored as a mat4; only the upper-left
    /// 3x3 is used in the shader.
    normal_matrix: [[f32; 4]; 4],
    material_index: u32,
}

/// Default texture indices, matching the order `VulkanRenderer::new` seeds them.
const WHITE_TEXTURE: u32 = 0;
const FLAT_NORMAL_TEXTURE: u32 = 1;

#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub(crate) struct GpuMaterial {
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4], // metallic, roughness, reflectance
    /// Indices into the set-2 texture array: [albedo, normal, metal-rough, emissive].
    tex_indices: [u32; 4],
}

/// Pack the engine's [`SceneLighting`] into the std140 layout the shader expects.
fn to_gpu_lighting(lighting: &SceneLighting, camera_pos: Vec3, extent: [u32; 2]) -> GpuLighting {
    let (w, h) = (extent[0] as f32, extent[1] as f32);
    let count = lighting.point_lights.len().min(MAX_POINT_LIGHTS);
    let mut point_lights = [GpuPointLight::ZERO; MAX_POINT_LIGHTS];
    for (slot, light) in point_lights
        .iter_mut()
        .zip(lighting.point_lights.iter().take(count))
    {
        slot.position = [
            light.position.x,
            light.position.y,
            light.position.z,
            light.range.max(1e-4),
        ];
        slot.color = [light.color.x, light.color.y, light.color.z, light.intensity];
    }

    // The shader wants the direction *toward* the light, so negate.
    let to_sun = (-lighting.sun.direction).normalize_or_zero();

    GpuLighting {
        camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
        ambient: [
            lighting.ambient_color.x,
            lighting.ambient_color.y,
            lighting.ambient_color.z,
            lighting.ambient_intensity,
        ],
        sun_direction: [to_sun.x, to_sun.y, to_sun.z, 0.0],
        sun_color: [
            lighting.sun.color.x,
            lighting.sun.color.y,
            lighting.sun.color.z,
            lighting.sun.intensity,
        ],
        params: [
            count as f32,
            lighting.shininess,
            lighting.specular_strength,
            0.0,
        ],
        viewport: [w, h, 1.0 / w, 1.0 / h],
        point_lights,
    }
}

/// GPU mirror of a [`PointLight`](crate::gfx::PointLight), padded to std140
/// (two `vec4`s).
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuPointLight {
    /// xyz = world position, w = range.
    position: [f32; 4],
    /// rgb = color, w = intensity.
    color: [f32; 4],
}

impl GpuPointLight {
    const ZERO: Self = Self {
        position: [0.0; 4],
        color: [0.0; 4],
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
    /// rgb = ambient color, w = ambient intensity.
    ambient: [f32; 4],
    /// xyz = normalized direction toward the sun.
    sun_direction: [f32; 4],
    /// rgb = sun color, w = sun intensity.
    sun_color: [f32; 4],
    /// x = point light count, y = shininess, z = specular strength.
    params: [f32; 4],
    /// x=w, y=h, z=1/w, w=1/h
    viewport: [f32; 4],
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
}

pub struct ForwardPass {
    pub render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    /// Pool we sub-allocate the per-frame lighting UBO from.
    uniform_buffer_allocator: SubbufferAllocator,
    /// Shared sampler used for every texture in the set-2 array.
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

        let pipeline = build_pipeline(device, &render_pass);

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
        )
        .unwrap();

        let ao_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        ).unwrap();

        Self {
            render_pass,
            pipeline,
            uniform_buffer_allocator,
            sampler,
            ao_sampler
        }
    }

    /// Build the set-1 material storage buffer + descriptor set. Cached by the
    /// renderer and only rebuilt when the material table changes.
    pub fn build_material_set(
        &self,
        ctx: &VkContext,
        materials: &[GpuMaterial],
    ) -> Arc<DescriptorSet> {
        let buffer = Buffer::from_iter(
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
        .expect("failed to allocate material buffer");

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
        let texture_array =
            (0..MAX_TEXTURES).map(|i| textures.get(i).cloned().unwrap_or_else(|| default_view.clone()));
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

    pub fn draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            vulkano::command_buffer::PrimaryAutoCommandBuffer,
        >,
        renderer: &VulkanRenderer,
        items: &[RenderItem],
        lighting: &SceneLighting,
        camera: &Camera,
        extent: [u32; 2],
        ao_view: Arc<ImageView>,
        material_set: Arc<DescriptorSet>,
        texture_set: Arc<DescriptorSet>,
    ) {
        let aspect = extent[0] as f32 / extent[1] as f32;
        let view_proj = camera.view_projection(aspect);

        // Upload this frame's lighting into a fresh sub-buffer and bind it as the
        // pipeline's set 0. The same descriptor stays bound for every draw.
        let lighting_buffer = self
            .uniform_buffer_allocator
            .allocate_sized::<GpuLighting>()
            .unwrap();
        *lighting_buffer.write().unwrap() = to_gpu_lighting(lighting, camera.position, extent);

        let lighting_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, lighting_buffer)],
            [],
        )
        .unwrap();

        let ao_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[3].clone(),
            [WriteDescriptorSet::image_view_sampler(0, ao_view, self.ao_sampler.clone())],
            [],
        ).unwrap();
        
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
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                vec![lighting_set, material_set, texture_set, ao_set],
            )
            .unwrap();

        for item in items {
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            let model = item.model;
            let material_index = item.material.0;
            // Normals transform by the inverse-transpose so they stay perpendicular
            // to surfaces under non-uniform scaling.
            let normal_matrix = Mat4::from_mat3(Mat3::from_mat4(model).inverse().transpose());
            let push = PushConstants {
                mvp: (view_proj * model).to_cols_array_2d(),
                model: model.to_cols_array_2d(),
                normal_matrix: normal_matrix.to_cols_array_2d(),
                material_index,
            };

            builder
                .push_constants(self.pipeline.layout().clone(), 0, push)
                .unwrap()
                .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(mesh.index_buffer.clone())
                .unwrap();
            unsafe {
                builder.draw_indexed(mesh.index_count, 1, 0, 0, 0).unwrap();
            }
        }
    }
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
    }
}

pub(super) fn to_gpu_material(m: &Material) -> GpuMaterial {
    // Missing maps fall back to the default textures, which make the sample a
    // no-op (white = ×1, flat normal = unchanged geometric normal).
    GpuMaterial {
        base_color: [m.base_color.x, m.base_color.y, m.base_color.z, 1.0],
        emissive: [m.emissive.x, m.emissive.y, m.emissive.z, 0.0],
        params: [m.metallic, m.roughness, m.reflectance, 0.0],
        tex_indices: [
            m.albedo_texture.map_or(WHITE_TEXTURE, |h| h.0),
            m.normal_texture.map_or(FLAT_NORMAL_TEXTURE, |h| h.0),
            m.metallic_roughness_texture.map_or(WHITE_TEXTURE, |h| h.0),
            m.emissive_texture.map_or(WHITE_TEXTURE, |h| h.0),
        ],
    }
}

fn build_pipeline(device: &Arc<Device>, render_pass: &Arc<RenderPass>) -> Arc<GraphicsPipeline> {
    let vs = vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let vertex_input_state = Vertex::per_vertex().definition(&vs).unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
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
        src: r"
            #version 460

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec3 color;
            layout(location = 3) in vec2 uv;
            layout(location = 4) in vec4 tangent; // xyz = tangent, w = handedness

            layout(location = 0) out vec3 v_world_pos;
            layout(location = 1) out vec3 v_normal;
            layout(location = 2) out vec3 v_tangent;
            layout(location = 3) out vec3 v_bitangent;
            layout(location = 4) out vec2 v_uv;
            layout(location = 5) out vec3 v_color;

            // Declared identically to the fragment shader so the two stages
            // share one push-constant range. `material_index` is unused here.
            layout(push_constant) uniform Push {
                mat4 mvp;
                mat4 model;
                mat4 normal_matrix;
                uint material_index;
            } push;

            void main() {
                vec4 world = push.model * vec4(position, 1.0);
                v_world_pos = world.xyz;

                // World-space TBN basis for tangent-space normal mapping.
                vec3 N = normalize(mat3(push.normal_matrix) * normal);
                vec3 T = normalize(mat3(push.model) * tangent.xyz);
                T = normalize(T - dot(T, N) * N);          // Gram-Schmidt
                vec3 B = cross(N, T) * tangent.w;          // handedness from w

                v_normal = N;
                v_tangent = T;
                v_bitangent = B;
                v_uv = uv;
                v_color = color;
                gl_Position = push.mvp * vec4(position, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) in vec3 v_world_pos;
            layout(location = 1) in vec3 v_normal;
            layout(location = 2) in vec3 v_tangent;
            layout(location = 3) in vec3 v_bitangent;
            layout(location = 4) in vec2 v_uv;
            layout(location = 5) in vec3 v_color;

            layout(location = 0) out vec4 f_color;

            // Keep in sync with MAX_POINT_LIGHTS / MAX_TEXTURES in forward.rs.
            const int MAX_POINT_LIGHTS = 16;
            const int MAX_TEXTURES = 64;
            const float PI = 3.14159265359;

            struct PointLight {
                vec4 position; // xyz = world position, w = range
                vec4 color;    // rgb = color,         w = intensity
            };

            layout(set = 0, binding = 0) uniform Lighting {
                vec4 camera_pos;    // xyz = camera world position
                vec4 ambient;       // rgb = color, w = intensity
                vec4 sun_direction; // xyz = direction toward the sun (normalized)
                vec4 sun_color;     // rgb = color, w = intensity
                vec4 params;        // x = point light count (y,z legacy, unused by PBR)
                vec4 viewport;      // x=w, y=h, z=1/w, w=1/h
                PointLight point_lights[MAX_POINT_LIGHTS];
            } lighting;

            // Mirrors GpuMaterial in forward.rs. std430 packs this exactly like
            // the Rust #[repr(C)] struct because every field is 16 bytes.
            struct GpuMaterial {
                vec4 base_color;   // rgb = albedo
                vec4 emissive;     // rgb = emissive
                vec4 params;       // x = metallic, y = roughness, z = reflectance
                uvec4 tex_indices; // x=albedo, y=normal, z=metal-rough, w=emissive
            };

            // Material table indexed by the per-draw material_index. A storage
            // buffer so the array can be sized at runtime (one entry per material).
            layout(set = 1, binding = 0, std430) readonly buffer Materials {
                GpuMaterial materials[];
            };

            // Textures are kept separate from the sampler: Metal/MoltenVK allows
            // only 16 sampler states per stage but many sampled images, so a
            // combined sampler2D[64] would blow the sampler limit. One shared
            // sampler + an array of texture2D stays well under it.
            layout(set = 2, binding = 0) uniform texture2D textures[MAX_TEXTURES];
            layout(set = 2, binding = 1) uniform sampler tex_sampler;

            // Screen-space ambient occlusion (blurred), sampled by screen-space UV.
            layout(set = 3, binding = 0) uniform sampler2D u_ao;

            // Index is dynamically uniform (from the material), so plain indexing
            // is legal without the nonuniform qualifier.
            vec4 sample_tex(uint index, vec2 uv) {
                return texture(sampler2D(textures[index], tex_sampler), uv);
            }

            // Declared identically to the vertex shader so the stages share one
            // push-constant range; only material_index is read here.
            layout(push_constant) uniform Push {
                mat4 mvp;
                mat4 model;
                mat4 normal_matrix;
                uint material_index;
            } push;

            // --- Cook-Torrance terms (metallic-roughness workflow) ---

            // GGX / Trowbridge-Reitz normal distribution.
            float distribution_ggx(float n_dot_h, float a) {
                float a2 = a * a;
                float d = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
                return a2 / max(PI * d * d, 1e-7);
            }

            // Smith height-correlated visibility (already folds in the 1/(4 NoL NoV) denom).
            float visibility_smith_ggx(float n_dot_v, float n_dot_l, float a) {
                float a2 = a * a;
                float gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
                float gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
                return 0.5 / max(gv + gl, 1e-5);
            }

            // Fresnel-Schlick reflectance.
            vec3 fresnel_schlick(float v_dot_h, vec3 f0) {
                return f0 + (1.0 - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
            }

            // Outgoing radiance toward the camera from one light direction L.
            vec3 brdf(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo,
                      float metallic, float roughness, vec3 f0) {
                float n_dot_l = max(dot(N, L), 0.0);
                if (n_dot_l <= 0.0) {
                    return vec3(0.0);
                }
                vec3 H = normalize(L + V);
                float n_dot_v = max(dot(N, V), 1e-4);
                float n_dot_h = max(dot(N, H), 0.0);
                float v_dot_h = max(dot(V, H), 0.0);

                float a = roughness * roughness; // perceptual -> linear roughness

                float D = distribution_ggx(n_dot_h, a);
                float Vis = visibility_smith_ggx(n_dot_v, n_dot_l, a);
                vec3 F = fresnel_schlick(v_dot_h, f0);

                vec3 specular = D * Vis * F;

                // Diffuse keeps the energy not reflected (1 - F) and not metallic.
                vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
                vec3 diffuse = kd * albedo / PI;

                return (diffuse + specular) * radiance * n_dot_l;
            }

            // Smooth, range-limited falloff (windowed inverse-square).
            float attenuate(float dist, float range) {
                float s = dist / max(range, 1e-4);
                if (s >= 1.0) return 0.0;
                float window = 1.0 - s * s;
                return (window * window) / max(dist * dist, 1e-4);
            }

            void main() {
                GpuMaterial m = materials[push.material_index];

                // Sample the maps. Missing maps point at the default textures, so
                // these multiplies become no-ops. Albedo/emissive images are sRGB
                // (decoded to linear on sample); metal-rough is linear data.
                vec3 albedo_tex = sample_tex(m.tex_indices.x, v_uv).rgb;
                vec4 mr_tex     = sample_tex(m.tex_indices.z, v_uv);
                vec3 emis_tex   = sample_tex(m.tex_indices.w, v_uv).rgb;

                // Vertex color tints the material albedo; drop `* v_color` for a
                // pure material/texture color.
                vec3  albedo      = m.base_color.rgb * v_color * albedo_tex;
                // glTF metallic-roughness convention: G = roughness, B = metallic.
                float metallic    = clamp(m.params.x * mr_tex.b, 0.0, 1.0);
                float roughness   = clamp(m.params.y * mr_tex.g, 0.04, 1.0); // floor avoids a singular highlight
                float reflectance = m.params.z;

                // Dielectric F0 from reflectance (0.5 -> ~4%); metals use albedo as F0.
                vec3 f0 = mix(vec3(0.16 * reflectance * reflectance), albedo, metallic);

                // Tangent-space normal map -> world space via the TBN basis.
                vec3 n_tangent = sample_tex(m.tex_indices.y, v_uv).xyz * 2.0 - 1.0;
                mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
                vec3 N = normalize(TBN * n_tangent);

                vec3 V = normalize(lighting.camera_pos.xyz - v_world_pos);

                // Crude diffuse ambient (stands in for image-based lighting),
                // attenuated by screen-space ambient occlusion.
                float ao = texture(u_ao, gl_FragCoord.xy * lighting.viewport.zw).r;
                vec3 color = lighting.ambient.rgb * lighting.ambient.w * albedo * ao;

                // Directional sun.
                {
                    vec3 L = normalize(lighting.sun_direction.xyz);
                    vec3 radiance = lighting.sun_color.rgb * lighting.sun_color.w;
                    color += brdf(N, V, L, radiance, albedo, metallic, roughness, f0);
                }

                // Point lights.
                int count = int(lighting.params.x);
                for (int i = 0; i < count; ++i) {
                    PointLight light = lighting.point_lights[i];
                    vec3 to_light = light.position.xyz - v_world_pos;
                    float dist = length(to_light);
                    float atten = attenuate(dist, light.position.w);
                    if (atten <= 0.0) continue;
                    vec3 L = to_light / max(dist, 1e-4);
                    vec3 radiance = light.color.rgb * light.color.w * atten;
                    color += brdf(N, V, L, radiance, albedo, metallic, roughness, f0);
                }

                // Emissive adds on top, unaffected by scene lighting.
                color += m.emissive.rgb * emis_tex;

                f_color = vec4(color, 1.0);
            }
        ",
    }
}
