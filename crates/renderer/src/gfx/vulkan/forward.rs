use std::sync::Arc;

use glam::{Mat3, Mat4, Vec3};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
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

use crate::gfx::{RenderItem, SceneLighting, Vertex, MAX_POINT_LIGHTS};
use crate::scene::Camera;

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
}

/// Pack the engine's [`SceneLighting`] into the std140 layout the shader expects.
fn to_gpu_lighting(lighting: &SceneLighting, camera_pos: Vec3) -> GpuLighting {
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
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
}

pub struct ForwardPass {
    pub render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    /// Pool we sub-allocate the per-frame lighting UBO from.
    uniform_buffer_allocator: SubbufferAllocator,
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

        Self {
            render_pass,
            pipeline,
            uniform_buffer_allocator,
        }
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
    ) {
        let aspect = extent[0] as f32 / extent[1] as f32;
        let view_proj = camera.view_projection(aspect);

        // Upload this frame's lighting into a fresh sub-buffer and bind it as the
        // pipeline's set 0. The same descriptor stays bound for every draw.
        let lighting_buffer = self
            .uniform_buffer_allocator
            .allocate_sized::<GpuLighting>()
            .unwrap();
        *lighting_buffer.write().unwrap() = to_gpu_lighting(lighting, camera.position);

        let lighting_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, lighting_buffer)],
            [],
        )
        .unwrap();

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
                lighting_set,
            )
            .unwrap();

        for item in items {
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            let model = item.model;
            // Normals transform by the inverse-transpose so they stay perpendicular
            // to surfaces under non-uniform scaling.
            let normal_matrix = Mat4::from_mat3(Mat3::from_mat4(model).inverse().transpose());
            let push = PushConstants {
                mvp: (view_proj * model).to_cols_array_2d(),
                model: model.to_cols_array_2d(),
                normal_matrix: normal_matrix.to_cols_array_2d(),
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

            layout(location = 0) out vec3 v_world_pos;
            layout(location = 1) out vec3 v_normal;
            layout(location = 2) out vec3 v_color;

            layout(push_constant) uniform Push {
                mat4 mvp;
                mat4 model;
                mat4 normal_matrix;
            } push;

            void main() {
                vec4 world = push.model * vec4(position, 1.0);
                v_world_pos = world.xyz;
                v_normal = mat3(push.normal_matrix) * normal;
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
            layout(location = 2) in vec3 v_color;

            layout(location = 0) out vec4 f_color;

            // Keep in sync with MAX_POINT_LIGHTS in forward.rs.
            const int MAX_POINT_LIGHTS = 16;

            struct PointLight {
                vec4 position; // xyz = world position, w = range
                vec4 color;    // rgb = color,         w = intensity
            };

            layout(set = 0, binding = 0) uniform Lighting {
                vec4 camera_pos;    // xyz = camera world position
                vec4 ambient;       // rgb = color,    w = intensity
                vec4 sun_direction; // xyz = direction toward the sun (normalized)
                vec4 sun_color;     // rgb = color,    w = intensity
                vec4 params;        // x = point light count, y = shininess, z = specular strength
                PointLight point_lights[MAX_POINT_LIGHTS];
            } lighting;

            // Blinn-Phong response for a single light direction L with incoming
            // radiance, given surface normal N and view direction V.
            vec3 shade(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo,
                       float shininess, float specular_strength) {
                float ndotl = max(dot(N, L), 0.0);
                vec3 diffuse = albedo * ndotl;

                // Half-vector specular; gated by ndotl so highlights don't bleed
                // onto faces turned away from the light.
                vec3 H = normalize(L + V);
                float spec = (ndotl > 0.0) ? pow(max(dot(N, H), 0.0), shininess) : 0.0;
                vec3 specular = vec3(specular_strength * spec);

                return radiance * (diffuse + specular);
            }

            // Smooth, range-limited falloff (windowed inverse-square).
            float attenuate(float dist, float range) {
                float s = dist / max(range, 1e-4);
                if (s >= 1.0) return 0.0;
                float window = 1.0 - s * s;
                return (window * window) / max(dist * dist, 1e-4);
            }

            void main() {
                vec3 N = normalize(v_normal);
                vec3 V = normalize(lighting.camera_pos.xyz - v_world_pos);
                vec3 albedo = v_color;

                float shininess = lighting.params.y;
                float specular_strength = lighting.params.z;

                // Ambient fill.
                vec3 color = lighting.ambient.rgb * lighting.ambient.w * albedo;

                // Directional sun.
                {
                    vec3 L = normalize(lighting.sun_direction.xyz);
                    vec3 radiance = lighting.sun_color.rgb * lighting.sun_color.w;
                    color += shade(N, V, L, radiance, albedo, shininess, specular_strength);
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
                    color += shade(N, V, L, radiance, albedo, shininess, specular_strength);
                }

                f_color = vec4(color, 1.0);
            }
        ",
    }
}
