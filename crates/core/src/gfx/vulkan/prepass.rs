//! The one geometry pass in front of shading: view-space normals, screen-space
//! motion, the specular colour a surface reflects with, and the depth all three
//! are read against.
//!
//! It started as SSAO's private prepass and is shared now because TAA needs the
//! same rasterisation for a different attachment. Everything downstream of it —
//! SSAO, TAA, motion blur, depth of field, screen-space reflections — reads the
//! targets it leaves rather than rasterising the scene again.
//!
//! Reflections are what made it sample material maps rather than only transform
//! vertices. A reflection has to be tinted by the surface's own `f0` and
//! sharpened or spread by its roughness, and neither survives as a per-object
//! constant once a metal-rough map is involved; the normal it reflects about is
//! the mapped one for the same reason, or reflections slide over bumps the lit
//! image clearly has. SSAO reads the better normals too.

use std::sync::Arc;

use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::MemoryTypeFilter;
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

use crate::gfx::{DrawList, Vertex};

use super::VulkanRenderer;
use super::context::VkContext;
use super::instances::GpuObject;
use super::swapchain::DEPTH_FORMAT;
use super::taa::FrameView;

/// Where `prepass.frag` declares the material texture array. One set later than
/// the forward pass's, because this pipeline carries a frame block the other
/// takes from set 0 — the array itself is binding 0 of it either way, which is
/// what [`VkContext::mark_texture_array_partial`] assumes.
const TEXTURE_SET: usize = 3;

pub(super) const NORMAL_FORMAT: Format = Format::R8G8B8A8_UNORM;

/// `rgb` = the surface's normal-incidence specular colour, `a` = perceptual
/// roughness.
///
/// `f0` rather than base colour and metallic, because `f0` is the only thing
/// downstream actually wants and packing it here means the mix a metal implies
/// is done once, next to the material that decides it. Eight bits costs a
/// dielectric's ~4% about a thousandth in absolute terms, which lands well
/// inside the error the split-sum approximation already carries.
pub(super) const MATERIAL_FORMAT: Format = Format::R8G8B8A8_UNORM;

/// Signed and float, unlike the normal target: a motion vector is a UV *delta*,
/// so it is negative half the time, and a UNORM encoding would need a bias that
/// costs precision exactly where the vectors are smallest and matter most.
pub(super) const VELOCITY_FORMAT: Format = Format::R16G16_SFLOAT;

/// The per-frame camera block. Shared with the SSAO passes, which reconstruct
/// view-space positions from the same depth this pass wrote and so must use the
/// same — jittered — projection to do it.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct FrameUbo {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    jitter: [f32; 4],
    /// `proj * view`, premultiplied on the CPU — the identical `Mat4` the
    /// forward pass pushes as a push constant.
    ///
    /// Last in the block rather than beside `proj`, and that position is
    /// load-bearing: `ssao.frag` and `contact_shadows.frag` declare only the
    /// first three matrices of this block, which is legal exactly as long as
    /// what they declare stays a prefix of what is uploaded. Appending keeps it
    /// one; inserting would silently hand both of them the wrong matrix.
    ///
    /// It exists so `prepass.vert` can compute `gl_Position` from the same
    /// expression `forward.vert` does. Multiplying `proj * (view * world)` and
    /// `(proj * view) * world` are equal in exact arithmetic and not in floating
    /// point, and the forward pass now depth-tests `EQUAL` against the depth
    /// this pass wrote — so a difference in the last bit is a surface that
    /// vanishes. See the `invariant gl_Position` in both shaders.
    view_proj: [[f32; 4]; 4],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PrepassPush {
    /// First object row of this instanced run; the shader adds `gl_InstanceIndex`.
    object_base: u32,
    /// Row of the set-2 material table this run draws with, as the forward pass
    /// pushes it. A run is one (mesh, material) pair, so one index covers it.
    material_index: u32,
}

pub struct GeometryPrepass {
    pub(super) render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    /// The same pass for a `Masked` run: back faces kept, and a fragment shader
    /// that cuts the texels below the material's cutoff away.
    ///
    /// A second pipeline rather than a flag the one shader tests, because a
    /// `discard` anywhere in a shader costs *every* draw through it its early
    /// depth test — and this pass exists to be the depth everything downstream
    /// reads. Two pipelines at startup is the price of keeping that cost on the
    /// foliage alone.
    masked_pipeline: Arc<GraphicsPipeline>,
    /// The material maps are sampled at the same texture coordinates and the
    /// same mip selection the forward pass uses, so the two rasterisations
    /// cannot disagree about what a surface is.
    sampler: Arc<Sampler>,
    uniform_allocator: SubbufferAllocator,
}

impl GeometryPrepass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let render_pass = build_render_pass(device);
        let pipeline = build_pipeline(
            ctx,
            &render_pass,
            prepass_fs::load(device.clone()).unwrap(),
            false,
        );
        let masked_pipeline = build_pipeline(
            ctx,
            &render_pass,
            prepass_fs_masked::load(device.clone()).unwrap(),
            true,
        );
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
        let uniform_allocator = SubbufferAllocator::new(
            ctx.memory_allocator.clone(),
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
            masked_pipeline,
            sampler,
            uniform_allocator,
        }
    }

    /// Upload the camera block once for every pass in the frame that reads it.
    pub(super) fn begin_frame(&self, view: &FrameView) -> Subbuffer<FrameUbo> {
        let frame = self.uniform_allocator.allocate_sized::<FrameUbo>().unwrap();
        *frame.write().unwrap() = FrameUbo {
            view: view.view.to_cols_array_2d(),
            proj: view.proj.to_cols_array_2d(),
            inv_proj: view.proj.inverse().to_cols_array_2d(),
            prev_view_proj: view.prev_view_proj.to_cols_array_2d(),
            jitter: [view.jitter.x, view.jitter.y, 0.0, 0.0],
            view_proj: view.view_proj.to_cols_array_2d(),
        };
        frame
    }

    /// The sampler this pass reads material maps through.
    ///
    /// Lent to the shadow pass's cutout pipeline rather than duplicated there:
    /// an alpha test that filtered differently from the two passes it has to
    /// agree with would cut the leaf out at a slightly different place in the
    /// shadow map than in the frame, which reads as the shadow being offset.
    pub(super) fn material_sampler(&self) -> &Arc<Sampler> {
        &self.sampler
    }

    /// The set-1 per-object descriptor set this pass binds.
    pub(super) fn build_object_set(
        &self,
        ctx: &VkContext,
        rows: &Subbuffer<[GpuObject]>,
        indices: &Subbuffer<[u32]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[1].clone(),
            [
                WriteDescriptorSet::buffer(0, rows.clone()),
                WriteDescriptorSet::buffer(1, indices.clone()),
            ],
            [],
        )
        .unwrap()
    }

    /// The set-2 material table, over the same buffer the forward pass reads.
    ///
    /// Its own set rather than the forward pass's, for the reason the object
    /// sets are kept apart: set compatibility is a property of the layout each
    /// pipeline declares, not of the buffer written into it. The *data* is
    /// shared, which is what keeps the two passes agreeing about a material.
    pub(super) fn build_material_set(
        &self,
        ctx: &VkContext,
        materials: &Subbuffer<[super::forward::GpuMaterial]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[2].clone(),
            [WriteDescriptorSet::buffer(0, materials.clone())],
            [],
        )
        .unwrap()
    }

    /// The set-3 texture array, filled exactly as the forward pass fills its
    /// own: every unused slot points at the white default so no descriptor is
    /// left unwritten.
    pub(super) fn build_texture_set(
        &self,
        ctx: &VkContext,
        textures: &[Arc<ImageView>],
    ) -> Arc<DescriptorSet> {
        let default_view = textures[0].clone();
        let texture_array = (0..ctx.texture_array_len(textures.len())).map(|index| {
            textures
                .get(index)
                .cloned()
                .unwrap_or_else(|| default_view.clone())
        });
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[TEXTURE_SET].clone(),
            [
                WriteDescriptorSet::image_view_array(0, 0, texture_array),
                WriteDescriptorSet::sampler(1, self.sampler.clone()),
            ],
            [],
        )
        .unwrap()
    }

    pub(super) fn record(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        draws: DrawList<'_>,
        extent: [u32; 2],
        frame: Subbuffer<FrameUbo>,
        decals: Subbuffer<super::forward::GpuDecals>,
        object_set: Arc<DescriptorSet>,
        material_set: Arc<DescriptorSet>,
        texture_set: Arc<DescriptorSet>,
    ) {
        let frame_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, frame),
                // The very buffer object the forward pass's set 0 binds. See
                // `ForwardPass::upload_decals` — one allocation bound twice is
                // what keeps the two passes from stamping different decals onto
                // the same pixel.
                WriteDescriptorSet::buffer(1, decals),
            ],
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
            .unwrap();

        let sets = vec![frame_set, object_set, material_set, texture_set];
        let mut bound: Option<bool> = None;

        // One instanced draw per (mesh, material) run, matching the forward pass.
        // The model and normal matrices come from the shared object buffer, so
        // this pass no longer recomputes an inverse-transpose per item.
        for run in draws.runs() {
            let item = draws.item(run.start);
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            // The same question the forward pass asks of the same flag word, so
            // the two passes cut a cutout out at the same place. See
            // `GpuMaterial::is_masked`.
            let wants_masked = renderer
                .materials
                .get(item.material.0 as usize)
                .is_some_and(super::forward::GpuMaterial::is_masked);
            let pipeline = if wants_masked {
                &self.masked_pipeline
            } else {
                &self.pipeline
            };
            if bound != Some(wants_masked) {
                builder
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        sets.clone(),
                    )
                    .unwrap();
                bound = Some(wants_masked);
            }
            let push = PrepassPush {
                object_base: run.start as u32,
                material_index: item.material.0,
            };
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
                    .unwrap()
            };
        }
    }
}

fn build_render_pass(device: &Arc<Device>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            normal:   { format: NORMAL_FORMAT,   samples: 1, load_op: Clear, store_op: Store },
            velocity: { format: VELOCITY_FORMAT, samples: 1, load_op: Clear, store_op: Store },
            material: { format: MATERIAL_FORMAT, samples: 1, load_op: Clear, store_op: Store },
            depth:    { format: DEPTH_FORMAT,    samples: 1, load_op: Clear, store_op: Store },
        },
        pass: { color: [normal, velocity, material], depth_stencil: {depth}}
    )
    .unwrap()
}

fn build_pipeline(
    ctx: &VkContext,
    render_pass: &Arc<RenderPass>,
    fragment: Arc<vulkano::shader::ShaderModule>,
    masked: bool,
) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
    let vs = prepass_vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = fragment.entry_point("main").unwrap();
    let vertex_input_state = Vertex::per_vertex().definition(&vs).unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    let mut layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
    ctx.mark_texture_array_partial(&mut layout_info, TEXTURE_SET);
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
        ctx.pipeline_cache(),
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                // Two-sided for a cutout, matching the forward pass. The two
                // rasterise the same triangles or the depth this pass leaves is
                // not the depth the frame was shaded against.
                cull_mode: if masked {
                    CullMode::None
                } else {
                    CullMode::Back
                },
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
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

mod prepass_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/prepass.vert" }
}
mod prepass_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/prepass.frag",
        include: ["shaders"],
    }
}
/// The same source with the cutout's `discard` compiled in. Two modules rather
/// than two files, because the difference really is one define — everything
/// about what this pass reports stays in the one file both compile from.
mod prepass_fs_masked {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/prepass.frag",
        include: ["shaders"],
        define: [("ORRIN_MASKED", "1")],
    }
}
