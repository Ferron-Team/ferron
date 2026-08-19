use std::sync::Arc;

use glam::Mat4;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, ClearDepthStencilImageInfo, CommandBufferUsage,
    PrimaryAutoCommandBuffer,
};
use vulkano::command_buffer::{ClearAttachment, ClearRect};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::ClearDepthStencilValue;
use vulkano::image::sampler::{
    BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::{ImageLayout, SampleCount};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, DepthBiasState, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as _, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{
    AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, RenderPass,
    RenderPassCreateInfo, Subpass, SubpassDescription,
};
use vulkano::sync::GpuFuture;

use crate::gfx::punctual::ShadowAtlas;
use crate::gfx::{DrawList, PositionVertex};

use super::VulkanRenderer;
use super::context::VkContext;
use super::instances::GpuObject;
use super::swapchain::DEPTH_FORMAT;

/// Where `shadow.frag` declares the material texture array, in the masked
/// variant that alpha-tests; the plain one declares no such set. Binding 0 of
/// it, which is what [`VkContext::mark_texture_array_partial`] assumes.
const TEXTURE_SET: usize = 2;

/// Per-run push constants: 68 bytes, comfortably inside the 128-byte guaranteed
/// `maxPushConstantsSize`. The cascade's matrix rides here rather than in a
/// uniform buffer because it changes once per pass, not once per draw.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    light_view_proj: [[f32; 4]; 4],
    object_base: u32,
}

/// The cutout variant's, four bytes longer: it also names the material row to
/// alpha-test against. Two structs rather than one with an unused tail, because
/// a push-constant member no stage reads can be stripped from the reflected
/// range and the write would then run past its end.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct MaskedPushConstants {
    light_view_proj: [[f32; 4]; 4],
    object_base: u32,
    material_index: u32,
}

pub struct ShadowPass {
    pub(super) render_pass: Arc<RenderPass>,
    /// The same pass for the punctual atlas, differing in one operation: the
    /// depth attachment is `DontCare` rather than `Clear`.
    ///
    /// A cascade is a full layer that is entirely re-rendered, so clearing it
    /// whole is exactly right. The atlas is not: it is one 4096² image holding
    /// up to 64 tiles, of which a typical scene lights a fraction — the demo
    /// uses 19 — and clearing the whole thing writes 67 MB a frame to blank 45
    /// tiles nobody will read. [`record_atlas`](ShadowPass::record_atlas) clears
    /// the tiles actually in use instead.
    ///
    /// Safe only because `atlas_shadow` in `shading.glsl` clamps every one of
    /// its nine taps to half a texel inside the tile it belongs to. Nothing can
    /// read a texel outside an assigned tile, so whatever is left there is
    /// unobservable. The cascades cannot make that promise — they deliberately
    /// let a tap fall past the edge onto the sampler's white border — which is
    /// the other reason these are two render passes and not one.
    ///
    /// Compatible with the same pipelines, and that is a Vulkan guarantee rather
    /// than a coincidence: render-pass compatibility is defined on attachment
    /// formats and sample counts, and explicitly excludes load and store
    /// operations.
    pub(super) atlas_render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    /// The caster pipeline for a `Masked` material: two-sided, and running the
    /// fragment shader's alpha test.
    ///
    /// Without it a cutout casts the shadow of the quad its texels are painted
    /// on, which is the single most visible way foliage goes wrong — the
    /// silhouette is right in the frame and a rectangle on the ground. It needs
    /// the material table and the texture array that the depth-only pipeline
    /// has no use for, so its layout has three sets where the plain one has one;
    /// set 0 is identical in both, which is what lets the object set stay bound
    /// across a switch.
    masked_pipeline: Arc<GraphicsPipeline>,
    /// 1x1 array depth image cleared to 1.0, bound when shadows are off so the
    /// forward shader samples "fully lit" with no second code path. Not a graph
    /// resource: with shadows off the graph has no cascade image at all, so
    /// there is nothing for the forward pass to declare a read of.
    lit_view: Arc<ImageView>,
    /// A 1x1 non-array depth texture of 1.0, bound as the punctual atlas when
    /// there is none. The cascades' `lit_view` cannot serve: the forward shader
    /// declares the atlas as a plain `texture2D` and the cascades as a
    /// `texture2DArray`, and a view satisfies one or the other.
    lit_atlas_view: Arc<ImageView>,
    pub constant_bias: f32,
    pub slope_bias: f32,
    /// The punctual maps get their own pair. A cascade is orthographic, so one
    /// depth unit is the same world distance everywhere in it and a constant
    /// bias means one thing; a punctual face is perspective, where the same
    /// offset is millimetres at the light and metres at its range. They are
    /// tuned apart because they cannot be tuned together.
    pub punctual_constant_bias: f32,
    pub punctual_slope_bias: f32,
}

impl ShadowPass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let render_pass = depth_only_render_pass(device, AttachmentLoadOp::Clear);
        let atlas_render_pass = depth_only_render_pass(device, AttachmentLoadOp::DontCare);
        let pipeline = build_pipeline(ctx, &render_pass, false);
        let masked_pipeline = build_pipeline(ctx, &render_pass, true);

        Self {
            render_pass,
            atlas_render_pass,
            pipeline,
            masked_pipeline,
            lit_view: build_lit_view(ctx),
            lit_atlas_view: build_lit_atlas_view(ctx),
            constant_bias: 1.25,
            slope_bias: 2.5,
            punctual_constant_bias: 2.0,
            punctual_slope_bias: 3.0,
        }
    }

    /// The "everything is lit" atlas, bound when nothing punctual casts.
    pub(super) fn lit_atlas_view(&self) -> Arc<ImageView> {
        self.lit_atlas_view.clone()
    }

    /// The "everything is lit" view, bound when shadows are disabled.
    #[allow(dead_code)]
    pub(super) fn lit_view(&self) -> Arc<ImageView> {
        self.lit_view.clone()
    }

    /// The per-object descriptor set every cascade binds.
    ///
    /// Built once per frame and handed to each cascade, rather than once per
    /// cascade: all of them read the same buffer through the same layout.
    pub(super) fn build_object_set(
        &self,
        ctx: &VkContext,
        rows: &Subbuffer<[GpuObject]>,
        indices: &Subbuffer<[u32]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, rows.clone()),
                WriteDescriptorSet::buffer(1, indices.clone()),
            ],
            [],
        )
        .unwrap()
    }

    /// The set-1 material table and set-2 texture array the cutout pipeline
    /// alpha-tests against, over the same buffer and the same views every other
    /// pass reads. Cached by the renderer beside the prepass's, and rebuilt on
    /// the same invalidations.
    pub(super) fn build_material_set(
        &self,
        ctx: &VkContext,
        materials: &Subbuffer<[super::forward::GpuMaterial]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.masked_pipeline.layout().set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(0, materials.clone())],
            [],
        )
        .unwrap()
    }

    pub(super) fn build_texture_set(
        &self,
        ctx: &VkContext,
        textures: &[Arc<ImageView>],
        sampler: &Arc<Sampler>,
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
            self.masked_pipeline.layout().set_layouts()[TEXTURE_SET].clone(),
            [
                WriteDescriptorSet::image_view_array(0, 0, texture_array),
                WriteDescriptorSet::sampler(1, sampler.clone()),
            ],
            [],
        )
        .unwrap()
    }

    /// Record one cascade's depth pass.
    ///
    /// `resolution` is the shadow map's, not the frame's — every other pass in
    /// the executor takes the swapchain extent, and using it here would render
    /// the cascade into a corner of its own map with nothing reporting an error.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        casters: DrawList<'_>,
        view_proj: Mat4,
        object_base: u32,
        resolution: u32,
        sets: &CasterSets,
    ) {
        self.set_tile(builder, [0, 0], resolution);
        // Dynamic so the editor's bias sliders tune acne live instead of
        // rebuilding the pipeline on every drag. `clamp` stays 0.0: a nonzero
        // one needs the `depth_bias_clamp` device feature. Set before any
        // pipeline is bound, which is fine and is the point of dynamic state: it
        // survives the pipeline switches `draw` makes between cutout runs and
        // the rest.
        builder
            .set_depth_bias(self.constant_bias, 0.0, self.slope_bias)
            .unwrap();

        self.draw(builder, renderer, casters, view_proj, object_base, sets);
    }

    /// Record every face of every punctual caster into one atlas.
    ///
    /// One render pass for all of them, which is the whole reason the atlas
    /// exists: a tile is a viewport, so the sixteen-light worst case is one
    /// barrier rather than ninety-six. `casters` and `bases` are indexed like
    /// `atlas.casters` — one culled list per *light*, drawn into each of its
    /// faces, because a point light's reach is small enough that culling its six
    /// frustums apart would cost more test than it saved draw.
    pub(super) fn record_atlas(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        atlas: &ShadowAtlas,
        casters: &[DrawList<'_>],
        bases: &[u32],
        sets: &CasterSets,
    ) {
        // Every tile this frame assigned, cleared before anything draws — and
        // *only* those, which is the point: the attachment's load operation is
        // `DontCare`, so the 45-odd tiles a typical scene leaves unassigned cost
        // nothing rather than 67 MB of blanking. See `atlas_render_pass`.
        //
        // Over every assigned face rather than only the ones with something to
        // draw: a light whose caster list came back empty still has its tile
        // sampled, and it has to read as far, or the light would be shadowed by
        // whatever the last frame left in that tile.
        let tiles: Vec<ClearRect> = atlas
            .casters
            .iter()
            .flat_map(|caster| {
                &atlas.faces[caster.first_face..caster.first_face + caster.face_count]
            })
            .map(|face| ClearRect {
                offset: face.tile.offset,
                extent: [face.tile.size; 2],
                array_layers: 0..1,
            })
            .collect();
        if tiles.is_empty() {
            return;
        }
        builder
            .clear_attachments(
                [ClearAttachment::Depth(1.0)].into_iter().collect(),
                tiles.into_iter().collect(),
            )
            .unwrap();

        builder
            .set_depth_bias(self.punctual_constant_bias, 0.0, self.punctual_slope_bias)
            .unwrap();

        for (index, caster) in atlas.casters.iter().enumerate() {
            let (Some(list), Some(&base)) = (casters.get(index), bases.get(index)) else {
                continue;
            };
            if list.is_empty() {
                continue;
            }
            for face in &atlas.faces[caster.first_face..caster.first_face + caster.face_count] {
                self.set_tile(builder, face.tile.offset, face.tile.size);
                self.draw(builder, renderer, *list, face.view_proj, base, sets);
            }
        }
    }

    /// Aim the rasteriser at one tile.
    ///
    /// The scissor is not redundant with the viewport. A viewport confines where
    /// NDC *lands*, and an implementation is allowed a guard band around the
    /// clip volume — so without a scissor a triangle straddling a tile's edge
    /// may write a fragment into the neighbour, which is another light's depth.
    fn set_tile(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        offset: [u32; 2],
        size: u32,
    ) {
        builder
            .set_viewport(
                0,
                [Viewport {
                    offset: [offset[0] as f32, offset[1] as f32],
                    extent: [size as f32, size as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .set_scissor(
                0,
                [Scissor {
                    offset,
                    extent: [size, size],
                }]
                .into_iter()
                .collect(),
            )
            .unwrap();
    }

    /// The draw loop both callers share: one instanced draw per (mesh, material)
    /// run, with the light's matrix pushed per run.
    ///
    /// Nothing is bound on entry — not the pipeline and not the object set —
    /// because a run's material decides both. Every tile of the atlas and every
    /// cascade re-enters here, so the first run of each rebinds; that is one
    /// bind per tile against a loop whose body is a draw call, and it is what
    /// lets the pipeline change *inside* a tile when the caster list mixes
    /// foliage with everything else.
    fn draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        casters: DrawList<'_>,
        view_proj: Mat4,
        object_base: u32,
        sets: &CasterSets,
    ) {
        let mut bound: Option<bool> = None;
        for run in casters.runs() {
            let item = casters.item(run.start);
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            // The same flag word the forward pass and the prepass read, so all
            // three cut the cutout out along the same line. A caster list with
            // no cutout in it never touches `masked_pipeline` and records
            // exactly what it recorded before this existed.
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
                        sets.for_pipeline(wants_masked),
                    )
                    .unwrap();
                bound = Some(wants_masked);
            }

            let object_base = object_base + run.start as u32;
            let light_view_proj = view_proj.to_cols_array_2d();
            // Two ranges, so two writes. The plain pipeline's layout has no
            // material index in its range and pushing one would run past its
            // end — see `MaskedPushConstants`.
            if wants_masked {
                builder
                    .push_constants(
                        pipeline.layout().clone(),
                        0,
                        MaskedPushConstants {
                            light_view_proj,
                            object_base,
                            material_index: item.material.0,
                        },
                    )
                    .unwrap();
            } else {
                builder
                    .push_constants(
                        pipeline.layout().clone(),
                        0,
                        PushConstants {
                            light_view_proj,
                            object_base,
                        },
                    )
                    .unwrap();
            }
            builder
                .bind_vertex_buffers(0, mesh.position_buffer.clone())
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

/// What a caster draw binds, in the two shapes the two pipelines want.
///
/// One struct rather than three parameters threaded through `record`,
/// `record_atlas` and `draw`, and it holds the *cutout* sets as an `Option`
/// because a frame is entitled to have no material table cached yet — nothing
/// here forces the alpha-testing pipeline into existence for a scene with no
/// foliage in it.
pub(super) struct CasterSets {
    pub objects: Arc<DescriptorSet>,
    pub materials: Arc<DescriptorSet>,
    pub textures: Arc<DescriptorSet>,
}

impl CasterSets {
    fn for_pipeline(&self, masked: bool) -> Vec<Arc<DescriptorSet>> {
        if masked {
            vec![
                self.objects.clone(),
                self.materials.clone(),
                self.textures.clone(),
            ]
        } else {
            vec![self.objects.clone()]
        }
    }
}

/// The sampler the forward pass reads the cascades through.
///
/// It must be an *immutable* sampler baked into the descriptor set layout, not
/// one written into a set: Metal makes a sampler's compare function a
/// compile-time property, so MoltenVK rejects a written comparison sampler
/// unless `mutable_comparison_samplers` is available, which on a portability
/// subset device it is not.
///
/// The conventions live here because this module owns the map: `Less` against a
/// depth map cleared to 1.0, and a white border so a fragment sampling outside a
/// cascade reads as lit rather than as shadowed.
pub(super) fn comparison_sampler(device: &Arc<Device>) -> Arc<Sampler> {
    Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::ClampToBorder; 3],
            border_color: BorderColor::FloatOpaqueWhite,
            // The comparison happens before the bilinear filter, so a single tap
            // is already a 2x2 percentage-closer result.
            compare: Some(CompareOp::Less),
            ..Default::default()
        },
    )
    .unwrap()
}

fn depth_only_render_pass(device: &Arc<Device>, load_op: AttachmentLoadOp) -> Arc<RenderPass> {
    let create_info = RenderPassCreateInfo {
        attachments: vec![AttachmentDescription {
            format: DEPTH_FORMAT,
            samples: SampleCount::Sample1,
            load_op,
            store_op: AttachmentStoreOp::Store,
            initial_layout: ImageLayout::DepthStencilAttachmentOptimal,
            final_layout: ImageLayout::DepthStencilAttachmentOptimal,
            ..Default::default()
        }],
        subpasses: vec![SubpassDescription {
            depth_stencil_attachment: Some(AttachmentReference {
                attachment: 0,
                layout: ImageLayout::DepthStencilAttachmentOptimal,
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    RenderPass::new(device.clone(), create_info).unwrap()
}

fn build_pipeline(
    ctx: &VkContext,
    render_pass: &Arc<RenderPass>,
    masked: bool,
) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
    let (vs, fs) = if masked {
        (
            vs_masked::load(device.clone()).unwrap(),
            fs_masked::load(device.clone()).unwrap(),
        )
    } else {
        (
            vs::load(device.clone()).unwrap(),
            fs::load(device.clone()).unwrap(),
        )
    };
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    let vertex_input_state = PositionVertex::per_vertex().definition(&vs).unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    // Only the masked variant samples anything — the plain one writes depth and
    // nothing else — so on that pipeline this finds no set to mark and does
    // nothing, which is why it can be called unconditionally.
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
                // Back-face culling matches the forward pass, which is only
                // correct because `fit_cascade` applies the same Y flip the
                // camera projection does — otherwise the winding is mirrored
                // here and this would cull front faces instead.
                cull_mode: if masked {
                    CullMode::None
                } else {
                    CullMode::Back
                },
                // Placeholders; the real values are set dynamically per frame.
                depth_bias: Some(DepthBiasState::default()),
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            // No color attachments to blend into.
            color_blend_state: None,
            // Scissor as well as viewport, because the atlas aims this pipeline
            // at one tile of a shared attachment and the viewport alone does
            // not promise a fragment stays inside it.
            dynamic_state: [
                DynamicState::Viewport,
                DynamicState::Scissor,
                DynamicState::DepthBias,
            ]
            .into_iter()
            .collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

/// A 1x1 depth *array* image cleared to 1.0, for the cascades.
///
/// The array view type is load-bearing: the forward shader declares the
/// cascades as `texture2DArray`, and a plain 2D view would not satisfy that
/// descriptor when shadows are off.
fn build_lit_view(ctx: &VkContext) -> Arc<ImageView> {
    build_lit_depth(ctx, ImageViewType::Dim2dArray)
}

/// The same, viewed as a plain 2D image, for the punctual atlas.
///
/// Two views over two allocations rather than one image viewed both ways,
/// because a view's type is fixed at creation and the two descriptors want
/// different ones. A pair of one-texel images is not worth avoiding.
fn build_lit_atlas_view(ctx: &VkContext) -> Arc<ImageView> {
    build_lit_depth(ctx, ImageViewType::Dim2d)
}

/// A 1x1 single-layer depth image cleared to 1.0 — "nothing is nearer than the
/// far plane", which every comparison reads as lit.
fn build_lit_depth(ctx: &VkContext, view_type: ImageViewType) -> Arc<ImageView> {
    let image = Image::new(
        ctx.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: DEPTH_FORMAT,
            extent: [1, 1, 1],
            array_layers: 1,
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("failed to allocate the shadows-off depth texture");

    let mut builder = AutoCommandBufferBuilder::primary(
        ctx.command_buffer_allocator.clone(),
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .clear_depth_stencil_image(ClearDepthStencilImageInfo {
            clear_value: ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
            ..ClearDepthStencilImageInfo::image(image.clone())
        })
        .unwrap();
    vulkano::sync::now(ctx.device.clone())
        .then_execute(ctx.queue.clone(), builder.build().unwrap())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    ImageView::new(
        image.clone(),
        ImageViewCreateInfo {
            view_type,
            ..ImageViewCreateInfo::from_image(&image)
        },
    )
    .unwrap()
}

mod vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/shadow.vert" }
}
mod fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/shadow.frag" }
}
/// The cutout variants of the two above, from the same two files: one define
/// turns the vertex shader's UV on and the fragment shader's alpha test with it.
mod vs_masked {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/shadow.vert",
        define: [("ORRIN_MASKED", "1")],
    }
}
mod fs_masked {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/shadow.frag",
        define: [("ORRIN_MASKED", "1")],
    }
}
