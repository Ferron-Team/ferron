//! Backing store for the compiled graph: the images it owns, and the
//! framebuffers its passes draw into.
//!
//! Nothing here decides *what* to allocate. Format, extent and sample count come
//! from the declaration; usage flags and the memoryless hint are derived by the
//! compiler from what passes said they would do. So an image cannot be created
//! missing a capability something needs, and cannot quietly carry one nothing
//! asked for.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::Vec3;
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::format::ClearValue;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageCreateInfo, ImageSubresourceRange, ImageType, ImageUsage};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo};

use crate::gfx::graph::{FrameGraph, ResourceId};

use super::contact_shadows::ContactShadowPass;
use super::forward::ForwardPass;
use super::frame::{Frame, PassBody};
use super::oit::OitPass;
use super::prepass::GeometryPrepass;
use super::refraction::RefractionPass;
use super::shadow::ShadowPass;
use super::ssao::SsaoPass;

/// Everything that distinguishes one allocation from another.
///
/// The name is in the key and has to be: two resources can be identical in every
/// other field and still have to be separate images. `subsurface_blur_x` and
/// `subsurface_blur_y` are exactly that pair — same format, same extent, and two
/// allocations precisely because a graph resource is unversioned, so a pass that
/// read and wrote one image would make "readers after writers" point both ways.
/// Keying without the name would silently alias them into one.
#[derive(Clone, PartialEq, Eq, Hash)]
struct ImageKey {
    name: String,
    format: vulkano::format::Format,
    extent: [u32; 3],
    usage: ImageUsage,
    samples: vulkano::image::SampleCount,
    array_layers: u32,
    mip_levels: u32,
    memoryless: bool,
}

/// Graph-owned images, indexed by [`ResourceId`].
pub(super) struct GraphImages {
    views: Vec<Option<Arc<ImageView>>>,
    /// Every image allocated at the current extent, for any graph, keyed by what
    /// makes it that image.
    ///
    /// A structural toggle — SSAO off, transparency on, the MSAA switch — makes
    /// `ensure_graph` recompile and call straight back here, and without a cache
    /// that means dropping and recreating every image and every framebuffer in
    /// the frame. `declare` and `compile` are device-free and take microseconds;
    /// the allocation is the entire cost, and it is a visible hitch.
    ///
    /// That matters beyond the editor's checkboxes: folding "is this queue
    /// empty?" into the frame's structure means the graph is recompiled whenever
    /// an object drifts on or off screen, which without this would hitch every
    /// few frames.
    ///
    /// Cleared on resize rather than allowed to accumulate: `Extent::Frame`
    /// resolves differently then, so every entry keyed at the old size is dead
    /// weight no lookup will ever hit again.
    cache: HashMap<ImageKey, Arc<ImageView>>,
    /// What [`Extent::Frame`](crate::gfx::graph::Extent::Frame) resolved to when
    /// these were allocated, so a resize is a comparison rather than a flag
    /// somebody has to remember to set.
    extent: [u32; 2],
}

impl GraphImages {
    pub fn allocate(
        memory: &Arc<StandardMemoryAllocator>,
        graph: &FrameGraph,
        extent: [u32; 2],
    ) -> Self {
        let mut images = Self {
            views: Vec::new(),
            cache: HashMap::new(),
            extent,
        };
        images.rebuild(memory, graph, extent);
        images
    }

    /// Point `views` at the images this graph declares, allocating only the ones
    /// the cache does not already hold.
    pub fn rebuild(
        &mut self,
        memory: &Arc<StandardMemoryAllocator>,
        graph: &FrameGraph,
        extent: [u32; 2],
    ) {
        // A target that is only ever an attachment never leaves the render pass
        // that wrote it, so ask for lazily-allocated memory: on MoltenVK it
        // becomes tile-only and the multisampled HDR and depth targets cost no
        // DRAM at all.
        //
        // "Backends with no lazy memory type fall back silently" is the whole
        // problem with reading this as a general win. *No* desktop driver
        // exposes a lazily-allocated memory type — not AMD's, not NVIDIA's — so
        // there the fallback is ordinary VRAM paying full write-and-resolve
        // traffic, and `msaa_hdr` at 1440p is 118 MB of it. This request is a
        // tiler optimisation that costs nothing to keep and buys nothing off a
        // tiler, which is why `FrameConfig::msaa` now defaults off rather than
        // why this filter changed.
        let lazy = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter {
                preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL
                    | MemoryPropertyFlags::LAZILY_ALLOCATED,
                ..MemoryTypeFilter::PREFER_DEVICE
            },
            ..Default::default()
        };

        if self.extent != extent {
            self.cache.clear();
            self.extent = extent;
        }

        let mut views = vec![None; graph.resource_count()];
        // Within one graph a key must be claimed at most once, or two resources
        // would be handed the same allocation — see [`ImageKey`] for why that is
        // a correctness bug and not just a surprise. Names are unique per graph,
        // so this only fires on a declaration mistake.
        let mut claimed: std::collections::HashSet<ImageKey> = HashSet::new();
        for (id, image) in graph.transient_images() {
            let extent = image.desc.extent.resolve(extent);
            let mip_levels = image.desc.mip_levels.min(max_mip_levels(extent));
            // Vulkan has no arrayed 3D image, so the two are a declaration
            // mistake together rather than a shape to resolve a winner for.
            assert!(
                image.desc.depth.is_none() || image.desc.array_layers.is_none(),
                "render graph: `{}` asked to be both a 3D image and a 2D array",
                graph.resource_name(id),
            );
            let key = ImageKey {
                name: graph.resource_name(id).to_string(),
                format: image.desc.format,
                extent: [extent[0], extent[1], image.desc.depth.unwrap_or(1)],
                usage: image.usage,
                samples: image.desc.samples,
                array_layers: image.desc.array_layers.unwrap_or(1),
                mip_levels,
                memoryless: image.memoryless,
            };
            assert!(
                claimed.insert(key.clone()),
                "render graph: two resources named `{}` describe the same image",
                key.name,
            );
            if let Some(view) = self.cache.get(&key) {
                views[id.index()] = Some(view.clone());
                continue;
            }

            let allocated = Image::new(
                memory.clone(),
                ImageCreateInfo {
                    image_type: match image.desc.depth {
                        Some(_) => ImageType::Dim3d,
                        None => ImageType::Dim2d,
                    },
                    format: image.desc.format,
                    extent: [extent[0], extent[1], image.desc.depth.unwrap_or(1)],
                    usage: image.usage,
                    samples: image.desc.samples,
                    array_layers: image.desc.array_layers.unwrap_or(1),
                    mip_levels,
                    ..Default::default()
                },
                if image.memoryless {
                    lazy.clone()
                } else {
                    AllocationCreateInfo::default()
                },
            )
            .unwrap_or_else(|error| {
                panic!(
                    "render graph: could not allocate `{}` ({:?}, {:?}): {error}",
                    graph.resource_name(id),
                    image.desc.format,
                    image.usage,
                )
            });

            // The view type comes from the declaration, not from the layer
            // count: an array image of one layer must still be viewed as an
            // array, because the sampler type is compiled into the pipeline and
            // cannot depend on how many cascades the settings happen to ask for.
            let view = ImageView::new(
                allocated.clone(),
                ImageViewCreateInfo {
                    view_type: match (image.desc.depth, image.desc.array_layers) {
                        (Some(_), _) => ImageViewType::Dim3d,
                        (None, Some(_)) => ImageViewType::Dim2dArray,
                        (None, None) => ImageViewType::Dim2d,
                    },
                    // A storage image descriptor takes exactly one level, so a
                    // view spanning the whole pyramid cannot claim that usage —
                    // it is the one a shader samples with `textureLod`, and the
                    // per-level storage views come from `mip_view`.
                    usage: if mip_levels > 1 {
                        image.usage - ImageUsage::STORAGE
                    } else {
                        image.usage
                    },
                    ..ImageViewCreateInfo::from_image(&allocated)
                },
            )
            .unwrap();
            self.cache.insert(key, view.clone());
            views[id.index()] = Some(view);
        }
        self.views = views;
    }

    pub fn is_stale(&self, extent: [u32; 2]) -> bool {
        self.extent != extent
    }

    /// The view for a graph-owned image.
    ///
    /// Panics for a resource the graph does not own, which can only happen if a
    /// pass reads a handle from a graph it was not declared against.
    pub fn view(&self, id: ResourceId) -> Arc<ImageView> {
        self.views[id.index()]
            .clone()
            .expect("render graph resource is not a graph-owned image")
    }

    /// The width and height a graph-owned image was allocated at, which is what
    /// a pass drawing into one at less than the frame's extent has to set its
    /// viewport to — the framebuffer's render area follows the attachment, but
    /// the viewport is dynamic state nothing derives.
    pub fn extent(&self, id: ResourceId) -> [u32; 2] {
        let extent = self.view(id).image().extent();
        [extent[0], extent[1]]
    }

    /// A single-layer 2D view of one layer of an array image.
    ///
    /// A framebuffer attachment has to be one layer, while the same image is
    /// sampled as a whole array — so the cascades need both kinds of view over
    /// the same allocation.
    pub fn layer_view(&self, id: ResourceId, layer: u32) -> Arc<ImageView> {
        let image = self.view(id).image().clone();

        let info = ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            subresource_range: ImageSubresourceRange {
                array_layers: layer..layer + 1,
                ..image.subresource_range()
            },
            ..ImageViewCreateInfo::from_image(&image)
        };

        ImageView::new(image, info).unwrap()
    }

    /// A single-level view of one mip of a pyramid, which is what a storage
    /// image descriptor requires. The whole-pyramid view `view` hands back is
    /// the one to sample.
    pub fn mip_view(&self, id: ResourceId, level: u32) -> Arc<ImageView> {
        let image = self.view(id).image().clone();

        let info = ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            subresource_range: ImageSubresourceRange {
                mip_levels: level..level + 1,
                ..image.subresource_range()
            },
            ..ImageViewCreateInfo::from_image(&image)
        };

        ImageView::new(image, info).unwrap()
    }

    /// How many levels the allocated image actually has, which is what a pass
    /// writing a pyramid must loop over — the declaration is a request, and a
    /// small window cannot honour it.
    pub fn mip_levels(&self, id: ResourceId) -> u32 {
        self.view(id).image().mip_levels()
    }
}

/// The deepest pyramid an extent can carry: halving stops at one texel.
fn max_mip_levels(extent: [u32; 2]) -> u32 {
    32 - extent[0].max(extent[1]).max(1).leading_zeros()
}

/// One framebuffer per pass that draws into graph-owned images, indexed by
/// `PassId`.
///
/// The tonemap and overlay passes have none: they target the acquired swapchain
/// image, which changes every frame, so their framebuffers are the ones the
/// swapchain builds once per image and indexes by acquire.
pub(super) struct PassFramebuffers {
    framebuffers: Vec<Option<Arc<Framebuffer>>>,
}

impl PassFramebuffers {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        frame: &Frame,
        images: &GraphImages,
        forward: &ForwardPass,
        prepass: &GeometryPrepass,
        ssao: &SsaoPass,
        contact_shadows: &ContactShadowPass,
        oit: &OitPass,
        refraction: &RefractionPass,
        shadow: &ShadowPass,
    ) -> Self {
        let ids = frame.ids;
        // Only scheduled passes get one. A culled pass's images are never
        // allocated, so building its framebuffer would ask for a view that does
        // not exist.
        let mut framebuffers = vec![None; frame.bodies.len()];
        for &pass_id in frame.graph.order() {
            let body = frame.bodies[pass_id.index()];
            framebuffers[pass_id.index()] = (|| {
                let (render_pass, attachments) = match body {
                    PassBody::GeometryPrepass => {
                        let prepass_ids = ids.prepass.expect("prepass without its resources");
                        (
                            prepass.render_pass.clone(),
                            vec![
                                images.view(prepass_ids.normal),
                                images.view(prepass_ids.velocity),
                                images.view(prepass_ids.material),
                                images.view(prepass_ids.depth),
                            ],
                        )
                    }
                    PassBody::SsaoResolve => {
                        let ssao_ids = ids.ssao.expect("SSAO pass without SSAO resources");
                        (ssao.ssao_rp.clone(), vec![images.view(ssao_ids.raw_ao)])
                    }
                    PassBody::SsaoBlur => {
                        let ssao_ids = ids.ssao.expect("SSAO pass without SSAO resources");
                        (ssao.blur_rp.clone(), vec![images.view(ssao_ids.ao)])
                    }
                    PassBody::ContactShadows => (
                        contact_shadows.render_pass.clone(),
                        vec![
                            images.view(
                                ids.contact_shadows
                                    .expect("contact shadow pass without its mask"),
                            ),
                        ],
                    ),
                    // Two shapes, and which one is a property of the graph: the
                    // frame that diffuses subsurface light resolves a second
                    // colour target, so it opens the render pass that has one.
                    // The attachment order is the order each render pass
                    // *declares*, not the order the graph accessed them in.
                    //
                    // Four shapes rather than two, because the sample count
                    // moves more than a number: at one sample there is no
                    // resolve, so `hdr_color` is the colour attachment itself
                    // and the depth is the geometry prepass's, attached
                    // read-only — the third image in this frame two render
                    // passes attach, alongside `oit_accumulate` and
                    // `refraction_draw`, and for the same reason.
                    PassBody::Forward => {
                        let borrowed_depth = || {
                            images.view(
                                ids.prepass
                                    .expect("a one-sample forward pass borrows the prepass depth")
                                    .depth,
                            )
                        };
                        match (ids.msaa, ids.subsurface) {
                            (Some(msaa), Some(sss)) => (
                                forward.subsurface_render_pass.clone(),
                                vec![
                                    images.view(msaa.hdr),
                                    images
                                        .view(sss.msaa_diffusible.expect(
                                            "a multisampled frame resolves its diffusible",
                                        )),
                                    images.view(msaa.depth),
                                    images.view(ids.hdr_color),
                                    images.view(sss.diffusible),
                                ],
                            ),
                            (Some(msaa), None) => (
                                forward.render_pass.clone(),
                                vec![
                                    images.view(msaa.hdr),
                                    images.view(msaa.depth),
                                    images.view(ids.hdr_color),
                                ],
                            ),
                            (None, Some(sss)) => (
                                forward.single_subsurface_render_pass.clone(),
                                vec![
                                    images.view(ids.hdr_color),
                                    images.view(sss.diffusible),
                                    borrowed_depth(),
                                ],
                            ),
                            (None, None) => (
                                forward.single_render_pass.clone(),
                                vec![images.view(ids.hdr_color), borrowed_depth()],
                            ),
                        }
                    }
                    // The prepass depth as the third attachment, read-only —
                    // the one image in this frame two render passes attach.
                    PassBody::OitAccumulate => {
                        let oit_ids = ids
                            .transparency
                            .expect("transparency pass without its targets");
                        let prepass_ids = ids
                            .prepass
                            .expect("the graph scheduled transparency with no prepass");
                        (
                            oit.render_pass.clone(),
                            vec![
                                images.view(oit_ids.accum),
                                images.view(oit_ids.reveal),
                                images.view(prepass_ids.depth),
                            ],
                        )
                    }
                    // The prepass depth read-only again, in the same shape and
                    // for the same reason — one colour target instead of two,
                    // because a refractive surface composites its own
                    // background rather than accumulating a weight beside it.
                    PassBody::RefractionDraw => {
                        let refraction_ids =
                            ids.refraction.expect("refraction pass without its targets");
                        let prepass_ids = ids
                            .prepass
                            .expect("the graph scheduled refraction with no prepass");
                        (
                            refraction.render_pass.clone(),
                            vec![
                                images.view(refraction_ids.accum),
                                images.view(prepass_ids.depth),
                            ],
                        )
                    }
                    // One layer of the cascade array, not the array view: a
                    // framebuffer attachment is a single layer, and handing it
                    // the whole array would make every cascade clear and
                    // overwrite the same pixels without erroring.
                    PassBody::ShadowCascade(cascade) => (
                        shadow.render_pass.clone(),
                        vec![images.layer_view(
                            ids.shadows.expect("shadow pass without a shadow image"),
                            cascade,
                        )],
                    ),
                    // The cascades' render pass, because it is the same render
                    // pass: one depth attachment of the same format, cleared and
                    // stored. A framebuffer only needs a *compatible* one, and a
                    // second declaration of the identical thing would be a
                    // second place for the format to drift.
                    PassBody::PunctualShadows => (
                        shadow.atlas_render_pass.clone(),
                        vec![
                            images.view(
                                ids.shadow_atlas
                                    .expect("punctual shadow pass without an atlas"),
                            ),
                        ],
                    ),
                    // The tonemap and overlay passes target the acquired
                    // swapchain image; the metering passes are dispatches and
                    // target no attachment at all.
                    PassBody::Tonemap
                    | PassBody::Overlay
                    | PassBody::SsrHiz
                    | PassBody::SsrSource
                    | PassBody::SsrTrace
                    | PassBody::SsrResolve
                    | PassBody::FogScatter
                    | PassBody::FogIntegrate
                    | PassBody::SubsurfaceBlurHorizontal
                    | PassBody::SubsurfaceBlurVertical
                    | PassBody::SubsurfaceComposite
                    | PassBody::OitComposite
                    | PassBody::RefractionScene
                    | PassBody::RefractionComposite
                    | PassBody::TaaResolve
                    | PassBody::DofPrefilter
                    | PassBody::DofTileMax
                    | PassBody::DofGather
                    | PassBody::DofComposite
                    | PassBody::MotionBlurTileMax
                    | PassBody::MotionBlurNeighbourMax
                    | PassBody::MotionBlurGather
                    | PassBody::LuminanceHistogram
                    | PassBody::LuminanceAverage
                    | PassBody::BloomPrefilter
                    | PassBody::BloomDownsample(_)
                    | PassBody::BloomUpsample(_) => return None,
                };
                Some(
                    Framebuffer::new(
                        render_pass,
                        FramebufferCreateInfo {
                            attachments,
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                )
            })();
        }
        Self { framebuffers }
    }

    pub fn get(&self, pass: usize) -> Option<Arc<Framebuffer>> {
        self.framebuffers[pass].clone()
    }
}

/// How each pass's attachments start the frame. `None` means "leave it": for the
/// resolve target and the swapchain image, every pixel is written anyway, so
/// clearing first is bandwidth spent on values nothing reads.
pub(super) fn clear_values(
    body: PassBody,
    framebuffer: &Framebuffer,
    background: Vec3,
) -> Vec<Option<ClearValue>> {
    let attachments = framebuffer.attachments().len();
    match body {
        // Flat +Z in the normal buffer, no motion in the velocity buffer, far in
        // depth. Zero velocity is what the sky and any unrasterised pixel are
        // left with, and the resolve reads that as "reproject with the camera
        // alone" rather than as a stationary surface.
        // A black `f0` at zero roughness in the material target, so a pixel
        // nothing rasterised reflects nothing rather than mirroring the sky at
        // whatever the previous contents implied.
        PassBody::GeometryPrepass => vec![
            Some([0.5, 0.5, 1.0, 0.0].into()),
            Some([0.0, 0.0, 0.0, 0.0].into()),
            Some([0.0, 0.0, 0.0, 0.0].into()),
            Some(1.0.into()),
        ],
        // 1.0 = fully unoccluded, so an untouched pixel darkens nothing.
        PassBody::SsaoResolve | PassBody::SsaoBlur => vec![Some([1.0, 0.0, 0.0, 0.0].into())],
        // 1.0 = fully lit, for the same reason: the pass writes every pixel, so
        // this only decides what a pixel would be if it somehow did not.
        PassBody::ContactShadows => vec![Some([1.0, 0.0, 0.0, 0.0].into())],
        // Two shapes, told apart by the framebuffer rather than by a second pass
        // body: the two differ in their attachments and in nothing else, so a
        // variant here would have to be threaded through every exhaustive match
        // in the executor to say something the framebuffer already says.
        //
        // The diffusible target is cleared to zero, and that clear is the feature's
        // mask: the skybox and the debug lines share this render pass and write
        // nothing to that attachment, so zero is what a pixel they covered holds —
        // and a zero radius is exactly "nothing scattered here".
        //
        // Four shapes, and the attachment count alone no longer tells them
        // apart: a multisampled frame without diffusion and a one-sample frame
        // with it both have three. What separates them is the sample count,
        // read off the first attachment — the same thing that decides which
        // render pass `PassFramebuffers::build` opened.
        PassBody::Forward => {
            let multisampled = framebuffer.attachments()[0].image().samples()
                != vulkano::image::SampleCount::Sample1;
            let color = Some([background.x, background.y, background.z, 1.0].into());
            let diffusible = Some([0.0, 0.0, 0.0, 0.0].into());
            match (multisampled, attachments) {
                (true, 5) => vec![color, diffusible, Some(1.0.into()), None, None],
                (true, _) => vec![color, Some(1.0.into()), None],
                // One sample: the colour target, the diffusible one if the frame
                // has it, and the prepass depth last. That depth is *loaded* —
                // it is the whole input to the `EQUAL` test — so it carries no
                // clear, exactly as it carries none for the two transparency
                // passes that borrow it.
                (false, 3) => vec![color, diffusible, None],
                (false, _) => vec![color, None],
            }
        }
        // Nothing accumulated, and everything revealed. The second is the one
        // that matters: revealage is a running *product* of `1 - alpha`, so a
        // clear of zero would hide the opaque frame everywhere rather than
        // nowhere. The depth attachment is read-only and carries no clear.
        PassBody::OitAccumulate => vec![
            Some([0.0, 0.0, 0.0, 0.0].into()),
            Some([1.0, 0.0, 0.0, 0.0].into()),
            None,
        ],
        // Nothing covered, so nothing replaced: the composite reads this as the
        // frame's own colour showing through in full. Zero rather than one is
        // what makes it an `over` of nothing instead of a black wall.
        PassBody::RefractionDraw => vec![Some([0.0, 0.0, 0.0, 0.0].into()), None],
        PassBody::Tonemap => vec![None],
        // Dispatches, so there is no render pass to clear anything in.
        PassBody::SubsurfaceBlurHorizontal
        | PassBody::SubsurfaceBlurVertical
        | PassBody::SubsurfaceComposite => Vec::new(),
        // No render pass, so nothing to clear.
        PassBody::Overlay
        | PassBody::FogScatter
        | PassBody::FogIntegrate
        | PassBody::SsrHiz
        | PassBody::SsrSource
        | PassBody::SsrTrace
        | PassBody::SsrResolve
        | PassBody::OitComposite
        | PassBody::RefractionScene
        | PassBody::RefractionComposite
        | PassBody::TaaResolve
        | PassBody::DofPrefilter
        | PassBody::DofTileMax
        | PassBody::DofGather
        | PassBody::DofComposite
        | PassBody::MotionBlurTileMax
        | PassBody::MotionBlurNeighbourMax
        | PassBody::MotionBlurGather
        | PassBody::LuminanceHistogram
        | PassBody::LuminanceAverage
        | PassBody::BloomPrefilter
        | PassBody::BloomDownsample(_)
        | PassBody::BloomUpsample(_) => Vec::new(),
        // A cascade is one layer, entirely re-rendered, so clearing it whole is
        // exactly what it wants: a texel nobody drew into reads as far, and
        // therefore as lit.
        PassBody::ShadowCascade(_) => vec![Some(1.0.into())],
        // The atlas is not. It holds up to 64 tiles of which a frame lights a
        // fraction, so it loads `DontCare` and `record_atlas` clears the tiles
        // actually assigned — see `ShadowPass::atlas_render_pass`.
        PassBody::PunctualShadows => vec![None],
    }
}

pub(super) fn begin_info(
    framebuffer: Arc<Framebuffer>,
    body: PassBody,
    background: Vec3,
) -> RenderPassBeginInfo {
    RenderPassBeginInfo {
        clear_values: clear_values(body, &framebuffer, background),
        ..RenderPassBeginInfo::framebuffer(framebuffer)
    }
}
