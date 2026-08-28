//! Backing store for the compiled graph: the images its passes render into and
//! sample.
//!
//! What each pass *does* with them — which are attachments, how they load and
//! store, what they clear to — is [`rendering`](super::rendering)'s, and was
//! a `RenderPass` and a `Framebuffer` per pass before dynamic rendering.
//!
//! Nothing here decides *what* to allocate. Format, extent and sample count come
//! from the declaration; usage flags and the memoryless hint are derived by the
//! compiler from what passes said they would do. So an image cannot be created
//! missing a capability something needs, and cannot quietly carry one nothing
//! asked for.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageCreateInfo, ImageSubresourceRange, ImageType, ImageUsage};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

use crate::gfx::graph::{FrameGraph, ResourceId};

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
