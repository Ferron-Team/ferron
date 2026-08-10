use std::sync::Arc;

use super::context::VkContext;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo};

pub const DEPTH_FORMAT: Format = Format::D32_SFLOAT;

/// Presentation mode — flip this to toggle vsync:
/// - `Fifo`: vsync ON, capped to refresh, no tearing (always supported).
/// - `Mailbox`: uncapped, no tearing (not always supported).
/// - `Immediate`: uncapped, may tear (not always supported).
///
/// Falls back to `Fifo` automatically if the surface doesn't support the choice.
pub const PRESENT_MODE: PresentMode = PresentMode::Fifo;

/// What the frame's last pass draws into: a real swapchain, or — offscreen — a
/// single ordinary image nobody presents.
///
/// One type rather than two because everything between the tonemap pass and this
/// point must not be able to tell the difference. An offscreen render exists to
/// be *evidence* about the windowed one, and it stops being that the moment the
/// two take different paths through the renderer. So `swapchain` is an option
/// and the rest of the struct is shared: the frame acquires index 0 instead of
/// waiting on a presentation engine, and skips the present at the end.
pub struct SwapchainState {
    /// `None` offscreen. The only two places that branch on it are the acquire
    /// and the present.
    pub swapchain: Option<Arc<Swapchain>>,
    /// The images behind `framebuffers`, kept so an offscreen render can copy
    /// the finished frame back out. Empty for a windowed target, which has
    /// nothing to read back.
    pub readback: Vec<Arc<Image>>,
    /// Carried rather than read off the swapchain, because offscreen there is no
    /// swapchain to read it off.
    pub format: Format,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    /// One view per swapchain image, parallel to `framebuffers`. An overlay
    /// (e.g. the editor UI) draws onto these directly, after the tonemap pass
    /// has written the scene into the same image.
    pub image_views: Vec<Arc<ImageView>>,
    pub extent: [u32; 2],
}

impl SwapchainState {
    pub fn new(
        ctx: &VkContext,
        surface: &Arc<Surface>,
        render_pass: &Arc<RenderPass>,
        format: Format,
        extent: [u32; 2],
    ) -> Self {
        let device = &ctx.device;
        let caps = device
            .physical_device()
            .surface_capabilities(surface, Default::default())
            .expect("failed to query surface capabilities");

        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();

        let present_mode = device
            .physical_device()
            .surface_present_modes(surface, Default::default())
            .map(|modes| {
                if modes.into_iter().any(|m| m == PRESENT_MODE) {
                    PRESENT_MODE
                } else {
                    PresentMode::Fifo
                }
            })
            .unwrap_or(PresentMode::Fifo);
        println!("Present mode: {present_mode:?}");

        // Prefer double-buffering, but stay within the surface's advertised range:
        // never below its minimum, and never above its maximum when it sets one
        // (max_image_count == None means unlimited). A surface whose max is 1
        // would otherwise fail creation against the unconditional `.max(2)`.
        let mut min_image_count = caps.min_image_count.max(2);
        if let Some(max) = caps.max_image_count {
            min_image_count = min_image_count.min(max);
        }

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count,
                image_format: format,
                image_extent: extent,
                image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                present_mode,
                composite_alpha,
                ..Default::default()
            },
        )
        .expect("failed to create swapchain");

        let (framebuffers, image_views) = build_framebuffers(render_pass, &images);

        Self {
            swapchain: Some(swapchain),
            readback: Vec::new(),
            format,
            framebuffers,
            image_views,
            extent,
        }
    }

    /// A single image the frame renders into and nobody presents.
    ///
    /// `TRANSFER_SRC` is the one usage a windowed target does not need: it is
    /// what lets the finished frame be copied back to host memory and written
    /// out as a PNG.
    pub fn offscreen(
        ctx: &VkContext,
        render_pass: &Arc<RenderPass>,
        format: Format,
        extent: [u32; 2],
    ) -> Self {
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSFER_DST
                    | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("failed to allocate the offscreen target");

        let images = [image];
        let (framebuffers, image_views) = build_framebuffers(render_pass, &images);

        Self {
            swapchain: None,
            readback: images.to_vec(),
            format,
            framebuffers,
            image_views,
            extent,
        }
    }

    // Returns false if the surface has zero area (minimized) and recreation is skipped.
    pub fn recreate(&mut self, render_pass: &Arc<RenderPass>, extent: [u32; 2]) -> bool {
        if extent[0] == 0 || extent[1] == 0 {
            return false;
        }
        // An offscreen target has a fixed extent decided by whoever asked for
        // the render; nothing can resize it out from under the frame.
        let Some(current) = self.swapchain.as_ref() else {
            return false;
        };

        let (swapchain, images) = current
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                ..current.create_info()
            })
            .expect("failed to recreate swapchain");

        self.swapchain = Some(swapchain);
        let (framebuffers, image_views) = build_framebuffers(render_pass, &images);
        self.framebuffers = framebuffers;
        self.image_views = image_views;
        self.extent = extent;
        true
    }
}

/// Build a framebuffer and keep its color view for each swapchain image. The
/// views are returned alongside so an overlay can target the same images.
fn build_framebuffers(
    render_pass: &Arc<RenderPass>,
    images: &[Arc<Image>],
) -> (Vec<Arc<Framebuffer>>, Vec<Arc<ImageView>>) {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            let framebuffer = Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view.clone()],
                    ..Default::default()
                },
            )
            .unwrap();
            (framebuffer, view)
        })
        .unzip()
}
