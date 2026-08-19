use std::sync::Arc;

use super::context::VkContext;
use crate::scene::{PresentSettings, VsyncMode};
use vulkano::device::{Device, DeviceOwned};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo};

pub const DEPTH_FORMAT: Format = Format::D32_SFLOAT;

/// The one place a [`VsyncMode`] becomes a Vulkan present mode.
fn present_mode(vsync: VsyncMode) -> PresentMode {
    match vsync {
        VsyncMode::Fifo => PresentMode::Fifo,
        VsyncMode::FifoRelaxed => PresentMode::FifoRelaxed,
        VsyncMode::Mailbox => PresentMode::Mailbox,
        VsyncMode::Immediate => PresentMode::Immediate,
    }
}

/// What the surface will actually honour of what was asked for.
///
/// Both halves fall back rather than fail: an unsupported present mode becomes
/// `Fifo`, which every driver has to offer, and an image count outside the
/// advertised range is clamped into it. A setting the surface cannot meet is a
/// worse frame, never a dead window — which matters more here than usual,
/// because these are the two knobs somebody reaches for *while* chasing a
/// number.
fn resolve(
    device: &Arc<Device>,
    surface: &Arc<Surface>,
    want: PresentSettings,
) -> (PresentMode, u32) {
    let physical = device.physical_device();
    let wanted = present_mode(want.vsync);
    let mode = physical
        .surface_present_modes(surface, Default::default())
        .map(|modes| {
            if modes.into_iter().any(|m| m == wanted) {
                wanted
            } else {
                PresentMode::Fifo
            }
        })
        .unwrap_or(PresentMode::Fifo);

    let mut images = want.images.max(1);
    if let Ok(caps) = physical.surface_capabilities(surface, Default::default()) {
        images = images.max(caps.min_image_count);
        if let Some(max) = caps.max_image_count {
            images = images.min(max);
        }
    }

    (mode, images)
}

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
        present: PresentSettings,
    ) -> Self {
        let device = &ctx.device;
        let caps = device
            .physical_device()
            .surface_capabilities(surface, Default::default())
            .expect("failed to query surface capabilities");

        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let (present_mode, min_image_count) = resolve(device, surface, present);

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
    //
    // `present` is re-resolved here rather than carried from construction: this is
    // the one path a present-mode or image-count change travels, so the two
    // settings reach the swapchain the same way an extent does and there is no
    // second place for them to be applied from.
    pub fn recreate(
        &mut self,
        render_pass: &Arc<RenderPass>,
        extent: [u32; 2],
        present: PresentSettings,
    ) -> bool {
        if extent[0] == 0 || extent[1] == 0 {
            return false;
        }
        // An offscreen target has a fixed extent decided by whoever asked for
        // the render; nothing can resize it out from under the frame.
        let Some(current) = self.swapchain.as_ref() else {
            return false;
        };

        let (present_mode, min_image_count) = resolve(current.device(), current.surface(), present);
        let (swapchain, images) = current
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                present_mode,
                min_image_count,
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

    /// What the surface actually honoured, as opposed to what was asked for.
    ///
    /// `None` offscreen, where there is no presentation engine to ask. Reported
    /// by the run banner and the performance panel for one reason: a present mode
    /// the driver quietly declined is the last thing a frame-time figure may omit.
    pub fn applied_present(&self) -> Option<(PresentMode, u32)> {
        let swapchain = self.swapchain.as_ref()?;
        Some((
            swapchain.create_info().present_mode,
            swapchain.image_count(),
        ))
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
