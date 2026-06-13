use std::sync::Arc;

use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo};

use super::context::VkContext;

pub const DEPTH_FORMAT: Format = Format::D32_SFLOAT;

pub struct SwapchainState {
    pub swapchain: Arc<Swapchain>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
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

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count.max(2),
                image_format: format,
                image_extent: extent,
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                ..Default::default()
            },
        )
        .expect("failed to create swapchain");

        let framebuffers =
            build_framebuffers(&ctx.memory_allocator, render_pass, &images, extent);

        Self {
            swapchain,
            framebuffers,
            extent,
        }
    }

    // Returns false if the surface has zero area (minimized) and recreation is skipped.
    pub fn recreate(
        &mut self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        render_pass: &Arc<RenderPass>,
        extent: [u32; 2],
    ) -> bool {
        if extent[0] == 0 || extent[1] == 0 {
            return false;
        }

        let (swapchain, images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                ..self.swapchain.create_info()
            })
            .expect("failed to recreate swapchain");

        self.swapchain = swapchain;
        self.framebuffers = build_framebuffers(memory_allocator, render_pass, &images, extent);
        self.extent = extent;
        true
    }
}

fn build_framebuffers(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    render_pass: &Arc<RenderPass>,
    images: &[Arc<Image>],
    extent: [u32; 2],
) -> Vec<Arc<Framebuffer>> {
    let depth = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: DEPTH_FORMAT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let color = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![color, depth.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect()
}
