mod context;
mod forward;
mod swapchain;

use std::sync::Arc;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents,
};
use vulkano::instance::Instance;
use vulkano::swapchain::{
    acquire_next_image, Surface, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::sync::{self, future::FenceSignalFuture};
use vulkano::{Validated, VulkanError};

use crate::scene::{Camera, CpuMesh, MeshHandle};

use self::context::VkContext;
use self::forward::{ForwardPass, GpuMesh};
use self::swapchain::SwapchainState;

use super::{RenderBackend, RenderItem, SceneLighting};

type FrameFuture = FenceSignalFuture<Box<dyn GpuFuture>>;

pub struct VulkanRenderer {
    pub(crate) ctx: VkContext,
    swapchain: SwapchainState,
    forward: ForwardPass,
    pub(crate) meshes: Vec<GpuMesh>,
    previous_frame_end: Option<FrameFuture>,
    recreate_swapchain: bool,
    pending_extent: [u32; 2],
}

impl VulkanRenderer {
    pub fn new(instance: &Arc<Instance>, surface: Arc<Surface>, extent: [u32; 2]) -> Self {
        let ctx = VkContext::new(instance, &surface);
        let format = swapchain_color_format(&ctx, &surface);
        let forward = ForwardPass::new(&ctx.device, &ctx.memory_allocator, format);
        let swapchain = SwapchainState::new(&ctx, &surface, &forward.render_pass, format, extent);

        Self {
            ctx,
            swapchain,
            forward,
            meshes: Vec::new(),
            previous_frame_end: None,
            recreate_swapchain: false,
            pending_extent: extent,
        }
    }
}

impl RenderBackend for VulkanRenderer {
    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle {
        let gpu = forward::upload_mesh(&self.ctx.memory_allocator, &mesh.vertices, &mesh.indices);
        let handle = MeshHandle(self.meshes.len() as u32);
        self.meshes.push(gpu);
        handle
    }

    fn resize(&mut self, extent: [u32; 2]) {
        self.pending_extent = extent;
        self.recreate_swapchain = true;
    }

    fn render(&mut self, items: &[RenderItem], lighting: &SceneLighting, camera: &Camera) {
        if self.pending_extent[0] == 0 || self.pending_extent[1] == 0 {
            return;
        }

        if self.recreate_swapchain {
            if self.swapchain.recreate(
                &self.ctx.memory_allocator,
                &self.forward.render_pass,
                self.pending_extent,
            ) {
                self.recreate_swapchain = false;
            } else {
                return;
            }
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };
        if suboptimal {
            self.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.ctx.command_buffer_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.02, 0.02, 0.03, 1.0].into()),
                        Some(1.0.into()),
                        None,
                    ],
                    ..RenderPassBeginInfo::framebuffer(
                        self.swapchain.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();

        self.forward.draw(
            &mut builder,
            self,
            items,
            lighting,
            camera,
            self.swapchain.extent,
        );

        builder.end_render_pass(Default::default()).unwrap();
        let command_buffer = builder.build().unwrap();

        if let Some(prev) = self.previous_frame_end.as_mut() {
            prev.cleanup_finished();
        }

        let future = self
            .previous_frame_end
            .take()
            .map(|f| f.boxed())
            .unwrap_or_else(|| sync::now(self.ctx.device.clone()).boxed())
            .join(acquire_future)
            .then_execute(self.ctx.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.ctx.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.swapchain.clone(),
                    image_index,
                ),
            )
            .boxed()
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(f) => self.previous_frame_end = Some(f),
            Err(VulkanError::OutOfDate) => self.recreate_swapchain = true,
            Err(e) => {
                eprintln!("failed to flush future: {e}");
            }
        }
    }
}

fn swapchain_color_format(ctx: &VkContext, surface: &Arc<Surface>) -> vulkano::format::Format {
    use vulkano::format::Format;
    use vulkano::swapchain::ColorSpace;
    ctx.device
        .physical_device()
        .surface_formats(surface, Default::default())
        .unwrap()
        .into_iter()
        .find(|(f, c)| {
            matches!(f, Format::B8G8R8A8_SRGB | Format::R8G8B8A8_SRGB)
                && *c == ColorSpace::SrgbNonLinear
        })
        .map(|(f, _)| f)
        .unwrap_or(Format::B8G8R8A8_SRGB)
}
