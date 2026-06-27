mod context;
mod forward;
mod hdr;
mod swapchain;
mod texture;
mod ssao;

use std::sync::Arc;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::instance::Instance;
use vulkano::swapchain::{
    acquire_next_image, Surface, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::sync::{self, future::FenceSignalFuture};
use vulkano::{Validated, VulkanError};

use crate::scene::{Camera, CpuMesh, HdrSettings, MaterialHandle, MeshHandle, SsaoSettings};

use self::context::VkContext;
use self::forward::{ForwardPass, GpuMesh, GpuMaterial};
use self::swapchain::SwapchainState;
use self::ssao::SsaoPass;
use self::hdr::{HdrPass, HDR_FORMAT};

use super::{Material, RenderBackend, RenderItem, SceneLighting, TextureHandle};

type FrameFuture = FenceSignalFuture<Box<dyn GpuFuture>>;

/// A hook that draws over the final swapchain image between the tonemap pass and
/// present (the editor UI). Given the future to wait on and that image's view, it
/// returns the future to present. A plain closure, so this module stays free of
/// any UI/egui types.
pub type Overlay<'a> =
    &'a mut dyn FnMut(Box<dyn GpuFuture>, Arc<ImageView>) -> Box<dyn GpuFuture>;

pub struct VulkanRenderer {
    pub(crate) ctx: VkContext,
    swapchain: SwapchainState,
    forward: ForwardPass,
    hdr: HdrPass,
    ssao: SsaoPass,
    pub(crate) meshes: Vec<GpuMesh>,
    pub(crate) materials: Vec<GpuMaterial>,
    /// Texture views indexed by `TextureHandle`. Index 0 is a 1x1 white texture
    /// and index 1 a flat normal map; materials without a given map point here.
    pub(crate) textures: Vec<Arc<ImageView>>,
    /// Cached set-1 (materials) and set-2 (textures) descriptor sets. `None` =
    /// dirty; rebuilt lazily in `render` after a `load_material`/`load_texture`.
    material_set: Option<Arc<DescriptorSet>>,
    texture_set: Option<Arc<DescriptorSet>>,
    previous_frame_end: Option<FrameFuture>,
    recreate_swapchain: bool,
    pending_extent: [u32; 2],
}

impl VulkanRenderer {
    pub fn new(instance: &Arc<Instance>, surface: Arc<Surface>, extent: [u32; 2]) -> Self {
        let ctx = VkContext::new(instance, &surface);
        let format = swapchain_color_format(&ctx, &surface);
        let forward = ForwardPass::new(&ctx.device, &ctx.memory_allocator, HDR_FORMAT);
        let hdr = HdrPass::new(&ctx, &forward.render_pass, format, extent);
        let ssao = SsaoPass::new(&ctx, extent);
        let swapchain = SwapchainState::new(&ctx, &surface, &hdr.tonemap_rp, format, extent);

        // Default textures so every material slot resolves to a valid view:
        // index 0 = white (a no-op multiply), index 1 = flat normal (0,0,1).
        let textures = vec![
            texture::upload_texture(&ctx, &[255, 255, 255, 255], [1, 1], Format::R8G8B8A8_UNORM),
            texture::upload_texture(&ctx, &[128, 128, 255, 255], [1, 1], Format::R8G8B8A8_UNORM),
        ];

        Self {
            ctx,
            swapchain,
            forward,
            hdr,
            ssao,
            meshes: Vec::new(),
            materials: vec![forward::to_gpu_material(&Material::default())],
            textures,
            material_set: None,
            texture_set: None,
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

    fn load_material(&mut self, material: &Material) -> MaterialHandle {
        let handle = MaterialHandle(self.materials.len() as u32);
        self.materials.push(forward::to_gpu_material(material));
        self.material_set = None; // invalidate cache
        handle
    }

    fn load_texture(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        srgb: bool,
    ) -> TextureHandle {
        // Color maps are authored in sRGB so the GPU decodes them to linear on
        // sample; data maps (normal, metallic-roughness) are already linear.
        let format = if srgb {
            Format::R8G8B8A8_SRGB
        } else {
            Format::R8G8B8A8_UNORM
        };
        let view = texture::upload_texture(&self.ctx, pixels, [width, height], format);
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(view);
        self.texture_set = None; // invalidate cache
        handle
    }

    fn resize(&mut self, extent: [u32; 2]) {
        self.pending_extent = extent;
        self.recreate_swapchain = true;
    }

    fn render(
        &mut self,
        items: &[RenderItem],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        hdr: &HdrSettings,
    ) {
        self.render_frame(items, lighting, camera, ssao, hdr, None);
    }
}

impl VulkanRenderer {
    pub fn queue(&self) -> Arc<Queue> {
        self.ctx.queue.clone()
    }

    pub fn color_format(&self) -> Format {
        self.swapchain.swapchain.image_format()
    }

    /// Like [`render`](RenderBackend::render) but composites `overlay` (the
    /// editor UI) onto the final image before present.
    pub fn render_with_overlay(
        &mut self,
        items: &[RenderItem],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        hdr: &HdrSettings,
        overlay: Overlay<'_>,
    ) {
        self.render_frame(items, lighting, camera, ssao, hdr, Some(overlay));
    }

    fn render_frame(
        &mut self,
        items: &[RenderItem],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        hdr: &HdrSettings,
        overlay: Option<Overlay<'_>>,
    ) {
        if self.pending_extent[0] == 0 || self.pending_extent[1] == 0 {
            return;
        }

        if self.recreate_swapchain {
            if self.swapchain.recreate(
                &self.hdr.tonemap_rp,
                self.pending_extent,
            ) {
                self.hdr.resize(
                    &self.ctx.memory_allocator,
                    &self.forward.render_pass,
                    self.pending_extent,
                );
                self.ssao.resize(&self.ctx.memory_allocator, self.pending_extent);
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

        // Drive the SSAO tunables from the world resource each frame. When SSAO
        // is disabled we skip the three passes and bind a 1x1 white AO view, so
        // the forward shader samples 1.0 (no occlusion) and is otherwise unchanged.
        self.ssao.radius = ssao.radius;
        self.ssao.bias = ssao.bias;
        self.ssao.power = ssao.power;
        self.hdr.exposure = hdr.exposure;

        // Material table and texture array are static after asset load, so cache
        // their descriptor sets and rebuild only when invalidated (set to None).
        if self.material_set.is_none() {
            self.material_set = Some(self.forward.build_material_set(&self.ctx, &self.materials));
        }
        if self.texture_set.is_none() {
            self.texture_set = Some(self.forward.build_texture_set(&self.ctx, &self.textures));
        }
        let material_set = self.material_set.clone().unwrap();
        let texture_set = self.texture_set.clone().unwrap();

        let ao_view = if ssao.enabled {
            self.ssao.record(&mut builder, self, items, camera, self.swapchain.extent);
            self.ssao.ao_view()
        } else {
            self.ssao.white_view()
        };

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.02, 0.02, 0.03, 1.0].into()),
                        Some(1.0.into()),
                        None,
                    ],
                    ..RenderPassBeginInfo::framebuffer(self.hdr.forward_framebuffer())
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
            ao_view,
            material_set,
            texture_set,
        );

        builder.end_render_pass(Default::default()).unwrap();

        // Tonemap the resolved HDR target into the acquired swapchain image.
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None],
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
        self.hdr
            .record_tonemap(&mut builder, &self.ctx, self.swapchain.extent);
        builder.end_render_pass(Default::default()).unwrap();

        let command_buffer = builder.build().unwrap();

        if let Some(prev) = self.previous_frame_end.as_mut() {
            prev.cleanup_finished();
        }

        let after_scene = self
            .previous_frame_end
            .take()
            .map(|f| f.boxed())
            .unwrap_or_else(|| sync::now(self.ctx.device.clone()).boxed())
            .join(acquire_future)
            .then_execute(self.ctx.queue.clone(), command_buffer)
            .unwrap()
            .boxed();

        // Let the overlay (editor UI) draw onto the same swapchain image before
        // present. Without one, present the tonemapped scene directly.
        let before_present = match overlay {
            Some(draw) => draw(
                after_scene,
                self.swapchain.image_views[image_index as usize].clone(),
            ),
            None => after_scene,
        };

        let future = before_present
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
