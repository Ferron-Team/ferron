use std::sync::Arc;

use ash::vk;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorBindingFlags;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::Instance;
use vulkano::memory::MemoryHeapFlags;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::swapchain::Surface;
use vulkano::{Version, VulkanObject};

pub struct VkContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// Whether a descriptor array may be left partly unwritten.
    ///
    /// The material texture array is [`MAX_TEXTURES`] slots wide and a scene
    /// fills a handful of them. Without this the rest have to be written with a
    /// stand-in view, and the cost of that is not the write: vulkano records a
    /// tracked resource use for every element of every array the bound pipeline
    /// declares, *per draw call*, so a frame of a hundred draws tracks nineteen
    /// thousand image uses and spends milliseconds of CPU doing it. Writing only
    /// the textures that exist is what removes them, and a partly written array
    /// is only legal with this.
    ///
    /// Vulkan 1.2 core. A device without it gets the filled array it always had
    /// — see [`VkContext::texture_array_len`].
    ///
    /// [`MAX_TEXTURES`]: crate::gfx::MAX_TEXTURES
    pub partially_bound: bool,
}

impl VkContext {
    /// How many elements of the material texture array to write.
    ///
    /// Every slot when the device cannot leave one unwritten, and only the
    /// textures that exist when it can. Paired with
    /// [`mark_texture_array_partial`](Self::mark_texture_array_partial): the
    /// short write is only legal on a layout that declared the binding
    /// partially bound, so the two must read the same flag, which is why
    /// neither takes it as an argument.
    pub fn texture_array_len(&self, loaded: usize) -> usize {
        if self.partially_bound {
            loaded.min(crate::gfx::MAX_TEXTURES)
        } else {
            crate::gfx::MAX_TEXTURES
        }
    }

    /// Declare `set`'s binding 0 — the material texture array, in each of the
    /// three layouts that has one — as partially bound, when the device allows
    /// it.
    ///
    /// A no-op on a pipeline whose shaders never declared the set, so the plain
    /// shadow pipeline (depth only, no material sampling) can be handed the same
    /// call as the masked one it shares a pass with.
    pub fn mark_texture_array_partial(
        &self,
        layout: &mut PipelineDescriptorSetLayoutCreateInfo,
        set: usize,
    ) {
        if !self.partially_bound {
            return;
        }
        if let Some(binding) = layout
            .set_layouts
            .get_mut(set)
            .and_then(|set_layout| set_layout.bindings.get_mut(&0))
        {
            binding.binding_flags |= DescriptorBindingFlags::PARTIALLY_BOUND;
        }
    }

    /// `surface` is `None` for an offscreen context — one that renders into an
    /// ordinary image and never presents. The only thing it changes is device
    /// selection: with no surface there is nothing to ask for presentation
    /// support, and no swapchain extension to require. Everything downstream is
    /// identical, which is the point — an offscreen render has to go through the
    /// same device, the same features and the same passes as a windowed one, or
    /// it would not be evidence about the windowed one.
    pub fn new(instance: &Arc<Instance>, surface: Option<&Arc<Surface>>) -> Self {
        let mut device_extensions = DeviceExtensions {
            khr_swapchain: surface.is_some(),
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(instance, surface, &device_extensions);

        println!(
            "Using device: {} ({:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        // On portability-subset devices (MoltenVK on macOS) the extension must be
        // enabled if present, and egui's font/texture image views use a
        // non-identity component swizzle, which needs `image_view_format_swizzle`.
        let swizzle = physical_device
            .supported_features()
            .image_view_format_swizzle;

        let anisotropy = physical_device.supported_features().sampler_anisotropy;

        // Promoted to core in 1.2, and only read there: reaching it through
        // `VK_EXT_descriptor_indexing` on a 1.1 device would drag in that
        // extension's own dependencies for a device old enough that the filled
        // array is the safer shape anyway.
        let partially_bound = physical_device.api_version() >= Version::V1_2
            && physical_device
                .supported_features()
                .descriptor_binding_partially_bound;

        // Without it every attachment of a pipeline must blend identically, and
        // the transparency accumulation's two do not: one sums and the other
        // multiplies, which is exactly what makes its draw order irrelevant.
        // Universally supported on desktop and on Metal; asserted rather than
        // fallen back on, because there is no second blend equation to fall back
        // to.
        assert!(
            physical_device.supported_features().independent_blend,
            "this device cannot blend two attachments differently, which the \
             transparency pass requires",
        );

        if physical_device
            .supported_extensions()
            .khr_portability_subset
        {
            device_extensions.khr_portability_subset = true;
        }

        // VK_EXT_memory_budget exposes live VRAM usage/budget per heap for the
        // performance overlay. Enable it when present; `vram_bytes` falls back to
        // reporting total heap size when it isn't.
        if physical_device.supported_extensions().ext_memory_budget {
            device_extensions.ext_memory_budget = true;
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    image_view_format_swizzle: swizzle,
                    sampler_anisotropy: anisotropy,
                    independent_blend: true,
                    descriptor_binding_partially_bound: partially_bound,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();
        assert_packed_color_is_storable(&device);
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            partially_bound,
        }
    }

    /// Live device-local VRAM as `(used, total)` bytes. `used` is `None` when
    /// `VK_EXT_memory_budget` isn't available (e.g. some drivers); `total` is the
    /// summed size of all device-local heaps and is always reported.
    ///
    /// On unified-memory devices (Apple Silicon) the "device-local" heap is system
    /// RAM, so `total` there is the shared pool, not a dedicated VRAM bank.
    pub fn vram_bytes(&self) -> (Option<u64>, u64) {
        let phys = self.device.physical_device();
        let mem_props = phys.memory_properties();

        // Collect device-local heap indices and their summed size up front; the
        // budget extension reports usage per heap against these same indices.
        let mut total = 0u64;
        let device_local: Vec<usize> = mem_props
            .memory_heaps
            .iter()
            .enumerate()
            .filter(|(_, h)| h.flags.intersects(MemoryHeapFlags::DEVICE_LOCAL))
            .map(|(i, h)| {
                total += h.size;
                i
            })
            .collect();

        let instance = self.device.instance();
        if !self.device.enabled_extensions().ext_memory_budget
            || instance.api_version() < Version::V1_1
        {
            return (None, total);
        }

        // Chain the budget struct onto a memory-properties2 query, mirroring how
        // vulkano makes the same call internally.
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        {
            let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe {
                (instance.fns().v1_1.get_physical_device_memory_properties2)(
                    phys.handle(),
                    &mut props2,
                );
            }
        }

        let used = device_local.iter().map(|&i| budget.heap_usage[i]).sum();
        (Some(used), total)
    }
}

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: Option<&Arc<Surface>>,
    extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices")
        .filter(|p| p.supported_extensions().contains(extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        // A graphics queue is the whole requirement offscreen.
                        // Presentation support is a property of a surface, and
                        // there is no surface to hold it against.
                        && surface.is_none_or(|surface| {
                            p.surface_support(i as u32, surface).unwrap_or(false)
                        })
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("no suitable physical device found")
}


/// Fail at startup if the frame's packed colour format cannot back a storage
/// image on this device.
///
/// [`HDR_FORMAT`](super::hdr::HDR_FORMAT) is what most of the optical chain
/// reads and writes, and roughly half of those passes are compute passes writing
/// it as a storage image. Vulkan's mandatory-format table guarantees
/// `B10G11R11_UFLOAT_PACK32` as a sampled image and a colour attachment but *not*
/// as a storage image — in practice every desktop driver supports it, and the
/// halved bandwidth is worth more than any other single change in the frame.
///
/// Checked here, once, rather than left to surface as a descriptor-write failure
/// deep inside the first frame that runs bloom. If this ever fires on real
/// hardware the fix is to make the format a device-chosen field of `FrameConfig`
/// and compile a second set of compute shaders for the wide format — which is
/// why the assertion says so.
fn assert_packed_color_is_storable(device: &Arc<Device>) {
    let supported = device
        .physical_device()
        .format_properties(super::hdr::HDR_FORMAT)
        .map(|properties| {
            properties
                .optimal_tiling_features
                .contains(vulkano::format::FormatFeatures::STORAGE_IMAGE)
        })
        .unwrap_or(false);

    assert!(
        supported,
        "this device cannot use {:?} as a storage image, which the frame's \
         colour chain requires. Vulkan does not mandate it, though every \
         desktop driver provides it; supporting such a device means making the \
         colour format a device-chosen `FrameConfig` field and building the \
         compute passes against both formats.",
        super::hdr::HDR_FORMAT,
    );
}