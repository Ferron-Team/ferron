use std::sync::Arc;

use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::Instance;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::Surface;

pub struct VkContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl VkContext {
    pub fn new(instance: &Arc<Instance>, surface: &Arc<Surface>) -> Self {
        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
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
        if physical_device.supported_extensions().khr_portability_subset {
            device_extensions.khr_portability_subset = true;
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
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();
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
        }
    }
}

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
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
                        && p.surface_support(i as u32, surface).unwrap_or(false)
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
