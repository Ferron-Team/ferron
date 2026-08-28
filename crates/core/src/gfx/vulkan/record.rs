//! Recording without vulkano's auto-synchronisation, and submitting without its
//! future chain.
//!
//! This is the seam tier 3 item 1 asks for. Everywhere else in `gfx/vulkan/`,
//! commands go into an [`AutoCommandBufferBuilder`], which tracks every
//! resource's state per recorded command on the CPU and derives its own
//! barriers — a second synchronisation compiler, running beside the one in
//! `gfx/graph/` whose plan nothing submits. What lands here records into
//! vulkano's raw [`RecordingCommandBuffer`] instead: no tracking, no derived
//! barriers, and the caller says what the dependencies are.
//!
//! [`AutoCommandBufferBuilder`]: vulkano::command_buffer::AutoCommandBufferBuilder
//!
//! # Why submission is hand-rolled
//!
//! Not for control — because vulkano 0.35 leaves no alternative. A finished raw
//! [`CommandBuffer`] does not implement `PrimaryCommandBufferAbstract`, and
//! cannot be made to from outside the crate: that trait requires
//! `fn resources_usage(&self) -> &CommandBufferResourcesUsage`, and every field
//! of `CommandBufferResourcesUsage` is `pub(crate)` with no constructor and no
//! `Default`. `GpuFuture::then_execute` takes that trait, and so does
//! `QueueGuard::submit` by way of `CommandBufferSubmitInfo`. So a raw command
//! buffer can only reach the queue through `ash`, which is already a direct
//! dependency for the memory-budget query in `context.rs`.
//!
//! The practical consequence is that "record through the unchecked path" and
//! "drop to `ash`" — options (b) and (c) of the item — are one change, not two.
//!
//! # What the caller owes
//!
//! [`RecordingCommandBuffer`] keeps nothing alive and checks nothing. Every
//! buffer, image, view, descriptor set and pipeline a recorded command names
//! must outlive the GPU's execution of it, and every dependency between two
//! recorded commands must be a barrier the caller wrote. [`submit_one_shot`]
//! discharges the lifetime half by waiting before it returns, which is why it is
//! the shape the upload paths use; a frame that does not wait has to hold its
//! own keep-alive list until the fence signals.
//!
//! # Raw and auto-synchronised buffers cannot share an image
//!
//! This is the constraint that decides how the rest of the migration is staged,
//! and it is not visible from either API's surface.
//!
//! Vulkano tracks, per `Image`, whether any command buffer has ever declared a
//! use of it (`Image::is_layout_initialized`). The first auto-synchronised
//! command buffer to touch an image assumes it is in `Undefined` on entry and
//! sets that flag; every later one assumes the image's canonical layout instead
//! (`RawImage::default_layout` — `ShaderReadOnlyOptimal` for a sampled texture).
//! A raw command buffer sets nothing, because it declares nothing.
//!
//! So an image uploaded here and then sampled by an auto-synchronised frame is
//! still, as far as vulkano knows, untouched: the frame emits an
//! `Undefined -> ShaderReadOnlyOptimal` barrier before its first sample
//! (`auto/builder.rs`, the `state.initial_layout` barrier), and `Undefined` as a
//! source layout is the spec's licence to **discard the contents**. The upload
//! that just happened is what would be discarded.
//!
//! Measured, no driver here takes that licence: the twelve `offscreen` captures
//! are byte-identical to the auto-synchronised path on both RADV GFX1201 and
//! lavapipe. That is evidence it is benign today, not that it is correct — it is
//! the one-vendor-scheduler failure mode the graph exists to remove, pointed the
//! other way, and a driver is free to start discarding at any release.
//!
//! There is no public lever to set the flag: `Image::layout_initialized` is
//! `pub(crate)`, and `VkImageCreateInfo::initialLayout` cannot be anything but
//! `Undefined` for an optimal-tiled device-local image. The consequence is that
//! the conversion cannot be staged per resource, only per *resource's whole
//! lifetime*: an image recorded into raw must never be named by an auto
//! builder. For textures that means the frame converts with them.
//!
//! # Barriers use pre-`synchronization2` flags
//!
//! `synchronization2` is not enabled on the device (`context.rs`), so
//! `pipeline_barrier` takes vulkano's `VK_VERSION_1_0` path, which narrows the
//! 64-bit `AccessFlags2` to 32 bits with an `as u32`. Any access bit at 32 or
//! above — `SHADER_SAMPLED_READ`, `SHADER_STORAGE_READ` and the rest of the
//! sync2-only set — truncates to zero there, which is a barrier that silently
//! carries no access mask. Until the feature is enabled, stay on flags that
//! existed in 1.0: `SHADER_READ` rather than `SHADER_SAMPLED_READ`.

use std::sync::Arc;

use ash::vk;
use vulkano::VulkanObject;
use vulkano::command_buffer::{CommandBufferBeginInfo, RecordingCommandBuffer};
use vulkano::command_buffer::{CommandBufferLevel, CommandBufferUsage};
use vulkano::image::{Image, ImageAspects, ImageLayout, ImageSubresourceRange};
use vulkano::sync::fence::{Fence, FenceCreateInfo};
use vulkano::sync::{AccessFlags, DependencyInfo, ImageMemoryBarrier, PipelineStages};

use super::context::VkContext;

/// Records a one-time command buffer with no auto-synchronisation, submits it,
/// and blocks until the GPU is done with it.
///
/// The wait is what makes this safe to hand a closure that borrows: nothing the
/// closure named can be dropped before the GPU has finished reading it, because
/// this does not return until then. That also makes it the wrong primitive for a
/// frame — see the module docs — and the right one for the load-time uploads,
/// which block on a fence today regardless.
pub(super) fn submit_one_shot(ctx: &VkContext, record: impl FnOnce(&mut RecordingCommandBuffer)) {
    let mut recording = RecordingCommandBuffer::new(
        ctx.command_buffer_allocator.clone(),
        ctx.queue.queue_family_index(),
        CommandBufferLevel::Primary,
        CommandBufferBeginInfo {
            usage: CommandBufferUsage::OneTimeSubmit,
            ..Default::default()
        },
    )
    .expect("failed to begin a raw command buffer");

    record(&mut recording);

    // SAFETY: the closure has returned, so nothing is still recording, and every
    // command it recorded is one this module's callers have paired with the
    // barriers it needs.
    let command_buffer = unsafe { recording.end() }.expect("failed to end a raw command buffer");

    let fence = Fence::new(ctx.device.clone(), FenceCreateInfo::default())
        .expect("failed to create the upload fence");

    let handles = [command_buffer.handle()];
    let submit = vk::SubmitInfo::default().command_buffers(&handles);

    // `Queue::with` is taken for the lock, not for what the guard offers: a
    // `VkQueue` is externally synchronised, and vulkano's own submissions take
    // the same lock.
    ctx.queue.clone().with(|_guard| {
        let fns = ctx.device.fns();
        // SAFETY: one submission of one command buffer that finished recording
        // above, on the queue whose family it was allocated from, under the
        // queue's lock, with a fresh unsignalled fence.
        unsafe { (fns.v1_0.queue_submit)(ctx.queue.handle(), 1, &submit, fence.handle()) }
            .result()
            .expect("failed to submit a raw command buffer");
    });

    fence
        .wait(None)
        .expect("failed to wait on the upload fence");
}

/// Every mip level and array layer of a colour image.
pub(super) fn whole_image(image: &Arc<Image>) -> ImageSubresourceRange {
    ImageSubresourceRange {
        aspects: ImageAspects::COLOR,
        mip_levels: 0..image.mip_levels(),
        array_layers: 0..image.array_layers(),
    }
}

/// One image barrier, recorded on its own.
///
/// Verbose by design: the point of this seam is that a transition is written
/// down where it happens rather than derived, so the call site reads as the
/// dependency it is.
pub(super) fn image_barrier(recording: &mut RecordingCommandBuffer, barrier: ImageMemoryBarrier) {
    // SAFETY: the caller is asserting the dependency this expresses is the one
    // the surrounding commands need; vulkano validates the barrier itself.
    unsafe {
        recording.pipeline_barrier(&DependencyInfo {
            image_memory_barriers: [barrier].into_iter().collect(),
            ..Default::default()
        })
    }
    .expect("invalid image barrier");
}

/// The transition an image needs before anything may copy or blit into it,
/// from a state whose contents do not matter.
pub(super) fn to_transfer_dst(
    image: Arc<Image>,
    subresource_range: ImageSubresourceRange,
    old_layout: ImageLayout,
) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::TOP_OF_PIPE,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStages::ALL_TRANSFER,
        dst_access: AccessFlags::TRANSFER_WRITE,
        old_layout,
        new_layout: ImageLayout::TransferDstOptimal,
        subresource_range,
        ..ImageMemoryBarrier::image(image)
    }
}

/// The transition a level needs between being written and being read as a blit
/// source.
pub(super) fn transfer_dst_to_src(
    image: Arc<Image>,
    subresource_range: ImageSubresourceRange,
) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::ALL_TRANSFER,
        src_access: AccessFlags::TRANSFER_WRITE,
        dst_stages: PipelineStages::ALL_TRANSFER,
        dst_access: AccessFlags::TRANSFER_READ,
        old_layout: ImageLayout::TransferDstOptimal,
        new_layout: ImageLayout::TransferSrcOptimal,
        subresource_range,
        ..ImageMemoryBarrier::image(image)
    }
}

/// The transition that hands a finished image to the shaders that sample it.
///
/// `new_layout` is `ShaderReadOnlyOptimal` because that is what vulkano calls an
/// image's canonical layout when its usage is `SAMPLED` plus transfer bits
/// (`RawImage::default_layout`), and every auto-synchronised command buffer that
/// later samples this image will assume it is in that layout on entry.
pub(super) fn to_shader_read(
    image: Arc<Image>,
    subresource_range: ImageSubresourceRange,
    old_layout: ImageLayout,
) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::ALL_TRANSFER,
        src_access: AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE,
        dst_stages: PipelineStages::FRAGMENT_SHADER,
        dst_access: AccessFlags::SHADER_READ,
        old_layout,
        new_layout: ImageLayout::ShaderReadOnlyOptimal,
        subresource_range,
        ..ImageMemoryBarrier::image(image)
    }
}
