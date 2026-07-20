//! Debug-line overlay pass — **scaffold; GPU plumbing left to the owner.**
//!
//! Draws the engine's per-frame [`DebugLine`] buffer as GPU line primitives into
//! the forward (HDR) render pass, right after the scene geometry. Sharing that
//! subpass is deliberate: the pass reuses the existing depth buffer, so debug
//! lines are correctly *occluded* by scene geometry, and it reuses the camera
//! view-projection. The one accepted cost is colour: the forward target is HDR
//! and tonemapped downstream, so a line's on-screen colour drifts from the exact
//! RGBA the script requested. Drawing post-tonemap would fix the colour but lose
//! depth occlusion — see the design notes on the issue.
//!
//! This module gives the rest of the engine a stable shape to call. The two
//! methods below compile and run as no-ops today; the `TODO(owner)` steps are
//! the renderer internals to fill in.

use std::sync::Arc;

use vulkano::buffer::BufferContents;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::vertex_input::Vertex as VertexTrait;
use vulkano::render_pass::RenderPass;

use crate::scene::{Camera, DebugLine};

/// One endpoint of a debug line: world-space position + RGBA colour. Two of
/// these make a segment, drawn with `PrimitiveTopology::LineList`.
#[derive(BufferContents, VertexTrait, Clone, Copy)]
#[repr(C)]
pub struct LineVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

pub struct LinePass {
    // TODO(owner): hold the `GraphicsPipeline` here, plus a way to stream this
    // frame's vertices — either a growable host-visible `Subbuffer<[LineVertex]>`
    // or a `SubbufferAllocator` (like `ForwardPass`'s uniform allocator).
}

impl LinePass {
    /// Build the line pipeline against the forward render pass's subpass 0.
    pub fn new(
        _device: &Arc<Device>,
        _memory_allocator: &Arc<StandardMemoryAllocator>,
        _render_pass: &Arc<RenderPass>,
    ) -> Self {
        // TODO(owner): build the pipeline, mirroring `forward::build_pipeline`
        // but with these differences:
        //   - InputAssemblyState { topology: PrimitiveTopology::LineList, .. }
        //   - RasterizationState::default() (no back-face cull for lines)
        //   - DepthStencilState with DepthState { write_enable: false, compare_op:
        //     CompareOp::Less, .. } — occluded by geometry, but writes no depth
        //   - MultisampleState with Sample4 (must match the forward subpass)
        //   - vertex_input_state from `LineVertex::per_vertex().definition(&vs)`
        //   - a push constant holding the camera view-projection (a `mat4`)
        //   - subpass 0 of `render_pass`
        //   - shaders `shaders/line.vert` / `shaders/line.frag` via the
        //     `vulkano_shaders::shader!` macro (see `forward`'s `vs`/`fs` mods)
        Self {}
    }

    /// Record this frame's lines into `builder`. Must be called *inside* the
    /// forward render pass, after the scene geometry. No-op until the pipeline is
    /// implemented.
    pub fn record(
        &mut self,
        _builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        lines: &[DebugLine],
        _camera: &Camera,
        _extent: [u32; 2],
    ) {
        if lines.is_empty() {
            return;
        }

        // Flatten each segment into its two endpoints. This CPU-side massaging is
        // the non-renderer half of the work, so it's done here; the GPU upload
        // and draw are the parts to implement.
        let mut vertices: Vec<LineVertex> = Vec::with_capacity(lines.len() * 2);
        for line in lines {
            vertices.push(LineVertex {
                position: line.from.to_array(),
                color: line.color,
            });
            vertices.push(LineVertex {
                position: line.to.to_array(),
                color: line.color,
            });
        }

        let _ = vertices; // TODO(owner): remove once the buffer below consumes it.

        // TODO(owner):
        //   1. Upload `vertices` into a host-visible vertex buffer (reuse/grow it
        //      across frames rather than allocating each frame).
        //   2. `set_viewport` for `extent`, `bind_pipeline_graphics`, and
        //      `push_constants` with `camera.view_projection(aspect)` where
        //      `aspect = extent[0] as f32 / extent[1] as f32`.
        //   3. `bind_vertex_buffers(0, buffer)` then
        //      `draw(vertices.len() as u32, 1, 0, 0)`.
    }
}
