//! Every mesh's geometry, in three buffers rather than three per mesh.
//!
//! A mesh used to own its position, surface and index buffers, which meant a
//! pass had to rebind all three between one mesh and the next — so a draw could
//! only ever be one mesh, and the draw *order* had to be a CPU decision. That is
//! the constraint that kept [`cull`](super::cull) recording one indirect draw
//! per batch per view: the GPU could count a batch's instances, but only the
//! host could say which mesh the next draw was for.
//!
//! Here all meshes share one position buffer, one surface buffer and one index
//! buffer, and a mesh becomes three numbers into them. A pass binds the three
//! once and every draw after that names its geometry through `firstIndex` and
//! `vertexOffset` — fields `VkDrawIndexedIndirectCommand` already carries, which
//! is what lets one multi-draw cover every batch a view kept.
//!
//! # Growth
//!
//! Meshes are appended and never freed — `load_mesh` pushes and nothing pops —
//! so placement is a bump allocator and a mesh's span is stable for the life of
//! the renderer. When a buffer runs out, a larger one is allocated and the old
//! contents are copied into its front, which leaves every span already handed
//! out still correct. The old buffer stays alive as long as a recorded frame
//! holds it: [`Recorder::bind_vertex_buffers`](super::record::Recorder) keeps
//! what it binds.
//!
//! # Why the two upload paths survive
//!
//! Wide-BAR devices write into the arena directly; narrow-BAR ones stage through
//! host memory, exactly as they did when the buffers were per mesh — the reason
//! for the split is where the memory lives, which sharing one allocation does
//! not change. See [`allocate`].

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{BufferCopy, CopyBufferInfoTyped};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::geom::Aabb;
use crate::gfx::{PositionVertex, SurfaceVertex, Vertex};
use glam::Vec3;

use super::context::VkContext;
use super::record::Recorder;

/// Vertices and indices a fresh arena holds before the first growth. Small
/// enough that a scene of a few cubes does not reserve megabytes, and a power of
/// two so the growth below never has to round.
const INITIAL_VERTICES: u64 = 1 << 14;
const INITIAL_INDICES: u64 = 1 << 15;

/// Where one mesh's geometry sits inside the shared buffers.
///
/// The three fields are `VkDrawIndexedIndirectCommand`'s, in its units: indices
/// for `first_index`, vertices for `vertex_offset`. `vertex_offset` is signed
/// because the command's is, and applies to *both* vertex bindings — which is
/// sound only because a mesh's positions and surfaces are appended together and
/// therefore share a vertex number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MeshSpan {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
}

/// The bump allocator over both watermarks.
///
/// Separate from the buffers so the arithmetic that decides where a mesh lands
/// can be tested without a device.
#[derive(Default)]
struct Placement {
    vertices: u32,
    indices: u32,
}

impl Placement {
    /// Where a mesh of this size goes, advancing both watermarks past it.
    fn place(&mut self, vertices: u32, indices: u32) -> MeshSpan {
        let span = MeshSpan {
            first_index: self.indices,
            index_count: indices,
            vertex_offset: self.vertices as i32,
        };
        self.vertices += vertices;
        self.indices += indices;
        span
    }
}

/// The capacity a buffer of `capacity` should grow to in order to hold `needed`,
/// or `None` when it already does.
///
/// Powers of two, so a scene that loads a thousand meshes copies the arena a
/// logarithmic number of times rather than a thousand.
fn grown(capacity: u64, needed: u64) -> Option<u64> {
    (needed > capacity).then(|| needed.next_power_of_two())
}

/// The shared geometry every opaque and blended pass binds.
pub(crate) struct MeshArena {
    positions: Subbuffer<[PositionVertex]>,
    surfaces: Subbuffer<[SurfaceVertex]>,
    indices: Subbuffer<[u32]>,
    placement: Placement,
    /// Whether the arena is memory the CPU can write into. Decided once, by the
    /// same BAR-width probe that used to decide it per mesh.
    host_visible: bool,
}

impl MeshArena {
    pub(super) fn new(ctx: &VkContext) -> Self {
        let host_visible = ctx.profile.wide_bar();
        Self {
            positions: allocate(
                ctx,
                BufferUsage::VERTEX_BUFFER,
                INITIAL_VERTICES,
                host_visible,
            ),
            surfaces: allocate(
                ctx,
                BufferUsage::VERTEX_BUFFER,
                INITIAL_VERTICES,
                host_visible,
            ),
            indices: allocate(
                ctx,
                BufferUsage::INDEX_BUFFER,
                INITIAL_INDICES,
                host_visible,
            ),
            placement: Placement::default(),
            host_visible,
        }
    }

    /// Bind everything a pass that shades a surface reads, plus the indices.
    ///
    /// Once per pass rather than once per draw, which is the point of the
    /// arena. Vertex bindings are command-buffer state and a secondary starts
    /// with none, so this belongs at the top of whatever records the draws —
    /// not around them.
    pub(super) fn bind(&self, builder: &mut Recorder) {
        builder
            .bind_vertex_buffers(0, (self.positions.clone(), self.surfaces.clone()))
            .bind_index_buffer(self.indices.clone());
    }

    /// The same for a depth-only pass, which reads position and texture
    /// coordinate and never the surface stream. See [`PositionVertex`].
    pub(super) fn bind_positions(&self, builder: &mut Recorder) {
        builder
            .bind_vertex_buffers(0, self.positions.clone())
            .bind_index_buffer(self.indices.clone());
    }

    /// Put a mesh into the arena and say where it landed.
    ///
    /// Also derives the object-space bounds, because upload is the last place
    /// the vertex data exists on the CPU.
    pub(super) fn upload(
        &mut self,
        ctx: &VkContext,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> (MeshSpan, Aabb) {
        let bounds = Aabb::from_points(vertices.iter().map(|v| Vec3::from(v.position)));

        // De-interleaved here because upload is the last place a vertex exists
        // as the one authored struct, and the only place that has to know the
        // two GPU streams are halves of it.
        let (positions, surfaces): (Vec<_>, Vec<_>) = vertices.iter().map(Vertex::split).unzip();

        self.reserve(ctx, positions.len() as u64, indices.len() as u64);
        let span = self
            .placement
            .place(positions.len() as u32, indices.len() as u32);

        let vertex_offset = span.vertex_offset as u64;
        if self.host_visible {
            write_directly(&self.positions, vertex_offset, &positions);
            write_directly(&self.surfaces, vertex_offset, &surfaces);
            write_directly(&self.indices, u64::from(span.first_index), indices);
        } else {
            let mut builder = Recorder::new(ctx);
            stage(
                ctx,
                &mut builder,
                &self.positions,
                vertex_offset,
                &positions,
            );
            stage(ctx, &mut builder, &self.surfaces, vertex_offset, &surfaces);
            stage(
                ctx,
                &mut builder,
                &self.indices,
                u64::from(span.first_index),
                indices,
            );
            builder.submit_and_wait(ctx);
        }

        (span, bounds)
    }

    /// Make room for one more mesh, growing whichever of the three buffers is
    /// short and copying what is already in it across.
    fn reserve(&mut self, ctx: &VkContext, vertices: u64, indices: u64) {
        let held_vertices = u64::from(self.placement.vertices);
        let held_indices = u64::from(self.placement.indices);
        let positions = grown(self.positions.len(), held_vertices + vertices);
        let surfaces = grown(self.surfaces.len(), held_vertices + vertices);
        let index_capacity = grown(self.indices.len(), held_indices + indices);
        if positions.is_none() && surfaces.is_none() && index_capacity.is_none() {
            return;
        }

        // One command buffer for up to three copies, and one wait: growth is
        // rare, but it happens while a scene is loading, where a round trip per
        // buffer would be three.
        let mut builder = Recorder::new(ctx);
        if let Some(capacity) = positions {
            self.positions = regrow(
                ctx,
                &mut builder,
                &self.positions,
                BufferUsage::VERTEX_BUFFER,
                capacity,
                held_vertices,
                self.host_visible,
            );
        }
        if let Some(capacity) = surfaces {
            self.surfaces = regrow(
                ctx,
                &mut builder,
                &self.surfaces,
                BufferUsage::VERTEX_BUFFER,
                capacity,
                held_vertices,
                self.host_visible,
            );
        }
        if let Some(capacity) = index_capacity {
            self.indices = regrow(
                ctx,
                &mut builder,
                &self.indices,
                BufferUsage::INDEX_BUFFER,
                capacity,
                held_indices,
                self.host_visible,
            );
        }
        builder.submit_and_wait(ctx);
    }
}

/// A larger buffer holding what the old one held.
///
/// The copy is recorded rather than done host-side because the arena may be
/// memory the CPU can write but not usefully read.
#[allow(clippy::too_many_arguments)]
fn regrow<T: BufferContents + Copy>(
    ctx: &VkContext,
    builder: &mut Recorder,
    old: &Subbuffer<[T]>,
    usage: BufferUsage,
    capacity: u64,
    held: u64,
    host_visible: bool,
) -> Subbuffer<[T]> {
    let new = allocate::<T>(ctx, usage, capacity, host_visible);
    if held > 0 {
        builder.copy_buffer(CopyBufferInfoTyped {
            regions: [BufferCopy {
                size: held,
                ..Default::default()
            }]
            .into_iter()
            .collect(),
            ..CopyBufferInfoTyped::buffers(old.clone(), new.clone())
        });
    }
    new
}

/// One `memcpy` into the arena — the wide-BAR path, and the reason a mesh load
/// on such a device still submits nothing.
fn write_directly<T: BufferContents + Copy>(buffer: &Subbuffer<[T]>, offset: u64, data: &[T]) {
    let mut write = buffer.write().expect("the arena is host-visible");
    write[offset as usize..offset as usize + data.len()].copy_from_slice(data);
}

/// Record a copy of `data` into `buffer` at `offset`, through host memory the
/// device can read — the narrow-BAR path.
fn stage<T: BufferContents + Copy>(
    ctx: &VkContext,
    builder: &mut Recorder,
    buffer: &Subbuffer<[T]>,
    offset: u64,
    data: &[T],
) {
    if data.is_empty() {
        return;
    }
    let staging = Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data.iter().copied(),
    )
    .expect("failed to allocate mesh staging buffer");

    builder.copy_buffer(CopyBufferInfoTyped {
        regions: [BufferCopy {
            dst_offset: offset,
            size: data.len() as u64,
            ..Default::default()
        }]
        .into_iter()
        .collect(),
        ..CopyBufferInfoTyped::buffers(staging, buffer.clone())
    });
}

/// One of the three arena buffers, put where the GPU can read it many times per
/// frame.
///
/// Static geometry is the one resource in the renderer that is written once and
/// then *re-read* by every pass in the frame — the prepass, the forward pass,
/// each shadow cascade, the atlas — which is what makes where it lives worth a
/// branch. The two vendor guides say the same thing about it from opposite
/// directions: AMD's, that device-local host-visible memory is for data "each
/// byte of which is accessed once by the GPU", and NVIDIA's, to look explicitly
/// for `DEVICE_LOCAL` when picking a memory type. Vertices are neither
/// write-once nor incidentally device-local.
///
/// The reason a branch beats picking one path is that both are right on some
/// machine:
///
/// - **Wide BAR** (Resizable BAR on, or a unified-memory part). The CPU writes
///   straight into video memory. The buffer is device-local *and* the upload is
///   one `memcpy` with no staging copy, no command buffer and no fence. Staging
///   here would be strictly worse: same destination, extra work.
/// - **Narrow BAR.** `PREFER_DEVICE | HOST_SEQUENTIAL_WRITE` reads as *required*
///   host-visible and only *preferred* device-local, so it resolves to the
///   legacy 256 MiB aperture — and when that fills, vulkano falls back to the
///   next type satisfying the requirement, which is ordinary system RAM. The
///   failure is silent and it is the bad one: every cascade then re-reads the
///   scene's geometry across PCIe, every frame, for as long as the scene is
///   open. Staging costs one copy at load and buys VRAM residency.
///
/// This is the branch that a machine with Resizable BAR *cannot* show you. On
/// such a machine the first path is taken throughout; the second exists for the
/// configuration that is still common on NVIDIA desktops and on anything with
/// the option switched off in firmware.
///
/// `TRANSFER_SRC` as well as `TRANSFER_DST` because growth reads the buffer it
/// replaces; the narrow-BAR path needs the latter anyway.
fn allocate<T: BufferContents>(
    ctx: &VkContext,
    usage: BufferUsage,
    len: u64,
    host_visible: bool,
) -> Subbuffer<[T]> {
    let memory_type_filter = if host_visible {
        MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
    } else {
        // No host-access filter, which is the entire point: it leaves the plain
        // `DEVICE_LOCAL` type — the one the aperture does not cover — as the
        // best match.
        MemoryTypeFilter::PREFER_DEVICE
    };
    Buffer::new_slice::<T>(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: usage | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter,
            ..Default::default()
        },
        len,
    )
    .expect("failed to allocate the mesh arena")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_are_appended_end_to_end() {
        let mut placement = Placement::default();
        let first = placement.place(4, 6);
        let second = placement.place(3, 3);

        assert_eq!(first.vertex_offset, 0);
        assert_eq!(first.first_index, 0);
        assert_eq!(first.index_count, 6);
        assert_eq!(second.vertex_offset, 4);
        assert_eq!(second.first_index, 6);
        assert_eq!(placement.vertices, 7);
        assert_eq!(placement.indices, 9);
    }

    /// A mesh's span has to keep meaning what it meant after the arena grows,
    /// because handles handed out earlier are never revisited.
    #[test]
    fn growth_preserves_earlier_spans() {
        let mut placement = Placement::default();
        let first = placement.place(4, 6);
        assert_eq!(grown(8, 4), None);
        assert_eq!(grown(8, 12), Some(16));
        let after = placement.place(2, 2);

        assert_eq!(first.first_index, 0);
        assert_eq!(after.first_index, 6);
        assert_eq!(after.vertex_offset, 4);
    }

    #[test]
    fn an_empty_mesh_takes_no_room() {
        let mut placement = Placement::default();
        placement.place(0, 0);
        let next = placement.place(1, 3);

        assert_eq!(next.first_index, 0);
        assert_eq!(next.vertex_offset, 0);
    }
}
