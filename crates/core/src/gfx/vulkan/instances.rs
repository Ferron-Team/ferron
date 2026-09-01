//! Per-object matrices that live on the GPU between frames.
//!
//! Every geometry pass reads the same three matrices per object — the model, its
//! inverse-transpose, and where the object stood last frame. The old shape wrote
//! all three into a fresh streaming buffer every frame, once per *list the object
//! appeared in*: an object the camera can see that also casts into four cascades
//! was written five times, 192 bytes each, whether or not it had moved since the
//! last frame. A static scene paid full upload bandwidth for matrices that were
//! bit-for-bit what the GPU already held.
//!
//! Here the rows are a device-local buffer indexed by *entity slot*, which is
//! stable for as long as the entity lives, so:
//!
//! - an object occupies one row no matter how many lists draw it, and
//! - a row is written only on the frames where its matrices actually changed.
//!
//! What each list carries instead is four bytes per entry — the row number — in
//! [`InstanceLists`], which is still streamed per frame because the lists
//! themselves change every frame. The vertex shaders gained one indirection for
//! it: `objects[instances[push.object_base + gl_InstanceIndex]]` rather than
//! `objects[push.object_base + gl_InstanceIndex]`.
//!
//! # Why a staged copy rather than a mapped write
//!
//! The rows outlive the frame that wrote them, so writing them in place would be
//! writing memory the previous frame may still be reading — `render_frame` only
//! calls `cleanup_finished` and never waits, so the CPU is free to record frame
//! N+1 while the GPU executes frame N. Instead the dirty rows go into a
//! per-frame staging subbuffer, which the allocator recycles only once the GPU
//! is done with it, and a `copy_buffer` recorded at the top of the frame's
//! command buffer moves them across. The copy is a command like any other, so
//! the ordering against the passes that read the rows is the same
//! synchronisation everything else in this renderer relies on.

use glam::Mat4;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::BufferCopy;
use vulkano::command_buffer::CopyBufferInfoTyped;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::gfx::RenderItem;

use super::context::VkContext;
use super::record::Recorder;

/// Per-object transforms, indexed through [`InstanceLists`] from a storage
/// buffer. std430 matches this `#[repr(C)]` layout exactly because every field
/// is a 64-byte `mat4` (a multiple of 16).
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct GpuObject {
    model: [[f32; 4]; 4],
    /// Inverse-transpose of `model`'s rotation/scale, for transforming normals
    /// correctly under non-uniform scaling. Stored as a mat4; only the upper-left
    /// 3x3 is used in the shader.
    normal_matrix: [[f32; 4]; 4],
    /// Last frame's `model`, for the motion vector the prepass writes. Uploaded
    /// for every pass rather than only the one that reads it: the buffer is
    /// shared, so the row's stride is shared too, and a second layout for the
    /// passes that ignore this field would be two ways for one object row to be
    /// wrong.
    prev_model: [[f32; 4]; 4],
}

/// How many rows a fresh store holds before the first growth. Large enough that
/// the built-in scenes never reallocate and small enough to be noise on a device
/// that only ever draws a handful of objects.
const INITIAL_ROWS: u32 = 1024;

/// What the GPU is believed to hold for one row.
///
/// The two source matrices rather than the packed [`GpuObject`], because these
/// are what the dirty test compares and `normal_matrix` is a function of `model`
/// — if the model matrix is unchanged the inverse-transpose is too, so storing
/// it would be a third of the memory for none of the answer.
///
/// Deliberately *not* keyed by entity generation. A slot reused by a different
/// entity that happens to stand exactly where the last one did needs no upload:
/// the row's contents are fully determined by this pair, so equal pairs mean the
/// row on the GPU is already right, whoever put it there.
#[derive(Clone, Copy, PartialEq)]
struct Resident {
    model: Mat4,
    prev_model: Mat4,
}

/// One entry of a draw order: which object row to draw, and which material to
/// draw it with.
///
/// The material rides here rather than in a push constant because a multi-draw
/// has no per-draw push — every batch of a pipeline variant is one command, and
/// the only thing that varies between them that a shader can read is what the
/// instance points at. The CPU path writes the same pairs so that one vertex
/// shader serves both. See [`cull`](super::cull).
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct InstanceEntry {
    pub row: u32,
    pub material: u32,
}

/// This frame's draw-order lists, as row numbers into [`InstanceStore`], and
/// where each list's block begins.
///
/// One buffer for every geometry pass in the frame, which is what
/// `instance_index` says in the graph. The opaque camera list indexes from zero
/// and needs no base.
pub(super) struct InstanceLists {
    pub indices: Subbuffer<[InstanceEntry]>,
    /// Where the blended items' indices start.
    pub transparent_base: u32,
    /// Where the refractive items' indices start.
    pub refractive_base: u32,
    pub cascade_bases: [u32; crate::gfx::shadows::MAX_CASCADES],
    pub punctual_bases: [u32; crate::gfx::punctual::MAX_SHADOW_LIGHTS],
}

/// The persistent row buffer, plus the mirror the dirty test reads.
pub(super) struct InstanceStore {
    /// Device-local, `TRANSFER_DST`: written only by the staged copy below.
    rows: Subbuffer<[GpuObject]>,
    /// How many rows `rows` holds. Kept alongside rather than read back off the
    /// buffer so that growth is a comparison against a `u32` slot.
    capacity: u32,
    /// What the GPU holds, per row, or `None` for a row never written. Indexed
    /// exactly as `rows` is.
    resident: Vec<Option<Resident>>,
    /// Host-visible scratch for the dirty rows of one frame. A `SubbufferAllocator`
    /// rather than a persistent buffer because this *is* per-frame data, and its
    /// recycling is what makes writing it safe while the previous frame is in
    /// flight.
    staging: SubbufferAllocator,
    /// Per-frame storage for the row numbers each list draws.
    index_allocator: SubbufferAllocator,
}

impl InstanceStore {
    pub(super) fn new(ctx: &VkContext) -> Self {
        let staging = SubbufferAllocator::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let index_allocator = SubbufferAllocator::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        Self {
            rows: allocate_rows(ctx, INITIAL_ROWS),
            capacity: INITIAL_ROWS,
            resident: vec![None; INITIAL_ROWS as usize],
            staging,
            index_allocator,
        }
    }

    /// The row buffer every geometry pass binds. Valid only after [`Self::sync`]
    /// has run for this frame, which is where a growth would have replaced it.
    pub(super) fn rows(&self) -> &Subbuffer<[GpuObject]> {
        &self.rows
    }

    /// Bring the GPU's rows up to date with `items`, recording the copy into
    /// `builder`.
    ///
    /// Returns how many rows had to be written, which is what the perf harness
    /// reads to tell a static scene from a moving one.
    pub(super) fn sync(
        &mut self,
        ctx: &VkContext,
        builder: &mut Recorder,
        items: &[RenderItem],
    ) -> usize {
        // Growth first, because the dirty test below is against the mirror this
        // may have just cleared. `+ 1` since `instance` is an index.
        let needed = items
            .iter()
            .map(|item| item.instance + 1)
            .max()
            .unwrap_or(0);
        if needed > self.capacity {
            // Next power of two, so a scene that spawns steadily reallocates a
            // logarithmic number of times rather than once per growth.
            let capacity = needed.next_power_of_two();
            self.rows = allocate_rows(ctx, capacity);
            self.capacity = capacity;
            // A new buffer holds nothing, so every row is dirty again. Cleared
            // rather than resized: the old contents describe the old buffer.
            self.resident.clear();
            self.resident.resize(capacity as usize, None);
        }

        // The dirty rows, packed into the staging buffer in the order they are
        // found, each remembering which row it is bound for.
        let mut dirty: Vec<(u32, GpuObject)> = Vec::new();
        for item in items {
            let slot = item.instance as usize;
            let current = Resident {
                model: item.model,
                prev_model: item.prev_model,
            };
            if self.resident[slot] == Some(current) {
                continue;
            }
            self.resident[slot] = Some(current);
            dirty.push((
                item.instance,
                GpuObject {
                    model: item.model.to_cols_array_2d(),
                    normal_matrix: Mat4::from_mat3(item.normal_matrix).to_cols_array_2d(),
                    prev_model: item.prev_model.to_cols_array_2d(),
                },
            ));
        }

        if dirty.is_empty() {
            return 0;
        }

        // Sorted so that rows adjacent on the GPU are adjacent in the staging
        // buffer too, which is what lets the run-coalescing below turn a scene
        // that moved wholesale into one region rather than thousands.
        dirty.sort_unstable_by_key(|&(slot, _)| slot);

        let staging = self
            .staging
            .allocate_slice::<GpuObject>(dirty.len() as u64)
            .unwrap();
        {
            let mut write = staging.write().unwrap();
            for (i, &(_, row)) in dirty.iter().enumerate() {
                write[i] = row;
            }
        }

        // One region per maximal run of consecutive rows. A frame in which
        // nothing moved records no copy at all; one in which everything moved
        // records a single region.
        let mut regions: Vec<BufferCopy> = Vec::new();
        let mut start = 0usize;
        while start < dirty.len() {
            let mut end = start + 1;
            while end < dirty.len() && dirty[end].0 == dirty[end - 1].0 + 1 {
                end += 1;
            }
            regions.push(BufferCopy {
                src_offset: start as u64,
                dst_offset: u64::from(dirty[start].0),
                size: (end - start) as u64,
                ..Default::default()
            });
            start = end;
        }

        builder.copy_buffer(CopyBufferInfoTyped {
            regions: regions.into_iter().collect(),
            ..CopyBufferInfoTyped::buffers(staging, self.rows.clone())
        });

        dirty.len()
    }

    /// Write this frame's draw orders as row numbers.
    ///
    /// The opaque `visible` list goes first so the forward and prepass passes
    /// keep indexing from zero; the blended ones follow, then each cascade's
    /// casters, then each punctual light's, and the returned bases say where.
    /// One buffer rather than one per list is what keeps `instance_index` a
    /// single resource in the graph rather than a convenient fiction.
    pub(super) fn upload_lists(
        &self,
        visible: crate::gfx::DrawList<'_>,
        transparent: crate::gfx::DrawList<'_>,
        refractive: crate::gfx::DrawList<'_>,
        casters: &[crate::gfx::DrawList<'_>],
        punctual: &[crate::gfx::DrawList<'_>],
    ) -> InstanceLists {
        let total: usize = visible.len()
            + transparent.len()
            + refractive.len()
            + casters.iter().map(crate::gfx::DrawList::len).sum::<usize>()
            + punctual
                .iter()
                .map(crate::gfx::DrawList::len)
                .sum::<usize>();
        // allocate_slice rejects length 0; an empty scene still needs a bindable
        // buffer, so round up to one (unwritten, unread) slot.
        let indices = self
            .index_allocator
            .allocate_slice::<InstanceEntry>(total.max(1) as u64)
            .unwrap();

        let transparent_base;
        let refractive_base;
        let mut cascade_bases = [0u32; crate::gfx::shadows::MAX_CASCADES];
        let mut punctual_bases = [0u32; crate::gfx::punctual::MAX_SHADOW_LIGHTS];
        {
            let mut rows = indices.write().unwrap();
            let mut next = 0usize;
            let mut write = |list: &crate::gfx::DrawList<'_>, next: &mut usize| {
                for i in 0..list.len() {
                    let item = list.item(i);
                    rows[*next] = InstanceEntry {
                        row: item.instance,
                        material: item.material.0,
                    };
                    *next += 1;
                }
            };
            write(&visible, &mut next);
            transparent_base = next as u32;
            write(&transparent, &mut next);
            refractive_base = next as u32;
            write(&refractive, &mut next);
            for (base, list) in cascade_bases.iter_mut().zip(casters) {
                *base = next as u32;
                write(list, &mut next);
            }
            for (base, list) in punctual_bases.iter_mut().zip(punctual) {
                *base = next as u32;
                write(list, &mut next);
            }
        }
        InstanceLists {
            indices,
            transparent_base,
            refractive_base,
            cascade_bases,
            punctual_bases,
        }
    }
}

/// Device-local rows, written only by the staged copy in [`InstanceStore::sync`].
fn allocate_rows(ctx: &VkContext, rows: u32) -> Subbuffer<[GpuObject]> {
    Buffer::new_slice::<GpuObject>(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        u64::from(rows),
    )
    .expect("failed to allocate the instance row buffer")
}
