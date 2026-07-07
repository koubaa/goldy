//! Shared primitives reused across the Vulkan, DX12, and Metal backends.
//!
//! Nothing here may import from a specific backend sub-module. All items are
//! `pub(super)` so that sibling modules (`vulkan/`, `dx12/`, `metal/`) can
//! re-export them locally without leaking internal details to crate consumers.
//!
//! When only the mock backend is enabled, most items here are unused except
//! push-layout helpers consumed by task-graph `DispatchBatch` emission.
#![cfg_attr(
    not(any(
        feature = "vulkan",
        all(feature = "dx12", target_os = "windows"),
        all(feature = "metal", target_os = "macos"),
    )),
    allow(dead_code)
)]

use crate::slang::OwnedLayoutCheck;
use crate::types::{DepthStencilState, OptimizationLevel, PrimitiveTopology, TextureFormat, VertexBufferLayout};
use anyhow::Result;

use super::DeviceHandle;
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "dx12", target_os = "windows"),
))]
use super::ShaderHandle;

// ──────────────────────────────────────────────────────────────────────────────
// Push-constant layout
// ──────────────────────────────────────────────────────────────────────────────

/// Maximum number of bindless resource indices in region A of [`PushLayout`].
pub const MAX_BINDLESS_SLOTS: usize = 16;
/// Maximum number of `u32` user parameters in region B of [`PushLayout`].
pub const MAX_USER_SLOTS: usize = 8;
/// Total size of [`PushLayout`] in bytes (must match the Slang `PushLayout` struct).
pub const TOTAL_PUSH_BYTES: usize = 128;
/// Byte stride of one entry in a `DispatchBatch` argument buffer.
///
/// Layout per entry: `[PushLayout (TOTAL_PUSH_BYTES)] [wg_x u32] [wg_y u32] [wg_z u32]`
pub const DISPATCH_BATCH_STRIDE: usize = TOTAL_PUSH_BYTES + 3 * 4;

/// Packed 128-byte push-constant / root-constant / `set_bytes` layout shared by
/// all three GPU backends.
///
/// ```text
/// Bytes  0–31:  16 × u16  bindless resource indices  (region A)
/// Bytes 32–63:  8  × u32  user parameters            (region B)
/// Bytes 64–127: 64 × u8   reserved / future           (region C)
/// ```
///
/// - Region A: bindless heap indices for `Scattered<T>`, `BufRO<T>`, textures, samplers, …
/// - Region B: per-dispatch scalar user params (`uint`, `float`, `int` …).
/// - Region C: zero-filled, reserved for future extension.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct PushLayout {
    pub bindless: [u16; MAX_BINDLESS_SLOTS],
    pub user: [u32; MAX_USER_SLOTS],
    pub _reserved: [u32; 16],
}

const _: () = assert!(std::mem::size_of::<PushLayout>() == TOTAL_PUSH_BYTES);

// Safety: PushLayout is a plain-old-data type with a statically-verified size
// and no padding bytes (verified by the size assertion above plus repr(C)).
unsafe impl bytemuck::Pod for PushLayout {}
unsafe impl bytemuck::Zeroable for PushLayout {}

impl PushLayout {
    /// Reinterpret the layout as a raw byte slice.
    ///
    /// Useful for Metal `set_bytes` and any backend that needs a `*const u8`.
    /// Prefer `bytemuck::bytes_of(layout)` when `bytemuck` is already in scope.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Push-layout fill helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Fill push layout for frame-table routing: indices live in the staging/table;
/// `_reserved[0]` carries the dispatch base offset within the row.
#[inline]
pub fn fill_frame_table_dispatch(layout: &mut PushLayout, dispatch_base: u32, user: &[u32]) {
    layout._reserved[crate::frame_table::dispatch_table_base_word_index()] = dispatch_base;
    for (i, &val) in user.iter().enumerate().take(MAX_USER_SLOTS) {
        layout.user[i] = val;
    }
}

/// `PushLayout._reserved` word carrying the submitting context's selector
/// bindless slot (`_rs1` in generated shader wrappers).
pub const FRAME_TABLE_SELECTOR_SLOT_WORD: usize = 1;
/// `PushLayout._reserved` word carrying the submitting context's table
/// bindless slot (`_rs2` in generated shader wrappers).
pub const FRAME_TABLE_TABLE_SLOT_WORD: usize = 2;

/// Record-time fill of the submitting context's per-context frame-table
/// bindless slots (`_rs1`/`_rs2`).
///
/// Context-agnostic lowering (task-graph analysis) leaves these words zero;
/// each backend sets them when recording for a known context so concurrent
/// contexts on one device never share mutable descriptor slots.
#[inline]
pub fn set_frame_table_slots(layout: &mut PushLayout, selector_slot: u32, table_slot: u32) {
    layout._reserved[FRAME_TABLE_SELECTOR_SLOT_WORD] = selector_slot;
    layout._reserved[FRAME_TABLE_TABLE_SLOT_WORD] = table_slot;
}

/// Patch the frame-table slot words of every entry in a `DispatchBatch`
/// argument blob (entries are `DISPATCH_BATCH_STRIDE` bytes apart).
///
/// `DispatchBatch` arg data is built during context-agnostic lowering with the
/// slot words zeroed; backends call this at record time for the submitting
/// context.
pub fn patch_dispatch_batch_frame_table_slots(arg_data: &mut [u8], count: usize, selector_slot: u32, table_slot: u32) {
    let sel_off = (MAX_BINDLESS_SLOTS * 2) + (MAX_USER_SLOTS * 4) + FRAME_TABLE_SELECTOR_SLOT_WORD * 4;
    let tab_off = (MAX_BINDLESS_SLOTS * 2) + (MAX_USER_SLOTS * 4) + FRAME_TABLE_TABLE_SLOT_WORD * 4;
    for i in 0..count {
        let base = i * DISPATCH_BATCH_STRIDE;
        if base + TOTAL_PUSH_BYTES > arg_data.len() {
            break;
        }
        arg_data[base + sel_off..base + sel_off + 4].copy_from_slice(&selector_slot.to_ne_bytes());
        arg_data[base + tab_off..base + tab_off + 4].copy_from_slice(&table_slot.to_ne_bytes());
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Slot allocator
// ──────────────────────────────────────────────────────────────────────────────

/// A simple free-list slot allocator.
///
/// Allocates monotonically-increasing `u32` indices, recycling freed slots
/// via a LIFO free list so the live counter stays bounded under churn.
#[derive(Debug, Clone)]
pub struct SlotAllocator {
    next: u32,
    free: Vec<u32>,
}

impl Default for SlotAllocator {
    fn default() -> Self {
        Self::new(0)
    }
}

impl SlotAllocator {
    /// Create an allocator whose first fresh slot is `start`.
    ///
    /// Use `start = 0` for the common case. Non-zero starts are used by the
    /// Metal backend where each resource category is offset within a single
    /// flat argument buffer.
    pub fn new(start: u32) -> Self {
        Self {
            next: start,
            free: Vec::new(),
        }
    }

    /// Allocate a slot. Pops a recycled slot if available, otherwise mints a
    /// new one by incrementing the internal counter.
    #[inline]
    pub fn alloc(&mut self) -> u32 {
        self.free.pop().unwrap_or_else(|| {
            let i = self.next;
            self.next += 1;
            i
        })
    }

    /// Return a slot to the free list for future reuse.
    #[inline]
    pub fn free(&mut self, slot: u32) {
        self.free.push(slot);
    }

    /// Number of slots currently live (allocated and not yet freed).
    #[inline]
    pub fn live_count(&self) -> u32 {
        self.next - self.free.len() as u32
    }

    /// Number of slots currently on the free list.
    ///
    /// Only compiled for tests and the Metal backend, which exercise slot
    /// recycling; other backends' production code uses [`Self::alloc`] /
    /// [`Self::free`] only.
    #[cfg(any(test, all(feature = "metal", target_os = "macos")))]
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// The next fresh slot that would be minted if the free list were empty.
    #[cfg(any(test, all(feature = "metal", target_os = "macos")))]
    #[inline]
    pub fn next_fresh(&self) -> u32 {
        self.next
    }

    /// Ensure the next allocated slot is at least `min` (reserves low indices).
    #[inline]
    pub fn ensure_minimum_next(&mut self, min: u32) {
        if self.next < min {
            self.next = min;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Deferred queue
// ──────────────────────────────────────────────────────────────────────────────

/// A keyed queue of deferred values, drained once the key threshold is met.
///
/// Used to implement `DeletionQueue` on all three backends: values are
/// GPU-owned resources tagged with a timeline key (fence value or
/// [`crate::timeline::TimelineValue`]). The backend calls [`drain_up_to`] when
/// the GPU timeline advances past the threshold.
///
/// [`drain_up_to`]: DeferredQueue::drain_up_to
pub struct DeferredQueue<K, V> {
    pending: Vec<(K, V)>,
}

impl<K, V> Default for DeferredQueue<K, V> {
    fn default() -> Self {
        Self { pending: Vec::new() }
    }
}

impl<K, V> DeferredQueue<K, V> {
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// Enqueue `value` to be released once the key threshold reaches `key`.
    #[inline]
    pub fn push(&mut self, key: K, value: V) {
        self.pending.push((key, value));
    }

    /// Number of entries currently pending.
    #[inline]
    pub fn len(&self) -> usize {
        self.pending.len()
    }
}

impl<K: PartialOrd + Copy, V> DeferredQueue<K, V> {
    /// Drain all entries whose key is `<= threshold`, returning them as an
    /// owned `Vec`. Entries with keys above `threshold` are kept in the queue.
    pub fn drain_up_to(&mut self, threshold: K) -> Vec<V> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let mut i = 0;
        let mut eligible: Vec<V> = Vec::new();
        while i < self.pending.len() {
            if self.pending[i].0 <= threshold {
                let (_, v) = self.pending.swap_remove(i);
                eligible.push(v);
            } else {
                i += 1;
            }
        }
        eligible
    }

}

impl<K, V> DeferredQueue<K, V> {
    /// Drain all entries unconditionally (device teardown).
    pub fn flush_all(&mut self) -> impl Iterator<Item = V> + '_ {
        self.pending.drain(..).map(|(_, v)| v)
    }

    /// Drain all entries whose key satisfies `ready`, returning them as an owned `Vec`.
    ///
    /// Unlike [`Self::drain_up_to`], this does not require `K: PartialOrd` — used when a
    /// single entry's readiness depends on a multi-part requirement (e.g. a per-context
    /// `(ContextHandle, u64)` snapshot) rather than one totally-ordered threshold.
    pub fn drain_where<F: Fn(&K) -> bool>(&mut self, ready: F) -> Vec<V> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let mut i = 0;
        let mut eligible: Vec<V> = Vec::new();
        while i < self.pending.len() {
            if ready(&self.pending[i].0) {
                let (_, v) = self.pending.swap_remove(i);
                eligible.push(v);
            } else {
                i += 1;
            }
        }
        eligible
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Staging belt core
// ──────────────────────────────────────────────────────────────────────────────

/// Byte alignment applied to every write offset in a staging belt chunk.
pub const STAGING_COPY_ALIGN: u64 = 256;
/// Default staging chunk size used by both Vulkan and DX12 belts.
pub const DEFAULT_STAGING_CHUNK_SIZE: u64 = 256 * 1024;

/// Backend-agnostic interface for a single staging-belt chunk.
///
/// Implementors own the GPU-visible upload buffer; the generic
/// [`StagingBeltCore`] manages the bump-allocator bookkeeping on top.
pub trait BeltChunk: Sized {
    /// Total capacity of this chunk in bytes.
    fn capacity(&self) -> u64;
    /// Current write cursor (bytes written so far, potentially unaligned).
    fn offset(&self) -> u64;
    /// Mutable reference to the write cursor.
    fn offset_mut(&mut self) -> &mut u64;
    /// Pointer to the start of the mapped host-visible memory region.
    ///
    /// # Safety
    /// The caller must only write within `[mapped_ptr(), mapped_ptr() + capacity())`.
    fn mapped_ptr(&self) -> *mut u8;

    /// Reset the write cursor to zero so the chunk can be reused.
    #[inline]
    fn reset(&mut self) {
        *self.offset_mut() = 0;
    }
}

/// Shared bump-allocator bookkeeping for staging belts.
///
/// Both the Vulkan and DX12 backends wrap this type, contributing only
/// their backend-specific chunk allocation, reclaim, and resource-handle
/// extraction.
pub struct StagingBeltCore<C> {
    pub free: Vec<C>,
    pub active: Vec<C>,
    /// Chunks in-flight, keyed by the fence/timeline token they must wait on.
    pub in_flight: Vec<(u64, Vec<C>)>,
    pub chunk_size: u64,
}

impl<C: BeltChunk> StagingBeltCore<C> {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            free: Vec::new(),
            active: Vec::new(),
            in_flight: Vec::new(),
            chunk_size,
        }
    }

    /// Write `data` into the belt and return `(active_chunk_index, start_offset)`.
    ///
    /// If the current active chunk has insufficient space, a free chunk is
    /// recycled or `alloc` is called to create a fresh one. The returned index
    /// into `self.active` lets the caller extract the backend-specific resource
    /// handle (e.g. `vk::Buffer` or `ID3D12Resource`).
    pub fn write(&mut self, data: &[u8], alloc: impl FnOnce(u64) -> Result<C>) -> Result<(usize, u64)> {
        if data.is_empty() {
            anyhow::bail!("StagingBeltCore::write: empty data");
        }
        let len = data.len() as u64;

        // Try to fit into the last active chunk.
        if let Some(ch) = self.active.last_mut() {
            let start = align_up(ch.offset(), STAGING_COPY_ALIGN);
            if start + len <= ch.capacity() {
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ch.mapped_ptr().add(start as usize), data.len());
                }
                *ch.offset_mut() = start + len;
                return Ok((self.active.len() - 1, start));
            }
        }

        let alloc_size = self.chunk_size.max(align_up(len, STAGING_COPY_ALIGN));

        // Linear scan from the back (most-recently-freed first) to avoid the
        // push-then-immediately-pop loop when every free chunk is smaller than
        // `len`.
        let mut chunk = if let Some(pos) = self.free.iter().rposition(|c| c.capacity() >= len) {
            let mut c = self.free.swap_remove(pos);
            c.reset();
            c
        } else {
            alloc(alloc_size)?
        };

        debug_assert_eq!(chunk.offset(), 0);
        let start = 0u64;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), chunk.mapped_ptr().add(start as usize), data.len());
        }
        *chunk.offset_mut() = start + len;
        self.active.push(chunk);
        Ok((self.active.len() - 1, start))
    }

    /// Move all active chunks to the in-flight list, tagged with `token`.
    pub fn finish(&mut self, token: u64) {
        if self.active.is_empty() {
            return;
        }
        self.in_flight.push((token, std::mem::take(&mut self.active)));
    }

    /// Drop free chunks whose capacity exceeds `self.chunk_size`.
    ///
    /// Safe to call at any time: free chunks are only moved here after their
    /// associated fence/timeline has signaled.
    pub fn trim_free(&mut self, mut destroy: impl FnMut(C)) {
        let limit = self.chunk_size;
        let mut i = 0;
        while i < self.free.len() {
            if self.free[i].capacity() > limit {
                destroy(self.free.swap_remove(i));
            } else {
                i += 1;
            }
        }
    }

    /// Destroy all chunks in every list (device teardown).
    pub fn destroy_all(&mut self, mut destroy: impl FnMut(C)) {
        for ch in self.free.drain(..) {
            destroy(ch);
        }
        for ch in self.active.drain(..) {
            destroy(ch);
        }
        for (_, chunks) in self.in_flight.drain(..) {
            for ch in chunks {
                destroy(ch);
            }
        }
    }
}

/// Round `x` up to the next multiple of alignment `a`.
#[inline]
pub fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal [`BeltChunk`] backed by a heap-allocated byte array.
    ///
    /// The heap allocation is stable across moves of the `MockChunk` struct
    /// itself, so `mapped_ptr()` remains valid after the chunk is pushed into a
    /// `Vec` inside [`StagingBeltCore`].
    struct MockChunk {
        backing: Vec<u8>,
        offset: u64,
    }

    impl MockChunk {
        fn new(capacity: u64) -> Self {
            Self {
                backing: vec![0u8; capacity as usize],
                offset: 0,
            }
        }
    }

    impl BeltChunk for MockChunk {
        fn capacity(&self) -> u64 {
            self.backing.len() as u64
        }
        fn offset(&self) -> u64 {
            self.offset
        }
        fn offset_mut(&mut self) -> &mut u64 {
            &mut self.offset
        }
        fn mapped_ptr(&self) -> *mut u8 {
            self.backing.as_ptr() as *mut u8
        }
    }

    fn new_core(chunk_size: u64) -> StagingBeltCore<MockChunk> {
        StagingBeltCore::new(chunk_size)
    }

    /// `finish()` moves all active chunks to `in_flight` tagged with the given
    /// token, leaving `active` empty.  A subsequent `write()` cannot touch the
    /// in-flight chunk — it must open a new active chunk.
    #[test]
    fn staging_belt_finish_isolates_in_flight_chunk() {
        let mut core = new_core(256);

        let payload_a = b"hello";
        let (idx_a, start_a) = core.write(payload_a, |sz| Ok(MockChunk::new(sz))).unwrap();
        assert_eq!(core.active.len(), 1, "one active chunk after first write");
        assert_eq!(core.in_flight.len(), 0);
        assert_eq!(
            &core.active[idx_a].backing[start_a as usize..start_a as usize + payload_a.len()],
            payload_a
        );

        core.finish(42);
        assert_eq!(core.active.len(), 0, "active drained by finish");
        assert_eq!(core.in_flight.len(), 1, "one in-flight batch");
        assert_eq!(core.in_flight[0].0, 42, "token preserved");
        assert_eq!(core.in_flight[0].1.len(), 1, "one chunk in batch");

        // A second write must NOT touch the in-flight chunk.
        let payload_b = b"world";
        let (idx_b, _start_b) = core.write(payload_b, |sz| Ok(MockChunk::new(sz))).unwrap();
        assert_eq!(core.active.len(), 1, "new active chunk opened");
        assert_eq!(core.in_flight.len(), 1, "in-flight batch is untouched");
        assert_eq!(idx_b, 0, "second write is in a fresh active chunk at index 0");

        // The in-flight chunk still holds payload_a unmodified.
        let inflight_chunk = &core.in_flight[0].1[0];
        assert_eq!(
            &inflight_chunk.backing[start_a as usize..start_a as usize + payload_a.len()],
            payload_a
        );
    }

    /// Once a chunk is moved from `in_flight` to `free` (simulating a reclaim),
    /// the next `write()` reuses it without calling the allocation closure.
    #[test]
    fn staging_belt_reclaimed_chunk_is_reused() {
        let mut core = new_core(256);

        core.write(b"first", |sz| Ok(MockChunk::new(sz))).unwrap();
        core.finish(1);
        assert_eq!(core.in_flight.len(), 1);

        // Simulate reclaim: token 1 has completed.
        let (token, mut chunks) = core.in_flight.remove(0);
        assert_eq!(token, 1);
        for ch in &mut chunks {
            ch.reset();
        }
        core.free.extend(chunks);
        assert_eq!(core.free.len(), 1, "chunk available for reuse");

        // The allocation closure must NOT be called — the free chunk is reused.
        let alloc_called = std::cell::Cell::new(false);
        core.write(b"second", |sz| {
            alloc_called.set(true);
            Ok(MockChunk::new(sz))
        })
        .unwrap();

        assert!(!alloc_called.get(), "free chunk must be reused, not a fresh allocation");
        assert_eq!(core.free.len(), 0, "free list drained");
        assert_eq!(core.active.len(), 1);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Clear-size resolution
// ──────────────────────────────────────────────────────────────────────────────

/// Resolve the effective byte count for a `clear_buffer` call.
///
/// A `size` of `0` is the sentinel for "clear to end of buffer"; in that case
/// the function returns `buffer_size.saturating_sub(offset)`.
#[inline]
pub fn resolve_clear_size(buffer_size: u64, offset: u64, size: u64) -> u64 {
    if size == 0 {
        buffer_size.saturating_sub(offset)
    } else {
        size
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shader / pipeline creation descriptors (shared across backends)
// ──────────────────────────────────────────────────────────────────────────────

/// Deferred shader creation parameters shared by every GPU backend.
#[derive(Debug)]
pub struct ShaderDesc<'a> {
    pub device: DeviceHandle,
    pub slang_source: &'a str,
    pub search_paths: &'a [&'a str],
    pub defines: &'a [(&'a str, &'a str)],
    pub optimization_level: OptimizationLevel,
    pub layout_checks: Vec<OwnedLayoutCheck>,
}

impl<'a> ShaderDesc<'a> {
    #[inline]
    pub fn new(
        device: DeviceHandle,
        slang_source: &'a str,
        search_paths: &'a [&'a str],
        defines: &'a [(&'a str, &'a str)],
        optimization_level: OptimizationLevel,
    ) -> Self {
        Self {
            device,
            slang_source,
            search_paths,
            defines,
            optimization_level,
            layout_checks: Vec::new(),
        }
    }

    #[inline]
    pub fn with_layout_checks(mut self, layout_checks: Vec<OwnedLayoutCheck>) -> Self {
        self.layout_checks = layout_checks;
        self
    }
}

/// Slang compile inputs for a single shader stage (Metal path; same field set as other backends).
#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
pub struct ShaderStageCompileDesc<'a> {
    pub slang_source: &'a str,
    pub search_paths: &'a [&'a str],
    pub entry_point: &'a str,
    pub stage: crate::slang::SlangStage,
    pub extra_defines: &'a [(&'a str, &'a str)],
    pub layout_checks: &'a [OwnedLayoutCheck],
    pub optimization_level: OptimizationLevel,
}

/// Rasterization state for a graphics pipeline, shared across Vulkan, DX12, and Metal.
#[derive(Debug, Clone, Copy)]
pub struct PipelineDesc<'a> {
    pub vertex_layout: &'a VertexBufferLayout,
    pub topology: PrimitiveTopology,
    pub target_format: TextureFormat,
    pub depth_stencil: Option<&'a DepthStencilState>,
}

impl<'a> PipelineDesc<'a> {
    #[inline]
    pub fn new(
        vertex_layout: &'a VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
    ) -> Self {
        Self {
            vertex_layout,
            topology,
            target_format,
            depth_stencil: None,
        }
    }

    #[inline]
    pub fn with_depth_stencil(mut self, depth_stencil: Option<&'a DepthStencilState>) -> Self {
        self.depth_stencil = depth_stencil;
        self
    }
}

/// Full graphics pipeline creation inputs for backends that resolve shaders by handle.
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "dx12", target_os = "windows"),
))]
#[derive(Debug, Clone, Copy)]
pub struct GraphicsPipelineCreateDesc<'a> {
    pub device_handle: DeviceHandle,
    pub vertex_shader: ShaderHandle,
    pub fragment_shader: ShaderHandle,
    pub raster: &'a PipelineDesc<'a>,
}
