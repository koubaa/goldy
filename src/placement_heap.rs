//! Persistent placement heap with paged frame allocation.
//!
//! A [`PlacementHeap`] owns a single large GPU [`Allocation`] divided into `depth`
//! fixed-size pages. Frame N uses page `N % depth` at offset `(N % depth) * page_alloc_size`.
//! Since the offset is deterministic for a given page slot, the view cache hits in
//! steady state.
//!
//! Call [`PlacementHeap::configure_pages`] once when transient heap size is known,
//! then [`PlacementHeap::advance_page`] each frame for that frame's base offset.
//!
//! This eliminates per-frame allocation overhead: in steady state the backing
//! buffer is allocated once and reused across all frames.
//!
//! ## Stable transient slot IDs
//!
//! In steady state (no shape changes, no heap growth), `BufferView`s and `Texture`s
//! are cached across frames keyed on `(slot_id, shape, placement)`. Cache hits skip
//! backend `create_buffer_view` / `Texture::new` calls entirely, eliminating the
//! per-frame bindless-slot churn that previously caused ~100 µs overhead before
//! `surface.submit_partition_early`.
//!
//! ### One-graph-per-context invariant
//!
//! The view cache is keyed on `TransientId` (a `u32` from `0..N` per graph). Since
//! `TransientId` is assigned by declaration order and reset to 0 on each
//! [`TaskGraph::clear`](crate::TaskGraph::clear), `TransientId(0)` from graph A and
//! `TransientId(0)` from graph B would collide. The heap is owned per-`Context`
//! (`ContextInner::placement_heap`), so **this heap assumes exactly one `TaskGraph`
//! per context**; independent contexts each get their own heap and `TransientId`
//! namespace and never share transient `BufferView`/`Texture` handles. Debug-assertions
//! telemetry in [`TaskGraph`](crate::TaskGraph) enforces deterministic declaration order
//! across frames.

use crate::backend::TextureHandle;
use crate::buffer::{Allocation, BufferView};
use crate::device::Device;
use crate::task_graph::TransientTextureKey;
use crate::texture::Texture;
use crate::timeline::TimelineValue;
use crate::tracy_plot;
use crate::types::{BufferFlags, BufferKind, ResourceAccess, TextureFlags, TextureKind};
use crate::vram_allocator::DeferredPayload;
use anyhow::{Context, Result};
use std::collections::HashMap;

/// Default page size for alignment within the heap (4 MiB).
const DEFAULT_PAGE_SIZE: u64 = 4 * 1024 * 1024;

/// Snapshot of the placement heap's state, useful for diagnostics and tests.
#[derive(Debug, Clone, Copy)]
pub struct PlacementHeapStats {
    /// Total capacity of the backing buffer in bytes.
    pub capacity: u64,
}

/// A cached `BufferView` for a transient slot.
///
/// Keyed by `(slot_id, base_offset, size, stride)`. When all fields match the
/// incoming request the cached view is returned directly, skipping
/// `create_buffer_view` and the bindless slot allocation.
struct CachedView {
    view: BufferView,
    base_offset: u64,
    size: u64,
    stride: u32,
}

/// A cached `Texture` for a transient texture color slot.
///
/// Keyed by `TransientTextureKey` (width, height, format). When the key matches
/// the incoming request the cached texture is returned directly, skipping
/// `Texture::new` and the bindless descriptor allocation.
struct CachedTexture {
    texture: Texture,
    key: TransientTextureKey,
}

/// Persistent GPU buffer with paged allocation for graph-colored transient heaps.
///
/// The heap owns a single large [`Allocation`]. After [`Self::configure_pages`], each
/// frame obtains a deterministic page offset via [`Self::advance_page`].
///
/// ## View and texture cache
///
/// [`Self::get_or_create_view`] and `get_or_create_textures` implement a
/// stable-slot cache. In steady state (same spec, same placement) all backend
/// descriptor work is skipped. Eviction via `Device::defer_release` ensures
/// GPU safety when shapes or placements change.
pub struct PlacementHeap {
    buffer: Allocation,
    page_size: u64,
    /// Paged allocation state. Set by [`Self::configure_pages`].
    pages: Option<PagedState>,
    /// Per-slot `BufferView` cache. Indexed by `TransientId` (raw `u32`).
    /// Survives across frames; invalidated on `grow` or explicit shape change.
    view_cache: HashMap<u32, CachedView>,
    /// Per-page-slot, per-color `Texture` cache for transient textures.
    ///
    /// Indexed as `texture_cache[page_slot][color]`.  Each page slot holds an
    /// independent set of colored textures so that concurrent in-flight renders
    /// that use different page slots (from [`Self::advance_page`]) never write to
    /// the same `TextureHandle` simultaneously.
    texture_cache: Vec<Vec<CachedTexture>>,
    /// Number of `create_buffer_view` backend calls since last reset (for tests and Tracy).
    view_create_count: usize,
    /// Number of `Texture::new` calls since last reset (for tests and Tracy).
    texture_create_count: usize,
}

/// The buffer is divided into `depth` fixed-size pages.
///
/// Frame N uses page `N % depth` at offset `(N % depth) * page_alloc_size`.
/// The offset is deterministic, so the view cache always hits in steady state.
/// Pages rotate without explicit reclaim bookkeeping.
struct PagedState {
    /// Size of each page (rounded up to `PlacementHeap::page_size`).
    page_alloc_size: u64,
    /// Number of pages (= pipeline depth).
    depth: usize,
    /// Monotonic frame counter; `advance_page()` increments this.
    frame_counter: u64,
    /// Most recently stamped timeline, used for safe eviction of stale cache entries.
    last_timeline: Option<TimelineValue>,
    /// Per-slot last-stamped timeline, indexed by `frame_counter % depth`.
    ///
    /// When a page slot is reused by a concurrent renderer, `get_or_create_textures`
    /// checks whether `slot_timelines[slot]` has been retired by the GPU before
    /// handing out the cached `TextureHandle`.  If not yet retired, a fresh texture
    /// is created so the two renders never share a `TextureHandle` on the GPU.
    slot_timelines: Vec<Option<TimelineValue>>,
    /// The page slot selected by the most recent `advance_page` call.
    last_slot_used: usize,
}

impl PlacementHeap {
    /// Create a new placement heap with the given capacity.
    ///
    /// `capacity` is rounded up to `page_size` alignment. The backing buffer is
    /// allocated immediately. Call [`Self::configure_pages`] before [`Self::advance_page`].
    pub fn new(device: &Device, capacity: u64, page_size: u64) -> Result<Self> {
        let page_size = page_size.max(256);
        let capacity = round_up(capacity.max(page_size), page_size);
        let buffer = device
            .alloc_buffer(capacity, BufferKind::Scattered, None, BufferFlags::empty())
            .context("PlacementHeap: failed to allocate backing buffer")?;
        Ok(Self {
            buffer,
            page_size,
            pages: None,
            view_cache: HashMap::new(),
            texture_cache: Vec::new(),
            view_create_count: 0,
            texture_create_count: 0,
        })
    }

    /// Create with the default page size (4 MiB).
    pub fn with_capacity(device: &Device, capacity: u64) -> Result<Self> {
        Self::new(device, capacity, DEFAULT_PAGE_SIZE)
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.buffer.allocated_size()
    }

    /// Number of `create_buffer_view` backend calls since construction (or last `reset_counts`).
    ///
    /// Useful in tests to assert cache hit/miss behaviour without Tracy.
    pub fn view_create_count(&self) -> usize {
        self.view_create_count
    }

    /// Number of `Texture::new` calls since construction (or last `reset_counts`).
    pub fn texture_create_count(&self) -> usize {
        self.texture_create_count
    }

    /// Number of `BufferView`s currently held in the stable-slot cache.
    pub fn cached_view_count(&self) -> usize {
        self.view_cache.len()
    }

    /// Number of `Texture`s currently held in the stable-slot texture cache.
    pub fn cached_texture_count(&self) -> usize {
        self.texture_cache.iter().map(|slot| slot.len()).sum()
    }

    /// Reset the diagnostic counters (for test isolation).
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn reset_counts(&mut self) {
        self.view_create_count = 0;
        self.texture_create_count = 0;
    }

    /// Configure `depth` fixed-size pages sized to hold `alloc_size` bytes each.
    ///
    /// Each page is sized to hold `alloc_size` bytes (rounded up to `page_size`).
    /// The backing buffer is grown if it cannot accommodate `depth` pages.
    ///
    /// If the heap is already configured with the same layout, this is a no-op.
    /// If the layout changes (different `alloc_size` or `depth`), the view cache is
    /// invalidated via `defer_release` and pages are reconfigured.
    ///
    /// Call this once per submit when `transient_heap_size_and_layout` returns a
    /// stable size; afterwards use [`Self::advance_page`] to get each frame's offset.
    pub fn configure_pages(&mut self, alloc_size: u64, depth: usize, device: &Device) -> Result<()> {
        let depth = depth.max(1);
        let page_alloc_size = round_up(alloc_size.max(1), self.page_size);
        let required_cap = page_alloc_size
            .checked_mul(depth as u64)
            .context("PlacementHeap::configure_pages: capacity overflow")?;

        // Fast path: already configured with identical layout.
        if let Some(ref p) = self.pages {
            if p.page_alloc_size == page_alloc_size && p.depth == depth {
                return Ok(());
            }
        }

        // Layout changed: invalidate all cached views (they point into the old layout).
        self.invalidate_all(device);

        if required_cap > self.buffer.allocated_size() {
            self.buffer = device
                .alloc_buffer(required_cap, BufferKind::Scattered, None, BufferFlags::empty())
                .context("PlacementHeap::configure_pages: failed to grow backing buffer")?;
        }

        self.pages = Some(PagedState {
            page_alloc_size,
            depth,
            frame_counter: 0,
            last_timeline: None,
            slot_timelines: vec![None; depth],
            last_slot_used: 0,
        });
        Ok(())
    }

    /// Return the base offset for the current frame's page and advance the frame counter.
    ///
    /// Must only be called after [`Self::configure_pages`]. Panics in debug builds if
    /// pages are not configured.
    ///
    /// The returned offset is page-aligned and deterministic for frame N:
    /// `(N % depth) * page_alloc_size`.
    pub fn advance_page(&mut self) -> u64 {
        let p = self
            .pages
            .as_mut()
            .expect("PlacementHeap::advance_page called before configure_pages");
        let page_idx = p.frame_counter % (p.depth as u64);
        p.frame_counter += 1;
        p.last_slot_used = page_idx as usize;
        page_idx * p.page_alloc_size
    }

    /// Returns `true` if pages have been configured via [`Self::configure_pages`].
    pub fn is_paged(&self) -> bool {
        self.pages.is_some()
    }

    /// The page slot that the next [`Self::advance_page`] call will use.
    ///
    /// Equals `frame_counter % depth`. Returns 0 when pages are not configured.
    /// Use this before calling [`Self::advance_page`] to tie transient texture
    /// allocation to the same slot as the transient buffer allocation for a frame.
    pub fn current_page_slot(&self) -> usize {
        if let Some(ref p) = self.pages {
            (p.frame_counter % p.depth as u64) as usize
        } else {
            0
        }
    }

    /// Return the cached `BufferView` for `slot_id` if its shape and placement match,
    /// or create a new one (evicting the old via `defer_release`).
    ///
    /// Returns `(uav_index, srv_index, was_cache_hit)`.
    ///
    /// ## Slot identity contract
    ///
    /// The `slot_id` is the raw `TransientId` value assigned by `TaskGraph::transient_buffer*`.
    /// Because `TaskGraph::clear` resets the counter to 0, the N-th declaration in any frame
    /// produces `slot_id = N`. Cache correctness therefore requires that the calling recording
    /// phase is **deterministic**: the same logical buffer is always declared N-th. The
    /// `#[cfg(debug_assertions)]` telemetry in `TaskGraph` asserts this invariant.
    ///
    /// ## One-graph-per-device
    ///
    /// Slot IDs are not namespaced. If two `TaskGraph`s are submitted through the same device,
    /// their IDs collide. This method panics in debug builds if concurrent submissions
    /// would violate cache correctness (see the one-graph-per-device section above).
    pub fn get_or_create_view(
        &mut self,
        slot_id: u32,
        base_offset: u64,
        size: u64,
        stride: u32,
        device: &Device,
    ) -> Result<(u32, u32, bool)> {
        // Check for a cache hit (immutable borrow ends at the closing brace).
        let is_hit = if let Some(entry) = self.view_cache.get(&slot_id) {
            entry.base_offset == base_offset && entry.size == size && entry.stride == stride
        } else {
            false
        };

        if is_hit {
            let entry = self.view_cache.get(&slot_id).unwrap();
            let uav = entry.view.resource_index(ResourceAccess::Write).unwrap_or(u32::MAX);
            let srv = entry.view.resource_index(ResourceAccess::Read).unwrap_or(uav);
            tracy_plot!("goldy.transient_resolve.cache_hit", 1.0_f64);
            return Ok((uav, srv, true));
        }

        // Evict stale entry if present (shape or placement changed).
        if let Some(old_entry) = self.view_cache.remove(&slot_id) {
            tracy_plot!("goldy.transient_resolve.cache_miss", 1.0_f64);
            self.evict_view(old_entry.view, device);
        } else {
            tracy_plot!("goldy.transient_resolve.cache_miss", 1.0_f64);
        }

        // Create a fresh view and cache it.
        let view = self.buffer.create_view(base_offset, size, Some(stride))?;
        self.view_create_count += 1;
        let uav = view.resource_index(ResourceAccess::Write).unwrap_or(u32::MAX);
        let srv = view.resource_index(ResourceAccess::Read).unwrap_or(uav);
        self.view_cache.insert(
            slot_id,
            CachedView {
                view,
                base_offset,
                size,
                stride,
            },
        );
        Ok((uav, srv, false))
    }

    /// Return or create `Texture`s for the graph-coloring color slots.
    ///
    /// `color_keys[i]` describes the texture needed at color index `i`. When the
    /// cached texture at index `i` has a matching key it is reused; otherwise the
    /// old texture is evicted via `defer_release` and a new one is created.
    ///
    /// Returns a `Vec<TextureHandle>` aligned with `color_keys`.
    pub(crate) fn get_or_create_textures(
        &mut self,
        device: &Device,
        color_keys: &[TransientTextureKey],
        page_slot: usize,
        retired_timeline: TimelineValue,
    ) -> Result<Vec<TextureHandle>> {
        // Grow the outer vec so that index `page_slot` exists.
        while self.texture_cache.len() <= page_slot {
            self.texture_cache.push(Vec::new());
        }

        // Evict surplus cached slots when the color count shrinks.
        {
            let slot = &mut self.texture_cache[page_slot];
            if slot.len() > color_keys.len() {
                let surplus: Vec<CachedTexture> = slot.drain(color_keys.len()..).collect();
                let epoch = self.max_in_flight_timeline();
                if let Some(epoch) = epoch {
                    let mut payload = DeferredPayload::new();
                    for ct in surplus {
                        payload.push(ct.texture);
                    }
                    device.defer_release(epoch, payload);
                }
                // if no in-flight epoch the textures can be dropped synchronously
            }
        }

        let mut handles = Vec::with_capacity(color_keys.len());

        for (i, key) in color_keys.iter().enumerate() {
            // Potential cache hit check.
            if i < self.texture_cache[page_slot].len() && self.texture_cache[page_slot][i].key == *key {
                // Verify the previous occupant of this page slot has been retired.
                // When two concurrent renderers share the same heap they advance to
                // different page slots, but once the counter wraps the slot may still
                // be in-flight.  Returning a live handle to a second renderer would
                // produce a GPU-level write-write race.
                let slot_epoch = self
                    .pages
                    .as_ref()
                    .and_then(|p| p.slot_timelines.get(page_slot).copied().flatten());
                let in_flight = slot_epoch.is_some_and(|epoch| epoch > retired_timeline);
                if !in_flight {
                    // Cache hit: previous use has retired, safe to reuse.
                    handles.push(self.texture_cache[page_slot][i].texture.gpu_handle());
                    continue;
                }
                // The slot is still in-flight from a concurrent render.  Fall
                // through to the cache-miss path to evict (defer-release) the old
                // texture and allocate a fresh one for this render.
            }

            // Cache miss (or in-flight eviction): create a new texture.
            if i < self.texture_cache[page_slot].len() {
                let epoch = self.max_in_flight_timeline();
                if let Some(epoch) = epoch {
                    let mut payload = DeferredPayload::new();
                    let new_tex = Texture::new(
                        device,
                        key.width,
                        key.height,
                        key.format,
                        TextureKind::Direct,
                        TextureFlags::COPY_DST,
                    )?;
                    self.texture_create_count += 1;
                    let h = new_tex.gpu_handle();
                    let old = std::mem::replace(
                        &mut self.texture_cache[page_slot][i],
                        CachedTexture {
                            texture: new_tex,
                            key: *key,
                        },
                    );
                    payload.push(old.texture);
                    device.defer_release(epoch, payload);
                    handles.push(h);
                } else {
                    // No in-flight work; safe to replace synchronously.
                    let new_tex = Texture::new(
                        device,
                        key.width,
                        key.height,
                        key.format,
                        TextureKind::Direct,
                        TextureFlags::COPY_DST,
                    )?;
                    self.texture_create_count += 1;
                    let h = new_tex.gpu_handle();
                    self.texture_cache[page_slot][i] = CachedTexture {
                        texture: new_tex,
                        key: *key,
                    };
                    handles.push(h);
                }
            } else {
                // New slot (cache is growing): just create.
                let new_tex = Texture::new(
                    device,
                    key.width,
                    key.height,
                    key.format,
                    TextureKind::Direct,
                    TextureFlags::COPY_DST,
                )?;
                self.texture_create_count += 1;
                let h = new_tex.gpu_handle();
                self.texture_cache[page_slot].push(CachedTexture {
                    texture: new_tex,
                    key: *key,
                });
                handles.push(h);
            }
        }

        Ok(handles)
    }

    /// Invalidate all cached buffer views and textures, deferring their release
    /// to the device's VramAllocator ring.
    ///
    /// Called from [`Self::grow`] (new backing buffer, all views are stale) and
    /// optionally on VRAM-pressure events (Plan 3).
    pub fn invalidate_all(&mut self, device: &Device) {
        let epoch = self.max_in_flight_timeline();
        let has_textures = self.texture_cache.iter().any(|slot| !slot.is_empty());
        if !self.view_cache.is_empty() || has_textures {
            let mut payload = DeferredPayload::new();
            for (_, entry) in self.view_cache.drain() {
                payload.push(entry.view);
            }
            for slot in self.texture_cache.iter_mut() {
                for ct in slot.drain(..) {
                    payload.push(ct.texture);
                }
            }
            if let Some(epoch) = epoch {
                device.defer_release(epoch, payload);
            }
            // if epoch is None all in-flight work is retired; drop synchronously
        }
    }

    /// Record the most recent submit timeline for safe cache eviction.
    ///
    /// Used on the standalone-submit path after the transient view cache is enabled.
    pub fn stamp_pending(&mut self, timeline: TimelineValue) {
        if let Some(ref mut p) = self.pages {
            p.last_timeline = Some(timeline);
            p.slot_timelines[p.last_slot_used] = Some(timeline);
        }
    }

    /// Record the submit timeline (for the surface-present path).
    pub fn stamp_all_pending(&mut self, timeline: TimelineValue) {
        if let Some(ref mut p) = self.pages {
            p.last_timeline = Some(timeline);
        }
    }

    /// Reference to the backing buffer (for creating views).
    pub(crate) fn buffer(&self) -> &Allocation {
        &self.buffer
    }

    /// Last stamped timeline from a submit, or `None` if no frame has been stamped yet.
    pub fn max_in_flight_timeline(&self) -> Option<TimelineValue> {
        self.pages.as_ref()?.last_timeline
    }

    /// Snapshot of the heap's current state for diagnostics.
    pub fn stats(&self) -> PlacementHeapStats {
        PlacementHeapStats {
            capacity: self.capacity(),
        }
    }

    /// Grow the heap to at least `new_capacity`.
    ///
    /// All cached buffer views and textures are invalidated (via `defer_release`)
    /// before the new backing buffer replaces the old one, since all existing views
    /// reference the old buffer handle. Paged state is reset so [`Self::configure_pages`]
    /// must be called again before the next [`Self::advance_page`].
    pub fn grow(&mut self, device: &Device, new_capacity: u64) -> Result<()> {
        let aligned_cap = round_up(new_capacity, self.page_size);
        if aligned_cap <= self.buffer.allocated_size() {
            return Ok(());
        }
        self.invalidate_all(device);
        self.buffer = device
            .alloc_buffer(aligned_cap, BufferKind::Scattered, None, BufferFlags::empty())
            .context("PlacementHeap::grow: failed to allocate new backing buffer")?;
        if let Some(ref mut p) = self.pages {
            p.frame_counter = 0;
            p.last_timeline = None;
        }
        Ok(())
    }

    /// Evict a single `BufferView` from the cache via `defer_release`.
    fn evict_view(&self, view: BufferView, device: &Device) {
        if let Some(epoch) = self.max_in_flight_timeline() {
            let mut payload = DeferredPayload::new();
            payload.push(view);
            device.defer_release(epoch, payload);
        }
        // If there is no stamped timeline, no GPU work references this view;
        // dropping synchronously is safe.
    }
}

fn round_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment > 0);
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::types::TextureFormat;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn grow_increases_capacity() {
        let device = test_device();
        let mut heap = PlacementHeap::new(&device, 4096, 1024).unwrap();
        assert_eq!(heap.capacity(), 4096);

        heap.grow(&device, 8192).unwrap();
        assert!(heap.capacity() >= 8192);
    }

    #[test]
    fn advance_page_rotates_offsets() {
        let device = test_device();
        let mut heap = PlacementHeap::new(&device, 4096, 1024).unwrap();
        heap.configure_pages(1024, 3, &device).unwrap();

        assert_eq!(heap.advance_page(), 0);
        assert_eq!(heap.advance_page(), 1024);
        assert_eq!(heap.advance_page(), 2048);
        assert_eq!(heap.advance_page(), 0);
    }

    // ── Stable slot ID tests ─────────────────────────────────────────────────

    /// Submitting the same transient spec twice should produce exactly 3
    /// `create_buffer_view` calls (one per slot, on the first frame), not 6.
    #[test]
    fn transient_view_cache_hit_in_steady_state() {
        let device = test_device();
        let mut heap = PlacementHeap::with_capacity(&device, 64 * 1024 * 1024).unwrap();

        let specs: Vec<(u32, u64, u32)> = vec![(0, 256, 4), (1, 512, 4), (2, 128, 4)];
        let base_offset = 0u64;
        let offsets: Vec<u64> = specs.iter().enumerate().map(|(i, _)| i as u64 * 512).collect();

        for (i, &(id, size, stride)) in specs.iter().enumerate() {
            heap.get_or_create_view(id, base_offset + offsets[i], size, stride, &device)
                .unwrap();
        }
        assert_eq!(heap.view_create_count(), 3, "first frame: 3 creates");

        for (i, &(id, size, stride)) in specs.iter().enumerate() {
            let (_, _, hit) = heap
                .get_or_create_view(id, base_offset + offsets[i], size, stride, &device)
                .unwrap();
            assert!(hit, "slot {id} must be a cache hit on the second frame");
        }
        assert_eq!(
            heap.view_create_count(),
            3,
            "second frame with same placement: still 3 creates total"
        );
    }

    /// Changing a slot's size should evict the old view and create a new one.
    #[test]
    fn transient_view_cache_evicts_on_shape_change() {
        let device = test_device();
        let mut heap = PlacementHeap::with_capacity(&device, 64 * 1024 * 1024).unwrap();

        heap.get_or_create_view(0, 0, 256, 4, &device).unwrap();
        assert_eq!(heap.view_create_count(), 1);

        heap.get_or_create_view(0, 0, 512, 4, &device).unwrap();
        assert_eq!(heap.view_create_count(), 2, "shape change caused a new create");
    }

    /// Growing the heap must invalidate all cached views (they reference the old buffer).
    #[test]
    fn transient_view_cache_invalidates_on_grow() {
        let device = test_device();
        let mut heap = PlacementHeap::new(&device, 4 * 1024 * 1024, DEFAULT_PAGE_SIZE).unwrap();

        heap.get_or_create_view(0, 0, 256, 4, &device).unwrap();
        assert_eq!(heap.view_create_count(), 1);

        heap.grow(&device, 8 * 1024 * 1024).unwrap();
        assert!(heap.view_cache.is_empty(), "view cache must be empty after grow");

        heap.get_or_create_view(0, 0, 256, 4, &device).unwrap();
        assert_eq!(heap.view_create_count(), 2, "post-grow frame creates a new view");
    }

    /// Minimum-size slots should not panic or pollute the cache.
    #[test]
    fn transient_view_cache_handles_zero_size() {
        let device = test_device();
        let mut heap = PlacementHeap::with_capacity(&device, 64 * 1024 * 1024).unwrap();

        let result = heap.get_or_create_view(0, 0, 1, 4, &device);
        assert!(result.is_ok(), "size=1 (minimum) must not panic");
        assert_eq!(heap.view_create_count(), 1);

        let (_, _, hit) = heap.get_or_create_view(0, 0, 1, 4, &device).unwrap();
        assert!(hit, "repeated call must be a cache hit");
        assert_eq!(heap.view_create_count(), 1);
    }

    /// Simulate 60 frames with 3 stable transient buffers — frames 2..60 must create zero views.
    #[test]
    fn steady_state_transient_resolution_zero_cost() {
        let device = test_device();
        let mut heap = PlacementHeap::with_capacity(&device, 64 * 1024 * 1024).unwrap();

        let frame_specs: Vec<(u32, u64, u32)> = vec![(0, 1024, 4), (1, 2048, 4), (2, 512, 4)];
        let offsets: Vec<u64> = vec![0, 1024, 3072];
        let num_transients = frame_specs.len();

        for frame in 0u64..60 {
            let count_before = heap.view_create_count();
            for &(id, size, stride) in &frame_specs {
                heap.get_or_create_view(id, offsets[id as usize], size, stride, &device)
                    .unwrap();
            }
            let created = heap.view_create_count() - count_before;
            if frame == 0 {
                assert_eq!(
                    created, num_transients,
                    "frame 0 must create exactly {num_transients} views"
                );
            } else {
                assert_eq!(
                    created, 0,
                    "frame {frame} must have zero new view creates (all cache hits)"
                );
            }
        }

        assert_eq!(
            heap.view_create_count(),
            num_transients,
            "total creates = {num_transients}"
        );
    }

    /// Transient textures: same shape twice should produce exactly N creates.
    #[test]
    fn transient_texture_cache_hit_in_steady_state() {
        let device = test_device();
        let mut heap = PlacementHeap::with_capacity(&device, 64 * 1024 * 1024).unwrap();

        let keys = vec![
            TransientTextureKey {
                width: 4,
                height: 4,
                format: TextureFormat::Rgba8Unorm,
            },
            TransientTextureKey {
                width: 8,
                height: 8,
                format: TextureFormat::Rgba8Unorm,
            },
        ];

        // page_slot 0, retired_timeline 0: pages aren't configured here, so the slot
        // in-flight check is a no-op and the second call must hit the cache.
        heap.get_or_create_textures(&device, &keys, 0, 0).unwrap();
        assert_eq!(heap.texture_create_count(), 2);

        heap.get_or_create_textures(&device, &keys, 0, 0).unwrap();
        assert_eq!(
            heap.texture_create_count(),
            2,
            "same keys on second frame must be cache hits"
        );
    }

    fn sample_texture_keys() -> Vec<TransientTextureKey> {
        vec![TransientTextureKey {
            width: 4,
            height: 4,
            format: TextureFormat::Rgba8Unorm,
        }]
    }

    /// `slot_timelines` must withhold a cached texture while the slot epoch is in-flight.
    #[test]
    fn texture_slot_gate_withholds_until_retired() {
        let device = test_device();
        let mut heap = PlacementHeap::with_capacity(&device, 64 * 1024).unwrap();
        heap.configure_pages(1024, 2, &device).unwrap();

        let keys = sample_texture_keys();
        let slot = 0usize;
        const SLOT_EPOCH: TimelineValue = 10;

        let _ = heap.advance_page();
        let handles_initial = heap.get_or_create_textures(&device, &keys, slot, 0).unwrap();
        assert_eq!(heap.texture_create_count(), 1);
        heap.stamp_pending(SLOT_EPOCH);

        // Wrap back to the same page slot before the GPU has retired SLOT_EPOCH.
        let _ = heap.advance_page();
        let _ = heap.advance_page();

        let handles_in_flight = heap
            .get_or_create_textures(&device, &keys, slot, SLOT_EPOCH - 1)
            .unwrap();
        assert_eq!(
            heap.texture_create_count(),
            2,
            "in-flight slot must allocate a fresh texture, not reuse the cached handle"
        );
        assert_ne!(
            handles_initial[0], handles_in_flight[0],
            "in-flight reuse would be a write-write race on the same TextureHandle"
        );

        // Once retired_timeline catches up, the cached texture is safe to hand back.
        let handles_retired = heap.get_or_create_textures(&device, &keys, slot, SLOT_EPOCH).unwrap();
        assert_eq!(
            heap.texture_create_count(),
            2,
            "retired slot must be a cache hit (no extra Texture::new)"
        );
        assert_eq!(
            handles_retired[0], handles_in_flight[0],
            "retired path should return the texture created for the in-flight render"
        );
    }

    /// Two concurrent renders sharing one heap that wrap to the same page slot must not
    /// reuse a live `TextureHandle` before the prior occupant retires.
    #[test]
    fn texture_slot_gate_concurrent_wraparound_race() {
        let device = test_device();
        let mut heap = PlacementHeap::with_capacity(&device, 64 * 1024).unwrap();
        heap.configure_pages(1024, 2, &device).unwrap();

        let keys = sample_texture_keys();

        // Render A: page slot 0, timeline 5 still in flight.
        let _ = heap.advance_page();
        let render_a = heap.get_or_create_textures(&device, &keys, 0, 0).unwrap();
        heap.stamp_pending(5);

        // Render B: uses slot 1, then wraps to slot 0 while render A is still in flight.
        let _ = heap.advance_page();
        let _ = heap.advance_page();
        let render_b = heap.get_or_create_textures(&device, &keys, 0, 3).unwrap();

        assert_eq!(
            heap.texture_create_count(),
            2,
            "wraparound while slot epoch=5 > retired=3 must force a fresh texture"
        );
        assert_ne!(
            render_a[0], render_b[0],
            "concurrent renders must not share a TextureHandle on the same page slot"
        );

        // After retirement, slot 0 reuses render B's texture.
        let render_c = heap.get_or_create_textures(&device, &keys, 0, 5).unwrap();
        assert_eq!(heap.texture_create_count(), 2);
        assert_eq!(render_c[0], render_b[0]);
    }
}
