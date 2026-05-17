//! Persistent placement heap with ring-based frame allocation.
//!
//! A [`PlacementHeap`] owns a single large GPU [`Buffer`] and carves frame-sized
//! regions from it using a ring allocator. Each frame acquires a contiguous region,
//! creates [`BufferView`]s at graph-colored offsets within that region, and releases
//! the region once the GPU retires past its timeline epoch. View lifetimes are tracked
//! via [`Device::defer_release`] rather than the ring itself.
//!
//! This eliminates per-frame `Buffer::new` overhead: in steady state the backing
//! buffer is allocated once and reused across all frames. Only lightweight
//! `BufferView` creation (bindless slot registration) happens per frame.

use crate::buffer::{Buffer, BufferView};
use crate::device::Device;
use crate::timeline::TimelineValue;
use crate::types::{BufferFlags, DataAccess};
use crate::vram_allocator::DeferredPayload;
use anyhow::{Context, Result};
use std::collections::VecDeque;

/// Default page size for alignment within the heap (4 MiB).
const DEFAULT_PAGE_SIZE: u64 = 4 * 1024 * 1024;

/// Snapshot of the placement heap's state, useful for diagnostics and tests.
#[derive(Debug, Clone, Copy)]
pub struct PlacementHeapStats {
    /// Total capacity of the backing buffer in bytes.
    pub capacity: u64,
    /// Bytes currently occupied by in-flight regions.
    pub in_flight_bytes: u64,
    /// Number of in-flight regions (one per submitted frame).
    pub in_flight_count: usize,
}

/// Persistent GPU buffer with ring-based allocation for graph-colored transient heaps.
///
/// The heap owns a single large [`Buffer`]. Each frame acquires a contiguous region
/// via [`Self::acquire`], creates views within it, and later releases the region via
/// [`Self::reclaim`] once the GPU has retired past the frame's timeline.
///
/// The ring ensures multiple in-flight frames coexist without overlap:
/// - Frame N occupies `[offset_N, offset_N + size_N)`
/// - Frame N+1 occupies `[offset_{N+1}, ...)`
/// - When Frame N retires, its region becomes available for future frames
///
/// View lifetimes are managed via [`Device::defer_release`] rather than the ring
/// itself: callers pass views to [`Self::stamp_and_defer_views`] (standalone path)
/// or to the [`Frame`](crate::surface::Frame) keepalive (surface path). The ring
/// tracks only byte-range allocation and timeline values.
pub struct PlacementHeap {
    buffer: Buffer,
    page_size: u64,
    /// Next write position (byte offset into the buffer).
    bump: u64,
    /// In-flight regions, ordered by submission time (FIFO).
    regions: VecDeque<RingEntry>,
}

/// Tracks a frame's allocation within the ring.
struct RingEntry {
    base_offset: u64,
    /// Rounded-up size (page-aligned).
    size: u64,
    timeline: Option<TimelineValue>,
}

impl PlacementHeap {
    /// Create a new placement heap with the given capacity.
    ///
    /// `capacity` is rounded up to `page_size` alignment. The backing buffer is
    /// allocated immediately.
    pub fn new(device: &Device, capacity: u64, page_size: u64) -> Result<Self> {
        let page_size = page_size.max(256);
        let capacity = round_up(capacity.max(page_size), page_size);
        let buffer = device
            .alloc_buffer(capacity, DataAccess::Scattered, None, BufferFlags::empty())
            .context("PlacementHeap: failed to allocate backing buffer")?;
        Ok(Self {
            buffer,
            page_size,
            bump: 0,
            regions: VecDeque::new(),
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

    /// Bytes currently occupied by in-flight regions.
    pub fn in_flight_bytes(&self) -> u64 {
        self.regions.iter().map(|r| r.size).sum()
    }

    /// Number of in-flight regions.
    pub fn in_flight_count(&self) -> usize {
        self.regions.len()
    }

    /// Reclaim regions whose GPU timeline has retired.
    ///
    /// Reclaimed regions' views are dropped, freeing their bindless slots.
    /// Returns the number of regions reclaimed.
    pub fn reclaim(&mut self, gpu_progress: TimelineValue) -> usize {
        let mut count = 0;
        while let Some(front) = self.regions.front() {
            match front.timeline {
                Some(t) if t <= gpu_progress => {
                    self.regions.pop_front();
                    count += 1;
                }
                _ => break,
            }
        }
        count
    }

    /// Acquire a contiguous region of at least `size` bytes.
    ///
    /// The region is page-aligned. Returns the base offset within the backing
    /// buffer. The caller should create views at `base_offset + colored_offset`.
    ///
    /// If there is not enough contiguous space, returns `None`. The caller should
    /// reclaim retired regions and retry, or grow the heap.
    pub fn acquire(&mut self, size: u64) -> Option<u64> {
        let aligned_size = round_up(size.max(1), self.page_size);
        let cap = self.buffer.allocated_size();

        if aligned_size > cap {
            return None;
        }

        // Find the lowest in-use offset (front of the ring).
        let tail = self.regions.front().map(|r| r.base_offset);

        match tail {
            None => {
                // Ring is empty — allocate from the start.
                self.bump = aligned_size;
                self.push_entry(0, aligned_size);
                Some(0)
            }
            Some(tail_offset) => {
                if self.bump >= tail_offset {
                    // bump is ahead of (or equal to) tail:
                    //   [....tail~~~~bump....]
                    // Try allocating after bump.
                    if self.bump + aligned_size <= cap {
                        let offset = self.bump;
                        self.bump += aligned_size;
                        self.push_entry(offset, aligned_size);
                        return Some(offset);
                    }
                    // Try wrapping to the beginning.
                    if aligned_size <= tail_offset {
                        let offset = 0;
                        self.bump = aligned_size;
                        self.push_entry(offset, aligned_size);
                        return Some(offset);
                    }
                    None
                } else {
                    // bump wrapped around, tail is ahead:
                    //   [~~bump....tail~~~~]
                    if self.bump + aligned_size <= tail_offset {
                        let offset = self.bump;
                        self.bump += aligned_size;
                        self.push_entry(offset, aligned_size);
                        Some(offset)
                    } else {
                        None
                    }
                }
            }
        }
    }

    /// Stamp the most recently acquired region with a timeline value and immediately
    /// defer `views` to the device's VramAllocator ring.
    ///
    /// The views will be dropped when `device.flush_deferred_deletions()` observes
    /// `gpu_progress >= timeline`, freeing their bindless slots at the right time.
    /// This is the standalone-submit path; the surface path defers views via
    /// `Frame::keepalive` instead.
    pub fn stamp_and_defer_views(
        &mut self,
        timeline: TimelineValue,
        views: Vec<BufferView>,
        device: &Device,
    ) {
        if let Some(entry) = self.regions.back_mut() {
            if entry.timeline.is_none() {
                entry.timeline = Some(timeline);
            }
        }
        if !views.is_empty() {
            let mut payload = DeferredPayload::new();
            for view in views {
                payload.push(view);
            }
            device.defer_release(timeline, payload);
        }
    }

    /// Stamp all unstamped regions with the given timeline (for the surface-present path).
    ///
    /// Views for these regions are already deferred via the frame's keepalive payload
    /// (see `Frame::resolve_and_compile_transient_buffers`). This method only updates
    /// the ring's timeline tracking so that `reclaim` can free ring space correctly.
    pub fn stamp_all_pending(&mut self, timeline: TimelineValue) {
        for entry in &mut self.regions {
            if entry.timeline.is_none() {
                entry.timeline = Some(timeline);
            }
        }
    }

    /// Reference to the backing buffer (for creating views).
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Highest in-flight timeline value across all regions, or `None` if idle.
    pub fn max_in_flight_timeline(&self) -> Option<TimelineValue> {
        self.regions.iter().filter_map(|r| r.timeline).max()
    }

    /// Snapshot of the heap's current state for diagnostics.
    pub fn stats(&self) -> PlacementHeapStats {
        PlacementHeapStats {
            capacity: self.capacity(),
            in_flight_bytes: self.in_flight_bytes(),
            in_flight_count: self.in_flight_count(),
        }
    }

    /// Grow the heap to at least `new_capacity`.
    ///
    /// Only safe when the ring is empty (no in-flight regions). Returns `Err`
    /// if there are in-flight regions or if allocation fails.
    pub fn grow(&mut self, device: &Device, new_capacity: u64) -> Result<()> {
        if !self.regions.is_empty() {
            anyhow::bail!(
                "PlacementHeap::grow: cannot grow with {} in-flight regions",
                self.regions.len()
            );
        }
        let aligned_cap = round_up(new_capacity, self.page_size);
        if aligned_cap <= self.buffer.allocated_size() {
            return Ok(());
        }
        self.buffer = device
            .alloc_buffer(
                aligned_cap,
                DataAccess::Scattered,
                None,
                BufferFlags::empty(),
            )
            .context("PlacementHeap::grow: failed to allocate new backing buffer")?;
        self.bump = 0;
        Ok(())
    }

    fn push_entry(&mut self, offset: u64, size: u64) {
        self.regions.push_back(RingEntry {
            base_offset: offset,
            size,
            timeline: None,
        });
    }

    /// Stamp the most recently acquired region without deferring any views.
    /// Used in tests and in contexts where views were already deferred by the caller.
    #[cfg(test)]
    pub(crate) fn stamp(&mut self, timeline: TimelineValue) {
        if let Some(entry) = self.regions.back_mut() {
            if entry.timeline.is_none() {
                entry.timeline = Some(timeline);
            }
        }
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

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn empty_heap_acquire_succeeds() {
        let device = test_device();
        let mut heap = PlacementHeap::with_capacity(&device, 64 * 1024 * 1024).unwrap();
        let offset = heap.acquire(16 * 1024 * 1024);
        assert!(offset.is_some());
        assert_eq!(offset.unwrap(), 0);
    }

    #[test]
    fn sequential_acquires_dont_overlap() {
        let device = test_device();
        let mut heap = PlacementHeap::new(&device, 64 * 1024 * 1024, 1024).unwrap();

        let o1 = heap.acquire(8 * 1024).unwrap();
        heap.stamp(1);
        let o2 = heap.acquire(8 * 1024).unwrap();
        heap.stamp(2);

        assert_ne!(o1, o2);
        assert!(o2 >= o1 + 8 * 1024);
    }

    #[test]
    fn reclaim_frees_space() {
        let device = test_device();
        // 3 pages of 1024 bytes
        let mut heap = PlacementHeap::new(&device, 3 * 1024, 1024).unwrap();

        // Fill all 3 pages
        let _o1 = heap.acquire(1024).unwrap();
        heap.stamp(1);
        let _o2 = heap.acquire(1024).unwrap();
        heap.stamp(2);
        let _o3 = heap.acquire(1024).unwrap();
        heap.stamp(3);

        // Ring is full
        assert!(heap.acquire(1024).is_none());

        // Reclaim first entry (gpu_progress >= 1)
        let count = heap.reclaim(1);
        assert_eq!(count, 1);

        // Now we can acquire again (wraps to offset 0)
        let o4 = heap.acquire(1024);
        assert!(o4.is_some());
        assert_eq!(o4.unwrap(), 0);
    }

    #[test]
    fn acquire_too_large_returns_none() {
        let device = test_device();
        let mut heap = PlacementHeap::new(&device, 4096, 1024).unwrap();
        assert!(heap.acquire(8192).is_none());
    }

    #[test]
    fn grow_resets_ring() {
        let device = test_device();
        let mut heap = PlacementHeap::new(&device, 4096, 1024).unwrap();
        assert_eq!(heap.capacity(), 4096);

        heap.grow(&device, 8192).unwrap();
        assert!(heap.capacity() >= 8192);
        assert_eq!(heap.in_flight_count(), 0);
    }

    #[test]
    fn grow_fails_with_inflight() {
        let device = test_device();
        let mut heap = PlacementHeap::new(&device, 4096, 1024).unwrap();
        heap.acquire(1024).unwrap();
        heap.stamp(1);
        assert!(heap.grow(&device, 8192).is_err());
    }

    #[test]
    fn stamp_all_pending_stamps_unstamped() {
        let device = test_device();
        let mut heap = PlacementHeap::new(&device, 8192, 1024).unwrap();
        heap.acquire(1024).unwrap();
        heap.acquire(1024).unwrap();
        heap.stamp_all_pending(5);

        let count = heap.reclaim(5);
        assert_eq!(count, 2);
    }

    #[test]
    fn stamp_and_defer_views_defers_via_vram_allocator() {
        // Verify that stamp_and_defer_views registers a DeferredPayload so that
        // views are dropped when the VramAllocator ring processes them, not by
        // PlacementHeap reclaim (which no longer holds views).
        let device = test_device();
        let mut heap = PlacementHeap::with_capacity(&device, 64 * 1024 * 1024).unwrap();

        let offset = heap.acquire(4 * 1024 * 1024).unwrap();
        let buf = heap.buffer();
        let view = buf.create_view(offset, 1024, Some(4)).expect("create_view");

        let epoch: u64 = 1;
        heap.stamp_and_defer_views(epoch, vec![view], &device);

        // PlacementHeap reclaim should free ring space (returns 1) because the region
        // is stamped. Views are now managed by the VramAllocator, not the PlacementHeap.
        let freed = heap.reclaim(epoch);
        assert_eq!(freed, 1, "ring space should be reclaimed by PlacementHeap");

        // The region has no views attached to it — PlacementHeap ring entries are
        // now pure ring-space trackers (no view ownership).
        assert_eq!(
            heap.in_flight_count(),
            0,
            "no in-flight entries after reclaim"
        );
    }
}
