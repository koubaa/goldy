//! Unified GPU memory allocation interface.
//!
//! Goldy has three independent GPU memory allocation paths:
//!
//! 1. **Transient sub-allocations** — [`TransientAllocator`] → [`BufferPool`] → [`Buffer::new`].
//!    Pluggable recycling policy via the [`TransientAllocator`] trait, but no control over
//!    *where* memory comes from.
//! 2. **Standalone named buffers** — consumers call [`Buffer::new`] directly for bump readback,
//!    staging, indirect dispatch, etc.
//! 3. **Textures** — [`TexturePool`] → [`Texture::new`]. No interception point.
//!
//! [`VramAllocator`] sits **below** all three pooling systems, providing a single customization
//! point for *where* GPU memory comes from. This enables:
//!
//! - **Unified memory control** — alias transient, standalone, and texture allocations into one
//!   address space or placement heap.
//! - **Backend-native strategies** — Metal `makeAliasable` placement heaps, Vulkan sparse
//!   binding, DX12 tiled resources as first-class allocator implementations.
//! - **Budgeting / telemetry** — VRAM caps, fragmentation monitoring, eviction policies.
//! - **Defragmentation** — move allocations and update bindless descriptors atomically.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────┐
//! │  Consumers (ekrano, user code)                 │
//! │  ┌─────────────┐  ┌──────────┐  ┌───────────┐ │
//! │  │ TransientAlloc│ │ Buffer:: │  │ Texture:: │ │
//! │  │ (recycling)  │  │ new()    │  │ new()     │ │
//! │  └──────┬───────┘  └────┬─────┘  └─────┬─────┘ │
//! │         │               │              │       │
//! │  ┌──────▼───────────────▼──────────────▼──────┐│
//! │  │           VramAllocator trait               ││
//! │  │  alloc_buffer / alloc_texture / free / ...  ││
//! │  └──────────────────┬──────────────────────────┘│
//! │                     │                           │
//! │  ┌──────────────────▼──────────────────────────┐│
//! │  │           GpuBackend (Metal/Vulkan/DX12)    ││
//! │  └─────────────────────────────────────────────┘│
//! └────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! The [`Device`] holds an [`Arc<dyn VramAllocator>`]. Call
//! [`Device::with_vram_allocator`] before creating any GPU resources to install
//! a custom allocator. The default ([`DefaultVramAllocator`]) delegates directly
//! to the backend with zero overhead.
//!
//! [`TransientAllocator`]: crate::transient_allocator::TransientAllocator
//! [`BufferPool`]: crate::buffer::BufferPool
//! [`Buffer::new`]: crate::buffer::Buffer::new
//! [`Texture::new`]: crate::texture::Texture::new
//! [`TexturePool`]: crate::texture_pool::TexturePool
//! [`Device`]: crate::device::Device
//! [`Device::with_vram_allocator`]: crate::device::Device::with_vram_allocator
//! [`VramAllocator`]: crate::vram_allocator::VramAllocator
//! [`DefaultVramAllocator`]: crate::vram_allocator::DefaultVramAllocator

use crate::buffer::Buffer;
use crate::device::Device;
use crate::texture::Texture;
use crate::timeline::TimelineValue;
use crate::types::*;
use anyhow::Result;
use std::any::Any;
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

// The GPU epoch that is currently being reclaimed on this thread by `boundary_crossed`.
//
// Set before dropping `DeferredPayload` entries so that `Buffer::drop` (→ the Metal
// `destroy_buffer` path) can use the reclamation epoch as the deletion-queue barrier
// instead of the conservative `timeline_scheduled_max`.  This allows the Metal heap
// allocator to free those buffers on the very next `process_deletion_queue_up_to_signaled`
// call, since `signaled_value >= reclamation_epoch` is already true by definition.
//
// `None` means we are NOT in a reclamation context; normal deletion semantics apply.
thread_local! {
    pub static RECLAMATION_EPOCH: Cell<Option<u64>> = const { Cell::new(None) };
}

// -----------------------------------------------------------------------
// DeferredPayload
// -----------------------------------------------------------------------

/// A type-erased bundle of GPU resources to be held alive until a GPU timeline epoch retires.
///
/// Passed to [`VramAllocator::defer_release`] to register resources for deferred dropping.
/// The allocator holds the payload until [`VramAllocator::reclaim`] determines that the
/// associated epoch has been reached, then drops all resources in the payload.
///
/// # Example
///
/// ```no_run
/// # use goldy::vram_allocator::DeferredPayload;
/// # use goldy::buffer::Buffer;
/// # fn example(buf: Buffer, view: goldy::buffer::BufferView) {
/// let mut payload = DeferredPayload::new();
/// payload.push(buf).push(view);
/// # }
/// ```
pub struct DeferredPayload(pub(crate) Vec<Box<dyn Any + Send>>);

impl DeferredPayload {
    /// Create an empty payload.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Add a resource to this payload. Returns `&mut Self` for chaining.
    pub fn push<T: Send + 'static>(&mut self, resource: T) -> &mut Self {
        self.0.push(Box::new(resource));
        self
    }

    /// Returns `true` if no resources have been added.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of resources in this payload.
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Default for DeferredPayload {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------
// Trait
// -----------------------------------------------------------------------

/// A pluggable strategy for allocating GPU memory (buffers and textures).
///
/// Implementations intercept every buffer and texture allocation, enabling unified
/// memory budgets, placement-heap strategies, and telemetry without changing call sites.
///
/// Methods take `&self` and must be internally synchronized (the trait is `Send + Sync`).
/// Use [`AtomicI64`] / [`AtomicU64`](std::sync::atomic::AtomicU64) for lock-free counters,
/// or a `Mutex` for more complex state.
pub trait VramAllocator: Send + Sync {
    /// Allocate a GPU buffer.
    ///
    /// The default implementation calls [`Buffer::new_with_stride_and_flags`] directly.
    /// Custom implementations may allocate from a placement heap, enforce a budget, or
    /// track the allocation for telemetry.
    fn alloc_buffer(
        &self,
        device: &Device,
        size: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<Buffer> {
        Buffer::new_with_stride_and_flags(device, size, access, element_stride, flags)
    }

    /// Allocate a GPU buffer with a pre-reserved capacity hint.
    ///
    /// The default implementation calls [`Buffer::new_with_capacity_hint_and_flags`].
    fn alloc_buffer_with_capacity(
        &self,
        device: &Device,
        initial_size: u64,
        expected_max: u64,
        access: DataAccess,
        flags: BufferFlags,
    ) -> Result<Buffer> {
        Buffer::new_with_capacity_hint_and_flags(device, initial_size, expected_max, access, flags)
    }

    /// Allocate a GPU texture.
    ///
    /// The default implementation calls [`Texture::new`] directly.
    fn alloc_texture(
        &self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: SpatialAccess,
        flags: TextureFlags,
    ) -> Result<Texture> {
        Texture::new(device, width, height, format, access, flags)
    }

    /// Notify the allocator that a buffer has been freed.
    ///
    /// Called automatically by [`Buffer::drop`] when the allocator is installed on the
    /// device. Implementations should decrement their tracked byte counts here.
    /// The `size` is the buffer's allocated size at the time of destruction.
    fn notify_buffer_freed(&self, _size: u64) {}

    /// Notify the allocator that a buffer was resized beyond its previous GPU allocation.
    ///
    /// Called by [`Buffer::resize_to`] / [`Buffer::resize_to_uninitialized`] when the
    /// backend grows the allocation in place (`new_size > self.allocated_size`).
    /// `old_allocated` and `new_allocated` are the backend `buffer_capacity()` values
    /// before and after the resize. Implementations should adjust their tracked byte
    /// counts by the delta.
    fn notify_buffer_resized(&self, _old_allocated: u64, _new_allocated: u64) {}

    /// Notify the allocator that a texture has been freed.
    ///
    /// Called automatically by [`Texture::drop`] when the allocator is installed on the
    /// device. The `byte_size` is [`Texture::byte_size`] at the time of destruction.
    fn notify_texture_freed(&self, _byte_size: usize) {}

    /// Net bytes allocated by this allocator (allocations minus frees).
    ///
    /// Returns 0 if the implementation does not track allocations.
    fn allocated_bytes(&self) -> u64 {
        0
    }

    /// Optional byte budget. Returns `None` if no budget is enforced.
    /// When set, [`alloc_buffer`](Self::alloc_buffer) and
    /// [`alloc_texture`](Self::alloc_texture) should return an error if
    /// the allocation would exceed the budget.
    fn budget(&self) -> Option<u64> {
        None
    }

    /// Strategy identifier for diagnostics and tracing.
    fn name(&self) -> &'static str;

    /// Register `payload` for deferred dropping after GPU timeline `epoch` retires.
    ///
    /// The allocator holds all resources in the payload alive until
    /// [`boundary_crossed`](Self::boundary_crossed) is called with an epoch >= the
    /// registered value, at which point the payload is dropped. Entries are expected to
    /// arrive roughly in epoch order.
    ///
    /// Custom allocators that manage their own memory (e.g. PTX slab allocators) should
    /// override this to integrate with their internal reclamation pipeline.
    ///
    /// The default implementation is a no-op: payloads are dropped immediately.
    fn defer_release(&self, _epoch: TimelineValue, _payload: DeferredPayload) {}

    /// Notify the allocator that the GPU has crossed dispatch boundary `epoch`.
    ///
    /// All deferred payloads registered at epochs `<= epoch` are immediately dropped.
    /// Returns the number of entries reclaimed.
    ///
    /// This is the event-driven reclamation entry point. It is called:
    /// - From the Metal completion handler (asynchronously, when a command buffer finishes)
    /// - From the host after `wait_until(epoch)` returns (synchronous safety net)
    /// - From DX12/Vulkan backend flush/wait paths with the current fence/semaphore value
    ///
    /// The call is idempotent: invoking it multiple times for the same epoch is safe.
    ///
    /// The default implementation is a no-op and returns 0.
    fn boundary_crossed(&self, _epoch: TimelineValue) -> usize {
        0
    }

    /// Peek at the oldest deferred epoch without freeing anything.
    ///
    /// Returns `None` if the deferred ring is empty. Used by the heap-pressure
    /// reclamation path to determine which epoch to wait on before retrying
    /// an allocation that failed due to heap exhaustion.
    fn peek_oldest_deferred_epoch(&self) -> Option<TimelineValue> {
        None
    }

    /// Drop all deferred payloads unconditionally, regardless of their epoch.
    ///
    /// Called by the device on shutdown, after waiting for the high-water timeline,
    /// to ensure nothing leaks across `destroy_device`. Also callable by consumers
    /// that know the GPU is idle and want to eagerly reclaim memory.
    ///
    /// The default implementation is a no-op.
    fn drain(&self) {}
}

// -----------------------------------------------------------------------
// Default implementation
// -----------------------------------------------------------------------

/// The default allocator: delegates directly to [`Buffer::new`] / [`Texture::new`]
/// with no tracking, budgeting, or overhead.
///
/// Installed automatically when a [`Device`] is created. Implements the full
/// deferred-release ring: [`VramAllocator::defer_release`], [`VramAllocator::boundary_crossed`],
/// and [`VramAllocator::drain`].
pub struct DefaultVramAllocator {
    deferred: Mutex<VecDeque<(TimelineValue, DeferredPayload)>>,
}

impl DefaultVramAllocator {
    /// Create a new default allocator.
    pub fn new() -> Self {
        Self {
            deferred: Mutex::new(VecDeque::new()),
        }
    }
}

impl Default for DefaultVramAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl VramAllocator for DefaultVramAllocator {
    fn name(&self) -> &'static str {
        "default"
    }

    fn defer_release(&self, epoch: TimelineValue, payload: DeferredPayload) {
        if payload.is_empty() {
            return;
        }
        self.deferred.lock().unwrap().push_back((epoch, payload));
    }

    fn boundary_crossed(&self, epoch: TimelineValue) -> usize {
        // Drain payloads whose epoch has completed.
        let drained: Vec<(TimelineValue, DeferredPayload)> = {
            let mut ring = self.deferred.lock().unwrap();
            let mut drained = Vec::new();
            while let Some((entry_epoch, _)) = ring.front() {
                if *entry_epoch <= epoch {
                    drained.push(ring.pop_front().unwrap());
                } else {
                    break;
                }
            }
            drained
        };
        let count = drained.len();
        // Drop payloads with RECLAMATION_EPOCH set so that backend destroy paths (e.g.
        // Metal's destroy_buffer) can use the epoch as a lower deletion-queue barrier.
        // This allows freed Metal heap allocations to become available on the very next
        // process_deletion_queue_up_to_signaled call rather than waiting for
        // timeline_scheduled_max to be GPU-signaled.
        RECLAMATION_EPOCH.with(|e| e.set(Some(epoch)));
        drop(drained); // payloads dropped here; Buffer::drop fires for each Buffer inside
        RECLAMATION_EPOCH.with(|e| e.set(None));
        count
    }

    fn drain(&self) {
        self.deferred.lock().unwrap().clear();
    }

    fn peek_oldest_deferred_epoch(&self) -> Option<TimelineValue> {
        self.deferred.lock().unwrap().front().map(|(epoch, _)| *epoch)
    }
}

// -----------------------------------------------------------------------
// Tracking allocator
// -----------------------------------------------------------------------

/// A `VramAllocator` that wraps another allocator and tracks total allocated bytes.
///
/// Optionally enforces a byte budget: allocations that would push the total above
/// the budget return an error instead of proceeding.
///
/// # Example
///
/// ```no_run
/// # use goldy::vram_allocator::{TrackingVramAllocator, DefaultVramAllocator};
/// # use std::sync::Arc;
/// // Track all allocations with a 512 MB budget:
/// let allocator = TrackingVramAllocator::with_budget(
///     Arc::new(DefaultVramAllocator::new()),
///     512 * 1024 * 1024,
/// );
/// ```
pub struct TrackingVramAllocator {
    inner: Arc<dyn VramAllocator>,
    /// Signed to handle potential over-decrement from mismatched free notifications
    /// without panicking. Steady-state value is non-negative.
    live_bytes: AtomicI64,
    budget_bytes: Option<u64>,
}

impl TrackingVramAllocator {
    /// Wrap `inner` with byte-level tracking but no budget.
    pub fn new(inner: Arc<dyn VramAllocator>) -> Self {
        Self {
            inner,
            live_bytes: AtomicI64::new(0),
            budget_bytes: None,
        }
    }

    /// Wrap `inner` with tracking and a byte budget.
    pub fn with_budget(inner: Arc<dyn VramAllocator>, budget_bytes: u64) -> Self {
        Self {
            inner,
            live_bytes: AtomicI64::new(0),
            budget_bytes: Some(budget_bytes),
        }
    }

    fn check_budget(&self, additional: u64) -> Result<()> {
        if let Some(cap) = self.budget_bytes {
            let current = self.live_bytes.load(Ordering::Relaxed) as u64;
            if current.saturating_add(additional) > cap {
                anyhow::bail!(
                    "VramAllocator budget exceeded: {current} + {additional} > {cap} \
                     (allocator={}, budget={})",
                    self.inner.name(),
                    bytesize(cap),
                );
            }
        }
        Ok(())
    }
}

impl VramAllocator for TrackingVramAllocator {
    fn alloc_buffer(
        &self,
        device: &Device,
        size: u64,
        access: DataAccess,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<Buffer> {
        self.check_budget(size)?;
        let buf = self
            .inner
            .alloc_buffer(device, size, access, element_stride, flags)?;
        self.live_bytes
            .fetch_add(buf.allocated_size() as i64, Ordering::Relaxed);
        Ok(buf)
    }

    fn alloc_buffer_with_capacity(
        &self,
        device: &Device,
        initial_size: u64,
        expected_max: u64,
        access: DataAccess,
        flags: BufferFlags,
    ) -> Result<Buffer> {
        self.check_budget(expected_max.max(initial_size))?;
        let buf = self.inner.alloc_buffer_with_capacity(
            device,
            initial_size,
            expected_max,
            access,
            flags,
        )?;
        self.live_bytes
            .fetch_add(buf.allocated_size() as i64, Ordering::Relaxed);
        Ok(buf)
    }

    fn alloc_texture(
        &self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: SpatialAccess,
        flags: TextureFlags,
    ) -> Result<Texture> {
        let estimated = (width as u64) * (height as u64) * (format.bytes_per_pixel() as u64);
        self.check_budget(estimated)?;
        let tex = self
            .inner
            .alloc_texture(device, width, height, format, access, flags)?;
        self.live_bytes
            .fetch_add(tex.byte_size() as i64, Ordering::Relaxed);
        Ok(tex)
    }

    fn notify_buffer_freed(&self, size: u64) {
        self.live_bytes.fetch_sub(size as i64, Ordering::Relaxed);
        self.inner.notify_buffer_freed(size);
    }

    fn notify_texture_freed(&self, byte_size: usize) {
        self.live_bytes
            .fetch_sub(byte_size as i64, Ordering::Relaxed);
        self.inner.notify_texture_freed(byte_size);
    }

    fn notify_buffer_resized(&self, old_allocated: u64, new_allocated: u64) {
        self.live_bytes.fetch_add(
            (new_allocated as i64).wrapping_sub(old_allocated as i64),
            Ordering::Relaxed,
        );
        self.inner.notify_buffer_resized(old_allocated, new_allocated);
    }

    fn allocated_bytes(&self) -> u64 {
        self.live_bytes.load(Ordering::Relaxed).max(0) as u64
    }

    fn budget(&self) -> Option<u64> {
        self.budget_bytes
    }

    fn name(&self) -> &'static str {
        "tracking"
    }

    fn defer_release(&self, epoch: TimelineValue, payload: DeferredPayload) {
        self.inner.defer_release(epoch, payload);
    }

    fn boundary_crossed(&self, epoch: TimelineValue) -> usize {
        self.inner.boundary_crossed(epoch)
    }

    fn drain(&self) {
        self.inner.drain();
    }

    fn peek_oldest_deferred_epoch(&self) -> Option<TimelineValue> {
        self.inner.peek_oldest_deferred_epoch()
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn bytesize(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn default_allocator_creates_buffer() {
        let device = test_device();
        let alloc = DefaultVramAllocator::new();
        let buf = alloc
            .alloc_buffer(
                &device,
                1024,
                DataAccess::Scattered,
                None,
                BufferFlags::empty(),
            )
            .unwrap();
        assert_eq!(buf.size(), 1024);
    }

    #[test]
    fn default_allocator_creates_texture() {
        let device = test_device();
        let alloc = DefaultVramAllocator::new();
        let tex = alloc
            .alloc_texture(
                &device,
                64,
                64,
                TextureFormat::Rgba8Unorm,
                SpatialAccess::Interpolated,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .unwrap();
        assert_eq!(tex.width(), 64);
        assert_eq!(tex.height(), 64);
    }

    #[test]
    fn tracking_allocator_tracks_bytes() {
        let alloc = Arc::new(TrackingVramAllocator::new(Arc::new(DefaultVramAllocator::new())));
        let base = test_device();
        let device = base.with_vram_allocator(Arc::clone(&alloc) as Arc<dyn VramAllocator>);

        assert_eq!(alloc.allocated_bytes(), 0);

        let buf = device
            .alloc_buffer(4096, DataAccess::Scattered, None, BufferFlags::empty())
            .unwrap();
        assert!(alloc.allocated_bytes() >= 4096);

        drop(buf);
        assert_eq!(alloc.allocated_bytes(), 0);
    }

    #[test]
    fn tracking_allocator_budget_enforcement() {
        let device = test_device();
        let alloc = TrackingVramAllocator::with_budget(Arc::new(DefaultVramAllocator::new()), 8192);

        let _buf = alloc
            .alloc_buffer(
                &device,
                4096,
                DataAccess::Scattered,
                None,
                BufferFlags::empty(),
            )
            .unwrap();

        let result = alloc.alloc_buffer(
            &device,
            8192,
            DataAccess::Scattered,
            None,
            BufferFlags::empty(),
        );
        assert!(result.is_err(), "should fail when over budget");
    }

    #[test]
    fn tracking_allocator_texture_tracking() {
        let alloc = Arc::new(TrackingVramAllocator::new(Arc::new(DefaultVramAllocator::new())));
        let base = test_device();
        let device = base.with_vram_allocator(Arc::clone(&alloc) as Arc<dyn VramAllocator>);

        let tex = device
            .alloc_texture(
                32,
                32,
                TextureFormat::Rgba8Unorm,
                SpatialAccess::Interpolated,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .unwrap();
        assert!(alloc.allocated_bytes() > 0);

        drop(tex);
        assert_eq!(alloc.allocated_bytes(), 0);
    }

    #[test]
    fn bytesize_formatting() {
        assert_eq!(bytesize(500), "500 B");
        assert_eq!(bytesize(1024), "1.0 KiB");
        assert_eq!(bytesize(1024 * 1024), "1.0 MiB");
        assert_eq!(bytesize(1024 * 1024 * 1024), "1.0 GiB");
    }

    // -----------------------------------------------------------------------
    // DeferredPayload tests
    // -----------------------------------------------------------------------

    #[test]
    fn deferred_payload_push_and_len() {
        let mut p = DeferredPayload::new();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        p.push(42u32).push("hello");
        assert!(!p.is_empty());
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn deferred_payload_default_is_empty() {
        let p = DeferredPayload::default();
        assert!(p.is_empty());
    }

    // -----------------------------------------------------------------------
    // DefaultVramAllocator deferred ring tests
    // -----------------------------------------------------------------------

    #[test]
    fn default_allocator_reclaim_drops_retired_entries() {
        let alloc = DefaultVramAllocator::new();

        // Track drops via an Arc.
        let alive = Arc::new(());
        let weak = Arc::downgrade(&alive);

        let mut p = DeferredPayload::new();
        p.push(alive);
        alloc.defer_release(5, p);

        // epoch=5, boundary=4 — should NOT reclaim yet.
        assert_eq!(alloc.boundary_crossed(4), 0);
        assert!(weak.upgrade().is_some(), "resource should still be alive");

        // boundary=5 — should reclaim and drop.
        assert_eq!(alloc.boundary_crossed(5), 1);
        assert!(
            weak.upgrade().is_none(),
            "resource should have been dropped"
        );
    }

    #[test]
    fn default_allocator_reclaim_preserves_future_entries() {
        let alloc = DefaultVramAllocator::new();

        let alive_early = Arc::new(1u32);
        let weak_early = Arc::downgrade(&alive_early);
        let alive_late = Arc::new(2u32);
        let weak_late = Arc::downgrade(&alive_late);

        let mut p1 = DeferredPayload::new();
        p1.push(alive_early);
        alloc.defer_release(2, p1);

        let mut p2 = DeferredPayload::new();
        p2.push(alive_late);
        alloc.defer_release(10, p2);

        // Reclaim only up to epoch 2.
        assert_eq!(alloc.boundary_crossed(2), 1);
        assert!(weak_early.upgrade().is_none(), "epoch=2 should be dropped");
        assert!(weak_late.upgrade().is_some(), "epoch=10 should survive");

        // Reclaim the rest.
        assert_eq!(alloc.boundary_crossed(10), 1);
        assert!(
            weak_late.upgrade().is_none(),
            "epoch=10 should now be dropped"
        );
    }

    #[test]
    fn default_allocator_drain_drops_all() {
        let alloc = DefaultVramAllocator::new();

        let alive = Arc::new(99u32);
        let weak = Arc::downgrade(&alive);

        let mut p = DeferredPayload::new();
        p.push(alive);
        alloc.defer_release(9999, p);

        // drain() should drop everything regardless of epoch.
        alloc.drain();
        assert!(weak.upgrade().is_none(), "drain should drop all resources");
    }

    #[test]
    fn default_allocator_empty_payload_skipped() {
        let alloc = DefaultVramAllocator::new();
        // Deferring an empty payload should not add an entry.
        alloc.defer_release(1, DeferredPayload::new());
        // boundary_crossed returns 0 (nothing was added).
        assert_eq!(alloc.boundary_crossed(100), 0);
    }

    // -----------------------------------------------------------------------
    // TrackingVramAllocator deferred delegation tests
    // -----------------------------------------------------------------------

    #[test]
    fn tracking_allocator_delegates_defer_and_reclaim() {
        let inner = Arc::new(DefaultVramAllocator::new());
        let tracking = TrackingVramAllocator::new(inner.clone());

        let alive = Arc::new(7u32);
        let weak = Arc::downgrade(&alive);

        let mut p = DeferredPayload::new();
        p.push(alive);
        tracking.defer_release(3, p);

        // Not yet retired.
        assert_eq!(tracking.boundary_crossed(2), 0);
        assert!(weak.upgrade().is_some());

        // Retired.
        assert_eq!(tracking.boundary_crossed(3), 1);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn tracking_allocator_delegates_drain() {
        let inner = Arc::new(DefaultVramAllocator::new());
        let tracking = TrackingVramAllocator::new(inner.clone());

        let alive = Arc::new(8u32);
        let weak = Arc::downgrade(&alive);

        let mut p = DeferredPayload::new();
        p.push(alive);
        tracking.defer_release(9999, p);

        tracking.drain();
        assert!(weak.upgrade().is_none());
    }

    // -----------------------------------------------------------------------
    // Phase 0 end-to-end routing test
    // -----------------------------------------------------------------------

    #[test]
    fn tracking_routes_through_transient_allocator() {
        use crate::transient_allocator::{TransientAllocatorConfig, TransientAllocatorStrategy};

        let tracking = Arc::new(TrackingVramAllocator::new(
            Arc::new(DefaultVramAllocator::new()) as Arc<dyn VramAllocator>,
        ));
        let base = test_device();
        let device =
            base.with_vram_allocator(Arc::clone(&tracking) as Arc<dyn VramAllocator>);

        let mut ta = TransientAllocatorStrategy::EpochRegions
            .create(&device, TransientAllocatorConfig::default())
            .unwrap();
        ta.begin_frame(&device, 65536).unwrap();
        let _view = ta.alloc(&device, 4096, None).unwrap();
        let bytes_after_alloc = tracking.allocated_bytes();
        assert!(
            bytes_after_alloc >= 4096,
            "transient alloc must be tracked; got {bytes_after_alloc}"
        );

        ta.end_frame(&device, 0);
        device.flush_deferred_deletions();
        drop(ta);

        assert_eq!(
            tracking.allocated_bytes(),
            0,
            "all transient bytes must be freed after epoch retirement"
        );
    }
}
