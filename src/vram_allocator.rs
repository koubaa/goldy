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
use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

// -----------------------------------------------------------------------
// DeferredPayload
// -----------------------------------------------------------------------

/// A type-erased bundle of GPU resources to be held alive until a GPU timeline epoch retires.
///
/// Passed to [`VramAllocator::defer_release`] to register resources for deferred dropping.
/// The allocator holds the payload until [`VramAllocator::boundary_crossed`] determines that the
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
// ParcelKind
// -----------------------------------------------------------------------

/// Zoning / telemetry label for a freed parcel (buffer-kind vs texture-kind).
///
/// Not used for separate accounting code paths — only passed to [`VramAllocator::notify_freed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParcelKind {
    Buffer,
    Texture,
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

    /// Notify the allocator that a deed-holding parcel has been freed.
    ///
    /// Called automatically from [`Buffer::drop`] / [`Texture::drop`] when the parcel
    /// was allocated through the device's [`VramAllocator`] (and carries a deed).
    /// Borrowing sub-parcels (e.g. [`crate::buffer::BufferView`]) never call this.
    ///
    /// `reserved` is the parcel's reserved backing size; `committed` is the runtime's
    /// handed-out estimate (logical size for buffers, [`Texture::byte_size`] for textures).
    fn notify_freed(&self, _reserved: u64, _committed: u64, _kind: ParcelKind) {}

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
    /// The allocator holds all resources in the payload alive until a subsequent call to
    /// [`boundary_crossed`](Self::boundary_crossed) observes `gpu_progress >= epoch`, at which point the
    /// payload is dropped. Entries are expected to arrive roughly in epoch order; calling
    /// [`boundary_crossed`](Self::boundary_crossed) drains from the front.
    ///
    /// Custom allocators that manage their own memory (e.g. PTX slab allocators) should
    /// override this to integrate with their internal reclamation pipeline.
    ///
    /// The default implementation is a no-op: payloads are dropped immediately.
    fn defer_release(&self, _epoch: TimelineValue, _payload: DeferredPayload) {}

    /// Reclaim all deferred payloads whose epoch is `<= gpu_progress`, dropping them.
    ///
    /// Returns the number of entries reclaimed. Typically called from
    /// [`Device::flush_deferred_deletions`](crate::device::Device::flush_deferred_deletions)
    /// at frame boundaries.
    ///
    /// The default implementation is a no-op and returns 0.
    fn boundary_crossed(&self, _gpu_progress: TimelineValue) -> usize {
        0
    }

    /// Returns `true` if there are deferred payloads waiting for GPU retirement.
    ///
    /// Callers that call `flush_deferred_deletions` and find that it reclaimed nothing
    /// can use this to decide whether a GPU wait would be productive.
    ///
    /// The default implementation always returns `false`.
    fn has_deferred_payloads(&self) -> bool {
        false
    }

    /// The oldest epoch currently in the deferred ring, if any.
    ///
    /// If non-`None`, waiting for this timeline value then calling
    /// [`boundary_crossed`](Self::boundary_crossed) would free the oldest batch of deferred resources.
    ///
    /// The default implementation returns `None`.
    fn oldest_deferred_epoch(&self) -> Option<TimelineValue> {
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

    fn boundary_crossed(&self, gpu_progress: TimelineValue) -> usize {
        let drained: Vec<(TimelineValue, DeferredPayload)> = {
            let mut ring = self.deferred.lock().unwrap();
            let mut drained = Vec::new();
            while let Some((epoch, _)) = ring.front() {
                if *epoch <= gpu_progress {
                    drained.push(ring.pop_front().unwrap());
                } else {
                    break;
                }
            }
            drained
        };
        let count = drained.len();
        drop(drained);
        count
    }

    fn has_deferred_payloads(&self) -> bool {
        !self.deferred.lock().unwrap().is_empty()
    }

    fn oldest_deferred_epoch(&self) -> Option<TimelineValue> {
        self.deferred
            .lock()
            .unwrap()
            .front()
            .map(|(epoch, _)| *epoch)
    }

    fn drain(&self) {
        self.deferred.lock().unwrap().clear();
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

    fn notify_freed(&self, reserved: u64, committed: u64, kind: ParcelKind) {
        self.live_bytes
            .fetch_sub(reserved as i64, Ordering::Relaxed);
        self.inner.notify_freed(reserved, committed, kind);
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

    fn boundary_crossed(&self, gpu_progress: TimelineValue) -> usize {
        self.inner.boundary_crossed(gpu_progress)
    }

    fn has_deferred_payloads(&self) -> bool {
        self.inner.has_deferred_payloads()
    }

    fn oldest_deferred_epoch(&self) -> Option<TimelineValue> {
        self.inner.oldest_deferred_epoch()
    }

    fn drain(&self) {
        self.inner.drain();
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

    /// Returns `(base, device, tracking)`. Keep `base` alive for the test body so the
    /// mock backend device handle remains valid on the cloned `device` handle.
    fn device_with_tracking() -> (Device, Device, Arc<TrackingVramAllocator>) {
        let base = test_device();
        let tracking = Arc::new(TrackingVramAllocator::new(Arc::new(
            DefaultVramAllocator::new(),
        )));
        let device = base.with_vram_allocator(tracking.clone());
        (base, device, tracking)
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
        let (_base, device, tracking) = device_with_tracking();

        assert_eq!(tracking.allocated_bytes(), 0);

        let buf = device
            .alloc_buffer(4096, DataAccess::Scattered, None, BufferFlags::empty())
            .unwrap();
        assert!(tracking.allocated_bytes() >= 4096);

        drop(buf);
        assert_eq!(tracking.allocated_bytes(), 0);
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
        let (_base, device, tracking) = device_with_tracking();

        let tex = device
            .alloc_texture(
                32,
                32,
                TextureFormat::Rgba8Unorm,
                SpatialAccess::Interpolated,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .unwrap();
        assert!(tracking.allocated_bytes() > 0);

        drop(tex);
        assert_eq!(tracking.allocated_bytes(), 0);
    }

    #[test]
    fn accounting_round_trip_mixed_parcels() {
        let (_base, device, tracking) = device_with_tracking();
        const N: usize = 8;

        for _ in 0..N {
            assert_eq!(tracking.allocated_bytes(), 0);

            let buf = device
                .alloc_buffer(1024, DataAccess::Scattered, None, BufferFlags::empty())
                .unwrap();
            let hinted = device
                .alloc_buffer_with_capacity(512, 4096, DataAccess::Scattered, BufferFlags::empty())
                .unwrap();
            let tex = device
                .alloc_texture(
                    16,
                    16,
                    TextureFormat::Rgba8Unorm,
                    SpatialAccess::Interpolated,
                    TextureFlags::COPY_DST,
                )
                .unwrap();

            let bytes_with_parcels = tracking.allocated_bytes();
            assert!(bytes_with_parcels > 0);

            let view = buf.create_view(0, 256, Some(4)).unwrap();
            let bytes_before_view_drop = tracking.allocated_bytes();
            drop(view);
            assert_eq!(
                tracking.allocated_bytes(),
                bytes_before_view_drop,
                "BufferView drop must not change allocator accounting"
            );

            drop(buf);
            drop(hinted);
            drop(tex);
            assert_eq!(tracking.allocated_bytes(), 0);
        }
    }

    #[test]
    fn deed_survives_with_vram_allocator_clone() {
        let base = test_device();
        let tracking = Arc::new(TrackingVramAllocator::new(Arc::new(
            DefaultVramAllocator::new(),
        )));
        let device = base.with_vram_allocator(tracking.clone());

        let buf = device
            .alloc_buffer(2048, DataAccess::Scattered, None, BufferFlags::empty())
            .unwrap();
        assert!(tracking.allocated_bytes() >= 2048);
        drop(buf);
        assert_eq!(tracking.allocated_bytes(), 0);
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
    fn default_allocator_boundary_crossed_drops_retired_entries() {
        let alloc = DefaultVramAllocator::new();

        // Track drops via an Arc.
        let alive = Arc::new(());
        let weak = Arc::downgrade(&alive);

        let mut p = DeferredPayload::new();
        p.push(alive);
        alloc.defer_release(5, p);

        // epoch=5, gpu_progress=4 — should NOT reclaim yet.
        assert_eq!(alloc.boundary_crossed(4), 0);
        assert!(weak.upgrade().is_some(), "resource should still be alive");

        // gpu_progress=5 — should reclaim and drop.
        assert_eq!(alloc.boundary_crossed(5), 1);
        assert!(
            weak.upgrade().is_none(),
            "resource should have been dropped"
        );
    }

    #[test]
    fn default_allocator_boundary_crossed_preserves_future_entries() {
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
    fn tracking_allocator_delegates_defer_and_boundary_crossed() {
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
}
