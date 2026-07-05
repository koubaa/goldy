//! Unified GPU memory allocation interface.
//!
//! [`VramAllocator`] is the single customization point for *where* GPU memory comes from.
//! It sits below Goldy's pooling layers; every allocation that goes through a
//! `Device::alloc_buffer` / `Device::alloc_buffer_with_capacity` / `Device::alloc_texture`
//! method passes through the installed allocator:
//!
//! - **Transient backing** — [`TransientAllocator`] → [`BufferPool`] → `Device::alloc_buffer`
//! - **Standalone named buffers** — `Device::alloc_buffer` / `Device::alloc_buffer_with_capacity`
//! - **Textures** — [`TexturePool`] → `Device::alloc_texture`
//!
//! **Accounting deed.** Each resource returned from a `Device::alloc_*` call carries a deed —
//! a `Weak` back-reference to the allocator. When the backing buffer or texture is dropped,
//! [`VramAllocator::notify_freed`] is called automatically so the allocator can update its
//! byte counters. Sub-range [`crate::BufferView`]s carry no deed and are never accounted.
//!
//! External callers use `Device::alloc_buffer` / `Device::alloc_texture` (and the
//! `alloc_buffer_with_*` helpers). Internal `Allocation::new_with_stride_and_flags` is
//! `pub(crate)` for allocator backends and in-crate tests only.
//!
//! **Deferred-reclamation ring.** The allocator also owns the timeline-gated reclamation ring.
//! Resources with in-flight GPU work are registered via [`VramAllocator::defer_release`] and
//! dropped by [`VramAllocator::boundary_crossed`] once the timeline retires past their epoch.
//! [`DefaultVramAllocator`] implements the ring and optional [`AllocationPolicy`] hooks.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────┐
//! │  Consumers (ekrano, user code)                    │
//! │  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │
//! │  │TransientAlloc│  │Device::alloc_│  │Texture  │ │
//! │  │ (recycling)  │  │buffer()      │  │Pool     │ │
//! │  └──────┬───────┘  └──────┬───────┘  └────┬────┘ │
//! │         │ via BufferPool  │               │      │
//! │  ┌──────▼─────────────────▼───────────────▼─────┐│
//! │  │              VramAllocator trait              ││
//! │  │  alloc_buffer / alloc_texture / notify_freed  ││
//! │  │  defer_release / boundary_crossed / drain     ││
//! │  └──────────────────┬────────────────────────────┘│
//! │                     │                             │
//! │  ┌──────────────────▼────────────────────────────┐│
//! │  │          GpuBackend (Metal/Vulkan/DX12)       ││
//! │  └───────────────────────────────────────────────┘│
//! └───────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! The [`Device`] holds an [`Arc<dyn VramAllocator>`]. The default
//! ([`DefaultVramAllocator`]) delegates directly to the backend and implements the deferred
//! ring at zero overhead when [`NoPolicy`] is installed.
//! Install a custom [`AllocationPolicy`] via
//! [`Device::set_allocation_policy`](crate::device::Device::set_allocation_policy) for byte
//! tracking and budget enforcement.
//!
//! [`TransientAllocator`]: crate::transient_allocator::TransientAllocator
//! [`BufferPool`]: crate::buffer::BufferPool
//! [`Texture`]: crate::Texture
//! [`TexturePool`]: crate::texture_pool::TexturePool
//! [`Device`]: crate::device::Device
//! [`VramAllocator`]: crate::vram_allocator::VramAllocator
//! [`DefaultVramAllocator`]: crate::vram_allocator::DefaultVramAllocator
//! [`AllocationPolicy`]: crate::allocation_policy::AllocationPolicy
//! [`BudgetPolicy`]: crate::allocation_policy::BudgetPolicy

use crate::allocation_policy::{AllocCommit, AllocFreeEvent, AllocRequest, AllocationPolicy, NoPolicy};
use crate::buffer::Allocation;
use crate::device::Device;
use crate::texture::TextureBacking;
use crate::timeline::TimelineValue;
use crate::types::*;
use anyhow::Result;
use std::any::Any;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};

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
/// # use goldy::{BufferPool, Device};
/// # fn example(device: &Device) -> anyhow::Result<()> {
/// let mut pool = BufferPool::new(device, 4096)?;
/// let view = pool.alloc::<u32>(16)?;
/// let mut payload = DeferredPayload::new();
/// payload.push(view);
/// # Ok(())
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
// ParcelType
// -----------------------------------------------------------------------

/// Zoning / telemetry label for a freed parcel (buffer-kind vs texture-kind).
///
/// Not used for separate accounting code paths — only passed to [`VramAllocator::notify_freed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParcelType {
    Buffer,
    Texture,
}

/// Weak back-reference from a deed-holding parcel to its [`VramAllocator`].
///
/// When the parcel is dropped, [`Self::notify_freed`] calls
/// [`VramAllocator::notify_freed`] so allocation policies can update accounting.
#[derive(Clone)]
pub(crate) struct ParcelDeed {
    allocator: std::sync::Weak<dyn VramAllocatorAlloc>,
}

impl ParcelDeed {
    pub(crate) fn new(allocator: std::sync::Weak<dyn VramAllocatorAlloc>) -> Self {
        Self { allocator }
    }

    pub fn notify_freed(&self, reserved: u64, committed: u64, kind: ParcelType) {
        if let Some(alloc) = self.allocator.upgrade() {
            alloc.notify_freed(reserved, committed, kind);
        }
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
/// Use [`AtomicI64`](std::sync::atomic::AtomicI64) / [`AtomicU64`](std::sync::atomic::AtomicU64) for lock-free counters,
/// or a `Mutex` for more complex state.
pub trait VramAllocator: Send + Sync {
    /// Notify the allocator that a deed-holding parcel has been freed.
    ///
    /// Called automatically when a deed-holding buffer or texture is dropped after allocation
    /// through the device's [`VramAllocator`].
    /// Borrowing sub-range views (e.g. [`crate::BufferView`]) never call this.
    ///
    /// `reserved` is the parcel's reserved backing size; `committed` is the runtime's
    /// handed-out estimate (logical size for buffers, [`crate::Texture::byte_size`] for textures).
    fn notify_freed(&self, _reserved: u64, _committed: u64, _kind: ParcelType) {}

    /// Net bytes allocated by this allocator (allocations minus frees).
    ///
    /// Returns 0 if the implementation does not track allocations.
    fn allocated_bytes(&self) -> u64 {
        0
    }

    /// Optional byte budget. Returns `None` if no budget is enforced.
    /// When set, buffer and texture allocation methods should return an error if
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
    /// [`Context::flush_deferred_deletions`](crate::Context::flush_deferred_deletions)
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

    /// Install an [`AllocationPolicy`] when the implementation supports it.
    ///
    /// The default implementation always fails: custom [`VramAllocator`] wrappers do
    /// not expose a policy slot. Only [`DefaultVramAllocator`] overrides this; it
    /// rejects a second install when a
    /// non-[`NoPolicy`] is already set.
    fn set_allocation_policy(&self, _policy: Arc<dyn AllocationPolicy>) -> Result<()> {
        anyhow::bail!("this VramAllocator does not support allocation policies")
    }

    /// Like [`Self::set_allocation_policy`] but succeeds when a policy is already installed.
    fn ensure_allocation_policy(&self, policy: Arc<dyn AllocationPolicy>) -> Result<()> {
        self.set_allocation_policy(policy)
    }
}

/// Crate-internal buffer and texture allocation hooks for [`Device::alloc_buffer`] /
/// [`Device::alloc_texture`].
pub(crate) trait VramAllocatorAlloc: VramAllocator {
    /// Allocate a GPU buffer.
    fn alloc_buffer(
        &self,
        device: &Device,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<Allocation> {
        Allocation::new_with_stride_and_flags(device, size, access, element_stride, flags)
    }

    /// Allocate a GPU buffer with a pre-reserved capacity hint.
    fn alloc_buffer_with_capacity(
        &self,
        device: &Device,
        initial_size: u64,
        expected_max: u64,
        access: BufferKind,
        flags: BufferFlags,
    ) -> Result<Allocation> {
        Allocation::new_with_capacity_hint_and_flags(device, initial_size, expected_max, access, flags)
    }

    /// Allocate a GPU texture.
    fn alloc_texture(
        &self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<TextureBacking> {
        TextureBacking::new(device, width, height, format, access, flags)
    }
}

// -----------------------------------------------------------------------
// Default implementation
// -----------------------------------------------------------------------

/// The default allocator: delegates directly to raw buffer / texture constructors
/// with no tracking, budgeting, or overhead.
///
/// Installed automatically when a [`Device`] is created. Implements the full
/// deferred-release ring: [`VramAllocator::defer_release`], [`VramAllocator::boundary_crossed`],
/// and [`VramAllocator::drain`].
///
/// **Device-owned ring:** the ring is device-installed and keyed by device-global timeline
/// epochs from [`crate::context::Context::defer_release`].
/// [`crate::context::Context::boundary_crossed`] drains entries
/// when `epoch <= device_retired` (max completed over all live contexts). Any context may
/// poll boundaries; multi-context deferral is sound under this conservative collapse.
/// Per-handle last-touch reclamation (tighter than `device_retired`) is a future optimization.
pub struct DefaultVramAllocator {
    deferred: Mutex<VecDeque<(TimelineValue, DeferredPayload)>>,
    policy: RwLock<Arc<dyn AllocationPolicy>>,
}

impl DefaultVramAllocator {
    /// Create a new default allocator with [`NoPolicy`].
    pub fn new() -> Self {
        Self {
            deferred: Mutex::new(VecDeque::new()),
            policy: RwLock::new(Arc::new(NoPolicy)),
        }
    }

    /// Create a default allocator with the given [`AllocationPolicy`].
    pub fn with_policy(policy: Arc<dyn AllocationPolicy>) -> Self {
        Self {
            deferred: Mutex::new(VecDeque::new()),
            policy: RwLock::new(policy),
        }
    }

    /// Install a custom allocation policy. Fails if one is already installed.
    pub fn set_policy(&self, policy: Arc<dyn AllocationPolicy>) -> Result<()> {
        let mut guard = self.policy.write().unwrap();
        if !guard.is_noop() {
            anyhow::bail!("allocation policy already installed");
        }
        *guard = policy;
        Ok(())
    }

    /// Install `policy` only while the default [`NoPolicy`] is still active.
    pub fn ensure_policy(&self, policy: Arc<dyn AllocationPolicy>) -> Result<()> {
        let mut guard = self.policy.write().unwrap();
        if guard.is_noop() {
            *guard = policy;
        }
        Ok(())
    }

    fn with_policy_read<R>(&self, f: impl FnOnce(&dyn AllocationPolicy) -> R) -> R {
        let policy = self.policy.read().unwrap();
        f(policy.as_ref())
    }
}

impl Default for DefaultVramAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl VramAllocatorAlloc for DefaultVramAllocator {
    fn alloc_buffer(
        &self,
        device: &Device,
        size: u64,
        access: BufferKind,
        element_stride: Option<u32>,
        flags: BufferFlags,
    ) -> Result<Allocation> {
        let req = AllocRequest {
            reserved_estimate: size,
            committed_estimate: size,
            kind: ParcelType::Buffer,
        };
        self.with_policy_read(|policy| policy.before_alloc(&req))?;
        let buf = Allocation::new_with_stride_and_flags(device, size, access, element_stride, flags)?;
        self.with_policy_read(|policy| {
            policy.after_alloc(&AllocCommit::from_buffer(&buf));
        });
        Ok(buf)
    }

    fn alloc_buffer_with_capacity(
        &self,
        device: &Device,
        initial_size: u64,
        expected_max: u64,
        access: BufferKind,
        flags: BufferFlags,
    ) -> Result<Allocation> {
        let estimate = expected_max.max(initial_size);
        let req = AllocRequest {
            reserved_estimate: estimate,
            committed_estimate: initial_size,
            kind: ParcelType::Buffer,
        };
        self.with_policy_read(|policy| policy.before_alloc(&req))?;
        let buf = Allocation::new_with_capacity_hint_and_flags(device, initial_size, expected_max, access, flags)?;
        self.with_policy_read(|policy| {
            policy.after_alloc(&AllocCommit::from_buffer(&buf));
        });
        Ok(buf)
    }

    fn alloc_texture(
        &self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<TextureBacking> {
        let estimated = (width as u64) * (height as u64) * (format.bytes_per_pixel() as u64);
        let req = AllocRequest {
            reserved_estimate: estimated,
            committed_estimate: estimated,
            kind: ParcelType::Texture,
        };
        self.with_policy_read(|policy| policy.before_alloc(&req))?;
        let tex = TextureBacking::new(device, width, height, format, access, flags)?;
        self.with_policy_read(|policy| {
            policy.after_alloc(&AllocCommit::from_texture(&tex));
        });
        Ok(tex)
    }
}

impl VramAllocator for DefaultVramAllocator {
    fn notify_freed(&self, reserved: u64, committed: u64, kind: ParcelType) {
        let event = AllocFreeEvent {
            reserved,
            committed,
            kind,
        };
        self.with_policy_read(|policy| policy.on_freed(&event));
    }

    fn allocated_bytes(&self) -> u64 {
        self.with_policy_read(|policy| policy.allocated_bytes())
    }

    fn budget(&self) -> Option<u64> {
        self.with_policy_read(|policy| policy.budget())
    }

    fn set_allocation_policy(&self, policy: Arc<dyn AllocationPolicy>) -> Result<()> {
        self.set_policy(policy)
    }

    fn ensure_allocation_policy(&self, policy: Arc<dyn AllocationPolicy>) -> Result<()> {
        self.ensure_policy(policy)
    }

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
        self.deferred.lock().unwrap().front().map(|(epoch, _)| *epoch)
    }

    fn drain(&self) {
        self.deferred.lock().unwrap().clear();
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

pub(crate) fn bytesize(bytes: u64) -> String {
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
    use crate::allocation_policy::BudgetPolicy;
    use crate::backend::mock::MockBackend;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    /// Device with a byte budget policy installed once for the test body.
    fn device_with_budget_policy(budget_bytes: u64) -> (Device, Arc<BudgetPolicy>) {
        let device = test_device();
        let policy = Arc::new(BudgetPolicy::with_budget(budget_bytes));
        device
            .set_allocation_policy(policy.clone())
            .expect("install budget policy in test fixture");
        (device, policy)
    }

    fn device_with_policy() -> (Device, Arc<BudgetPolicy>) {
        let device = test_device();
        let policy = Arc::new(BudgetPolicy::new());
        device
            .set_allocation_policy(policy.clone())
            .expect("install budget policy in test fixture");
        (device, policy)
    }

    mod allocation_policy {
        use super::*;

        #[test]
        fn budget_rejects_over_cap() {
            let (device, policy) = device_with_budget_policy(8192);

            let buf = device
                .alloc_buffer(4096, BufferKind::Scattered, None, BufferFlags::empty())
                .unwrap();
            assert_eq!(policy.allocated_bytes(), 4096);

            let err = device.alloc_buffer(8192, BufferKind::Scattered, None, BufferFlags::empty());
            assert!(err.is_err(), "allocation policy budget should reject second alloc");
            assert_eq!(policy.allocated_bytes(), 4096);

            drop(buf);
            assert_eq!(policy.allocated_bytes(), 0);
        }

        #[test]
        fn second_install_fails_on_default_allocator() {
            let device = test_device();
            device
                .set_allocation_policy(Arc::new(BudgetPolicy::new()))
                .expect("first policy install");
            let err = device.set_allocation_policy(Arc::new(BudgetPolicy::new()));
            assert!(err.is_err(), "second policy install should fail");
            assert!(
                err.unwrap_err().to_string().contains("already installed"),
                "expected duplicate-install error, got different failure"
            );
        }
    }

    #[test]
    fn default_allocator_creates_buffer() {
        let device = test_device();
        let alloc = DefaultVramAllocator::new();
        let buf = alloc
            .alloc_buffer(&device, 1024, BufferKind::Scattered, None, BufferFlags::empty())
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
                TextureKind::Interpolated,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .unwrap();
        assert_eq!(tex.width(), 64);
        assert_eq!(tex.height(), 64);
    }

    #[test]
    fn budget_policy_tracks_bytes() {
        let (device, policy) = device_with_policy();

        assert_eq!(policy.allocated_bytes(), 0);

        let buf = device
            .alloc_buffer(4096, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();
        assert!(policy.allocated_bytes() >= 4096);

        drop(buf);
        assert_eq!(policy.allocated_bytes(), 0);
    }

    #[test]
    fn budget_policy_tracks_textures() {
        let (device, policy) = device_with_policy();

        let tex = device
            .alloc_texture(
                32,
                32,
                TextureFormat::Rgba8Unorm,
                TextureKind::Interpolated,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .unwrap();
        assert!(policy.allocated_bytes() > 0);

        drop(tex);
        assert_eq!(policy.allocated_bytes(), 0);
    }

    #[test]
    fn accounting_round_trip_mixed_parcels() {
        let (device, policy) = device_with_policy();
        const N: usize = 8;

        for _ in 0..N {
            assert_eq!(policy.allocated_bytes(), 0);

            let buf = device
                .alloc_buffer(1024, BufferKind::Scattered, None, BufferFlags::empty())
                .unwrap();
            let hinted = device
                .alloc_buffer_with_capacity(512, 4096, BufferKind::Scattered, BufferFlags::empty())
                .unwrap();
            let tex = device
                .alloc_texture(
                    16,
                    16,
                    TextureFormat::Rgba8Unorm,
                    TextureKind::Interpolated,
                    TextureFlags::COPY_DST,
                )
                .unwrap();

            let bytes_with_parcels = policy.allocated_bytes();
            assert!(bytes_with_parcels > 0);

            let view = buf.create_view(0, 256, Some(4)).unwrap();
            let bytes_before_view_drop = policy.allocated_bytes();
            drop(view);
            assert_eq!(
                policy.allocated_bytes(),
                bytes_before_view_drop,
                "BufferView drop must not change allocator accounting"
            );

            drop(buf);
            drop(hinted);
            drop(tex);
            assert_eq!(policy.allocated_bytes(), 0);
        }
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
        assert!(weak.upgrade().is_none(), "resource should have been dropped");
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
        assert!(weak_late.upgrade().is_none(), "epoch=10 should now be dropped");
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
}
