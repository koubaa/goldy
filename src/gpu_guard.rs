//! RAII guard that defers GPU resource lifetime until a timeline epoch retires.
//!
//! [`GpuGuard`] pairs a set of GPU resources with a [`TimelineValue`] epoch. When the
//! guard is dropped, the resources are handed to the device's [`VramAllocator`] for
//! deferred release — they remain alive until [`Device::flush_deferred_deletions`]
//! observes `gpu_progress >= epoch`, at which point they are dropped.
//!
//! This is the correct, non-blocking pattern for ensuring owned resources outlive the
//! GPU work that references them without blocking the CPU.
//!
//! # Example
//!
//! ```no_run
//! # use goldy::{Device, GpuGuard};
//! # use goldy::buffer::Buffer;
//! # use goldy::task_graph::TaskGraph;
//! # fn example(device: &Device, buf: Buffer) -> anyhow::Result<()> {
//! let mut graph = TaskGraph::new(); // ... build your graph using buf ...
//! let tv = device.submit(&mut graph)?;
//!
//! let mut guard = GpuGuard::new(device, tv);
//! guard.hold(buf); // buf is now safe: dropped only after tv retires
//!
//! // Drop order in a struct: guard before device → correct.
//! // Or let the guard go out of scope; flush_deferred_deletions will reclaim it.
//! # Ok(())
//! # }
//! ```
//!
//! # Comparison with blocking
//!
//! Without `GpuGuard`, safe resource cleanup requires `context.wait_until(tv)` before
//! dropping — which stalls the CPU. `GpuGuard` defers the drop to the next
//! [`flush_deferred_deletions`](crate::device::Device::flush_deferred_deletions) call
//! once the timeline has naturally advanced, keeping the CPU–GPU pipeline full.
//!
//! [`TimelineValue`]: crate::timeline::TimelineValue
//! [`VramAllocator`]: crate::vram_allocator::VramAllocator

use crate::device::Device;
use crate::timeline::TimelineValue;
use crate::vram_allocator::DeferredPayload;
use std::any::Any;

/// RAII guard that defers dropping of GPU resources until a GPU timeline epoch retires.
///
/// Construct with [`GpuGuard::new`], add resources with [`hold`](Self::hold), then drop the
/// guard (or let it go out of scope). On drop, all held resources are passed to the
/// device's [`VramAllocator`] for deferred release — no blocking occurs.
///
/// See the [module documentation](self) for usage examples and a comparison with blocking.
///
/// [`VramAllocator`]: crate::vram_allocator::VramAllocator
pub struct GpuGuard {
    device: Device,
    epoch: TimelineValue,
    held: Vec<Box<dyn Any + Send>>,
}

impl GpuGuard {
    /// Create a new guard tied to `device` and `epoch`.
    ///
    /// Resources added via [`hold`](Self::hold) will not be dropped until
    /// `context.gpu_progress() >= epoch`.
    pub fn new(device: &Device, epoch: TimelineValue) -> Self {
        Self {
            device: device.clone(),
            epoch,
            held: Vec::new(),
        }
    }

    /// Hold `resource` alive until this guard's epoch retires.
    ///
    /// Returns `&mut Self` for chaining: `guard.hold(buf).hold(view)`.
    pub fn hold<T: Send + 'static>(&mut self, resource: T) -> &mut Self {
        self.held.push(Box::new(resource));
        self
    }

    /// The GPU timeline epoch this guard is waiting on.
    pub fn epoch(&self) -> TimelineValue {
        self.epoch
    }

    /// Returns `true` if no resources have been added to this guard.
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }

    /// Number of resources currently held by this guard.
    pub fn resource_count(&self) -> usize {
        self.held.len()
    }
}

impl Drop for GpuGuard {
    fn drop(&mut self) {
        if self.held.is_empty() {
            return;
        }
        let mut payload = DeferredPayload::new();
        for resource in self.held.drain(..) {
            payload.0.push(resource);
        }
        self.device.defer_release(self.epoch, payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::task_graph::TaskGraph;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn gpu_guard_new_is_empty() {
        let device = test_device();
        let guard = GpuGuard::new(&device, 5);
        assert!(guard.is_empty());
        assert_eq!(guard.resource_count(), 0);
        assert_eq!(guard.epoch(), 5);
    }

    #[test]
    fn gpu_guard_hold_tracks_count() {
        let device = test_device();
        let mut guard = GpuGuard::new(&device, 1);
        guard.hold(42u32).hold("hello").hold(vec![1u8, 2, 3]);
        assert!(!guard.is_empty());
        assert_eq!(guard.resource_count(), 3);
    }

    #[test]
    fn gpu_guard_empty_drop_is_noop() {
        let device = test_device();
        // An empty guard should not add anything to the deferred ring.
        let guard = GpuGuard::new(&device, 1);
        drop(guard);
        // After drop, flush: nothing to reclaim because nothing was deferred.
        device.flush_deferred_deletions();
        // No panic, no assertion failure — just confirming no crash.
    }

    #[test]
    fn gpu_guard_resources_not_dropped_before_epoch() {
        let device = test_device();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).unwrap();

        let alive = std::sync::Arc::new(99u32);
        let weak = std::sync::Arc::downgrade(&alive);

        let mut guard = GpuGuard::new(&device, tv + 100);
        guard.hold(alive);

        // Dropping the guard enqueues the Arc into the ring at epoch tv+100.
        drop(guard);

        // flush_deferred_deletions at current gpu_progress (tv) must NOT drop it.
        device.flush_deferred_deletions();
        assert!(
            weak.upgrade().is_some(),
            "resource must not be dropped before its epoch"
        );
    }

    #[test]
    fn gpu_guard_resources_dropped_after_epoch_and_flush() {
        let device = test_device();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).unwrap();

        let alive = std::sync::Arc::new(42u32);
        let weak = std::sync::Arc::downgrade(&alive);

        let mut guard = GpuGuard::new(&device, tv);
        guard.hold(alive);
        drop(guard);

        // Advance GPU to tv and flush — resource must be dropped.
        device.wait_until_impl(tv).unwrap();
        device.flush_deferred_deletions();
        assert!(
            weak.upgrade().is_none(),
            "resource must be dropped after epoch retires and flush_deferred_deletions is called"
        );
    }

    #[test]
    fn gpu_guard_multiple_guards_different_epochs() {
        // The mock backend marks every submission as completed immediately, so we use
        // one real submitted epoch (tv) and one far-future epoch (tv + 100) that hasn't
        // been submitted and therefore remains unretired after wait_until(tv).
        let device = test_device();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).unwrap();
        let tv_future = tv + 100;

        let alive_past = std::sync::Arc::new(1u32);
        let weak_past = std::sync::Arc::downgrade(&alive_past);
        let alive_future = std::sync::Arc::new(2u32);
        let weak_future = std::sync::Arc::downgrade(&alive_future);

        let mut guard_past = GpuGuard::new(&device, tv);
        guard_past.hold(alive_past);
        drop(guard_past);

        let mut guard_future = GpuGuard::new(&device, tv_future);
        guard_future.hold(alive_future);
        drop(guard_future);

        // Advance to tv — past guard's resource should be reclaimed, future's should not.
        device.wait_until_impl(tv).unwrap();
        device.flush_deferred_deletions();
        assert!(
            weak_past.upgrade().is_none(),
            "epoch=tv resource should be dropped after wait_until(tv)"
        );
        assert!(
            weak_future.upgrade().is_some(),
            "epoch=tv+100 resource should survive flush at tv"
        );

        // Advance to tv_future — future guard's resource should now be reclaimed.
        device.wait_until_impl(tv_future).unwrap();
        device.flush_deferred_deletions();
        assert!(
            weak_future.upgrade().is_none(),
            "epoch=tv+100 resource should be dropped after wait_until(tv+100)"
        );
    }

    #[test]
    fn gpu_guard_resources_cleaned_up_on_device_drop() {
        let device = test_device();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).unwrap();

        let alive = std::sync::Arc::new(7u32);
        let weak = std::sync::Arc::downgrade(&alive);

        // Guard with a far-future epoch — normal flush won't reclaim.
        let mut guard = GpuGuard::new(&device, tv + 9999);
        guard.hold(alive);
        drop(guard);

        // Device drop should drain everything.
        drop(device);
        assert!(
            weak.upgrade().is_none(),
            "device drop must clean up all deferred resources"
        );
    }

    #[test]
    fn gpu_guard_hold_multiple_resource_types() {
        let device = test_device();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).unwrap();

        // Allocate a real buffer and hold it in a guard.
        let buf = device
            .alloc_buffer(
                1024,
                crate::types::DataAccess::Scattered,
                None,
                crate::types::BufferFlags::empty(),
            )
            .unwrap();
        let alive = std::sync::Arc::new(0u32);
        let weak = std::sync::Arc::downgrade(&alive);

        let mut guard = GpuGuard::new(&device, tv);
        guard.hold(buf).hold(alive);
        assert_eq!(guard.resource_count(), 2);
        drop(guard);

        device.wait_until_impl(tv).unwrap();
        device.flush_deferred_deletions();
        assert!(weak.upgrade().is_none());
    }
}
