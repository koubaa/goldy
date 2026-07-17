//! RAII guard that defers GPU resource lifetime until a timeline epoch retires.
//!
//! [`GpuGuard`] pairs a set of GPU resources with a [`TimelineValue`] epoch. When the
//! guard is dropped, the resources are handed to the device's [`VramAllocator`] for
//! deferred release — they remain alive until [`crate::Context::flush_deferred_deletions`]
//! observes `gpu_progress >= epoch`, at which point they are dropped.
//!
//! This is the correct, non-blocking pattern for ensuring owned resources outlive the
//! GPU work that references them without blocking the CPU.
//!
//! # Example
//!
//! ```no_run
//! # use goldy::{Context, GpuGuard, RetainedPool, BufferKind, BufferFlags, Scheme};
//! # use std::sync::Arc;
//! # fn example(context: &Context, pool: &mut RetainedPool) -> anyhow::Result<()> {
//! let buf = pool.acquire_buffer(256, BufferKind::Scattered, None, BufferFlags::empty(), None)?;
//! let mut scheme = Scheme::new(context); // ... record nodes using &buf ...
//! let tv = scheme.submit()?.timeline_value();
//!
//! let mut guard = GpuGuard::new(context, tv);
//! guard.hold(buf); // buf is now safe: dropped only after tv retires
//!
//! // Drop order in a struct: guard before context → correct.
//! // Or let the guard go out of scope; flush_deferred_deletions will reclaim it.
//! # Ok(())
//! # }
//! ```
//!
//! # Comparison with blocking
//!
//! Without `GpuGuard`, safe resource cleanup requires `context.wait_until(tv)` before
//! dropping — which stalls the CPU. `GpuGuard` defers the drop to the next
//! [`flush_deferred_deletions`](crate::context::Context::flush_deferred_deletions) call
//! once the timeline has naturally advanced, keeping the CPU–GPU pipeline full.
//!
//! [`TimelineValue`]: crate::timeline::TimelineValue
//! [`VramAllocator`]: crate::vram_allocator::VramAllocator

use crate::context::Context;
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
    context: Context,
    epoch: TimelineValue,
    held: Vec<Box<dyn Any + Send>>,
}

impl GpuGuard {
    /// Create a new guard tied to `context` and `epoch`.
    ///
    /// Resources added via [`hold`](Self::hold) will not be dropped until
    /// `context.gpu_progress() >= epoch`.
    pub fn new(context: &Context, epoch: TimelineValue) -> Self {
        Self {
            context: context.clone(),
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
        self.context.defer_release(self.epoch, payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::context::Context;
    use crate::device::Device;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn test_ctx() -> Context {
        test_device().create_context().unwrap()
    }

    #[test]
    fn gpu_guard_new_is_empty() {
        let ctx = test_ctx();
        let guard = GpuGuard::new(&ctx, 5);
        assert!(guard.is_empty());
        assert_eq!(guard.resource_count(), 0);
        assert_eq!(guard.epoch(), 5);
    }

    #[test]
    fn gpu_guard_hold_tracks_count() {
        let ctx = test_ctx();
        let mut guard = GpuGuard::new(&ctx, 1);
        guard.hold(42u32).hold("hello").hold(vec![1u8, 2, 3]);
        assert!(!guard.is_empty());
        assert_eq!(guard.resource_count(), 3);
    }

    #[test]
    fn gpu_guard_empty_drop_is_noop() {
        let ctx = test_ctx();
        // An empty guard should not add anything to the deferred ring.
        let guard = GpuGuard::new(&ctx, 1);
        drop(guard);
        // After drop, flush: nothing to reclaim because nothing was deferred.
        ctx.flush_deferred_deletions();
        // No panic, no assertion failure — just confirming no crash.
    }

    fn scheme_submit(ctx: &Context) -> TimelineValue {
        crate::test_support::scheme_advance_timeline(ctx)
    }

    #[test]
    fn gpu_guard_resources_not_dropped_before_epoch() {
        let ctx = test_ctx();
        let tv = scheme_submit(&ctx);

        let alive = std::sync::Arc::new(99u32);
        let weak = std::sync::Arc::downgrade(&alive);

        let mut guard = GpuGuard::new(&ctx, tv + 100);
        guard.hold(alive);

        // Dropping the guard enqueues the Arc into the ring at epoch tv+100.
        drop(guard);

        // flush_deferred_deletions at current gpu_progress (tv) must NOT drop it.
        ctx.flush_deferred_deletions();
        assert!(
            weak.upgrade().is_some(),
            "resource must not be dropped before its epoch"
        );
    }

    #[test]
    fn gpu_guard_resources_dropped_after_epoch_and_flush() {
        let ctx = test_ctx();
        let tv = scheme_submit(&ctx);

        let alive = std::sync::Arc::new(42u32);
        let weak = std::sync::Arc::downgrade(&alive);

        let mut guard = GpuGuard::new(&ctx, tv);
        guard.hold(alive);
        drop(guard);

        // Advance GPU to tv and flush — resource must be dropped.
        ctx.wait_until(tv).unwrap();
        ctx.flush_deferred_deletions();
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
        let ctx = test_ctx();
        let tv = scheme_submit(&ctx);
        let tv_future = tv + 100;

        let alive_past = std::sync::Arc::new(1u32);
        let weak_past = std::sync::Arc::downgrade(&alive_past);
        let alive_future = std::sync::Arc::new(2u32);
        let weak_future = std::sync::Arc::downgrade(&alive_future);

        let mut guard_past = GpuGuard::new(&ctx, tv);
        guard_past.hold(alive_past);
        drop(guard_past);

        let mut guard_future = GpuGuard::new(&ctx, tv_future);
        guard_future.hold(alive_future);
        drop(guard_future);

        // Advance to tv — past guard's resource should be reclaimed, future's should not.
        ctx.wait_until(tv).unwrap();
        ctx.flush_deferred_deletions();
        assert!(
            weak_past.upgrade().is_none(),
            "epoch=tv resource should be dropped after wait_until(tv)"
        );
        assert!(
            weak_future.upgrade().is_some(),
            "epoch=tv+100 resource should survive flush at tv"
        );

        // Advance to tv_future — future guard's resource should now be reclaimed.
        ctx.wait_until(tv_future).unwrap();
        ctx.flush_deferred_deletions();
        assert!(
            weak_future.upgrade().is_none(),
            "epoch=tv+100 resource should be dropped after wait_until(tv+100)"
        );
    }

    #[test]
    fn gpu_guard_resources_cleaned_up_on_device_drop() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let tv = scheme_submit(&ctx);

        let alive = std::sync::Arc::new(7u32);
        let weak = std::sync::Arc::downgrade(&alive);

        // Guard with a far-future epoch — normal flush won't reclaim.
        let mut guard = GpuGuard::new(&ctx, tv + 9999);
        guard.hold(alive);
        drop(guard);

        // Device drop should drain everything.
        drop(ctx);
        drop(device);
        assert!(
            weak.upgrade().is_none(),
            "device drop must clean up all deferred resources"
        );
    }

    #[test]
    fn gpu_guard_hold_multiple_resource_types() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let tv = scheme_submit(&ctx);

        // Allocate a real buffer and hold it in a guard.
        let buf = device
            .alloc_buffer(
                1024,
                crate::types::BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
            )
            .unwrap();
        let alive = std::sync::Arc::new(0u32);
        let weak = std::sync::Arc::downgrade(&alive);

        let mut guard = GpuGuard::new(&ctx, tv);
        guard.hold(buf).hold(alive);
        assert_eq!(guard.resource_count(), 2);
        drop(guard);

        ctx.wait_until(tv).unwrap();
        ctx.flush_deferred_deletions();
        assert!(weak.upgrade().is_none());
    }
}
