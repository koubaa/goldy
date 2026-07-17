//! U0 baseline characterization tests for the unified boundary event refactor.
//!
//! These four tests lock the current reclamation contract that later units (U1–U9)
//! must preserve. Each test exercises the recycle-after-epoch path (not just
//! construction) and includes a negative assertion that fails if reclamation is
//! skipped.
//!
//! Contract under test:
//! 1. VRAM deferred ring empties after `submit + wait + flush`
//! 2. `HeapTransientAllocator` returns freed ranges after epoch retirement
//!
//! Run with: `cargo test -p goldy boundary_reclamation`

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::backend::mock::MockBackend;
    use crate::context::Context;
    use crate::device::Device;
    use crate::signal::Signal;
    use crate::test_support::scheme_advance_timeline;
    use crate::transient_allocator::{
        BumpResetAllocator, HeapTransientAllocator, TransientAllocator, TransientAllocatorConfig,
    };
    use crate::types::BufferFlags;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn test_ctx(device: &Device) -> Context {
        device.create_context().unwrap()
    }

    fn small_config() -> TransientAllocatorConfig {
        TransientAllocatorConfig {
            initial_size: 4 * 1024,
            alignment: 256,
            flags: BufferFlags::empty(),
        }
    }

    fn scheme_submit(ctx: &Context) -> crate::TimelineValue {
        let tv = scheme_advance_timeline(ctx);
        assert!(tv > 0, "scheme submit must advance timeline");
        tv
    }

    fn heap_config() -> TransientAllocatorConfig {
        TransientAllocatorConfig {
            initial_size: 64 * 1024,
            alignment: 256,
            flags: BufferFlags::empty(),
        }
    }

    /// The VRAM deferred ring must drain after submit + wait + flush.
    ///
    /// Negative check: after `wait_until` but *before* `flush_deferred_deletions`,
    /// the payload is still alive and `has_deferred_payloads()` is true.
    #[test]
    fn u0_vram_ring_empties_after_submit_wait_flush() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let tv = scheme_submit(&ctx);

        let alive = Arc::new(99u32);
        let weak = Arc::downgrade(&alive);
        ctx.defer_until(tv, alive);

        assert!(ctx.has_deferred_payloads(), "VRAM ring must hold payload before flush");
        assert!(weak.upgrade().is_some(), "payload must stay alive before flush");

        ctx.wait_until(tv).expect("wait");

        // GPU has retired the epoch, but reclaim has not run yet.
        assert!(
            ctx.has_deferred_payloads(),
            "VRAM ring must still hold payload after wait, before flush"
        );
        assert!(
            weak.upgrade().is_some(),
            "payload must stay alive after wait, before flush"
        );

        ctx.flush_deferred_deletions();

        assert!(
            !ctx.has_deferred_payloads(),
            "VRAM ring must be empty after submit + wait + flush"
        );
        assert!(weak.upgrade().is_none(), "payload must be dropped after flush");
    }

    /// HeapTransient freed ranges return to the free list once `gpu_progress >= epoch`.
    #[test]
    fn u0_heap_free_range_recycles_after_epoch() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let mut alloc = HeapTransientAllocator::new(&device, heap_config()).unwrap();

        alloc.begin_frame(&device, 0).unwrap();
        let v1 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        let offset = v1.offset();
        let tv = scheme_submit(&ctx);
        let far_epoch = tv + 100;

        alloc.free(offset, 1024, Some(far_epoch));
        alloc.end_frame(&device, far_epoch);

        alloc.begin_frame(&device, 0).unwrap();
        let blocked = alloc.alloc(&device, 1024, Some(4)).unwrap();
        assert_ne!(
            blocked.offset(),
            offset,
            "freed range must not be reused before epoch retires"
        );

        ctx.wait_until(far_epoch).expect("wait");
        alloc.begin_frame(&device, 0).unwrap();
        let reused = alloc.alloc(&device, 1024, Some(4)).expect("alloc after reclaim");
        assert_eq!(
            reused.offset(),
            offset,
            "freed range should be reused after wait + begin_frame"
        );
    }

    /// `poll_signals_and_service` routes `BoundaryCrossed` into `boundary_crossed(epoch)`.
    #[test]
    fn u3_signal_boundary_crossed_services_vram_ring() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let tv = scheme_submit(&ctx);

        let alive = Arc::new(42u32);
        let weak = Arc::downgrade(&alive);
        ctx.defer_until(tv, alive);
        assert!(ctx.has_deferred_payloads());

        let signals = ctx.poll_signals_and_service();
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, Signal::BoundaryCrossed { epoch } if *epoch == tv)),
            "submit should post BoundaryCrossed for epoch {tv}"
        );
        assert!(
            !ctx.has_deferred_payloads(),
            "VRAM ring must empty after poll_signals_and_service"
        );
        assert!(weak.upgrade().is_none(), "payload must drop after service");
    }

    /// Pull-side `flush_deferred_deletions` must reclaim without draining signals.
    ///
    /// Locks the plan invariant that `gpu_progress()` is the authoritative retirement
    /// horizon: reclamation must not depend solely on the client draining
    /// `Signal::BoundaryCrossed` from the poller queue.
    #[test]
    fn pull_path_reclaims_without_signal_drain() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let tv = scheme_submit(&ctx);

        let alive = Arc::new(77u32);
        let weak = Arc::downgrade(&alive);
        ctx.defer_until(tv, alive);
        assert!(ctx.has_deferred_payloads());

        ctx.wait_until(tv).expect("wait");
        ctx.flush_deferred_deletions();

        assert!(
            !ctx.has_deferred_payloads(),
            "VRAM ring must empty via pull path without polling signals"
        );
        assert!(weak.upgrade().is_none(), "payload must drop after pull flush");
    }

    /// `boundary_crossed` is idempotent: calling it twice for the same epoch reclaims once.
    #[test]
    fn boundary_crossed_is_idempotent_for_same_epoch() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let tv = scheme_submit(&ctx);

        let alive = Arc::new(11u32);
        let weak = Arc::downgrade(&alive);
        ctx.defer_until(tv, alive);
        assert!(ctx.has_deferred_payloads());

        ctx.wait_until(tv).expect("wait");

        ctx.boundary_crossed(tv);
        assert!(
            !ctx.has_deferred_payloads(),
            "first boundary_crossed must drain the VRAM ring"
        );
        assert!(weak.upgrade().is_none(), "payload must drop on first boundary_crossed");

        // Second call for the same epoch must be a no-op (no double-free panic).
        ctx.boundary_crossed(tv);
        assert!(
            !ctx.has_deferred_payloads(),
            "second boundary_crossed must remain a no-op"
        );
    }

    /// A stale (lower) epoch after a higher-water reclaim must not under-reclaim or double-free.
    #[test]
    fn boundary_crossed_stale_epoch_is_no_op_after_high_water() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let tv1 = scheme_submit(&ctx);
        let tv2 = scheme_submit(&ctx);
        assert!(tv2 > tv1, "mock timeline must advance");

        let alive1 = Arc::new(1u32);
        let alive2 = Arc::new(2u32);
        let weak1 = Arc::downgrade(&alive1);
        let weak2 = Arc::downgrade(&alive2);
        ctx.defer_until(tv1, alive1);
        ctx.defer_until(tv2, alive2);
        assert!(ctx.has_deferred_payloads());

        ctx.wait_until(tv2).expect("wait");

        // High-water reclaim retires both payloads (epoch <= tv2).
        ctx.boundary_crossed(tv2);
        assert!(!ctx.has_deferred_payloads());
        assert!(weak1.upgrade().is_none());
        assert!(weak2.upgrade().is_none());

        // Stale lower epoch must not panic or resurrect payloads.
        ctx.boundary_crossed(tv1);
        assert!(!ctx.has_deferred_payloads());
    }

    /// BumpReset must NOT reset the pool before the prior frame's epoch retires.
    ///
    /// `begin_frame` gates the reset on `gpu_progress() >= last_epoch`. If the epoch
    /// has not retired, it must block via `wait_until`. This test uses a far epoch
    /// that the mock backend has not reached and verifies that `begin_frame` calls
    /// `wait_until` (advancing mock progress) before resetting.
    #[test]
    fn u6_bump_reset_waits_before_reset() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let mut alloc = BumpResetAllocator::new(&device, small_config()).expect("create");

        alloc.begin_frame(&device, 0).expect("begin 1");
        let v1 = alloc.alloc(&device, 512, Some(4)).expect("alloc");
        let original_offset = v1.offset();
        let tv = scheme_submit(&ctx);

        // Retire to a far epoch the GPU has NOT reached.
        let far_epoch = tv + 100;
        alloc.end_frame(&device, far_epoch);

        // begin_frame must wait on far_epoch (mock wait_until advances progress),
        // then reset. If the wait were skipped, the pool would reset while the
        // GPU is still using the buffer — a use-after-free.
        alloc.begin_frame(&device, 0).expect("begin 2");

        // After reset the bump pointer is at 0, so the next alloc should land at
        // the same offset as the first (confirming reset happened).
        let v2 = alloc.alloc(&device, 512, Some(4)).expect("alloc after reset");
        assert_eq!(
            v2.offset(),
            original_offset,
            "bump should reset to 0 after wait + begin_frame"
        );
        // Device-global retirement must reach far_epoch (begin_frame wait_until on allocator ctx).
        assert!(
            device.timeline_retired() >= far_epoch,
            "begin_frame must have called wait_until to advance device retirement"
        );
    }
}
