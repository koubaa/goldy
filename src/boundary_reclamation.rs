//! U0 baseline characterization tests for the unified boundary event refactor.
//!
//! These tests lock the VRAM deferred-ring reclamation contract that later units
//! must preserve. Each test exercises the recycle-after-epoch path (not just
//! construction) and includes a negative assertion that fails if reclamation is
//! skipped.
//!
//! Contract under test:
//! 1. VRAM deferred ring empties after `submit + wait + flush`
//! 2. `poll_signals_and_service` / pull-path flush both reclaim correctly
//!
//! Run with: `cargo test -p goldy boundary_reclamation`

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::backend::mock::MockBackend;
    use crate::context::Context;
    use crate::device::Device;
    use crate::test_support::scheme_advance_timeline;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn test_ctx(device: &Device) -> Context {
        device.create_context().unwrap()
    }

    fn scheme_submit(ctx: &Context) -> crate::timeline::TimelineValue {
        let tv = scheme_advance_timeline(ctx);
        assert!(tv > 0, "scheme submit must advance timeline");
        tv
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

    /// `poll_signals_and_service` routes boundary-crossed into `boundary_crossed(epoch)`.
    #[test]
    fn u3_signal_boundary_crossed_services_vram_ring() {
        let device = test_device();
        let ctx = test_ctx(&device);
        let tv = scheme_submit(&ctx);

        let alive = Arc::new(42u32);
        let weak = Arc::downgrade(&alive);
        ctx.defer_until(tv, alive);
        assert!(ctx.has_deferred_payloads());

        let _signals = ctx.poll_signals_and_service();
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
}
