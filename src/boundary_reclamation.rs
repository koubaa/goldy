//! U0 baseline characterization tests for the unified boundary event refactor.
//!
//! These four tests lock the current reclamation contract that later units (U1–U9)
//! must preserve. Each test exercises the recycle-after-epoch path (not just
//! construction) and includes a negative assertion that fails if reclamation is
//! skipped.
//!
//! Contract under test:
//! 1. VRAM deferred ring empties after `submit + wait + flush`
//! 2. `EpochRegionsAllocator` recycles regions after epoch retirement
//! 3. `HeapTransientAllocator` returns freed ranges after epoch retirement
//! 4. `PlacementHeap` ring reclaims stamped regions once `gpu_progress >= epoch`
//!
//! Run with: `cargo test -p goldy boundary_reclamation`

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::backend::mock::MockBackend;
    use crate::device::Device;
    use crate::placement_heap::PlacementHeap;
    use crate::signal::Signal;
    use crate::task_graph::TaskGraph;
    use crate::transient_allocator::{
        BumpResetAllocator, EpochRegionsAllocator, HeapTransientAllocator, TransientAllocator,
        TransientAllocatorConfig,
    };
    use crate::types::BufferFlags;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn small_config() -> TransientAllocatorConfig {
        TransientAllocatorConfig {
            initial_size: 4 * 1024,
            min_region_size: 4 * 1024,
            max_regions: 3,
            alignment: 256,
            flags: BufferFlags::empty(),
        }
    }

    fn heap_config() -> TransientAllocatorConfig {
        TransientAllocatorConfig {
            initial_size: 64 * 1024,
            min_region_size: 64 * 1024,
            max_regions: 3,
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
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).expect("submit");

        let alive = Arc::new(99u32);
        let weak = Arc::downgrade(&alive);
        device.defer_until(tv, alive);

        assert!(
            device.has_deferred_payloads(),
            "VRAM ring must hold payload before flush"
        );
        assert!(
            weak.upgrade().is_some(),
            "payload must stay alive before flush"
        );

        device.wait_until(tv).expect("wait");

        // GPU has retired the epoch, but reclaim has not run yet.
        assert!(
            device.has_deferred_payloads(),
            "VRAM ring must still hold payload after wait, before flush"
        );
        assert!(
            weak.upgrade().is_some(),
            "payload must stay alive after wait, before flush"
        );

        device.flush_deferred_deletions();

        assert!(
            !device.has_deferred_payloads(),
            "VRAM ring must be empty after submit + wait + flush"
        );
        assert!(
            weak.upgrade().is_none(),
            "payload must be dropped after flush"
        );
    }

    /// EpochRegions regions transition Empty once `gpu_progress >= epoch` at `begin_frame`.
    #[test]
    fn u0_epoch_regions_recycles_after_epoch() {
        let device = test_device();
        let mut alloc = EpochRegionsAllocator::new(&device, small_config()).expect("create");

        alloc.begin_frame(&device, 0).expect("begin 1");
        let _view = alloc.alloc(&device, 1024, Some(4)).expect("alloc");
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).expect("submit");

        alloc.end_frame(&device, tv);
        assert!(
            alloc.retired_count() >= 1,
            "active region should be DeferredReclaim after end_frame"
        );
        assert_eq!(
            alloc.empty_count(),
            0,
            "no Empty regions immediately after end_frame"
        );

        device.wait_until(tv).expect("wait");

        // Epoch retired on GPU, but begin_frame has not run yet.
        assert!(
            alloc.retired_count() >= 1,
            "region must stay DeferredReclaim before begin_frame"
        );

        alloc.begin_frame(&device, 0).expect("begin after wait");
        assert_eq!(
            alloc.retired_count(),
            0,
            "DeferredReclaim should be gone after wait + begin_frame"
        );
    }

    /// EpochRegions must NOT recycle a region while `gpu_progress < epoch`.
    ///
    /// This is the use-after-free guard for the U4 direct-compare reclaim: `begin_frame`
    /// runs `reclaim_retired_regions`, but a region retired to a not-yet-retired epoch must
    /// stay `DeferredReclaim` until the GPU actually reaches it.
    #[test]
    fn u4_epoch_regions_not_recycled_before_epoch() {
        let device = test_device();
        let mut alloc = EpochRegionsAllocator::new(&device, small_config()).expect("create");

        alloc.begin_frame(&device, 0).expect("begin 1");
        let _view = alloc.alloc(&device, 1024, Some(4)).expect("alloc");
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).expect("submit");

        // Retire the region to a far epoch the GPU has NOT reached.
        let far_epoch = tv + 100;
        alloc.end_frame(&device, far_epoch);
        assert!(
            alloc.retired_count() >= 1,
            "region should be DeferredReclaim after end_frame"
        );

        // begin_frame runs reclaim_retired_regions, but gpu_progress (== tv) < far_epoch,
        // so the region must NOT recycle (it would be a use-after-free).
        alloc.begin_frame(&device, 0).expect("begin 2");
        assert!(
            alloc.retired_count() >= 1,
            "region must not recycle before its epoch retires"
        );

        // Once the epoch retires, the next begin_frame recycles it.
        device.wait_until(far_epoch).expect("wait");
        alloc.begin_frame(&device, 0).expect("begin 3");
        assert_eq!(
            alloc.retired_count(),
            0,
            "region recycles once gpu_progress >= epoch"
        );
    }

    /// HeapTransient freed ranges return to the free list once `gpu_progress >= epoch`.
    #[test]
    fn u0_heap_free_range_recycles_after_epoch() {
        let device = test_device();
        let mut alloc = HeapTransientAllocator::new(&device, heap_config()).unwrap();

        alloc.begin_frame(&device, 0).unwrap();
        let v1 = alloc.alloc(&device, 1024, Some(4)).unwrap();
        let offset = v1.offset();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).expect("submit");
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

        device.wait_until(far_epoch).expect("wait");
        alloc.begin_frame(&device, 0).unwrap();
        let reused = alloc
            .alloc(&device, 1024, Some(4))
            .expect("alloc after reclaim");
        assert_eq!(
            reused.offset(),
            offset,
            "freed range should be reused after wait + begin_frame"
        );
    }

    /// PlacementHeap ring must release stamped regions once gpu_progress >= epoch.
    #[test]
    fn u0_placement_heap_ring_reclaims_after_epoch() {
        let device = test_device();
        let mut heap = PlacementHeap::new(&device, 3 * 1024, 1024).unwrap();

        let _o1 = heap.acquire(1024).unwrap();
        heap.stamp(1);
        let _o2 = heap.acquire(1024).unwrap();
        heap.stamp(2);
        let _o3 = heap.acquire(1024).unwrap();
        heap.stamp(3);

        assert!(
            heap.acquire(1024).is_none(),
            "ring must be full with three stamped regions"
        );
        assert_eq!(heap.in_flight_count(), 3);

        // Before reclaim at epoch 1: still full.
        assert!(
            heap.acquire(1024).is_none(),
            "ring must stay full before reclaim"
        );

        let reclaimed = heap.reclaim(1);
        assert_eq!(reclaimed, 1, "one region should reclaim at epoch 1");

        let o4 = heap.acquire(1024).expect("space available after reclaim");
        assert_eq!(o4, 0, "reclaimed region should wrap to offset 0");
        assert_eq!(heap.in_flight_count(), 3, "two old + one new in flight");
    }

    /// `Device::boundary_crossed` drives placement-heap ring reclaim for device-owned heaps.
    #[test]
    fn u7_boundary_crossed_reclaims_placement_heap_ring() {
        let device = test_device();
        let mut heap = PlacementHeap::new(&device, 3 * 1024, 1024).unwrap();

        let _o1 = heap.acquire(1024).unwrap();
        heap.stamp(1);
        let _o2 = heap.acquire(1024).unwrap();
        heap.stamp(2);
        let _o3 = heap.acquire(1024).unwrap();
        heap.stamp(3);
        assert_eq!(heap.in_flight_count(), 3);

        *device.inner.placement_heap.lock().unwrap() = Some(heap);

        device.boundary_crossed(1);

        let stats = device
            .placement_heap_stats()
            .expect("device-owned placement heap");
        assert_eq!(
            stats.in_flight_count, 2,
            "boundary_crossed(epoch=1) must reclaim one ring region"
        );
    }

    /// `poll_signals_and_service` routes `BoundaryCrossed` into `boundary_crossed(epoch)`.
    #[test]
    fn u3_signal_boundary_crossed_services_vram_ring() {
        let device = test_device();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).expect("submit");

        let alive = Arc::new(42u32);
        let weak = Arc::downgrade(&alive);
        device.defer_until(tv, alive);
        assert!(device.has_deferred_payloads());

        let signals = device.poll_signals_and_service();
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, Signal::BoundaryCrossed { epoch } if *epoch == tv)),
            "submit should post BoundaryCrossed for epoch {tv}"
        );
        assert!(
            !device.has_deferred_payloads(),
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
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).expect("submit");

        let alive = Arc::new(77u32);
        let weak = Arc::downgrade(&alive);
        device.defer_until(tv, alive);
        assert!(device.has_deferred_payloads());

        device.wait_until(tv).expect("wait");
        device.flush_deferred_deletions();

        assert!(
            !device.has_deferred_payloads(),
            "VRAM ring must empty via pull path without polling signals"
        );
        assert!(
            weak.upgrade().is_none(),
            "payload must drop after pull flush"
        );
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
        let mut alloc = BumpResetAllocator::new(&device, small_config()).expect("create");

        alloc.begin_frame(&device, 0).expect("begin 1");
        let v1 = alloc.alloc(&device, 512, Some(4)).expect("alloc");
        let original_offset = v1.offset();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).expect("submit");

        // Retire to a far epoch the GPU has NOT reached.
        let far_epoch = tv + 100;
        alloc.end_frame(&device, far_epoch);

        // begin_frame must wait on far_epoch (mock wait_until advances progress),
        // then reset. If the wait were skipped, the pool would reset while the
        // GPU is still using the buffer — a use-after-free.
        alloc.begin_frame(&device, 0).expect("begin 2");

        // After reset the bump pointer is at 0, so the next alloc should land at
        // the same offset as the first (confirming reset happened).
        let v2 = alloc
            .alloc(&device, 512, Some(4))
            .expect("alloc after reset");
        assert_eq!(
            v2.offset(),
            original_offset,
            "bump should reset to 0 after wait + begin_frame"
        );
        // gpu_progress should now be >= far_epoch (mock wait_until advanced it).
        assert!(
            device.gpu_progress() >= far_epoch,
            "begin_frame must have called wait_until to advance progress"
        );
    }
}
