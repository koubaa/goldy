//! Smoke tests for [`goldy::FrameOrchestrator`] end-frame paths.

use goldy::{DeviceDescriptor, FrameOrchestrator, Instance, RequestAdapterOptions};

#[test]
fn orchestrator_double_begin_fails() {
    let instance = Instance::new().expect("instance");
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("device");
    let ctx = device.create_context().expect("context");

    let mut orch = FrameOrchestrator::new(&ctx, 3);
    let h = orch.begin_frame().expect("begin");

    assert!(orch.begin_frame().is_err());

    orch.end_frame_externally_ordered(h).expect("end");
}

#[test]
fn orchestrator_reclaim_empty_is_ok() {
    let instance = Instance::new().expect("instance");
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("device");
    let ctx = device.create_context().expect("context");

    let mut orch = FrameOrchestrator::new(&ctx, 2);
    orch.reclaim().unwrap();
}

#[test]
fn end_frame_externally_ordered_leaves_ring_empty() {
    let instance = Instance::new().expect("instance");
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("device");
    let ctx = device.create_context().expect("context");

    let mut orch = FrameOrchestrator::new(&ctx, 1);
    let h = orch.begin_frame().expect("begin");
    orch.end_frame_externally_ordered(h).expect("end externally");
    assert_eq!(orch.pending_frames(), 0);
    assert!(!orch.has_open_frame());

    // Next begin must not block: no retirement slot was created.
    let h2 = orch.begin_frame().expect("begin 2");
    orch.end_frame_externally_ordered(h2).expect("end 2");
    assert_eq!(orch.pending_frames(), 0);
}
