//! Smoke tests for [`goldy::FrameOrchestrator`] and [`goldy::Device::submit_pipelined`].

use goldy::{DeviceDescriptor, FrameOrchestrator, Instance, RequestAdapterOptions, TaskGraph};

#[test]
fn orchestrator_double_begin_fails() {
    let instance = Instance::new().expect("instance");
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("device");
    let ctx = device.create_context().expect("context");

    let mut orch: FrameOrchestrator<()> = FrameOrchestrator::new(&ctx, 3);
    let h = orch
        .begin_frame(|_d, _r| Ok::<_, std::convert::Infallible>(()))
        .expect("begin");

    assert!(orch
        .begin_frame(|_d, _r| Ok::<_, std::convert::Infallible>(()))
        .is_err());

    orch.end_frame_standalone(h, &mut TaskGraph::new(), None, ())
        .expect("end");
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

    let mut orch: FrameOrchestrator<()> = FrameOrchestrator::new(&ctx, 2);
    orch.reclaim(|_d, _r| Ok::<_, std::convert::Infallible>(())).unwrap();
}
