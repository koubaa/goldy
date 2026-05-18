//! Smoke tests for [`goldy::FrameOrchestrator`] and [`goldy::Device::submit_pipelined`].

use goldy::{DeviceType, FrameOrchestrator, Instance, TaskGraph};

#[test]
fn orchestrator_double_begin_fails() {
    let instance = Instance::new().expect("instance");
    let device = instance.create_device(DeviceType::Cpu).expect("cpu device");

    let mut orch: FrameOrchestrator<()> = FrameOrchestrator::new(&device, 3);
    let h = orch
        .begin_frame(|_d, _r| Ok::<_, std::convert::Infallible>(()))
        .expect("begin");

    assert!(orch
        .begin_frame(|_d, _r| Ok::<_, std::convert::Infallible>(()))
        .is_err());

    orch.end_frame_standalone(h, TaskGraph::new(), None, ())
        .expect("end");
}

#[test]
fn orchestrator_reclaim_empty_is_ok() {
    let instance = Instance::new().expect("instance");
    let device = instance.create_device(DeviceType::Cpu).expect("cpu device");

    let mut orch: FrameOrchestrator<()> = FrameOrchestrator::new(&device, 2);
    orch.reclaim(|_d, _r| Ok::<_, std::convert::Infallible>(()))
        .unwrap();
}
