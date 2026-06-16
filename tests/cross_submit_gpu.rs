//! GPU integration tests for epoch-driven cross-scheme synchronization.
//!
//! Exact integer readback assertions (no FLIP). Gated on real backends.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    types::ResourceAccess, BufferKind, ComputePipeline, Device, DeviceDescriptor, Grant, Instance, NodeAccess,
    ReadGrant, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, Submission,
};
use std::sync::Arc;
use submission::submission_context;

fn make_device() -> Device {
    let instance = Instance::new().expect("instance");
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("device")
}

const INC_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(Scattered<uint> buf, ThreadId id) {
    buf[id.x] = buf[id.x] + 1u;
}
"#;

const READ_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(BufRO<uint> buf, ThreadId id) {
    let _ = buf[id.x];
}
"#;

const OVERWRITE_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(Scattered<uint> buf, ThreadId id) {
    buf[id.x] = 42u;
}
"#;

fn read_u32(grant: &ReadGrant<goldy::GrantBuffer>, submission: &Submission) -> u32 {
    let loan = grant.consume(submission).expect("grant");
    bytemuck::cast_slice::<u8, u32>(&loan)[0]
}

#[test]
fn saxpy_style_chain_closed_form() {
    let device = make_device();
    let ctx = submission_context(&device);
    let shader = ShaderModule::from_slang(&device, INC_SHADER).expect("shader");
    let pipe = ComputePipeline::new(&device, &shader).expect("pipe");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
        .expect("buf");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("inc", &pipe)
        .bind_parcel(&buf, NodeAccess::ReadWrite)
        .bind_views(&[buf.handle(ResourceAccess::Write).expect("uav")])
        .dispatch(1, 1, 1);
    let grant = scheme.grant_read(&buf).expect("grant");

    const STEPS: u32 = 50;
    for _ in 0..STEPS {
        scheme.submit().expect("submit");
    }
    let submission = scheme.submit().expect("final");
    assert_eq!(read_u32(&grant, &submission), STEPS + 1);
}

#[test]
fn war_write_after_read_pipelined_overwrite() {
    let device = make_device();
    let ctx = submission_context(&device);
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("shader");
    let write_shader = ShaderModule::from_slang(&device, OVERWRITE_SHADER).expect("shader");
    let read_pipe = ComputePipeline::new(&device, &read_shader).expect("pipe");
    let write_pipe = ComputePipeline::new(&device, &write_shader).expect("pipe");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer_with_data(&[7u32; 1], BufferKind::Scattered)
        .expect("buf");

    // Independent schemes, no CPU wait — writer must synchronize against the reader's epoch (WAR).
    let mut reader = Scheme::new(&ctx);
    reader
        .node("read", &read_pipe)
        .bind_parcel(&buf, NodeAccess::Read)
        .bind_views(&[buf.handle(ResourceAccess::Read).expect("srv")])
        .dispatch(1, 1, 1);
    reader.submit().expect("read");

    let mut writer = Scheme::new(&ctx);
    writer
        .node("write", &write_pipe)
        .bind_parcel(&buf, NodeAccess::Write)
        .bind_views(&[buf.handle(ResourceAccess::Write).expect("uav")])
        .dispatch(1, 1, 1);
    let grant = writer.grant_read(&buf).expect("grant");
    let submission = writer.submit().expect("write");

    assert_eq!(read_u32(&grant, &submission), 42);
}
