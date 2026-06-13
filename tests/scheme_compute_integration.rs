//! Scheme compute integration tests — migrated from `TaskGraph` coverage.
//!
//! Retained worker schemes: parcels + `bind_resources_typed` + [`Scheme::submit`].
//! CPU→GPU parcel writes (including zero-fills) use [`upload::write_to_parcel`] — a
//! separate upload submission per call (one-node upload [`Scheme`] on `ctx`),
//! serialized against worker schemes by queue order. Callers do not use
//! `TaskGraph::clear_buffer` or `TaskGraph::write_buffer` directly.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

#[path = "common/submission.rs"]
mod submission;
#[path = "common/upload.rs"]
mod upload;

use goldy::{
    types::{BufferFlags, ResourceAccess, TextureFlags, TextureFormat, TextureKind},
    BufferKind, ComputePipeline, Device, DeviceDescriptor, Instance, NodeAccess, Parcel, RequestAdapterOptions,
    RetainedPool, Sampler, Scheme, ShaderModule, StructuredBufferElement,
};
use std::sync::Arc;
use submission::submission_context;
use upload::write_to_parcel;

fn make_device() -> Device {
    let instance = Instance::new().expect("Failed to create instance");

    #[cfg(all(feature = "dx12", target_os = "windows"))]
    if std::env::var("GOLDY_DX12_ALLOW_WARP").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")) {
        if let Ok(adapter) = instance.request_adapter(&RequestAdapterOptions {
            power_preference: goldy::PowerPreference::None,
            force_fallback_adapter: true,
        }) {
            if let Ok(dev) = adapter.request_device(&DeviceDescriptor::default()) {
                return dev;
            }
        }
    }

    instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("Failed to request adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("Failed to create device")
}

fn readback_parcel_u32(device: &Device, parcel: &Parcel, count: usize) -> Vec<u32> {
    let mut output = vec![0u8; count * 4];
    parcel.read_to_cpu(device, &mut output).expect("readback");
    bytemuck::cast_slice(&output).to_vec()
}

fn write_zeros_to_parcel(ctx: &goldy::Context, parcel: &Parcel, byte_len: usize) {
    // Upload micro-scheme: separate from worker Scheme.
    write_to_parcel(ctx, parcel, &vec![0u8; byte_len]).expect("write_to_parcel zeros");
}

const DOUBLE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x] * 2;
}
"#;

const IN_PLACE_DOUBLE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] * 2;
}
"#;

const ADD_TEN_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + 10;
}
"#;

const FILL_42_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = 42;
}
"#;

const FILL_99_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = 99;
}
"#;

const SUM_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> a, Scattered<uint> b, Scattered<uint> out, ThreadId id) {
    out[id.x] = a[id.x] + b[id.x];
}
"#;

const COPY_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x];
}
"#;

const INCREMENT_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + 1;
}
"#;

const FILL_SRC_WITH_INDEX_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = id.x;
}
"#;

const WRITE_IOTA_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    if (id.x < 64) data[id.x] = id.x + 1;
}
"#;

const SIX_SLOT_SUM_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> a, Scattered<uint> b, Scattered<uint> c,
             Scattered<uint> d, Scattered<uint> e, Scattered<uint> out,
             ThreadId id) {
    uint idx = id.x;
    if (idx >= 16) return;
    out[idx] = a[idx] + b[idx] + c[idx] + d[idx] + e[idx];
}
"#;

const MINIMAL_SHADER: &str = r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
}
"#;

// ---------------------------------------------------------------------------
// Migrated from task_graph_integration.rs
// ---------------------------------------------------------------------------

#[test]
fn scheme_graph_linear_chain() {
    let device = make_device();
    let ctx = submission_context(&device);

    let double_pipe =
        ComputePipeline::new(&device, &ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap()).unwrap();
    let add_pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap()).unwrap();

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let src = pool
        .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
        .unwrap();
    let dst = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("double", &double_pipe)
        .bind_parcel(&src, NodeAccess::Read)
        .bind_parcel(&dst, NodeAccess::Write)
        .bind_resources_typed(&[
            src.handle(ResourceAccess::ReadWrite).unwrap(),
            dst.handle(ResourceAccess::Write).unwrap(),
        ])
        .dispatch(1, 1, 1);
    scheme
        .node("add_ten", &add_pipe)
        .bind_parcel(&dst, NodeAccess::ReadWrite)
        .bind_resources_typed(&[dst.handle(ResourceAccess::ReadWrite).unwrap()])
        .dispatch(1, 1, 1);

    scheme.submit().unwrap();
    let frame = scheme.submit().unwrap();
    assert_eq!(scheme.replay_stats().records, 1, "linear chain records once");
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        scheme.replay_stats().resubmit_hits,
        1,
        "second submit must resubmit without re-record"
    );
    frame.wait(&ctx).unwrap();

    let result = readback_parcel_u32(&device, &dst, 64);
    for (i, &val) in result.iter().enumerate() {
        let expected = (i as u32) * 2 + 10;
        assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
    }
}

#[test]
fn scheme_graph_independent_dispatches() {
    let device = make_device();
    let ctx = submission_context(&device);

    let pipe_42 = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap()).unwrap();
    let pipe_99 = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, FILL_99_SHADER).unwrap()).unwrap();

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf_a = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();
    let buf_b = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("fill_a", &pipe_42)
        .bind_parcel(&buf_a, NodeAccess::Write)
        .bind_resources_typed(&[buf_a.handle(ResourceAccess::Write).unwrap()])
        .dispatch(1, 1, 1);
    scheme
        .node("fill_b", &pipe_99)
        .bind_parcel(&buf_b, NodeAccess::Write)
        .bind_resources_typed(&[buf_b.handle(ResourceAccess::Write).unwrap()])
        .dispatch(1, 1, 1);

    let frame = scheme.submit().unwrap();
    frame.wait(&ctx).unwrap();

    for &v in &readback_parcel_u32(&device, &buf_a, 64) {
        assert_eq!(v, 42);
    }
    for &v in &readback_parcel_u32(&device, &buf_b, 64) {
        assert_eq!(v, 99);
    }
}

#[test]
fn scheme_graph_diamond_dependency() {
    let device = make_device();
    let ctx = submission_context(&device);

    let fill_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, FILL_SRC_WITH_INDEX_SHADER).unwrap(),
    )
    .unwrap();
    let double_pipe =
        ComputePipeline::new(&device, &ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap()).unwrap();
    let sum_pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, SUM_SHADER).unwrap()).unwrap();

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let src = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();
    let y = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();
    let z = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();
    let out = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();

    let src_rw = src.handle(ResourceAccess::ReadWrite).unwrap();
    let y_w = y.handle(ResourceAccess::Write).unwrap();
    let z_w = z.handle(ResourceAccess::Write).unwrap();
    let out_w = out.handle(ResourceAccess::Write).unwrap();

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("fill_src", &fill_pipe)
        .bind_parcel(&src, NodeAccess::Write)
        .bind_resources_typed(&[src_rw])
        .dispatch(1, 1, 1);
    scheme
        .node("double_to_y", &double_pipe)
        .bind_parcel(&src, NodeAccess::Read)
        .bind_parcel(&y, NodeAccess::Write)
        .bind_resources_typed(&[src_rw, y_w])
        .dispatch(1, 1, 1);
    scheme
        .node("double_to_z", &double_pipe)
        .bind_parcel(&src, NodeAccess::Read)
        .bind_parcel(&z, NodeAccess::Write)
        .bind_resources_typed(&[src_rw, z_w])
        .dispatch(1, 1, 1);
    scheme
        .node("sum_yz", &sum_pipe)
        .bind_parcel(&y, NodeAccess::Read)
        .bind_parcel(&z, NodeAccess::Read)
        .bind_parcel(&out, NodeAccess::Write)
        .bind_resources_typed(&[y_w, z_w, out_w])
        .dispatch(1, 1, 1);

    let frame = scheme.submit().unwrap();
    assert_eq!(scheme.replay_stats().records, 1, "diamond records once");
    frame.wait(&ctx).unwrap();

    let result = readback_parcel_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        let expected = (i as u32) * 4;
        assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
    }
}

#[test]
fn scheme_graph_fill_readback() {
    let device = make_device();
    let ctx = submission_context(&device);

    let pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap()).unwrap();

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("fill", &pipe)
        .bind_parcel(&buf, NodeAccess::Write)
        .bind_resources_typed(&[buf.handle(ResourceAccess::Write).unwrap()])
        .dispatch(1, 1, 1);

    let frame = scheme.submit().unwrap();
    frame.wait(&ctx).unwrap();

    for &v in &readback_parcel_u32(&device, &buf, 64) {
        assert_eq!(v, 42);
    }
}

#[test]
fn scheme_zeros_then_dispatch_reads_zeros() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer_with_data(&(1..=64).collect::<Vec<u32>>(), BufferKind::Scattered)
        .unwrap();
    let out = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();

    write_zeros_to_parcel(&ctx, &buf, 64 * 4);

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("copy", &copy_pipe)
        .bind_parcel(&buf, NodeAccess::Read)
        .bind_parcel(&out, NodeAccess::Write)
        .bind_resources_typed(&[
            buf.handle(ResourceAccess::ReadWrite).unwrap(),
            out.handle(ResourceAccess::Write).unwrap(),
        ])
        .dispatch(1, 1, 1);

    let frame = scheme.submit().unwrap();
    frame.wait(&ctx).unwrap();

    for (i, &val) in readback_parcel_u32(&device, &out, 64).iter().enumerate() {
        assert_eq!(val, 0, "element {i}: expected 0 after zero write, got {val}");
    }
}

#[test]
fn scheme_write_then_dispatch_reads_uploaded_data() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();
    let out = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();

    let known_data: Vec<u32> = (100..164).collect();
    write_to_parcel(&ctx, &buf, bytemuck::cast_slice(&known_data)).expect("write_to_parcel");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("copy", &copy_pipe)
        .bind_parcel(&buf, NodeAccess::Read)
        .bind_parcel(&out, NodeAccess::Write)
        .bind_resources_typed(&[
            buf.handle(ResourceAccess::ReadWrite).unwrap(),
            out.handle(ResourceAccess::Write).unwrap(),
        ])
        .dispatch(1, 1, 1);

    let frame = scheme.submit().unwrap();
    frame.wait(&ctx).unwrap();

    for (i, &val) in readback_parcel_u32(&device, &out, 64).iter().enumerate() {
        assert_eq!(val, known_data[i], "element {i}");
    }
}

#[test]
fn scheme_stress_zeros_then_dispatch_large() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

    const N: usize = 16384;
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer_with_data(&(1..=N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
        .unwrap();
    let out = pool
        .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();

    write_zeros_to_parcel(&ctx, &buf, N * 4);

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("copy", &copy_pipe)
        .bind_parcel(&buf, NodeAccess::Read)
        .bind_parcel(&out, NodeAccess::Write)
        .bind_resources_typed(&[
            buf.handle(ResourceAccess::ReadWrite).unwrap(),
            out.handle(ResourceAccess::Write).unwrap(),
        ])
        .dispatch((N / 64) as u32, 1, 1);

    let frame = scheme.submit().unwrap();
    frame.wait(&ctx).unwrap();

    let nonzero_count = readback_parcel_u32(&device, &out, N)
        .iter()
        .filter(|&&v| v != 0)
        .count();
    assert_eq!(nonzero_count, 0, "expected all zeros after zero write");
}

#[test]
fn scheme_stress_many_zero_writes_many_dispatches() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

    const N: usize = 1024;
    const NUM_BUFS: usize = 8;
    let mut pool = RetainedPool::new(Arc::new(device.clone()));

    let mut srcs = Vec::new();
    let mut outs = Vec::new();
    for _ in 0..NUM_BUFS {
        srcs.push(
            pool.acquire_buffer_with_data(&(1..=N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
                .unwrap(),
        );
        outs.push(
            pool.acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
                .unwrap(),
        );
    }

    for src in &srcs {
        write_zeros_to_parcel(&ctx, src, N * 4);
    }

    let mut scheme = Scheme::new(&ctx);
    for (src, out) in srcs.iter().zip(outs.iter()) {
        scheme
            .node("copy", &copy_pipe)
            .bind_parcel(src, NodeAccess::Read)
            .bind_parcel(out, NodeAccess::Write)
            .bind_resources_typed(&[
                src.handle(ResourceAccess::ReadWrite).unwrap(),
                out.handle(ResourceAccess::Write).unwrap(),
            ])
            .dispatch((N / 64) as u32, 1, 1);
    }

    let frame = scheme.submit().unwrap();
    frame.wait(&ctx).unwrap();

    for (i, out) in outs.iter().enumerate() {
        let nonzero_count = readback_parcel_u32(&device, out, N).iter().filter(|&&v| v != 0).count();
        assert_eq!(nonzero_count, 0, "buffer {i}: expected all zeros after zero write");
    }
}

#[test]
fn scheme_stress_write_then_dispatch_chain() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();

    const N: usize = 1024;
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer_with_data(&(1..=N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
        .unwrap();
    let out = pool
        .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();

    let known_data: Vec<u32> = (0..N as u32).map(|i| i * 7 + 42).collect();
    write_to_parcel(&ctx, &buf, bytemuck::cast_slice(&known_data)).expect("write_to_parcel");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("copy", &copy_pipe)
        .bind_parcel(&buf, NodeAccess::Read)
        .bind_parcel(&out, NodeAccess::Write)
        .bind_resources_typed(&[
            buf.handle(ResourceAccess::ReadWrite).unwrap(),
            out.handle(ResourceAccess::Write).unwrap(),
        ])
        .dispatch((N / 64) as u32, 1, 1);

    let frame = scheme.submit().unwrap();
    frame.wait(&ctx).unwrap();

    for (i, &val) in readback_parcel_u32(&device, &out, N).iter().enumerate() {
        assert_eq!(val, known_data[i], "element {i}");
    }
}

#[test]
fn scheme_stress_two_phase_submission() {
    let device = make_device();
    let ctx = submission_context(&device);

    let double_pipe =
        ComputePipeline::new(&device, &ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap()).unwrap();
    let add_pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap()).unwrap();

    const N: usize = 4096;
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer_with_data(&(0..N as u32).collect::<Vec<u32>>(), BufferKind::Scattered)
        .unwrap();
    let tmp = pool
        .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();

    let buf_rw = buf.handle(ResourceAccess::ReadWrite).unwrap();
    let tmp_rw = tmp.handle(ResourceAccess::ReadWrite).unwrap();

    {
        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("double", &double_pipe)
            .bind_parcel(&buf, NodeAccess::Read)
            .bind_parcel(&tmp, NodeAccess::Write)
            .bind_resources_typed(&[buf_rw, tmp_rw])
            .dispatch((N / 64) as u32, 1, 1);
        let frame = scheme.submit().unwrap();
        frame.wait(&ctx).unwrap();
    }

    {
        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("add_ten", &add_pipe)
            .bind_parcel(&tmp, NodeAccess::ReadWrite)
            .bind_resources_typed(&[tmp_rw])
            .dispatch((N / 64) as u32, 1, 1);
        let frame = scheme.submit().unwrap();
        frame.wait(&ctx).unwrap();
    }

    for (i, &val) in readback_parcel_u32(&device, &tmp, N).iter().enumerate() {
        let expected = (i as u32) * 2 + 10;
        assert_eq!(val, expected, "element {i}");
    }
}

#[test]
fn scheme_stress_rapid_submissions() {
    let device = make_device();
    let ctx = submission_context(&device);

    let add_pipe = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap()).unwrap();

    const N: usize = 256;
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer_with_data(&vec![0u32; N], BufferKind::Scattered)
        .unwrap();

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("add_ten", &add_pipe)
        .bind_parcel(&buf, NodeAccess::ReadWrite)
        .bind_resources_typed(&[buf.handle(ResourceAccess::ReadWrite).unwrap()])
        .dispatch((N / 64) as u32, 1, 1);

    const ROUNDS: u32 = 20;
    let mut last_frame = None;
    for _ in 0..ROUNDS {
        last_frame = Some(scheme.submit().unwrap());
    }
    last_frame.expect("submit").wait(&ctx).unwrap();

    assert_eq!(scheme.replay_stats().records, 1, "rapid submissions record once");
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        scheme.replay_stats().resubmit_hits,
        u64::from(ROUNDS) - 1,
        "remaining submits are retention hits"
    );

    let expected = ROUNDS * 10;
    for (i, &val) in readback_parcel_u32(&device, &buf, N).iter().enumerate() {
        assert_eq!(val, expected, "element {i}");
    }
}

// ---------------------------------------------------------------------------
// Migrated from compute_integration.rs
// ---------------------------------------------------------------------------

#[test]
fn scheme_compute_dispatch_empty() {
    let device = make_device();
    let ctx = submission_context(&device);

    let pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, MINIMAL_SHADER).expect("shader"),
    )
    .expect("pipeline");

    let mut scheme = Scheme::new(&ctx);
    scheme.node("n0", &pipe).dispatch(1, 1, 1);
    scheme.submit().expect("submit");
}

#[test]
fn scheme_compute_write_and_readback() {
    let device = make_device();
    let ctx = submission_context(&device);

    let pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, IN_PLACE_DOUBLE_SHADER).expect("shader"),
    )
    .expect("pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buffer = pool
        .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
        .expect("buffer");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("double", &pipe)
        .bind_parcel(&buffer, NodeAccess::ReadWrite)
        .bind_resources_typed(&[buffer.handle(ResourceAccess::ReadWrite).expect("handle")])
        .dispatch(1, 1, 1);
    let frame = scheme.submit().expect("submit");
    frame.wait(&ctx).expect("wait");

    for (i, &val) in readback_parcel_u32(&device, &buffer, 64).iter().enumerate() {
        assert_eq!(val, (i as u32) * 2, "element {i}");
    }
}

#[test]
fn scheme_compute_with_uav_parcel() {
    let device = make_device();
    let ctx = submission_context(&device);

    let pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, IN_PLACE_DOUBLE_SHADER).expect("shader"),
    )
    .expect("pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buffer = pool
        .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
        .expect("buffer");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("double", &pipe)
        .bind_parcel(&buffer, NodeAccess::ReadWrite)
        .bind_resources_typed(&[buffer.handle(ResourceAccess::ReadWrite).expect("handle")])
        .dispatch(1, 1, 1);
    let frame = scheme.submit().expect("submit");
    frame.wait(&ctx).expect("wait");

    for (i, &val) in readback_parcel_u32(&device, &buffer, 64).iter().enumerate() {
        assert_eq!(val, (i as u32) * 2, "element {i}");
    }
}

#[test]
fn scheme_compute_with_srv_and_uav_parcels() {
    let device = make_device();
    let ctx = submission_context(&device);

    let pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
    )
    .expect("pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let input = pool
        .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
        .expect("input");
    let output = pool
        .acquire_buffer_with_data(&vec![0u32; 64], BufferKind::Scattered)
        .expect("output");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("copy", &pipe)
        .bind_parcel(&input, NodeAccess::Read)
        .bind_parcel(&output, NodeAccess::Write)
        .bind_resources_typed(&[
            input.handle(ResourceAccess::ReadWrite).expect("in"),
            output.handle(ResourceAccess::Write).expect("out"),
        ])
        .dispatch(1, 1, 1);
    let frame = scheme.submit().expect("submit");
    frame.wait(&ctx).expect("wait");

    let input_vals = readback_parcel_u32(&device, &input, 64);
    let output_vals = readback_parcel_u32(&device, &output, 64);
    assert_eq!(output_vals, input_vals, "copy must reproduce input in output");
}

#[test]
fn scheme_parcel_write_zeros_full() {
    let device = make_device();
    let ctx = submission_context(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let parcel = pool
        .acquire_buffer_with_data(&vec![0xDEAD_BEEFu32; 64], BufferKind::Scattered)
        .expect("parcel");

    let token = write_to_parcel(&ctx, &parcel, &vec![0u8; 64 * 4]).expect("write_to_parcel zeros");
    token.wait(&ctx).expect("wait");

    for (i, &val) in readback_parcel_u32(&device, &parcel, 64).iter().enumerate() {
        assert_eq!(val, 0, "element {i} should be 0 after zero write");
    }
}

#[test]
fn scheme_parcel_write_zeros_partial() {
    let device = make_device();
    let ctx = submission_context(&device);

    const SENTINEL: u32 = 0xDEAD_BEEF;
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let parcel = pool
        .acquire_buffer_with_data(&vec![SENTINEL; 64], BufferKind::Scattered)
        .expect("parcel");

    let mut data = vec![SENTINEL; 64];
    for slot in data.iter_mut().take(64).skip(16).take(16) {
        *slot = 0;
    }
    let token = write_to_parcel(&ctx, &parcel, bytemuck::cast_slice(&data)).expect("write_to_parcel");
    token.wait(&ctx).expect("wait");

    for (i, &val) in readback_parcel_u32(&device, &parcel, 64).iter().enumerate() {
        let expected = if (16..32).contains(&i) { 0 } else { SENTINEL };
        assert_eq!(val, expected, "element {i}");
    }
}

#[test]
fn scheme_parcel_write_zeros_to_end() {
    let device = make_device();
    let ctx = submission_context(&device);

    const SENTINEL: u32 = 0xCAFE_BABE;
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let parcel = pool
        .acquire_buffer_with_data(&vec![SENTINEL; 64], BufferKind::Scattered)
        .expect("parcel");

    let mut data = vec![SENTINEL; 64];
    for slot in data.iter_mut().skip(32) {
        *slot = 0;
    }
    let token = write_to_parcel(&ctx, &parcel, bytemuck::cast_slice(&data)).expect("write_to_parcel");
    token.wait(&ctx).expect("wait");

    for (i, &val) in readback_parcel_u32(&device, &parcel, 64).iter().enumerate() {
        let expected = if i < 32 { SENTINEL } else { 0 };
        assert_eq!(val, expected, "element {i}");
    }
}

#[test]
fn scheme_zeros_before_copy_dispatch() {
    let device = make_device();
    let ctx = submission_context(&device);

    let pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
    )
    .expect("pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let input = pool
        .acquire_buffer_with_data(&vec![0xDEAD_BEEFu32; 64], BufferKind::Scattered)
        .expect("input");
    let output = pool
        .acquire_buffer_with_data(&vec![0xFFFF_FFFFu32; 64], BufferKind::Scattered)
        .expect("output");

    write_zeros_to_parcel(&ctx, &input, 64 * 4);

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("copy", &pipe)
        .bind_parcel(&input, NodeAccess::Read)
        .bind_parcel(&output, NodeAccess::Write)
        .bind_resources_typed(&[
            input.handle(ResourceAccess::ReadWrite).expect("in"),
            output.handle(ResourceAccess::Write).expect("out"),
        ])
        .dispatch(1, 1, 1);
    let frame = scheme.submit().expect("submit");
    frame.wait(&ctx).expect("wait");

    for (i, &val) in readback_parcel_u32(&device, &output, 64).iter().enumerate() {
        assert_eq!(val, 0, "output[{i}] should be 0 (copied from zeroed input)");
    }
}

/// GPU ordering: copy scheme writes 42s → `write_to_parcel` zeros output → increment scheme.
/// Correct result is 1 (0 + 1). Cross-scheme serialization replaces in-graph `clear_buffer`.
#[test]
fn scheme_write_to_parcel_zeros_between_submissions() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
    )
    .expect("copy pipeline");
    let inc_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, INCREMENT_SHADER).expect("shader"),
    )
    .expect("inc pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let input = pool
        .acquire_buffer_with_data(&vec![42u32; 64], BufferKind::Scattered)
        .expect("input");
    let output = pool
        .acquire_buffer_with_data(&vec![0u32; 64], BufferKind::Scattered)
        .expect("output");

    let out_rw = output.handle(ResourceAccess::ReadWrite).expect("out");

    {
        let mut copy_scheme = Scheme::new(&ctx);
        copy_scheme
            .node("copy", &copy_pipe)
            .bind_parcel(&input, NodeAccess::Read)
            .bind_parcel(&output, NodeAccess::Write)
            .bind_resources_typed(&[
                input.handle(ResourceAccess::ReadWrite).expect("in"),
                output.handle(ResourceAccess::Write).expect("out w"),
            ])
            .dispatch(1, 1, 1);
        copy_scheme.submit().expect("copy submit 0");
        copy_scheme.submit().expect("copy submit 1");
        assert_eq!(copy_scheme.replay_stats().records, 1, "copy scheme records once");
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            copy_scheme.replay_stats().resubmit_hits,
            1,
            "second copy submit must resubmit without re-record"
        );
    }

    // No wait here: zero write must serialize after copy via queue order alone.
    write_zeros_to_parcel(&ctx, &output, 64 * 4);

    let frame = {
        let mut inc_scheme = Scheme::new(&ctx);
        inc_scheme
            .node("inc", &inc_pipe)
            .bind_parcel(&output, NodeAccess::ReadWrite)
            .bind_resources_typed(&[out_rw])
            .dispatch(1, 1, 1);
        inc_scheme.submit().expect("inc submit")
    };
    frame.wait(&ctx).expect("wait after inc");

    for (i, &val) in readback_parcel_u32(&device, &output, 64).iter().enumerate() {
        assert_eq!(
            val, 1,
            "output[{i}]: expected 1 (write_to_parcel zeroed before increment), got {val}"
        );
    }
}

/// Cross-scheme ordering: an upload micro-scheme may return its [`SchemeFrame`] without
/// waiting; the next worker [`Scheme::submit`] on the same context still sees the upload.
#[test]
fn scheme_upload_frame_unwaited_serializes_before_worker_submit() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
    )
    .expect("pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let input = pool
        .acquire_buffer_with_data(&vec![0xDEAD_BEEFu32; 64], BufferKind::Scattered)
        .expect("input");
    let output = pool
        .acquire_buffer_with_data(&vec![0u32; 64], BufferKind::Scattered)
        .expect("output");

    const PATTERN: u32 = 0xCAFE_BABE;
    let upload_data = vec![PATTERN; 64];
    let _upload_frame = write_to_parcel(&ctx, &input, bytemuck::cast_slice(&upload_data)).expect("write_to_parcel");
    // Deliberately no wait on upload frame — queue order must serialize before worker submit.

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("copy", &copy_pipe)
        .bind_parcel(&input, NodeAccess::Read)
        .bind_parcel(&output, NodeAccess::Write)
        .bind_resources_typed(&[
            input.handle(ResourceAccess::ReadWrite).expect("in"),
            output.handle(ResourceAccess::Write).expect("out"),
        ])
        .dispatch(1, 1, 1);
    let worker_frame = scheme.submit().expect("worker submit");
    worker_frame.wait(&ctx).expect("wait on worker frame only");

    for (i, &val) in readback_parcel_u32(&device, &output, 64).iter().enumerate() {
        assert_eq!(
            val, PATTERN,
            "output[{i}]: upload must be visible without waiting on upload frame"
        );
    }
}

#[test]
fn scheme_compute_many_resource_slots() {
    let device = make_device();
    let ctx = submission_context(&device);

    let pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, SIX_SLOT_SUM_SHADER).expect("shader"),
    )
    .expect("pipeline");

    const N: usize = 16;
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let a = pool
        .acquire_buffer_with_data(&[1u32; N], BufferKind::Scattered)
        .expect("a");
    let b = pool
        .acquire_buffer_with_data(&[2u32; N], BufferKind::Scattered)
        .expect("b");
    let c = pool
        .acquire_buffer_with_data(&[3u32; N], BufferKind::Scattered)
        .expect("c");
    let d = pool
        .acquire_buffer_with_data(&[4u32; N], BufferKind::Scattered)
        .expect("d");
    let e = pool
        .acquire_buffer_with_data(&[5u32; N], BufferKind::Scattered)
        .expect("e");
    let out = pool
        .acquire_buffer_with_data(&[0u32; N], BufferKind::Scattered)
        .expect("out");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("sum", &pipe)
        .bind_parcel(&a, NodeAccess::Read)
        .bind_parcel(&b, NodeAccess::Read)
        .bind_parcel(&c, NodeAccess::Read)
        .bind_parcel(&d, NodeAccess::Read)
        .bind_parcel(&e, NodeAccess::Read)
        .bind_parcel(&out, NodeAccess::Write)
        .bind_resources_typed(&[
            a.handle(ResourceAccess::ReadWrite).expect("a"),
            b.handle(ResourceAccess::ReadWrite).expect("b"),
            c.handle(ResourceAccess::ReadWrite).expect("c"),
            d.handle(ResourceAccess::ReadWrite).expect("d"),
            e.handle(ResourceAccess::ReadWrite).expect("e"),
            out.handle(ResourceAccess::Write).expect("out"),
        ])
        .dispatch(1, 1, 1);

    let frame = scheme.submit().expect("submit");
    frame.wait(&ctx).expect("wait");

    for (i, &val) in readback_parcel_u32(&device, &out, N).iter().enumerate() {
        assert_eq!(val, 15, "out[{i}] expected 15, got {val}");
    }
}

#[test]
fn scheme_regular_buffer_write_then_copy() {
    let device = make_device();
    let ctx = submission_context(&device);

    let write_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, WRITE_IOTA_SHADER).expect("shader"),
    )
    .expect("write pipeline");
    let copy_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, COPY_SHADER).expect("shader"),
    )
    .expect("copy pipeline");

    const N: usize = 64;
    let byte_size = N * 4;

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let scratch = pool
        .acquire_buffer(
            byte_size as u64,
            BufferKind::Scattered,
            None,
            BufferFlags::empty(),
            None,
        )
        .expect("scratch");
    let output = pool
        .acquire_buffer(
            byte_size as u64,
            BufferKind::Scattered,
            None,
            BufferFlags::empty(),
            None,
        )
        .expect("output");

    let scratch_rw = scratch.handle(ResourceAccess::ReadWrite).expect("scratch");
    let out_w = output.handle(ResourceAccess::Write).expect("out");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("write_iota", &write_pipe)
        .bind_parcel(&scratch, NodeAccess::Write)
        .bind_resources_typed(&[scratch_rw])
        .dispatch(1, 1, 1);
    scheme
        .node("copy_out", &copy_pipe)
        .bind_parcel(&scratch, NodeAccess::Read)
        .bind_parcel(&output, NodeAccess::Write)
        .bind_resources_typed(&[scratch_rw, out_w])
        .dispatch(1, 1, 1);

    let frame = scheme.submit().expect("submit");
    frame.wait(&ctx).expect("wait");

    let expected: Vec<u32> = (1..=N as u32).collect();
    assert_eq!(readback_parcel_u32(&device, &output, N), expected);
}

// ---------------------------------------------------------------------------
// Duplicated from compute_integration.rs
// ---------------------------------------------------------------------------

const PARTICLE_SHADER: &str = r#"
import goldy_exp;

struct Particle {
    float2 position;
    float2 velocity;
};

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<Particle> particles, ThreadId id) {
    uint idx = id.x;
    if (idx >= 4) return;
    Particle p = particles[idx];
    p.position += float2(0.01, 0.01);
    particles[idx] = p;
}
"#;

const TYPED_PAIR_SHADER: &str = r#"
import goldy_exp;

struct Pair { uint a; uint b; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(BufRO<Pair> input, Scattered<Pair> output, ThreadId id) {
    uint idx = id.x;
    if (idx >= 8) return;
    Pair p = input[idx];
    output[idx].a = p.a + p.b;
    output[idx].b = p.a * p.b;
}
"#;

const WRITE_TEXTURE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(DirectSpatial<float4> output, ThreadId id) {
    uint2 dims;
    output.GetDimensions(dims.x, dims.y);
    if (id.x < dims.x && id.y < dims.y) {
        output[int2(id.x, id.y)] = float4(1.0, 0.0, 0.0, 1.0);
    }
}
"#;

const WAVE_SCAN_64_UNIFORM: &str = r#"
import goldy_exp;
groupshared uint sh_scratch[32];
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
    uint ix  = local_id.x;
    uint val = 1u;
    uint lc  = WaveGetLaneCount();
    uint nw  = 64 / lc;
    uint wave_ix = ix / lc;
    uint inclusive = WavePrefixSum(val) + val;
    uint total     = WaveActiveSum(val);
    if (WaveIsFirstLane())
        sh_scratch[wave_ix] = total;
    GroupMemoryBarrierWithGroupSync();
    if (ix == 0) {
        uint run = 0;
        for (uint i = 0; i < nw; i++) {
            uint s = sh_scratch[i]; sh_scratch[i] = run; run += s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    OUT[ix] = sh_scratch[wave_ix] + inclusive;
}
"#;

const WAVE_SCAN_64_RAMP: &str = r#"
import goldy_exp;
groupshared uint sh_scratch[32];
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
    uint ix  = local_id.x;
    uint val = ix + 1u;
    uint lc  = WaveGetLaneCount();
    uint nw  = 64 / lc;
    uint wave_ix = ix / lc;
    uint inclusive = WavePrefixSum(val) + val;
    uint total     = WaveActiveSum(val);
    if (WaveIsFirstLane())
        sh_scratch[wave_ix] = total;
    GroupMemoryBarrierWithGroupSync();
    if (ix == 0) {
        uint run = 0;
        for (uint i = 0; i < nw; i++) {
            uint s = sh_scratch[i]; sh_scratch[i] = run; run += s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    OUT[ix] = sh_scratch[wave_ix] + inclusive;
}
"#;

const WAVE_SCAN_256_UNIFORM: &str = r#"
import goldy_exp;
groupshared uint sh_scratch[64];
[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
    uint ix  = local_id.x;
    uint val = 1u;
    uint lc  = WaveGetLaneCount();
    uint nw  = 256 / lc;
    uint wave_ix = ix / lc;
    uint inclusive = WavePrefixSum(val) + val;
    uint total     = WaveActiveSum(val);
    if (WaveIsFirstLane())
        sh_scratch[wave_ix] = total;
    GroupMemoryBarrierWithGroupSync();
    if (ix == 0) {
        uint run = 0;
        for (uint i = 0; i < nw; i++) {
            uint s = sh_scratch[i]; sh_scratch[i] = run; run += s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    OUT[ix] = sh_scratch[wave_ix] + inclusive;
}
"#;

const REDUCE_64_UNIFORM: &str = r#"
import goldy_exp;
groupshared uint sh_scratch[64];
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
    uint ix  = local_id.x;
    uint val = 1u;
    sh_scratch[ix] = val;
    for (uint i = 0; i < 6; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (ix + (1u << i) < 64u)
            val = val + sh_scratch[ix + (1u << i)];
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[ix] = val;
    }
    OUT[ix] = val;
}
"#;

const INCLUSIVE_SCAN_64_UNIFORM: &str = r#"
import goldy_exp;
groupshared uint sh_scratch[64];
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
    uint ix  = local_id.x;
    uint val = 1u;
    sh_scratch[ix] = val;
    for (uint i = 0; i < 6; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (ix >= (1u << i))
            val = sh_scratch[ix - (1u << i)] + val;
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[ix] = val;
    }
    OUT[ix] = val;
}
"#;

const BROADCAST_64: &str = r#"
import goldy_exp;
groupshared uint sh_slot[1];
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
    uint ix = local_id.x;
    if (ix == 0) sh_slot[0] = 42u;
    GroupMemoryBarrierWithGroupSync();
    OUT[ix] = sh_slot[0];
}
"#;

const UPPER_BOUND_64: &str = r#"
import goldy_exp;
groupshared uint sh_ps[64];
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
    uint ix = local_id.x;
    sh_ps[ix] = ix + 1u;
    GroupMemoryBarrierWithGroupSync();
    OUT[ix] = workgroup_upper_bound<6>(ix, sh_ps);
}
"#;

const DUAL_VIEW_WRITE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(4, 4, 1)]
void cs_main(DirectSpatial<float4> dst, ThreadId id) {
    uint x = id.x;
    uint y = id.y;
    dst[uint2(x, y)] = float4(float(x) / 255.0, float(y) / 255.0, 0.0, 1.0);
}
"#;

const DUAL_VIEW_READ_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(4, 4, 1)]
void cs_main(Interpolated<float4> src, Filter smp, Scattered<uint> out, ThreadId id) {
    uint x = id.x;
    uint y = id.y;
    float2 uv = (float2(x, y) + 0.5) / float2(4.0, 4.0);
    float4 v = src.Sample(smp, uv);
    uint r = uint(v.x * 255.0 + 0.5);
    uint g = uint(v.y * 255.0 + 0.5);
    uint b = uint(v.z * 255.0 + 0.5);
    uint a = uint(v.w * 255.0 + 0.5);
    out[y * 4 + x] = r | (g << 8) | (b << 16) | (a << 24);
}
"#;

fn dispatch_u32_write_scheme(device: &Device, ctx: &goldy::Context, shader_src: &str, out: &Parcel, byte_len: u64) {
    let shader = ShaderModule::from_slang(device, shader_src).expect("compile shader");
    let pipeline = ComputePipeline::new(device, &shader).expect("create pipeline");
    let out_w = out.handle(ResourceAccess::Write).expect("out handle");

    let mut scheme = Scheme::new(ctx);
    scheme
        .node("n0", &pipeline)
        .bind_parcel(out, NodeAccess::Write)
        .bind_resources_typed(&[out_w])
        .dispatch(1, 1, 1);
    let frame = scheme.submit().expect("submit");
    frame.wait(&ctx).expect("wait");

    let count = (byte_len / 4) as usize;
    let _ = readback_parcel_u32(device, out, count);
}

#[test]
fn scheme_compute_with_struct_buffer() {
    let device = make_device();
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, PARTICLE_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Particle {
        position: [f32; 2],
        velocity: [f32; 2],
    }
    impl StructuredBufferElement for Particle {}

    let particles = vec![
        Particle {
            position: [0.0, 0.0],
            velocity: [0.1, 0.0],
        };
        4
    ];

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buffer = pool
        .acquire_buffer_with_data(&particles, BufferKind::Scattered)
        .expect("buffer");
    let handle = buffer.handle(ResourceAccess::ReadWrite).expect("handle");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("n0", &pipeline)
        .bind_parcel(&buffer, NodeAccess::ReadWrite)
        .bind_resources_typed(&[handle])
        .dispatch(1, 1, 1);
    scheme.submit().expect("submit with struct buffer");
}

#[test]
fn scheme_scattered_typed_variable_assignment() {
    let device = make_device();
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, TYPED_PAIR_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Pair {
        a: u32,
        b: u32,
    }
    impl StructuredBufferElement for Pair {}

    let input_data: Vec<Pair> = (0..8).map(|i| Pair { a: i + 1, b: i + 10 }).collect();
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let input = pool
        .acquire_buffer_with_data(&input_data, BufferKind::Scattered)
        .expect("input");
    let output = pool
        .acquire_buffer_with_data(&[Pair { a: 0, b: 0 }; 8], BufferKind::Scattered)
        .expect("output");

    // BufRO<Pair> must bind the SRV slot; ReadWrite yields UAV and reads zeros on WARP.
    let in_h = input.handle(ResourceAccess::Read).expect("in");
    let out_h = output.handle(ResourceAccess::Write).expect("out");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("typed_copy", &pipeline)
        .bind_parcel(&input, NodeAccess::Read)
        .bind_parcel(&output, NodeAccess::Write)
        .bind_resources_typed(&[in_h, out_h])
        .dispatch(1, 1, 1);
    let frame = scheme.submit().expect("submit");
    frame.wait(&ctx).expect("wait");

    let mut raw = vec![0u8; 8 * std::mem::size_of::<Pair>()];
    output.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[Pair] = bytemuck::cast_slice(&raw);

    for i in 0..8u32 {
        let expected_a = (i + 1) + (i + 10);
        let expected_b = (i + 1) * (i + 10);
        assert_eq!(result[i as usize].a, expected_a, "output[{i}].a");
        assert_eq!(result[i as usize].b, expected_b, "output[{i}].b");
    }
}

#[test]
fn scheme_compute_write_to_texture() {
    let device = make_device();
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, WRITE_TEXTURE_SHADER).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    let width = 16u32;
    let height = 16u32;
    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let texture = pool
        .acquire_texture(
            width,
            height,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
            None,
        )
        .expect("texture");
    let tex_w = texture.handle(ResourceAccess::Write).expect("tex write");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("write_tex", &pipeline)
        .bind_parcel(&texture, NodeAccess::Write)
        .bind_resources_typed(&[tex_w])
        .dispatch(wg_x, wg_y, 1);
    let frame = scheme.submit().expect("submit");
    frame.wait(&ctx).expect("wait");

    let mut output = vec![0u8; (width * height * 4) as usize];
    texture
        .detach_texture()
        .expect("detach texture parcel")
        .read_to_cpu(&mut output)
        .expect("readback");

    let nonzero = output.iter().filter(|&&b| b != 0).count();
    assert!(nonzero > 0, "texture readback all zeros");
    assert_eq!(output[0], 255, "R channel");
    assert_eq!(output[1], 0, "G channel");
    assert_eq!(output[2], 0, "B channel");
    assert_eq!(output[3], 255, "A channel");
}

#[test]
fn scheme_wave_inclusive_scan_uniform_64() {
    let device = make_device();
    let ctx = submission_context(&device);
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let out = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("out");
    dispatch_u32_write_scheme(&device, &ctx, WAVE_SCAN_64_UNIFORM, &out, 64 * 4);
    let result = readback_parcel_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, i as u32 + 1, "wave_scan_uniform_64[{i}]");
    }
}

#[test]
fn scheme_wave_inclusive_scan_ramp_64() {
    let device = make_device();
    let ctx = submission_context(&device);
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let out = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("out");
    dispatch_u32_write_scheme(&device, &ctx, WAVE_SCAN_64_RAMP, &out, 64 * 4);
    let result = readback_parcel_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        let k = (i + 1) as u32;
        let expected = k * (k + 1) / 2;
        assert_eq!(val, expected, "wave_scan_ramp_64[{i}]");
    }
}

#[test]
fn scheme_wave_inclusive_scan_uniform_256() {
    let device = make_device();
    let ctx = submission_context(&device);
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let out = pool
        .acquire_buffer(256 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("out");
    dispatch_u32_write_scheme(&device, &ctx, WAVE_SCAN_256_UNIFORM, &out, 256 * 4);
    let result = readback_parcel_u32(&device, &out, 256);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, i as u32 + 1, "wave_scan_uniform_256[{i}]");
    }
}

#[test]
fn scheme_workgroup_reduce_uint_correct() {
    let device = make_device();
    let ctx = submission_context(&device);
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let out = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("out");
    dispatch_u32_write_scheme(&device, &ctx, REDUCE_64_UNIFORM, &out, 64 * 4);
    let result = readback_parcel_u32(&device, &out, 64);
    assert_eq!(result[0], 64, "workgroup_reduce thread 0");
}

#[test]
fn scheme_workgroup_inclusive_scan_uint_correct() {
    let device = make_device();
    let ctx = submission_context(&device);
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let out = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("out");
    dispatch_u32_write_scheme(&device, &ctx, INCLUSIVE_SCAN_64_UNIFORM, &out, 64 * 4);
    let result = readback_parcel_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, i as u32 + 1, "workgroup_inclusive_scan[{i}]");
    }
}

#[test]
fn scheme_workgroup_broadcast_correct() {
    let device = make_device();
    let ctx = submission_context(&device);
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let out = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("out");
    dispatch_u32_write_scheme(&device, &ctx, BROADCAST_64, &out, 64 * 4);
    let result = readback_parcel_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, 42, "workgroup_broadcast[{i}]");
    }
}

#[test]
fn scheme_workgroup_upper_bound_linear() {
    let device = make_device();
    let ctx = submission_context(&device);
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let out = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("out");
    dispatch_u32_write_scheme(&device, &ctx, UPPER_BOUND_64, &out, 64 * 4);
    let result = readback_parcel_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, i as u32, "workgroup_upper_bound[{i}]");
    }
}

#[test]
fn scheme_texture_dual_view_round_trip() {
    const W: u32 = 4;
    const H: u32 = 4;
    const N: usize = (W * H) as usize;

    let device = make_device();
    let ctx = submission_context(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let tex = pool
        .acquire_texture(
            W,
            H,
            TextureFormat::Rgba8Unorm,
            TextureKind::DirectInterpolated,
            TextureFlags::empty(),
            None,
        )
        .expect("texture");
    let out = pool
        .acquire_buffer((N * 4) as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("out");

    let write_shader = ShaderModule::from_slang(&device, DUAL_VIEW_WRITE_SHADER).expect("write shader");
    let write_pipeline = ComputePipeline::new(&device, &write_shader).expect("write pipeline");
    let read_shader = ShaderModule::from_slang(&device, DUAL_VIEW_READ_SHADER).expect("read shader");
    let read_pipeline = ComputePipeline::new(&device, &read_shader).expect("read pipeline");
    let sampler = Sampler::nearest(&device).expect("sampler");

    let tex_w = tex.handle(ResourceAccess::Write).expect("tex write");
    let tex_r = tex.handle(ResourceAccess::Read).expect("tex read");
    let smp_r = sampler.handle(ResourceAccess::Read).expect("sampler");
    let out_w = out.handle(ResourceAccess::Write).expect("out write");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("write", &write_pipeline)
        .bind_parcel(&tex, NodeAccess::Write)
        .bind_resources_typed(&[tex_w])
        .dispatch(1, 1, 1);
    scheme
        .node("read", &read_pipeline)
        .bind_parcel(&tex, NodeAccess::Read)
        .bind_parcel(&out, NodeAccess::Write)
        .bind_resources_typed(&[tex_r, smp_r, out_w])
        .dispatch(1, 1, 1);
    let frame = scheme.submit().expect("submit");
    frame.wait(&ctx).expect("wait");

    let mut raw = vec![0u8; N * 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[u32] = bytemuck::cast_slice(&raw);

    for y in 0..H as usize {
        for x in 0..W as usize {
            let expected_r = x as u8;
            let expected_g = y as u8;
            let packed = result[y * W as usize + x];
            let r = (packed & 0xFF) as u8;
            let g = ((packed >> 8) & 0xFF) as u8;
            let b = ((packed >> 16) & 0xFF) as u8;
            let a = ((packed >> 24) & 0xFF) as u8;
            assert_eq!(r, expected_r, "r mismatch at ({x},{y})");
            assert_eq!(g, expected_g, "g mismatch at ({x},{y})");
            assert_eq!(b, 0, "b mismatch at ({x},{y})");
            assert_eq!(a, 255, "a mismatch at ({x},{y})");
        }
    }
}

#[test]
fn scheme_two_contexts_both_submit_and_complete() {
    let device = make_device();
    let ctx_a = submission_context(&device);
    let ctx_b = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, IN_PLACE_DOUBLE_SHADER).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf_a = pool
        .acquire_buffer_with_data(&(0..64).collect::<Vec<u32>>(), BufferKind::Scattered)
        .expect("buf_a");
    let buf_b = pool
        .acquire_buffer_with_data(&(100..164).collect::<Vec<u32>>(), BufferKind::Scattered)
        .expect("buf_b");

    let buf_a_rw = buf_a.handle(ResourceAccess::ReadWrite).expect("buf_a");
    let buf_b_rw = buf_b.handle(ResourceAccess::ReadWrite).expect("buf_b");

    let mut scheme_a = Scheme::new(&ctx_a);
    scheme_a
        .node("n0", &pipeline)
        .bind_parcel(&buf_a, NodeAccess::ReadWrite)
        .bind_resources_typed(&[buf_a_rw])
        .dispatch(1, 1, 1);
    let frame_a = scheme_a.submit().expect("ctx_a submit");

    let mut scheme_b = Scheme::new(&ctx_b);
    scheme_b
        .node("n0", &pipeline)
        .bind_parcel(&buf_b, NodeAccess::ReadWrite)
        .bind_resources_typed(&[buf_b_rw])
        .dispatch(1, 1, 1);
    let frame_b = scheme_b.submit().expect("ctx_b submit");

    frame_a.wait(&ctx_a).expect("ctx_a wait");
    frame_b.wait(&ctx_b).expect("ctx_b wait");

    let result_a = readback_parcel_u32(&device, &buf_a, 64);
    let result_b = readback_parcel_u32(&device, &buf_b, 64);
    for i in 0..64 {
        assert_eq!(result_a[i], i as u32 * 2, "buf_a[{i}]");
        assert_eq!(result_b[i], (100 + i as u32) * 2, "buf_b[{i}]");
    }
}

#[test]
fn scheme_two_contexts_reclaim_independently() {
    let device = make_device();
    let ctx_a = submission_context(&device);
    let _ctx_b = submission_context(&device);

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let buf = pool
        .acquire_buffer(256, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("buf");

    let mut scheme = Scheme::new(&ctx_a);
    let frame_a = scheme.submit().expect("ctx_a submit");
    drop(buf);

    frame_a.wait(&ctx_a).expect("ctx_a wait");
    ctx_a.flush_deferred_deletions();

    assert_eq!(
        ctx_a.deferred_deletion_pending_count(),
        0,
        "ctx_a must reclaim without waiting for ctx_b (no device-global horizon)"
    );
}
