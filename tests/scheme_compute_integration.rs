//! Scheme compute integration tests — migrated from `TaskGraph` coverage.
//!
//! Retained worker schemes: parcels + `bind_resources_typed` + [`Scheme::submit`].
//! CPU→GPU parcel writes (including zero-fills) use [`goldy::write_to_parcel`] — a
//! separate upload submission per call (internally a one-node write graph on `ctx`),
//! serialized against worker schemes by queue order. Callers do not use
//! `TaskGraph::clear_buffer` or `TaskGraph::write_buffer` directly.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    types::{BufferFlags, ResourceAccess},
    write_to_parcel, BufferKind, ComputePipeline, Device, DeviceDescriptor, Instance, NodeAccess,
    Parcel, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule,
};
use std::sync::Arc;
use submission::submission_context;

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
    // Upload micro-scheme: TaskGraph::write_parcel internally, separate from worker Scheme.
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

    let double_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap(),
    )
    .unwrap();
    let add_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap(),
    )
    .unwrap();

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
    let tv = scheme.submit().unwrap();
    assert_eq!(scheme.replay_stats().records, 1, "linear chain records once");
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        scheme.replay_stats().resubmit_hits,
        1,
        "second submit must resubmit without re-record"
    );
    ctx.wait_until(tv).unwrap();

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

    let pipe_42 = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap(),
    )
    .unwrap();
    let pipe_99 = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, FILL_99_SHADER).unwrap(),
    )
    .unwrap();

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

    let tv = scheme.submit().unwrap();
    ctx.wait_until(tv).unwrap();

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
    let double_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap(),
    )
    .unwrap();
    let sum_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, SUM_SHADER).unwrap(),
    )
    .unwrap();

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

    let tv = scheme.submit().unwrap();
    assert_eq!(scheme.replay_stats().records, 1, "diamond records once");
    ctx.wait_until(tv).unwrap();

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

    let pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap(),
    )
    .unwrap();

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

    let tv = scheme.submit().unwrap();
    ctx.wait_until(tv).unwrap();

    for &v in &readback_parcel_u32(&device, &buf, 64) {
        assert_eq!(v, 42);
    }
}

#[test]
fn scheme_zeros_then_dispatch_reads_zeros() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, COPY_SHADER).unwrap(),
    )
    .unwrap();

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

    let tv = scheme.submit().unwrap();
    ctx.wait_until(tv).unwrap();

    for (i, &val) in readback_parcel_u32(&device, &out, 64).iter().enumerate() {
        assert_eq!(val, 0, "element {i}: expected 0 after zero write, got {val}");
    }
}

#[test]
fn scheme_write_then_dispatch_reads_uploaded_data() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, COPY_SHADER).unwrap(),
    )
    .unwrap();

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

    let tv = scheme.submit().unwrap();
    ctx.wait_until(tv).unwrap();

    for (i, &val) in readback_parcel_u32(&device, &out, 64).iter().enumerate() {
        assert_eq!(val, known_data[i], "element {i}");
    }
}

#[test]
fn scheme_stress_zeros_then_dispatch_large() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, COPY_SHADER).unwrap(),
    )
    .unwrap();

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

    let tv = scheme.submit().unwrap();
    ctx.wait_until(tv).unwrap();

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

    let copy_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, COPY_SHADER).unwrap(),
    )
    .unwrap();

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

    let tv = scheme.submit().unwrap();
    ctx.wait_until(tv).unwrap();

    for (i, out) in outs.iter().enumerate() {
        let nonzero_count = readback_parcel_u32(&device, out, N)
            .iter()
            .filter(|&&v| v != 0)
            .count();
        assert_eq!(
            nonzero_count, 0,
            "buffer {i}: expected all zeros after zero write"
        );
    }
}

#[test]
fn scheme_stress_write_then_dispatch_chain() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, COPY_SHADER).unwrap(),
    )
    .unwrap();

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

    let tv = scheme.submit().unwrap();
    ctx.wait_until(tv).unwrap();

    for (i, &val) in readback_parcel_u32(&device, &out, N).iter().enumerate() {
        assert_eq!(val, known_data[i], "element {i}");
    }
}

#[test]
fn scheme_stress_two_phase_submission() {
    let device = make_device();
    let ctx = submission_context(&device);

    let double_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap(),
    )
    .unwrap();
    let add_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap(),
    )
    .unwrap();

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
        let tv = scheme.submit().unwrap();
        ctx.wait_until(tv).unwrap();
    }

    {
        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("add_ten", &add_pipe)
            .bind_parcel(&tmp, NodeAccess::ReadWrite)
            .bind_resources_typed(&[tmp_rw])
            .dispatch((N / 64) as u32, 1, 1);
        let tv = scheme.submit().unwrap();
        ctx.wait_until(tv).unwrap();
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

    let add_pipe = ComputePipeline::new(
        &device,
        &ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap(),
    )
    .unwrap();

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
    let mut last_tv = 0;
    for _ in 0..ROUNDS {
        last_tv = scheme.submit().unwrap();
    }
    ctx.wait_until(last_tv).unwrap();

    assert_eq!(
        scheme.replay_stats().records,
        1,
        "rapid submissions record once"
    );
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
    let tv = scheme.submit().expect("submit");
    ctx.wait_until(tv).expect("wait");

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
    let tv = scheme.submit().expect("submit");
    ctx.wait_until(tv).expect("wait");

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
    let tv = scheme.submit().expect("submit");
    ctx.wait_until(tv).expect("wait");

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
    ctx.wait_until(token.timeline_value()).expect("wait");

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
    ctx.wait_until(token.timeline_value()).expect("wait");

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
    ctx.wait_until(token.timeline_value()).expect("wait");

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
    let tv = scheme.submit().expect("submit");
    ctx.wait_until(tv).expect("wait");

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

    let tv = {
        let mut inc_scheme = Scheme::new(&ctx);
        inc_scheme
            .node("inc", &inc_pipe)
            .bind_parcel(&output, NodeAccess::ReadWrite)
            .bind_resources_typed(&[out_rw])
            .dispatch(1, 1, 1);
        inc_scheme.submit().expect("inc submit")
    };
    ctx.wait_until(tv).expect("wait after inc");

    for (i, &val) in readback_parcel_u32(&device, &output, 64).iter().enumerate() {
        assert_eq!(
            val, 1,
            "output[{i}]: expected 1 (write_to_parcel zeroed before increment), got {val}"
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
    let a = pool.acquire_buffer_with_data(&[1u32; N], BufferKind::Scattered).expect("a");
    let b = pool.acquire_buffer_with_data(&[2u32; N], BufferKind::Scattered).expect("b");
    let c = pool.acquire_buffer_with_data(&[3u32; N], BufferKind::Scattered).expect("c");
    let d = pool.acquire_buffer_with_data(&[4u32; N], BufferKind::Scattered).expect("d");
    let e = pool.acquire_buffer_with_data(&[5u32; N], BufferKind::Scattered).expect("e");
    let out = pool.acquire_buffer_with_data(&[0u32; N], BufferKind::Scattered).expect("out");

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

    let tv = scheme.submit().expect("submit");
    ctx.wait_until(tv).expect("wait");

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
        .acquire_buffer(byte_size as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .expect("scratch");
    let output = pool
        .acquire_buffer(byte_size as u64, BufferKind::Scattered, None, BufferFlags::empty(), None)
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

    let tv = scheme.submit().expect("submit");
    ctx.wait_until(tv).expect("wait");

    let expected: Vec<u32> = (1..=N as u32).collect();
    assert_eq!(readback_parcel_u32(&device, &output, N), expected);
}
