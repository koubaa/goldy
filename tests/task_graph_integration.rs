//! Task graph integration tests.
//!
//! These tests verify that `TaskGraph` produces correct GPU results using
//! real backends, exercising dependency analysis, barrier insertion, and wave
//! scheduling on actual hardware.
//! They are only compiled when at least one backend feature is enabled.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

use goldy::{
    types::{BufferFlags, ResourceAccess},
    Buffer, BufferKind, ComputePipeline, NodeAccess, RetainedPool, ShaderModule, TaskGraph,
};
use std::sync::Arc;

mod common;
#[path = "common/submission.rs"]
mod submission;
use submission::submission_context;

fn test_alloc_buffer_with_data<T: goldy::StructuredBufferElement>(
    device: &goldy::Device,
    data: &[T],
    kind: goldy::BufferKind,
) -> goldy::Buffer {
    use std::sync::Arc;
    goldy::RetainedPool::new(Arc::new(device.clone()))
        .acquire_buffer_with_data(data, kind)
        .expect("acquire_buffer_with_data")
}

fn test_alloc_buffer(
    device: &goldy::Device,
    size: u64,
    kind: goldy::BufferKind,
    stride: Option<u32>,
    flags: goldy::types::BufferFlags,
) -> goldy::Buffer {
    use std::sync::Arc;
    goldy::RetainedPool::new(Arc::new(device.clone()))
        .acquire_buffer(size, kind, stride, flags, None)
        .expect("acquire_buffer")
}

/// Doubles each element: out[i] = in[i] * 2
const DOUBLE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x] * 2;
}
"#;

/// Adds 10 to each element in-place: buf[i] += 10
const ADD_TEN_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + 10;
}
"#;

/// Writes constant 42 to each element: buf[i] = 42
const FILL_42_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = 42;
}
"#;

/// Writes constant 99 to each element: buf[i] = 99
const FILL_99_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = 99;
}
"#;

/// Sums two buffers: out[i] = a[i] + b[i]
const SUM_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> a, Scattered<uint> b, Scattered<uint> out, ThreadId id) {
    out[id.x] = a[id.x] + b[id.x];
}
"#;

/// Copies input to output: out[i] = in[i]
const COPY_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x];
}
"#;

fn make_device() -> goldy::Device {
    let instance = goldy::Instance::new().expect("Failed to create instance");

    // When running on headless CI with WARP enabled, prefer the explicit WARP adapter
    // over the Microsoft Basic Render Driver (MSBR). MSBR lacks DXGI_ADAPTER_FLAG_SOFTWARE
    // on some CI runners so it's misclassified as DiscreteGpu, but its D3D12 compute
    // implementation faults silently (e.g. Signal AV after GPU work).
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    if std::env::var("GOLDY_DX12_ALLOW_WARP").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")) {
        if let Ok(adapter) = instance.request_adapter(&goldy::RequestAdapterOptions {
            power_preference: goldy::PowerPreference::None,
            force_fallback_adapter: true,
        }) {
            if let Ok(dev) = adapter.request_device(&goldy::DeviceDescriptor::default()) {
                return dev;
            }
        }
    }

    instance
        .request_adapter(&goldy::RequestAdapterOptions::default())
        .expect("Failed to request adapter")
        .request_device(&goldy::DeviceDescriptor::default())
        .expect("Failed to create device")
}

fn readback_u32(device: &goldy::Device, buffer: &Buffer, count: usize) -> Vec<u32> {
    let mut output = vec![0u8; count * 4];
    buffer.read_to_cpu(device, &mut output).expect("readback");
    bytemuck::cast_slice(&output).to_vec()
}

// ---------------------------------------------------------------------------
// TaskGraph dispatch tests
// ---------------------------------------------------------------------------

/// Linear chain: double then add 10. Exercises RAW dependency.
/// Expected: out[i] = i * 2 + 10
// Scheme migration: see scheme_graph_linear_chain
#[test]
fn graph_linear_chain() {
    let device = make_device();
    let ctx = submission_context(&device);

    let double_shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap();
    let add_shader = ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap();
    let double_pipe = ComputePipeline::new(&device, &double_shader).unwrap();
    let add_pipe = ComputePipeline::new(&device, &add_shader).unwrap();

    let input: Vec<u32> = (0..64).collect();
    let src = test_alloc_buffer_with_data(&device, &input, BufferKind::Scattered);
    let dst = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());

    // `DOUBLE_SHADER` reads `src` as `Scattered<uint>` (RWStructuredBuffer / UAV on DX12).
    // Bind the UAV index — `ResourceAccess::Read` returns the SRV slot, which WARP reads as zeros.
    let src_idx = src.resource_index(ResourceAccess::ReadWrite).unwrap();
    let dst_idx = dst.resource_index(ResourceAccess::Write).unwrap();

    let mut graph = TaskGraph::new();
    graph
        .node("double", &double_pipe)
        .with_buffer(&*src, NodeAccess::Read)
        .with_buffer(&*dst, NodeAccess::Write)
        .with_resource_slots_slice(&[src_idx, dst_idx])
        .dispatch(1, 1, 1);

    graph
        .node("add_ten", &add_pipe)
        .with_buffer(&*dst, NodeAccess::ReadWrite)
        .with_resource_slots_slice(&[dst_idx])
        .dispatch(1, 1, 1);

    graph.dispatch(&ctx).unwrap();

    let result = readback_u32(&device, &dst, 64);
    for (i, &val) in result.iter().enumerate() {
        let expected = (i as u32) * 2 + 10;
        assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
    }
}

/// Two independent dispatches writing to separate buffers.
/// Exercises the no-barrier path (both should land in wave 0).
// Scheme migration: see scheme_graph_independent_dispatches
#[test]
fn graph_independent_dispatches() {
    let device = make_device();
    let ctx = submission_context(&device);

    let fill42_shader = ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap();
    let fill99_shader = ShaderModule::from_slang(&device, FILL_99_SHADER).unwrap();
    let pipe_42 = ComputePipeline::new(&device, &fill42_shader).unwrap();
    let pipe_99 = ComputePipeline::new(&device, &fill99_shader).unwrap();

    let buf_a = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());
    let buf_b = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());

    // `FILL_42_SHADER` writes `buf_a` via `Scattered<uint>` (UAV on DX12).
    // `ResourceAccess::Read` returns the SRV slot — use Write for the UAV index.
    let idx_a = buf_a.resource_index(ResourceAccess::Write).unwrap();
    let idx_b = buf_b.resource_index(ResourceAccess::Write).unwrap();

    let mut graph = TaskGraph::new();
    graph
        .node("fill_a", &pipe_42)
        .with_buffer(&*buf_a, NodeAccess::Write)
        .with_resource_slots_slice(&[idx_a])
        .dispatch(1, 1, 1);
    graph
        .node("fill_b", &pipe_99)
        .with_buffer(&*buf_b, NodeAccess::Write)
        .with_resource_slots_slice(&[idx_b])
        .dispatch(1, 1, 1);

    graph.dispatch(&ctx).unwrap();

    let result_a = readback_u32(&device, &buf_a, 64);
    let result_b = readback_u32(&device, &buf_b, 64);

    for &v in &result_a {
        assert_eq!(v, 42);
    }
    for &v in &result_b {
        assert_eq!(v, 99);
    }
}

/// Diamond dependency: A writes X, B and C read X and write Y/Z, D reads Y+Z.
///
///       A (fill src with i)
///      / \
///     B   C  (double into Y / double into Z)
///      \ /
///       D    (sum Y+Z into out)
///
/// Expected: out[i] = i*2 + i*2 = i*4
// Scheme migration: see scheme_graph_diamond_dependency
#[test]
fn graph_diamond_dependency() {
    let device = make_device();
    let ctx = submission_context(&device);

    let fill_shader_src = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = id.x;
}
"#;
    let fill_shader = ShaderModule::from_slang(&device, fill_shader_src).unwrap();
    let fill_pipe = ComputePipeline::new(&device, &fill_shader).unwrap();

    let double_shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap();
    let double_pipe = ComputePipeline::new(&device, &double_shader).unwrap();

    let sum_shader = ShaderModule::from_slang(&device, SUM_SHADER).unwrap();
    let sum_pipe = ComputePipeline::new(&device, &sum_shader).unwrap();

    let src = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());
    let y = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());
    let z = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());
    let out = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());

    // `src` is accessed as `Scattered<uint>` by both the fill node (write) and the
    // double nodes (read-via-RWStructuredBuffer). All `Scattered` params need the UAV
    // index; `ResourceAccess::Read` would return the SRV slot and read zeros on WARP.
    let src_idx = src.resource_index(ResourceAccess::ReadWrite).unwrap();
    let y_idx = y.resource_index(ResourceAccess::Write).unwrap();
    let z_idx = z.resource_index(ResourceAccess::Write).unwrap();
    let out_idx = out.resource_index(ResourceAccess::Write).unwrap();

    let mut graph = TaskGraph::new();

    // A: fill src with thread index
    graph
        .node("fill_src", &fill_pipe)
        .with_buffer(&*src, NodeAccess::Write)
        .with_resource_slots_slice(&[src_idx])
        .dispatch(1, 1, 1);

    // B: double src -> y
    graph
        .node("double_to_y", &double_pipe)
        .with_buffer(&*src, NodeAccess::Read)
        .with_buffer(&*y, NodeAccess::Write)
        .with_resource_slots_slice(&[src_idx, y_idx])
        .dispatch(1, 1, 1);

    // C: double src -> z
    graph
        .node("double_to_z", &double_pipe)
        .with_buffer(&*src, NodeAccess::Read)
        .with_buffer(&*z, NodeAccess::Write)
        .with_resource_slots_slice(&[src_idx, z_idx])
        .dispatch(1, 1, 1);

    // D: sum y + z -> out
    graph
        .node("sum_yz", &sum_pipe)
        .with_buffer(&*y, NodeAccess::Read)
        .with_buffer(&*z, NodeAccess::Read)
        .with_buffer(&*out, NodeAccess::Write)
        .with_resource_slots_slice(&[y_idx, z_idx, out_idx])
        .dispatch(1, 1, 1);

    graph.dispatch(&ctx).unwrap();

    let result = readback_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        let expected = (i as u32) * 4;
        assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
    }
}

/// Two-pass chained compute through TaskGraph (double then add ten).
// Scheme migration: see scheme_graph_linear_chain
#[test]
fn graph_two_pass_chained_compute() {
    let device = make_device();
    let ctx = submission_context(&device);

    let double_shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap();
    let add_shader = ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap();
    let double_pipe = ComputePipeline::new(&device, &double_shader).unwrap();
    let add_pipe = ComputePipeline::new(&device, &add_shader).unwrap();

    let input: Vec<u32> = (0..64).collect();
    let src = test_alloc_buffer_with_data(&device, &input, BufferKind::Scattered);
    let dst = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());

    let src_idx = src.resource_index(ResourceAccess::ReadWrite).unwrap();
    let dst_idx = dst.resource_index(ResourceAccess::Write).unwrap();

    let mut graph = TaskGraph::new();
    graph
        .node("double", &double_pipe)
        .with_buffer(&*src, NodeAccess::Read)
        .with_buffer(&*dst, NodeAccess::Write)
        .with_resource_slots_slice(&[src_idx, dst_idx])
        .dispatch(1, 1, 1);
    graph
        .node("add_ten", &add_pipe)
        .with_buffer(&*dst, NodeAccess::ReadWrite)
        .with_resource_slots_slice(&[dst_idx])
        .dispatch(1, 1, 1);
    graph.dispatch(&ctx).unwrap();

    let result = readback_u32(&device, &dst, 64);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, (i as u32) * 2 + 10, "element {i}");
    }
}

/// Non-blocking submit via TaskGraph.
// Scheme migration: see scheme_graph_fill_readback
#[test]
fn graph_nonblocking_submit() {
    let device = make_device();
    let ctx = submission_context(&device);

    let fill_shader = ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap();
    let pipe = ComputePipeline::new(&device, &fill_shader).unwrap();

    let buf = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());
    let idx = buf.resource_index(ResourceAccess::Write).unwrap();

    let mut graph = TaskGraph::new();
    graph
        .node("fill", &pipe)
        .with_buffer(&*buf, NodeAccess::Write)
        .with_resource_slots_slice(&[idx])
        .dispatch(1, 1, 1);

    let tv = graph.submit(&ctx).unwrap();
    ctx.wait_until(tv).unwrap();

    let result = readback_u32(&device, &buf, 64);
    for &v in &result {
        assert_eq!(v, 42);
    }
}

// ---------------------------------------------------------------------------
// New integration tests: clear_buffer and write_buffer nodes
//
// These tests exercise the exact scenario that caused the DX12 race condition
// with `ComputeGraph::prelude`. Clears and writes are now first-class graph
// nodes subject to dependency analysis, so the correct barrier is inserted
// between the clear/write and the downstream dispatch on every backend.
// ---------------------------------------------------------------------------

/// `graph.clear_buffer()` followed by a dispatch that reads the buffer.
///
/// Previously this was done via `graph.prelude.push(ClearBuffer{..})` which
/// bypassed dependency analysis. On DX12 this caused a race because
/// `ClearUnorderedAccessViewUint` synchronizes under
/// `D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW`, which is distinct from
/// `D3D12_BARRIER_SYNC_COMPUTE_SHADING`. The TaskGraph refactor promotes the
/// clear to a first-class node so the analyzer emits the required barrier.
// Scheme migration: see scheme_zeros_then_dispatch_reads_zeros
#[test]
fn clear_then_dispatch_reads_zeros() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    // Allocate and fill a buffer with nonzero values.
    let nonzero: Vec<u32> = (1..=64).collect();
    let buf = test_alloc_buffer_with_data(&device, &nonzero, BufferKind::Scattered);
    let out = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());

    let buf_idx = buf.resource_index(ResourceAccess::Write).unwrap();
    let out_idx = out.resource_index(ResourceAccess::Write).unwrap();

    // Build a graph: clear buf → copy buf→out
    // The analyzer must insert a barrier between the clear and the copy.
    let mut graph = TaskGraph::new();
    graph.clear_parcel(&*buf, 0, 64 * 4).unwrap();
    graph
        .node("copy", &copy_pipe)
        .with_buffer(&*buf, NodeAccess::Read)
        .with_buffer(&*out, NodeAccess::Write)
        .with_resource_slots_slice(&[buf_idx, out_idx])
        .dispatch(1, 1, 1);

    graph.dispatch(&ctx).unwrap();

    // The copy reads from buf *after* the clear, so output must be all zeros.
    let result = readback_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val, 0,
            "element {i}: expected 0 after clear, got {val} (DX12 race check)"
        );
    }
}

/// `graph.write_buffer()` followed by a dispatch that reads the buffer.
///
/// The CPU data must be visible to the GPU dispatch. The TaskGraph analyzer
/// inserts a barrier between the write node and the read dispatch, ensuring
/// the upload completes before the shader accesses the buffer on all backends.
// Scheme migration: see scheme_write_then_dispatch_reads_uploaded_data
#[test]
fn write_then_dispatch_reads_uploaded_data() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    // Buffer starts empty (zeroed).
    let buf = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());
    let out = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());

    let buf_idx = buf.resource_index(ResourceAccess::Write).unwrap();
    let out_idx = out.resource_index(ResourceAccess::Write).unwrap();

    // Known data to upload: values 100..163.
    let known_data: Vec<u32> = (100..164).collect();
    let data_bytes: Vec<u8> = bytemuck::cast_slice(&known_data).to_vec();

    // Build a graph: write known_data into buf → copy buf→out
    let mut graph = TaskGraph::new();
    graph.write_parcel(&*buf, 0, data_bytes).unwrap();
    graph
        .node("copy", &copy_pipe)
        .with_buffer(&*buf, NodeAccess::Read)
        .with_buffer(&*out, NodeAccess::Write)
        .with_resource_slots_slice(&[buf_idx, out_idx])
        .dispatch(1, 1, 1);

    graph.dispatch(&ctx).unwrap();

    let result = readback_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        let expected = known_data[i];
        assert_eq!(
            val, expected,
            "element {i}: expected {expected} after write_buffer, got {val}"
        );
    }
}

/// `graph.write_parcel()` on a retained buffer parcel, then a dispatch that reads it.
///
/// Mirrors [`write_then_dispatch_reads_uploaded_data`] but uses the opaque parcel
/// path that goldy-doom will use for per-frame uniform uploads.
// Scheme migration: see scheme_write_then_dispatch_reads_uploaded_data
#[test]
fn write_parcel_then_dispatch_reads_uploaded_data() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let parcel = pool
        .acquire_buffer(64 * 4, BufferKind::Scattered, None, BufferFlags::empty(), None)
        .unwrap();
    let out = test_alloc_buffer(&device, 64 * 4, BufferKind::Scattered, None, BufferFlags::empty());

    let known_data: Vec<u32> = (200..264).collect();
    let data_bytes: Vec<u8> = bytemuck::cast_slice(&known_data).to_vec();

    let src_idx = parcel.resource_index(ResourceAccess::ReadWrite).unwrap();
    let out_idx = out.resource_index(ResourceAccess::Write).unwrap();

    let mut graph = TaskGraph::new();
    graph.write_parcel(&parcel, 0, data_bytes).unwrap();
    graph
        .node("copy", &copy_pipe)
        .with_parcel(&parcel, NodeAccess::Read)
        .with_buffer(&*out, NodeAccess::Write)
        .with_resource_slots_slice(&[src_idx, out_idx])
        .dispatch(1, 1, 1);

    let tv = graph.submit(&ctx).unwrap();
    ctx.wait_until(tv).unwrap();

    let result = readback_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        let expected = known_data[i];
        assert_eq!(
            val, expected,
            "element {i}: expected {expected} after write_parcel, got {val}"
        );
    }

    let refs = parcel.last_referenced();
    assert_eq!(refs.len(), 1, "bind_parcel should stamp at submit");
    assert_eq!(*refs.values().next().unwrap(), tv);
}

// ---------------------------------------------------------------------------
// GPU synchronization stress tests
//
// These tests target the same barrier scenarios that cause `many_bins_test`
// flakiness in ekrano issue #26. They isolate specific patterns at the goldy
// layer to narrow down which GPU sync path is broken.
//
// Each test should be run in a tight loop (e.g. 100x) to detect intermittent
// failures:
//   cargo test -p goldy --test task_graph_integration <name> -- --nocapture
// ---------------------------------------------------------------------------

/// Stress: clear a large buffer then verify every element is zero.
///
/// Uses 16K u32 elements (64 KiB) dispatched across 256 workgroups of 64 threads.
/// This is the minimal repro shape for the `ClearBuffer → compute dispatch`
/// barrier path. If the post-clear barrier is missing, some workgroups may read
/// stale (nonzero) data.
// Scheme migration: see scheme_stress_zeros_then_dispatch_large
#[test]
fn stress_clear_then_dispatch_large() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    const N: usize = 16384;
    let nonzero: Vec<u32> = (1..=N as u32).collect();
    let buf = test_alloc_buffer_with_data(&device, &nonzero, BufferKind::Scattered);
    let out = test_alloc_buffer(
        &device,
        (N * 4) as u64,
        BufferKind::Scattered,
        None,
        BufferFlags::empty(),
    );

    let buf_idx = buf.resource_index(ResourceAccess::Write).unwrap();
    let out_idx = out.resource_index(ResourceAccess::Write).unwrap();

    let mut graph = TaskGraph::new();
    graph.clear_parcel(&*buf, 0, (N * 4) as u64).unwrap();
    graph
        .node("copy", &copy_pipe)
        .with_buffer(&*buf, NodeAccess::Read)
        .with_buffer(&*out, NodeAccess::Write)
        .with_resource_slots_slice(&[buf_idx, out_idx])
        .dispatch((N / 64) as u32, 1, 1);

    graph.dispatch(&ctx).unwrap();

    let result = readback_u32(&device, &out, N);
    let nonzero_count = result.iter().filter(|&&v| v != 0).count();
    assert_eq!(
        nonzero_count, 0,
        "expected all zeros after clear, but {nonzero_count}/{N} elements were nonzero"
    );
}

/// Stress: many clears + many dispatches reading the cleared buffers in one graph.
///
/// Mimics Ekrano's pattern of clearing multiple pool buffers then dispatching
/// shaders that read them all. If any clear → dispatch barrier is missing,
/// some dispatches may read stale data.
// Scheme migration: see scheme_stress_many_zero_writes_many_dispatches
#[test]
fn stress_many_clears_many_dispatches() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    const N: usize = 1024;
    const NUM_BUFS: usize = 8;

    let mut srcs = Vec::new();
    let mut outs = Vec::new();
    for _ in 0..NUM_BUFS {
        let nonzero: Vec<u32> = (1..=N as u32).collect();
        srcs.push(test_alloc_buffer_with_data(&device, &nonzero, BufferKind::Scattered));
        outs.push(test_alloc_buffer(
            &device,
            (N * 4) as u64,
            BufferKind::Scattered,
            None,
            BufferFlags::empty(),
        ));
    }

    let mut graph = TaskGraph::new();
    for src in &srcs {
        graph.clear_parcel(&*src, 0, (N * 4) as u64).expect("clear_parcel");
    }
    for (src, out) in srcs.iter().zip(outs.iter()) {
        let src_idx = src.resource_index(ResourceAccess::Read).unwrap();
        let out_idx = out.resource_index(ResourceAccess::Write).unwrap();
        graph
            .node("copy", &copy_pipe)
            .with_buffer(&*src, NodeAccess::Read)
            .with_buffer(&*out, NodeAccess::Write)
            .with_resource_slots_slice(&[src_idx, out_idx])
            .dispatch((N / 64) as u32, 1, 1);
    }

    graph.dispatch(&ctx).unwrap();

    for (i, out) in outs.iter().enumerate() {
        let result = readback_u32(&device, out, N);
        let nonzero_count = result.iter().filter(|&&v| v != 0).count();
        assert_eq!(
            nonzero_count, 0,
            "buffer {i}: expected all zeros, but {nonzero_count}/{N} elements were nonzero"
        );
    }
}

/// Stress: clear → write → dispatch chain, exercising both clear and write barriers.
///
/// Buffer is first filled with nonzero data, then cleared to zero, then
/// overwritten with known data via write_buffer, then a dispatch copies it out.
/// Exercises Clear→Write (WAW) and Write→Dispatch (RAW) barrier insertion.
// Scheme migration: see scheme_stress_write_then_dispatch_chain
#[test]
fn stress_clear_write_dispatch_chain() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    const N: usize = 1024;
    let nonzero: Vec<u32> = (1..=N as u32).collect();
    let buf = test_alloc_buffer_with_data(&device, &nonzero, BufferKind::Scattered);
    let out = test_alloc_buffer(
        &device,
        (N * 4) as u64,
        BufferKind::Scattered,
        None,
        BufferFlags::empty(),
    );

    let buf_idx = buf.resource_index(ResourceAccess::Write).unwrap();
    let out_idx = out.resource_index(ResourceAccess::Write).unwrap();

    let known_data: Vec<u32> = (0..N as u32).map(|i| i * 7 + 42).collect();
    let data_bytes: Vec<u8> = bytemuck::cast_slice(&known_data).to_vec();

    let mut graph = TaskGraph::new();
    graph.clear_parcel(&*buf, 0, (N * 4) as u64).unwrap();
    graph.write_parcel(&*buf, 0, data_bytes).unwrap();
    graph
        .node("copy", &copy_pipe)
        .with_buffer(&*buf, NodeAccess::Read)
        .with_buffer(&*out, NodeAccess::Write)
        .with_resource_slots_slice(&[buf_idx, out_idx])
        .dispatch((N / 64) as u32, 1, 1);

    graph.dispatch(&ctx).unwrap();

    let result = readback_u32(&device, &out, N);
    for (i, &val) in result.iter().enumerate() {
        let expected = known_data[i];
        assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
    }
}

/// Stress: two-phase submission — submit one graph, then submit a second graph
/// that reads the output of the first.
///
/// Tests inter-submission synchronization (tail barrier correctness).
/// Mimics Ekrano's coarse→fine two-phase rendering where the backend may
/// split a single graph into multiple command buffers internally.
// Scheme migration: see scheme_stress_two_phase_submission
#[test]
fn stress_two_phase_submission() {
    let device = make_device();
    let ctx = submission_context(&device);

    let double_shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap();
    let add_shader = ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap();
    let double_pipe = ComputePipeline::new(&device, &double_shader).unwrap();
    let add_pipe = ComputePipeline::new(&device, &add_shader).unwrap();

    const N: usize = 4096;
    let input: Vec<u32> = (0..N as u32).collect();
    let buf = test_alloc_buffer_with_data(&device, &input, BufferKind::Scattered);
    let buf_idx = buf.resource_index(ResourceAccess::Write).unwrap();

    // Phase 1: double in-place. Use a tmp buffer since DOUBLE_SHADER reads
    // from one buffer and writes to another.
    let tmp = test_alloc_buffer(
        &device,
        (N * 4) as u64,
        BufferKind::Scattered,
        None,
        BufferFlags::empty(),
    );
    let tmp_idx = tmp.resource_index(ResourceAccess::Write).unwrap();

    {
        let mut graph = TaskGraph::new();
        graph
            .node("double", &double_pipe)
            .with_buffer(&*buf, NodeAccess::Read)
            .with_buffer(&*tmp, NodeAccess::Write)
            .with_resource_slots_slice(&[buf_idx, tmp_idx])
            .dispatch((N / 64) as u32, 1, 1);
        let tv = graph.submit(&ctx).unwrap();
        ctx.wait_until(tv).unwrap();
    }

    // Phase 2: add 10 to the doubled values.
    {
        let mut graph = TaskGraph::new();
        graph
            .node("add_ten", &add_pipe)
            .with_buffer(&*tmp, NodeAccess::ReadWrite)
            .with_resource_slots_slice(&[tmp_idx])
            .dispatch((N / 64) as u32, 1, 1);
        graph.dispatch(&ctx).unwrap();
    }

    let result = readback_u32(&device, &tmp, N);
    for (i, &val) in result.iter().enumerate() {
        let expected = (i as u32) * 2 + 10;
        assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
    }
}

/// Stress: rapid back-to-back submissions without waiting.
///
/// Submit N graphs in quick succession, each depending on the previous one's
/// output, using only `submit` (non-blocking). Wait only at the end.
/// This stresses the queue fence synchronization path.
// Scheme migration: see scheme_stress_rapid_submissions
#[test]
fn stress_rapid_submissions() {
    let device = make_device();
    let ctx = submission_context(&device);

    let add_shader = ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap();
    let add_pipe = ComputePipeline::new(&device, &add_shader).unwrap();

    const N: usize = 256;
    let zeros: Vec<u32> = vec![0; N];
    let buf = test_alloc_buffer_with_data(&device, &zeros, BufferKind::Scattered);
    let idx = buf.resource_index(ResourceAccess::Write).unwrap();

    const ROUNDS: u32 = 20;
    let mut last_tv = None;
    for _ in 0..ROUNDS {
        let mut graph = TaskGraph::new();
        graph
            .node("add_ten", &add_pipe)
            .with_buffer(&*buf, NodeAccess::ReadWrite)
            .with_resource_slots_slice(&[idx])
            .dispatch((N / 64) as u32, 1, 1);
        last_tv = Some(graph.submit(&ctx).unwrap());
    }

    ctx.wait_until(last_tv.unwrap()).unwrap();

    let result = readback_u32(&device, &buf, N);
    let expected = ROUNDS * 10;
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
    }
}

/// Stress: clear large buffer with indirect dispatch reading it.
///
/// Exercises the ClearBuffer → DispatchIndirect path. The indirect argument
/// buffer is written by one dispatch, then a second dispatch is launched
/// indirectly. The cleared data buffer must be fully zero when the indirect
/// dispatch reads it.
#[test]
fn stress_clear_then_indirect_dispatch() {
    let device = make_device();
    let ctx = submission_context(&device);

    // Shader that writes dispatch args: (N/64, 1, 1) at offset 0
    let write_args_shader_src = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> args, ThreadId id) {
    args[0] = 4;  // 256/64 workgroups
    args[1] = 1;
    args[2] = 1;
}
"#;
    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();
    let write_args_shader = ShaderModule::from_slang(&device, write_args_shader_src).unwrap();
    let write_args_pipe = ComputePipeline::new(&device, &write_args_shader).unwrap();

    const N: usize = 256;

    let nonzero: Vec<u32> = (1..=N as u32).collect();
    let buf = test_alloc_buffer_with_data(&device, &nonzero, BufferKind::Scattered);
    let out = test_alloc_buffer(
        &device,
        (N * 4) as u64,
        BufferKind::Scattered,
        None,
        BufferFlags::empty(),
    );
    let args = test_alloc_buffer(&device, 12, BufferKind::Scattered, None, BufferFlags::empty());

    let buf_idx = buf.resource_index(ResourceAccess::Write).unwrap();
    let out_idx = out.resource_index(ResourceAccess::Write).unwrap();
    let args_idx = args.resource_index(ResourceAccess::Write).unwrap();

    let mut graph = TaskGraph::new();
    graph.clear_parcel(&*buf, 0, (N * 4) as u64).unwrap();
    graph
        .node("write_args", &write_args_pipe)
        .with_buffer(&*args, NodeAccess::Write)
        .with_resource_slots_slice(&[args_idx])
        .dispatch(1, 1, 1);
    graph
        .node("copy_indirect", &copy_pipe)
        .with_buffer(&*buf, NodeAccess::Read)
        .with_buffer(&*out, NodeAccess::Write)
        .with_buffer(&*args, NodeAccess::Read)
        .with_resource_slots_slice(&[buf_idx, out_idx])
        .dispatch_indirect_parcel(&*args, 0)
        .unwrap();

    graph.dispatch(&ctx).unwrap();

    let result = readback_u32(&device, &out, N);
    let nonzero_count = result.iter().filter(|&&v| v != 0).count();
    assert_eq!(
        nonzero_count, 0,
        "expected all zeros after clear + indirect dispatch, but {nonzero_count}/{N} were nonzero"
    );
}

/// Stress: write → dispatch → write → dispatch chain.
///
/// Two write_buffer nodes each followed by a dispatch that reads the buffer,
/// all in one graph. The second write overwrites what the first dispatch read.
/// Tests WAW and RAW barrier insertion in sequence.
// Scheme migration: see scheme_stress_alternating_write_dispatch
#[test]
fn stress_alternating_write_dispatch() {
    let device = make_device();
    let ctx = submission_context(&device);

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    const N: usize = 256;

    let buf = test_alloc_buffer(
        &device,
        (N * 4) as u64,
        BufferKind::Scattered,
        None,
        BufferFlags::empty(),
    );
    let out1 = test_alloc_buffer(
        &device,
        (N * 4) as u64,
        BufferKind::Scattered,
        None,
        BufferFlags::empty(),
    );
    let out2 = test_alloc_buffer(
        &device,
        (N * 4) as u64,
        BufferKind::Scattered,
        None,
        BufferFlags::empty(),
    );

    let buf_idx = buf.resource_index(ResourceAccess::Write).unwrap();
    let out1_idx = out1.resource_index(ResourceAccess::Write).unwrap();
    let out2_idx = out2.resource_index(ResourceAccess::Write).unwrap();

    let data1: Vec<u32> = (0..N as u32).map(|i| i + 100).collect();
    let data2: Vec<u32> = (0..N as u32).map(|i| i + 200).collect();
    let bytes1: Vec<u8> = bytemuck::cast_slice(&data1).to_vec();
    let bytes2: Vec<u8> = bytemuck::cast_slice(&data2).to_vec();

    let mut graph = TaskGraph::new();
    // Phase 1: write data1 → copy to out1
    graph.write_parcel(&*buf, 0, bytes1).unwrap();
    graph
        .node("copy1", &copy_pipe)
        .with_buffer(&*buf, NodeAccess::Read)
        .with_buffer(&*out1, NodeAccess::Write)
        .with_resource_slots_slice(&[buf_idx, out1_idx])
        .dispatch((N / 64) as u32, 1, 1);
    // Phase 2: write data2 (overwrites buf) → copy to out2
    graph.write_parcel(&*buf, 0, bytes2).unwrap();
    graph
        .node("copy2", &copy_pipe)
        .with_buffer(&*buf, NodeAccess::Read)
        .with_buffer(&*out2, NodeAccess::Write)
        .with_resource_slots_slice(&[buf_idx, out2_idx])
        .dispatch((N / 64) as u32, 1, 1);

    graph.dispatch(&ctx).unwrap();

    let result1 = readback_u32(&device, &out1, N);
    for (i, &val) in result1.iter().enumerate() {
        assert_eq!(val, data1[i], "out1[{i}]: expected {}, got {val}", data1[i]);
    }
    let result2 = readback_u32(&device, &out2, N);
    for (i, &val) in result2.iter().enumerate() {
        assert_eq!(val, data2[i], "out2[{i}]: expected {}, got {val}", data2[i]);
    }
}
