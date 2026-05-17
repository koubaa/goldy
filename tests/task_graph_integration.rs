//! Task graph integration tests.
//!
//! These tests verify that `TaskGraph` produces correct GPU results using
//! real backends, exercising dependency analysis, barrier insertion, and wave
//! scheduling on actual hardware.
//! They are only compiled when at least one backend feature is enabled.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

use goldy::{
    Buffer, ComputeEncoder, ComputePipeline, DataAccess, NodeAccess, ShaderModule, TaskGraph,
};

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
    if std::env::var("GOLDY_DX12_ALLOW_WARP")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        if let Ok(dev) = instance.create_device_for_adapter(goldy::WARP_ADAPTER_ID) {
            return dev;
        }
    }

    instance
        .create_device(goldy::DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(goldy::DeviceType::IntegratedGpu))
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
#[test]
fn graph_linear_chain() {
    let device = make_device();

    let double_shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap();
    let add_shader = ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap();
    let double_pipe = ComputePipeline::new(&device, &double_shader).unwrap();
    let add_pipe = ComputePipeline::new(&device, &add_shader).unwrap();

    let input: Vec<u32> = (0..64).collect();
    let src = Buffer::with_data(&device, &input, DataAccess::Scattered).unwrap();
    let dst = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();

    let src_idx = src.bindless_index().unwrap();
    let dst_idx = dst.bindless_index().unwrap();

    let mut graph = TaskGraph::new();
    graph
        .node("double", &double_pipe)
        .bind_buffer(&src, NodeAccess::Read)
        .bind_buffer(&dst, NodeAccess::Write)
        .bind_resources_raw_slice(&[src_idx, dst_idx])
        .dispatch(1, 1, 1);

    graph
        .node("add_ten", &add_pipe)
        .bind_buffer(&dst, NodeAccess::ReadWrite)
        .bind_resources_raw_slice(&[dst_idx])
        .dispatch(1, 1, 1);

    graph.dispatch(&device).unwrap();

    let result = readback_u32(&device, &dst, 64);
    for (i, &val) in result.iter().enumerate() {
        let expected = (i as u32) * 2 + 10;
        assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
    }
}

/// Two independent dispatches writing to separate buffers.
/// Exercises the no-barrier path (both should land in wave 0).
#[test]
fn graph_independent_dispatches() {
    let device = make_device();

    let fill42_shader = ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap();
    let fill99_shader = ShaderModule::from_slang(&device, FILL_99_SHADER).unwrap();
    let pipe_42 = ComputePipeline::new(&device, &fill42_shader).unwrap();
    let pipe_99 = ComputePipeline::new(&device, &fill99_shader).unwrap();

    let buf_a = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();
    let buf_b = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();

    let idx_a = buf_a.bindless_index().unwrap();
    let idx_b = buf_b.bindless_index().unwrap();

    let mut graph = TaskGraph::new();
    graph
        .node("fill_a", &pipe_42)
        .bind_buffer(&buf_a, NodeAccess::Write)
        .bind_resources_raw_slice(&[idx_a])
        .dispatch(1, 1, 1);
    graph
        .node("fill_b", &pipe_99)
        .bind_buffer(&buf_b, NodeAccess::Write)
        .bind_resources_raw_slice(&[idx_b])
        .dispatch(1, 1, 1);

    graph.dispatch(&device).unwrap();

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
#[test]
fn graph_diamond_dependency() {
    let device = make_device();

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

    let src = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();
    let y = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();
    let z = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();
    let out = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();

    let src_idx = src.bindless_index().unwrap();
    let y_idx = y.bindless_index().unwrap();
    let z_idx = z.bindless_index().unwrap();
    let out_idx = out.bindless_index().unwrap();

    let mut graph = TaskGraph::new();

    // A: fill src with thread index
    graph
        .node("fill_src", &fill_pipe)
        .bind_buffer(&src, NodeAccess::Write)
        .bind_resources_raw_slice(&[src_idx])
        .dispatch(1, 1, 1);

    // B: double src -> y
    graph
        .node("double_to_y", &double_pipe)
        .bind_buffer(&src, NodeAccess::Read)
        .bind_buffer(&y, NodeAccess::Write)
        .bind_resources_raw_slice(&[src_idx, y_idx])
        .dispatch(1, 1, 1);

    // C: double src -> z
    graph
        .node("double_to_z", &double_pipe)
        .bind_buffer(&src, NodeAccess::Read)
        .bind_buffer(&z, NodeAccess::Write)
        .bind_resources_raw_slice(&[src_idx, z_idx])
        .dispatch(1, 1, 1);

    // D: sum y + z -> out
    graph
        .node("sum_yz", &sum_pipe)
        .bind_buffer(&y, NodeAccess::Read)
        .bind_buffer(&z, NodeAccess::Read)
        .bind_buffer(&out, NodeAccess::Write)
        .bind_resources_raw_slice(&[y_idx, z_idx, out_idx])
        .dispatch(1, 1, 1);

    graph.dispatch(&device).unwrap();

    let result = readback_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        let expected = (i as u32) * 4;
        assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
    }
}

/// TaskGraph produces the same result as manual ComputeEncoder for the same workload.
#[test]
fn graph_matches_encoder() {
    let device = make_device();

    let double_shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap();
    let add_shader = ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap();
    let double_pipe = ComputePipeline::new(&device, &double_shader).unwrap();
    let add_pipe = ComputePipeline::new(&device, &add_shader).unwrap();

    let input: Vec<u32> = (0..64).collect();

    // --- Run via ComputeEncoder (manual barriers) ---
    let src_enc = Buffer::with_data(&device, &input, DataAccess::Scattered).unwrap();
    let dst_enc = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();

    let src_enc_idx = src_enc.bindless_index().unwrap();
    let dst_enc_idx = dst_enc.bindless_index().unwrap();

    {
        let mut encoder = ComputeEncoder::new();
        {
            let mut pass = encoder.begin_compute_pass();
            pass.set_pipeline(&double_pipe);
            pass.bind_resources_raw(&[src_enc_idx, dst_enc_idx]);
            pass.dispatch(1, 1, 1);
        }
        encoder.dispatch(&device).unwrap();
    }
    {
        let mut encoder = ComputeEncoder::new();
        {
            let mut pass = encoder.begin_compute_pass();
            pass.set_pipeline(&add_pipe);
            pass.bind_resources_raw(&[dst_enc_idx]);
            pass.dispatch(1, 1, 1);
        }
        encoder.dispatch(&device).unwrap();
    }

    let result_enc = readback_u32(&device, &dst_enc, 64);

    // --- Run via TaskGraph ---
    let src_graph = Buffer::with_data(&device, &input, DataAccess::Scattered).unwrap();
    let dst_graph = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();

    let src_graph_idx = src_graph.bindless_index().unwrap();
    let dst_graph_idx = dst_graph.bindless_index().unwrap();

    let mut graph = TaskGraph::new();
    graph
        .node("double", &double_pipe)
        .bind_buffer(&src_graph, NodeAccess::Read)
        .bind_buffer(&dst_graph, NodeAccess::Write)
        .bind_resources_raw_slice(&[src_graph_idx, dst_graph_idx])
        .dispatch(1, 1, 1);
    graph
        .node("add_ten", &add_pipe)
        .bind_buffer(&dst_graph, NodeAccess::ReadWrite)
        .bind_resources_raw_slice(&[dst_graph_idx])
        .dispatch(1, 1, 1);
    graph.dispatch(&device).unwrap();

    let result_graph = readback_u32(&device, &dst_graph, 64);

    // Both should produce identical results
    assert_eq!(
        result_enc, result_graph,
        "TaskGraph should match ComputeEncoder output"
    );
}

/// Non-blocking submit via TaskGraph.
#[test]
fn graph_nonblocking_submit() {
    let device = make_device();

    let fill_shader = ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap();
    let pipe = ComputePipeline::new(&device, &fill_shader).unwrap();

    let buf = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();
    let idx = buf.bindless_index().unwrap();

    let mut graph = TaskGraph::new();
    graph
        .node("fill", &pipe)
        .bind_buffer(&buf, NodeAccess::Write)
        .bind_resources_raw_slice(&[idx])
        .dispatch(1, 1, 1);

    let tv = graph.submit(&device).unwrap();
    device.wait_until(tv).unwrap();

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
#[test]
fn clear_then_dispatch_reads_zeros() {
    let device = make_device();

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    // Allocate and fill a buffer with nonzero values.
    let nonzero: Vec<u32> = (1..=64).collect();
    let buf = Buffer::with_data(&device, &nonzero, DataAccess::Scattered).unwrap();
    let out = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();

    let buf_idx = buf.bindless_index().unwrap();
    let out_idx = out.bindless_index().unwrap();

    // Build a graph: clear buf → copy buf→out
    // The analyzer must insert a barrier between the clear and the copy.
    let mut graph = TaskGraph::new();
    graph.clear_buffer(&buf, 0, 64 * 4);
    graph
        .node("copy", &copy_pipe)
        .bind_buffer(&buf, NodeAccess::Read)
        .bind_buffer(&out, NodeAccess::Write)
        .bind_resources_raw_slice(&[buf_idx, out_idx])
        .dispatch(1, 1, 1);

    graph.dispatch(&device).unwrap();

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
#[test]
fn write_then_dispatch_reads_uploaded_data() {
    let device = make_device();

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    // Buffer starts empty (zeroed).
    let buf = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();
    let out = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();

    let buf_idx = buf.bindless_index().unwrap();
    let out_idx = out.bindless_index().unwrap();

    // Known data to upload: values 100..163.
    let known_data: Vec<u32> = (100..164).collect();
    let data_bytes: Vec<u8> = bytemuck::cast_slice(&known_data).to_vec();

    // Build a graph: write known_data into buf → copy buf→out
    let mut graph = TaskGraph::new();
    graph.write_buffer(&buf, 0, data_bytes);
    graph
        .node("copy", &copy_pipe)
        .bind_buffer(&buf, NodeAccess::Read)
        .bind_buffer(&out, NodeAccess::Write)
        .bind_resources_raw_slice(&[buf_idx, out_idx])
        .dispatch(1, 1, 1);

    graph.dispatch(&device).unwrap();

    let result = readback_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        let expected = known_data[i];
        assert_eq!(
            val, expected,
            "element {i}: expected {expected} after write_buffer, got {val}"
        );
    }
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
#[test]
fn stress_clear_then_dispatch_large() {
    let device = make_device();

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    const N: usize = 16384;
    let nonzero: Vec<u32> = (1..=N as u32).collect();
    let buf = Buffer::with_data(&device, &nonzero, DataAccess::Scattered).unwrap();
    let out = Buffer::new(&device, (N * 4) as u64, DataAccess::Scattered).unwrap();

    let buf_idx = buf.bindless_index().unwrap();
    let out_idx = out.bindless_index().unwrap();

    let mut graph = TaskGraph::new();
    graph.clear_buffer(&buf, 0, (N * 4) as u64);
    graph
        .node("copy", &copy_pipe)
        .bind_buffer(&buf, NodeAccess::Read)
        .bind_buffer(&out, NodeAccess::Write)
        .bind_resources_raw_slice(&[buf_idx, out_idx])
        .dispatch((N / 64) as u32, 1, 1);

    graph.dispatch(&device).unwrap();

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
#[test]
fn stress_many_clears_many_dispatches() {
    let device = make_device();

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    const N: usize = 1024;
    const NUM_BUFS: usize = 8;

    let mut srcs = Vec::new();
    let mut outs = Vec::new();
    for _ in 0..NUM_BUFS {
        let nonzero: Vec<u32> = (1..=N as u32).collect();
        srcs.push(Buffer::with_data(&device, &nonzero, DataAccess::Scattered).unwrap());
        outs.push(Buffer::new(&device, (N * 4) as u64, DataAccess::Scattered).unwrap());
    }

    let mut graph = TaskGraph::new();
    for src in &srcs {
        graph.clear_buffer(src, 0, (N * 4) as u64);
    }
    for (src, out) in srcs.iter().zip(outs.iter()) {
        let src_idx = src.bindless_index().unwrap();
        let out_idx = out.bindless_index().unwrap();
        graph
            .node("copy", &copy_pipe)
            .bind_buffer(src, NodeAccess::Read)
            .bind_buffer(out, NodeAccess::Write)
            .bind_resources_raw_slice(&[src_idx, out_idx])
            .dispatch((N / 64) as u32, 1, 1);
    }

    graph.dispatch(&device).unwrap();

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
#[test]
fn stress_clear_write_dispatch_chain() {
    let device = make_device();

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    const N: usize = 1024;
    let nonzero: Vec<u32> = (1..=N as u32).collect();
    let buf = Buffer::with_data(&device, &nonzero, DataAccess::Scattered).unwrap();
    let out = Buffer::new(&device, (N * 4) as u64, DataAccess::Scattered).unwrap();

    let buf_idx = buf.bindless_index().unwrap();
    let out_idx = out.bindless_index().unwrap();

    let known_data: Vec<u32> = (0..N as u32).map(|i| i * 7 + 42).collect();
    let data_bytes: Vec<u8> = bytemuck::cast_slice(&known_data).to_vec();

    let mut graph = TaskGraph::new();
    graph.clear_buffer(&buf, 0, (N * 4) as u64);
    graph.write_buffer(&buf, 0, data_bytes);
    graph
        .node("copy", &copy_pipe)
        .bind_buffer(&buf, NodeAccess::Read)
        .bind_buffer(&out, NodeAccess::Write)
        .bind_resources_raw_slice(&[buf_idx, out_idx])
        .dispatch((N / 64) as u32, 1, 1);

    graph.dispatch(&device).unwrap();

    let result = readback_u32(&device, &out, N);
    for (i, &val) in result.iter().enumerate() {
        let expected = known_data[i];
        assert_eq!(
            val, expected,
            "element {i}: expected {expected}, got {val}"
        );
    }
}

/// Stress: two-phase submission — submit one graph, then submit a second graph
/// that reads the output of the first.
///
/// Tests inter-submission synchronization (tail barrier correctness).
/// Mimics Ekrano's coarse→fine two-phase rendering where flush_mid_frame
/// submits the coarse graph and then the fine graph reads its output.
#[test]
fn stress_two_phase_submission() {
    let device = make_device();

    let double_shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).unwrap();
    let add_shader = ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap();
    let double_pipe = ComputePipeline::new(&device, &double_shader).unwrap();
    let add_pipe = ComputePipeline::new(&device, &add_shader).unwrap();

    const N: usize = 4096;
    let input: Vec<u32> = (0..N as u32).collect();
    let buf = Buffer::with_data(&device, &input, DataAccess::Scattered).unwrap();
    let buf_idx = buf.bindless_index().unwrap();

    // Phase 1: double in-place. Use a tmp buffer since DOUBLE_SHADER reads
    // from one buffer and writes to another.
    let tmp = Buffer::new(&device, (N * 4) as u64, DataAccess::Scattered).unwrap();
    let tmp_idx = tmp.bindless_index().unwrap();

    {
        let mut graph = TaskGraph::new();
        graph
            .node("double", &double_pipe)
            .bind_buffer(&buf, NodeAccess::Read)
            .bind_buffer(&tmp, NodeAccess::Write)
            .bind_resources_raw_slice(&[buf_idx, tmp_idx])
            .dispatch((N / 64) as u32, 1, 1);
        let tv = graph.submit(&device).unwrap();
        device.wait_until(tv).unwrap();
    }

    // Phase 2: add 10 to the doubled values.
    {
        let mut graph = TaskGraph::new();
        graph
            .node("add_ten", &add_pipe)
            .bind_buffer(&tmp, NodeAccess::ReadWrite)
            .bind_resources_raw_slice(&[tmp_idx])
            .dispatch((N / 64) as u32, 1, 1);
        graph.dispatch(&device).unwrap();
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
#[test]
fn stress_rapid_submissions() {
    let device = make_device();

    let add_shader = ShaderModule::from_slang(&device, ADD_TEN_SHADER).unwrap();
    let add_pipe = ComputePipeline::new(&device, &add_shader).unwrap();

    const N: usize = 256;
    let zeros: Vec<u32> = vec![0; N];
    let buf = Buffer::with_data(&device, &zeros, DataAccess::Scattered).unwrap();
    let idx = buf.bindless_index().unwrap();

    const ROUNDS: u32 = 20;
    let mut last_tv = None;
    for _ in 0..ROUNDS {
        let mut graph = TaskGraph::new();
        graph
            .node("add_ten", &add_pipe)
            .bind_buffer(&buf, NodeAccess::ReadWrite)
            .bind_resources_raw_slice(&[idx])
            .dispatch((N / 64) as u32, 1, 1);
        last_tv = Some(graph.submit(&device).unwrap());
    }

    device.wait_until(last_tv.unwrap()).unwrap();

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
    let write_args_shader =
        ShaderModule::from_slang(&device, write_args_shader_src).unwrap();
    let write_args_pipe = ComputePipeline::new(&device, &write_args_shader).unwrap();

    const N: usize = 256;

    let nonzero: Vec<u32> = (1..=N as u32).collect();
    let buf = Buffer::with_data(&device, &nonzero, DataAccess::Scattered).unwrap();
    let out = Buffer::new(&device, (N * 4) as u64, DataAccess::Scattered).unwrap();
    let args = Buffer::new(&device, 12, DataAccess::Scattered).unwrap();

    let buf_idx = buf.bindless_index().unwrap();
    let out_idx = out.bindless_index().unwrap();
    let args_idx = args.bindless_index().unwrap();

    let mut graph = TaskGraph::new();
    graph.clear_buffer(&buf, 0, (N * 4) as u64);
    graph
        .node("write_args", &write_args_pipe)
        .bind_buffer(&args, NodeAccess::Write)
        .bind_resources_raw_slice(&[args_idx])
        .dispatch(1, 1, 1);
    graph
        .node("copy_indirect", &copy_pipe)
        .bind_buffer(&buf, NodeAccess::Read)
        .bind_buffer(&out, NodeAccess::Write)
        .bind_buffer(&args, NodeAccess::Read)
        .bind_resources_raw_slice(&[buf_idx, out_idx])
        .dispatch_indirect(&args, 0);

    graph.dispatch(&device).unwrap();

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
#[test]
fn stress_alternating_write_dispatch() {
    let device = make_device();

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).unwrap();
    let copy_pipe = ComputePipeline::new(&device, &copy_shader).unwrap();

    const N: usize = 256;

    let buf = Buffer::new(&device, (N * 4) as u64, DataAccess::Scattered).unwrap();
    let out1 = Buffer::new(&device, (N * 4) as u64, DataAccess::Scattered).unwrap();
    let out2 = Buffer::new(&device, (N * 4) as u64, DataAccess::Scattered).unwrap();

    let buf_idx = buf.bindless_index().unwrap();
    let out1_idx = out1.bindless_index().unwrap();
    let out2_idx = out2.bindless_index().unwrap();

    let data1: Vec<u32> = (0..N as u32).map(|i| i + 100).collect();
    let data2: Vec<u32> = (0..N as u32).map(|i| i + 200).collect();
    let bytes1: Vec<u8> = bytemuck::cast_slice(&data1).to_vec();
    let bytes2: Vec<u8> = bytemuck::cast_slice(&data2).to_vec();

    let mut graph = TaskGraph::new();
    // Phase 1: write data1 → copy to out1
    graph.write_buffer(&buf, 0, bytes1);
    graph
        .node("copy1", &copy_pipe)
        .bind_buffer(&buf, NodeAccess::Read)
        .bind_buffer(&out1, NodeAccess::Write)
        .bind_resources_raw_slice(&[buf_idx, out1_idx])
        .dispatch((N / 64) as u32, 1, 1);
    // Phase 2: write data2 (overwrites buf) → copy to out2
    graph.write_buffer(&buf, 0, bytes2);
    graph
        .node("copy2", &copy_pipe)
        .bind_buffer(&buf, NodeAccess::Read)
        .bind_buffer(&out2, NodeAccess::Write)
        .bind_resources_raw_slice(&[buf_idx, out2_idx])
        .dispatch((N / 64) as u32, 1, 1);

    graph.dispatch(&device).unwrap();

    let result1 = readback_u32(&device, &out1, N);
    for (i, &val) in result1.iter().enumerate() {
        assert_eq!(val, data1[i], "out1[{i}]: expected {}, got {val}", data1[i]);
    }
    let result2 = readback_u32(&device, &out2, N);
    for (i, &val) in result2.iter().enumerate() {
        assert_eq!(val, data2[i], "out2[{i}]: expected {}, got {val}", data2[i]);
    }
}
