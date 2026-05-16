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
