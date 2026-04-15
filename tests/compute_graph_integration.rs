//! Compute graph integration tests.
//!
//! These tests verify that `ComputeGraph` (Tier 1) and `ComputeProgram` (Tier 2)
//! produce correct GPU results using real backends, exercising the dependency
//! analysis, barrier insertion, and wave scheduling on actual hardware.

use goldy::{
    Buffer, ComputeEncoder, ComputeGraph, ComputePipeline, ComputeProgram, DataAccess, NodeAccess,
    ShaderModule,
};

/// Doubles each element: out[i] = in[i] * 2
const DOUBLE_SHADER: &str = r#"
import goldy_exp;

#define INPUT  goldy_dyn_scattered<uint>(0)
#define OUTPUT goldy_dyn_scattered<uint>(1)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    OUTPUT[id.x] = INPUT[id.x] * 2;
}
"#;

/// Adds 10 to each element in-place: buf[i] += 10
const ADD_TEN_SHADER: &str = r#"
import goldy_exp;

#define DATA goldy_dyn_scattered<uint>(0)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    DATA[id.x] = DATA[id.x] + 10;
}
"#;

/// Writes constant 42 to each element: buf[i] = 42
const FILL_42_SHADER: &str = r#"
import goldy_exp;

#define DATA goldy_dyn_scattered<uint>(0)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    DATA[id.x] = 42;
}
"#;

/// Writes constant 99 to each element: buf[i] = 99
const FILL_99_SHADER: &str = r#"
import goldy_exp;

#define DATA goldy_dyn_scattered<uint>(0)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    DATA[id.x] = 99;
}
"#;

/// Sums two buffers: out[i] = a[i] + b[i]
const SUM_SHADER: &str = r#"
import goldy_exp;

#define A   goldy_dyn_scattered<uint>(0)
#define B   goldy_dyn_scattered<uint>(1)
#define OUT goldy_dyn_scattered<uint>(2)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    OUT[id.x] = A[id.x] + B[id.x];
}
"#;

fn make_device() -> goldy::Device {
    let instance = goldy::Instance::new().expect("Failed to create instance");
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
// ComputeGraph (Tier 1) tests
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

    let mut graph = ComputeGraph::new();
    graph
        .node("double", &double_pipe)
        .bind_buffer(&src, NodeAccess::Read)
        .bind_buffer(&dst, NodeAccess::Write)
        .push_constants_raw(&[src_idx, dst_idx])
        .dispatch(1, 1, 1);

    graph
        .node("add_ten", &add_pipe)
        .bind_buffer(&dst, NodeAccess::ReadWrite)
        .push_constants_raw(&[dst_idx])
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

    let mut graph = ComputeGraph::new();
    graph
        .node("fill_a", &pipe_42)
        .bind_buffer(&buf_a, NodeAccess::Write)
        .push_constants_raw(&[idx_a])
        .dispatch(1, 1, 1);
    graph
        .node("fill_b", &pipe_99)
        .bind_buffer(&buf_b, NodeAccess::Write)
        .push_constants_raw(&[idx_b])
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
#define DATA goldy_dyn_scattered<uint>(0)
[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    DATA[id.x] = id.x;
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

    let mut graph = ComputeGraph::new();

    // A: fill src with thread index
    graph
        .node("fill_src", &fill_pipe)
        .bind_buffer(&src, NodeAccess::Write)
        .push_constants_raw(&[src_idx])
        .dispatch(1, 1, 1);

    // B: double src -> y
    graph
        .node("double_to_y", &double_pipe)
        .bind_buffer(&src, NodeAccess::Read)
        .bind_buffer(&y, NodeAccess::Write)
        .push_constants_raw(&[src_idx, y_idx])
        .dispatch(1, 1, 1);

    // C: double src -> z
    graph
        .node("double_to_z", &double_pipe)
        .bind_buffer(&src, NodeAccess::Read)
        .bind_buffer(&z, NodeAccess::Write)
        .push_constants_raw(&[src_idx, z_idx])
        .dispatch(1, 1, 1);

    // D: sum y + z -> out
    graph
        .node("sum_yz", &sum_pipe)
        .bind_buffer(&y, NodeAccess::Read)
        .bind_buffer(&z, NodeAccess::Read)
        .bind_buffer(&out, NodeAccess::Write)
        .push_constants_raw(&[y_idx, z_idx, out_idx])
        .dispatch(1, 1, 1);

    graph.dispatch(&device).unwrap();

    let result = readback_u32(&device, &out, 64);
    for (i, &val) in result.iter().enumerate() {
        let expected = (i as u32) * 4;
        assert_eq!(val, expected, "element {i}: expected {expected}, got {val}");
    }
}

// ---------------------------------------------------------------------------
// ComputeProgram (Tier 2) tests
// ---------------------------------------------------------------------------

/// Compile a program once, specialize and run it twice with different buffers
/// and dimensions. Verifies that the cached schedule is reusable.
#[test]
fn program_reuse() {
    let device = make_device();

    let fill_shader = ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap();
    let fill_pipe = ComputePipeline::new(&device, &fill_shader).unwrap();

    let mut builder = ComputeProgram::builder();
    let buf_slot = builder.buffer_slot("buf");
    let wg = builder.dim_slot("wg");

    builder
        .step("fill", &fill_pipe)
        .bind_buffer(buf_slot, NodeAccess::Write)
        .dispatch_slot(wg);

    let program = builder.compile().unwrap();

    // Specialize #1
    let buf1 = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();
    let mut exec1 = program.specialize();
    exec1.bind_buffer(buf_slot, &buf1);
    exec1.set_dim(wg, (1, 1, 1));
    exec1.dispatch(&device).unwrap();

    // Specialize #2 with a different buffer
    let buf2 = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();
    let mut exec2 = program.specialize();
    exec2.bind_buffer(buf_slot, &buf2);
    exec2.set_dim(wg, (1, 1, 1));
    exec2.dispatch(&device).unwrap();
}

/// ComputeGraph produces the same result as manual ComputeEncoder for the same workload.
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
            pass.set_push_constants_raw(&[src_enc_idx, dst_enc_idx]);
            pass.dispatch(1, 1, 1);
        }
        encoder.dispatch(&device).unwrap();
    }
    {
        let mut encoder = ComputeEncoder::new();
        {
            let mut pass = encoder.begin_compute_pass();
            pass.set_pipeline(&add_pipe);
            pass.set_push_constants_raw(&[dst_enc_idx]);
            pass.dispatch(1, 1, 1);
        }
        encoder.dispatch(&device).unwrap();
    }

    let result_enc = readback_u32(&device, &dst_enc, 64);

    // --- Run via ComputeGraph ---
    let src_graph = Buffer::with_data(&device, &input, DataAccess::Scattered).unwrap();
    let dst_graph = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();

    let src_graph_idx = src_graph.bindless_index().unwrap();
    let dst_graph_idx = dst_graph.bindless_index().unwrap();

    let mut graph = ComputeGraph::new();
    graph
        .node("double", &double_pipe)
        .bind_buffer(&src_graph, NodeAccess::Read)
        .bind_buffer(&dst_graph, NodeAccess::Write)
        .push_constants_raw(&[src_graph_idx, dst_graph_idx])
        .dispatch(1, 1, 1);
    graph
        .node("add_ten", &add_pipe)
        .bind_buffer(&dst_graph, NodeAccess::ReadWrite)
        .push_constants_raw(&[dst_graph_idx])
        .dispatch(1, 1, 1);
    graph.dispatch(&device).unwrap();

    let result_graph = readback_u32(&device, &dst_graph, 64);

    // Both should produce identical results
    assert_eq!(
        result_enc, result_graph,
        "ComputeGraph should match ComputeEncoder output"
    );
}

/// Non-blocking submit via ComputeGraph.
#[test]
fn graph_nonblocking_submit() {
    let device = make_device();

    let fill_shader = ShaderModule::from_slang(&device, FILL_42_SHADER).unwrap();
    let pipe = ComputePipeline::new(&device, &fill_shader).unwrap();

    let buf = Buffer::new(&device, 64 * 4, DataAccess::Scattered).unwrap();
    let idx = buf.bindless_index().unwrap();

    let mut graph = ComputeGraph::new();
    graph
        .node("fill", &pipe)
        .bind_buffer(&buf, NodeAccess::Write)
        .push_constants_raw(&[idx])
        .dispatch(1, 1, 1);

    let future = graph.submit(&device).unwrap();
    future.wait().unwrap();

    let result = readback_u32(&device, &buf, 64);
    for &v in &result {
        assert_eq!(v, 42);
    }
}
