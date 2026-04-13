//! Compute pipeline integration tests.
//!
//! These tests verify compute pipeline functionality with actual GPU backends.

mod common;

use goldy::{
    types::{SpatialAccess, TextureFlags, TextureFormat},
    Buffer, BufferPool, ComputeEncoder, ComputePipeline, DataAccess, DeviceType, Instance,
    ShaderModule, Texture,
};

/// Simple compute shader that doubles each value in a buffer.
const DOUBLE_SHADER: &str = r#"
import goldy_exp;

#define DATA goldy_dyn_scattered<uint>(0)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    DATA[id.x] = DATA[id.x] * 2;
}
"#;

/// Compute shader that reads from one buffer and writes to another.
const COPY_SHADER: &str = r#"
import goldy_exp;

#define INPUT goldy_dyn_scattered<uint>(0)
#define OUTPUT goldy_dyn_scattered<uint>(1)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    OUTPUT[id.x] = INPUT[id.x];
}
"#;

#[test]
fn test_compute_pipeline_creation() {
    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader =
        ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("Failed to compile shader");

    let pipeline = ComputePipeline::new(&device, &shader);

    assert!(
        pipeline.is_ok(),
        "Failed to create compute pipeline: {:?}",
        pipeline.err()
    );
}

#[test]
fn test_compute_pipeline_no_bindings() {
    // A minimal compute shader with no bindings
    const MINIMAL_SHADER: &str = r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    // Do nothing - just test pipeline creation
}
"#;

    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader =
        ShaderModule::from_slang(&device, MINIMAL_SHADER).expect("Failed to compile shader");

    let pipeline = ComputePipeline::new(&device, &shader);

    assert!(
        pipeline.is_ok(),
        "Failed to create minimal compute pipeline: {:?}",
        pipeline.err()
    );
}

#[test]
fn test_compute_dispatch_empty() {
    // Test dispatching a compute shader with no resources
    const MINIMAL_SHADER: &str = r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
}
"#;

    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader =
        ShaderModule::from_slang(&device, MINIMAL_SHADER).expect("Failed to compile shader");

    let pipeline =
        ComputePipeline::new(&device, &shader).expect("Failed to create compute pipeline");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.dispatch(1, 1, 1);
    }

    let result = encoder.dispatch(&device);
    assert!(result.is_ok(), "Failed to dispatch: {:?}", result.err());
}

#[test]
fn test_compute_with_uav_buffer() {
    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader =
        ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("Failed to compile shader");

    // Create buffer with initial data
    let initial_data: Vec<u32> = (0..64).collect();
    let buffer = Buffer::with_data(&device, &initial_data, DataAccess::Scattered)
        .expect("Failed to create buffer");

    let pipeline =
        ComputePipeline::new(&device, &shader).expect("Failed to create compute pipeline");

    // Dispatch compute
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        // Pass buffer indices via push constants
        pass.set_push_constants(&[&buffer]);
        pass.dispatch(1, 1, 1); // 64 threads total
    }

    let result = encoder.dispatch(&device);
    assert!(result.is_ok(), "Failed to dispatch: {:?}", result.err());

    // Note: We can't easily read back the buffer without mapping support
    // This test just verifies the dispatch doesn't crash
}

#[test]
fn test_compute_with_srv_and_uav() {
    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("Failed to compile shader");

    // Create input buffer (read-only)
    let input_data: Vec<u32> = (0..64).collect();
    let input_buffer = Buffer::with_data(&device, &input_data, DataAccess::Scattered)
        .expect("Failed to create input buffer");

    // Create output buffer (read-write)
    let output_data: Vec<u32> = vec![0; 64];
    let output_buffer = Buffer::with_data(&device, &output_data, DataAccess::Scattered)
        .expect("Failed to create output buffer");

    let pipeline =
        ComputePipeline::new(&device, &shader).expect("Failed to create compute pipeline");

    // Dispatch compute
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        // Pass buffer indices via push constants
        // Order matches shader slots: [input (slot 0), output (slot 1)]
        pass.set_push_constants(&[&input_buffer, &output_buffer]);
        pass.dispatch(1, 1, 1); // 64 threads
    }

    let result = encoder.dispatch(&device);
    assert!(
        result.is_ok(),
        "Failed to dispatch with SRV+UAV: {:?}",
        result.err()
    );
}

/// Compute shader that increments each value by 1.
const INCREMENT_SHADER: &str = r#"
import goldy_exp;

#define DATA goldy_dyn_scattered<uint>(0)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    DATA[id.x] = DATA[id.x] + 1;
}
"#;

/// Compute shader that sums six input buffers into an output buffer.
/// Exercises bindless slots 0–5 (slot indices 4+ were broken before the 16-slot fix).
const SIX_SLOT_SUM_SHADER: &str = r#"
import goldy_exp;

#define A   goldy_dyn_scattered<uint>(0)
#define B   goldy_dyn_scattered<uint>(1)
#define C   goldy_dyn_scattered<uint>(2)
#define D   goldy_dyn_scattered<uint>(3)
#define E   goldy_dyn_scattered<uint>(4)
#define OUT goldy_dyn_scattered<uint>(5)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= 16) return;
    OUT[idx] = A[idx] + B[idx] + C[idx] + D[idx] + E[idx];
}
"#;

/// Helper: create a device (discrete or integrated).
fn make_device() -> goldy::Device {
    let instance = goldy::Instance::new().expect("Failed to create instance");
    instance
        .create_device(goldy::DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(goldy::DeviceType::IntegratedGpu))
        .expect("Failed to create device")
}

// ─── Buffer read_to_cpu / clear tests ────────────────────────────────────────

/// Write data via a compute shader then read it back, verifying correctness
/// of the full GPU staging round-trip (write → dispatch → readback).
#[test]
fn test_compute_write_and_readback() {
    let device = make_device();

    let shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let initial: Vec<u32> = (0..64).collect();
    let buffer =
        Buffer::with_data(&device, &initial, DataAccess::Scattered).expect("create buffer");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants(&[&buffer]);
        pass.dispatch(1, 1, 1); // 64 threads, each doubles one element
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut output = vec![0u8; 64 * 4];
    buffer
        .read_to_cpu(&device, &mut output)
        .expect("read_to_cpu");

    let result: &[u32] = bytemuck::cast_slice(&output);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val,
            (i as u32) * 2,
            "element {} expected {} got {}",
            i,
            i * 2,
            val
        );
    }
}

/// `Buffer::clear` (standalone, immediate) zeros the whole buffer.
#[test]
fn test_buffer_clear_standalone() {
    let device = make_device();

    let data: Vec<u32> = vec![0xDEAD_BEEF; 64];
    let buffer = Buffer::with_data(&device, &data, DataAccess::Scattered).expect("create buffer");

    buffer.clear(&device, 0, 0).expect("clear (full)");

    let mut output = vec![0u8; 64 * 4];
    buffer
        .read_to_cpu(&device, &mut output)
        .expect("read_to_cpu");

    let result: &[u32] = bytemuck::cast_slice(&output);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val, 0,
            "element {} should be 0 after clear, got {:#x}",
            i, val
        );
    }
}

/// `Buffer::clear` with an explicit range zeros only that slice.
#[test]
fn test_buffer_clear_partial() {
    let device = make_device();

    // 64 u32s = 256 bytes. Clear bytes 64–128 (elements 16–31).
    let sentinel = 0xDEAD_BEEFu32;
    let data: Vec<u32> = vec![sentinel; 64];
    let buffer = Buffer::with_data(&device, &data, DataAccess::Scattered).expect("create buffer");

    buffer.clear(&device, 64, 64).expect("partial clear");

    let mut output = vec![0u8; 64 * 4];
    buffer
        .read_to_cpu(&device, &mut output)
        .expect("read_to_cpu");

    let result: &[u32] = bytemuck::cast_slice(&output);
    for (i, &val) in result.iter().enumerate() {
        let expected = if (16..32).contains(&i) { 0 } else { sentinel };
        assert_eq!(
            val, expected,
            "element {} expected {:#x} got {:#x}",
            i, expected, val
        );
    }
}

/// `Buffer::clear` with `size = 0` clears from offset to end of buffer.
#[test]
fn test_buffer_clear_to_end() {
    let device = make_device();

    // Fill with sentinel, then clear from element 32 to end (offset 128, size 0).
    let sentinel = 0xCAFE_BABEu32;
    let data: Vec<u32> = vec![sentinel; 64];
    let buffer = Buffer::with_data(&device, &data, DataAccess::Scattered).expect("create buffer");

    buffer.clear(&device, 128, 0).expect("clear to end");

    let mut output = vec![0u8; 64 * 4];
    buffer
        .read_to_cpu(&device, &mut output)
        .expect("read_to_cpu");

    let result: &[u32] = bytemuck::cast_slice(&output);
    for (i, &val) in result.iter().enumerate() {
        let expected = if i < 32 { sentinel } else { 0 };
        assert_eq!(
            val, expected,
            "element {} expected {:#x} got {:#x}",
            i, expected, val
        );
    }
}

// ─── Batched ClearBuffer in compute encoder ───────────────────────────────────

/// `ComputePass::clear_buffer` batches the clear into the command stream.
/// Clears input before the copy dispatch; output should be all zeros.
#[test]
fn test_compute_batched_clear_before_dispatch() {
    let device = make_device();

    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let input: Vec<u32> = vec![0xDEAD_BEEF; 64];
    let input_buf =
        Buffer::with_data(&device, &input, DataAccess::Scattered).expect("input buffer");
    let output_buf = Buffer::with_data(&device, &vec![0xFFFF_FFFFu32; 64], DataAccess::Scattered)
        .expect("output buffer");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        // Clear input before the copy — output should receive zeros.
        pass.clear_buffer(&input_buf, 0, 0);
        pass.set_pipeline(&pipeline);
        pass.set_push_constants(&[&input_buf, &output_buf]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut out = vec![0u8; 64 * 4];
    output_buf.read_to_cpu(&device, &mut out).expect("readback");

    let result: &[u32] = bytemuck::cast_slice(&out);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val, 0,
            "output[{}] should be 0 (copied from cleared input), got {:#x}",
            i, val
        );
    }
}

/// GPU ordering: Dispatch A writes values → ClearBuffer → Dispatch B increments.
/// Correct result is 1 (0 + 1). An ordering bug would give 43 (42 + 1 without the clear).
#[test]
fn test_compute_clear_between_dispatches() {
    let device = make_device();

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile copy");
    let copy_pipeline = ComputePipeline::new(&device, &copy_shader).expect("copy pipeline");

    let inc_shader = ShaderModule::from_slang(&device, INCREMENT_SHADER).expect("compile inc");
    let inc_pipeline = ComputePipeline::new(&device, &inc_shader).expect("inc pipeline");

    // Input with 42s; output starts empty.
    let input_buf =
        Buffer::with_data(&device, &vec![42u32; 64], DataAccess::Scattered).expect("input");
    let output_buf =
        Buffer::with_data(&device, &vec![0u32; 64], DataAccess::Scattered).expect("output");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        // Pass 1: copy 42s into output.
        pass.set_pipeline(&copy_pipeline);
        pass.set_push_constants(&[&input_buf, &output_buf]);
        pass.dispatch(1, 1, 1);
        // Clear output — must happen AFTER the copy dispatch.
        pass.clear_buffer(&output_buf, 0, 0);
        // Pass 2: increment output (zeros → 1s).
        pass.set_pipeline(&inc_pipeline);
        pass.set_push_constants(&[&output_buf]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut out = vec![0u8; 64 * 4];
    output_buf.read_to_cpu(&device, &mut out).expect("readback");

    let result: &[u32] = bytemuck::cast_slice(&out);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val, 1,
            "output[{}]: expected 1 (clear was ordered after copy), got {} \
             (if 43: clear happened before copy; ordering broken)",
            i, val
        );
    }
}

// ─── Indirect dispatch ────────────────────────────────────────────────────────

/// `dispatch_indirect` reads workgroup counts from a buffer.
/// Write [1,1,1] as the dispatch args → shader runs 64 threads → doubles values.
#[test]
fn test_compute_dispatch_indirect() {
    let device = make_device();

    let shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    // Dispatch args: 1 workgroup in each dimension (3 × u32 = 12 bytes).
    let args: [u32; 3] = [1, 1, 1];
    let args_buf = Buffer::with_data(&device, &args, DataAccess::Scattered).expect("args buffer");

    let data: Vec<u32> = (0..64).collect();
    let data_buf = Buffer::with_data(&device, &data, DataAccess::Scattered).expect("data buffer");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants(&[&data_buf]);
        pass.dispatch_indirect(&args_buf, 0);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut output = vec![0u8; 64 * 4];
    data_buf
        .read_to_cpu(&device, &mut output)
        .expect("readback");

    let result: &[u32] = bytemuck::cast_slice(&output);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val,
            (i as u32) * 2,
            "element {} expected {} got {}",
            i,
            i * 2,
            val
        );
    }
}

/// `dispatch_indirect` returns an error when the args buffer has been destroyed.
#[test]
fn test_dispatch_indirect_invalid_buffer() {
    let device = make_device();

    let shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let data_buf =
        Buffer::with_data(&device, &vec![1u32; 64], DataAccess::Scattered).expect("data");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants(&[&data_buf]);

        // Record indirect dispatch with a temp buffer, then drop the buffer.
        // The encoder stores the raw handle; after drop it's stale.
        {
            let temp = Buffer::with_data(&device, &[1u32, 1, 1], DataAccess::Scattered)
                .expect("temp buffer");
            pass.dispatch_indirect(&temp, 0);
        } // temp dropped — backend destroys the buffer here
    }

    let result = encoder.dispatch(&device);
    assert!(
        result.is_err(),
        "Expected error dispatching with a destroyed indirect args buffer"
    );
}

// ─── Many push-constant slots (>4, exercises 16-slot expansion) ───────────────

/// Shader using 6 bindless slots (0–5). Before the 16-slot expansion, slots 4+
/// were mapped to garbage indices and this test would produce wrong results.
#[test]
fn test_compute_many_push_constant_slots() {
    let device = make_device();

    let shader = ShaderModule::from_slang(&device, SIX_SLOT_SUM_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    // Each input buffer contains a constant value; OUT[i] = sum = 1+2+3+4+5 = 15.
    const N: usize = 16;
    let a = Buffer::with_data(&device, &[1u32; N], DataAccess::Scattered).expect("a");
    let b = Buffer::with_data(&device, &[2u32; N], DataAccess::Scattered).expect("b");
    let c = Buffer::with_data(&device, &[3u32; N], DataAccess::Scattered).expect("c");
    let d = Buffer::with_data(&device, &[4u32; N], DataAccess::Scattered).expect("d");
    let e = Buffer::with_data(&device, &[5u32; N], DataAccess::Scattered).expect("e");
    let out = Buffer::with_data(&device, &[0u32; N], DataAccess::Scattered).expect("out");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants(&[&a, &b, &c, &d, &e, &out]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut output = vec![0u8; N * 4];
    out.read_to_cpu(&device, &mut output).expect("readback");

    let result: &[u32] = bytemuck::cast_slice(&output);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val, 15,
            "out[{}] expected 15 (1+2+3+4+5), got {} — slot index 4+ may be misbound",
            i, val
        );
    }
}

/// Test that uses a struct type (like Particle) - exercises same Metal code path as compute_particles.
#[test]
fn test_compute_with_struct_buffer() {
    const PARTICLE_SHADER: &str = r#"
import goldy_exp;

struct Particle {
    float2 position;
    float2 velocity;
};

#define PARTICLES goldy_dyn_scattered<Particle>(0)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= 4) return;
    Particle p = PARTICLES[idx];
    p.position += float2(0.01, 0.01);
    PARTICLES[idx] = p;
}
"#;

    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader =
        ShaderModule::from_slang(&device, PARTICLE_SHADER).expect("Failed to compile shader");

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Particle {
        position: [f32; 2],
        velocity: [f32; 2],
    }

    let particles = vec![
        Particle {
            position: [0.0, 0.0],
            velocity: [0.1, 0.0],
        };
        4
    ];

    let buffer = Buffer::with_data(&device, &particles, DataAccess::Scattered)
        .expect("Failed to create buffer");

    let pipeline =
        ComputePipeline::new(&device, &shader).expect("Failed to create compute pipeline");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants(&[&buffer]);
        pass.dispatch(1, 1, 1);
    }

    let result = encoder.dispatch(&device);
    assert!(
        result.is_ok(),
        "Failed to dispatch with struct buffer: {:?}",
        result.err()
    );
}

// ─── Buffer views: sub-buffer descriptor binding ──────────────────────────────

/// Two views into one buffer. Shader copies from view A to view B.
/// Proves that sub-buffer descriptors with offset/range work end-to-end.
#[test]
fn test_buffer_view_copy_between_sub_regions() {
    let device = make_device();

    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    const N: usize = 64;
    let mut data = vec![0u32; N * 2];
    // First half: source values 1..=64
    for (i, slot) in data.iter_mut().take(N).enumerate() {
        *slot = (i + 1) as u32;
    }
    // Second half: zeros (destination)

    let pool_buf =
        Buffer::with_data(&device, &data, DataAccess::Scattered).expect("create pool buffer");

    let view_a = pool_buf
        .create_view(0, (N * 4) as u64, Some(4))
        .expect("create view A");
    let view_b = pool_buf
        .create_view((N * 4) as u64, (N * 4) as u64, Some(4))
        .expect("create view B");

    let idx_a = view_a.bindless_index().expect("view A bindless index");
    let idx_b = view_b.bindless_index().expect("view B bindless index");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants_raw(&[idx_a, idx_b]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    // Read back the entire pool buffer and check the second half
    let mut output = vec![0u8; N * 2 * 4];
    pool_buf
        .read_to_cpu(&device, &mut output)
        .expect("read_to_cpu");

    let result: &[u32] = bytemuck::cast_slice(&output);
    for (i, &val) in result[N..].iter().enumerate() {
        assert_eq!(
            val,
            (i + 1) as u32,
            "dest[{}]: expected {} (copied from source view), got {}",
            i,
            i + 1,
            val
        );
    }
}

/// Shader doubles values in a view — the other half of the buffer must be untouched.
#[test]
fn test_buffer_view_isolation() {
    let device = make_device();

    let shader = ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    const N: usize = 64;
    let mut data = vec![0u32; N * 2];
    data[..N].fill(100); // first half: sentinel
    for (i, slot) in data[N..].iter_mut().enumerate() {
        *slot = (i + 1) as u32; // second half: values to double
    }

    let pool_buf =
        Buffer::with_data(&device, &data, DataAccess::Scattered).expect("create pool buffer");

    // View only the second half
    let view = pool_buf
        .create_view((N * 4) as u64, (N * 4) as u64, Some(4))
        .expect("create view");

    let idx = view.bindless_index().expect("view bindless index");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants_raw(&[idx]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut output = vec![0u8; N * 2 * 4];
    pool_buf
        .read_to_cpu(&device, &mut output)
        .expect("read_to_cpu");

    let result: &[u32] = bytemuck::cast_slice(&output);

    // First half must be untouched
    for (i, &val) in result[..N].iter().enumerate() {
        assert_eq!(
            val, 100,
            "sentinel[{}] was modified (expected 100, got {})",
            i, val
        );
    }

    // Second half must be doubled
    for (i, &val) in result[N..].iter().enumerate() {
        let expected = ((i + 1) as u32) * 2;
        assert_eq!(
            val, expected,
            "view[{}]: expected {} (doubled), got {}",
            i, expected, val
        );
    }
}

// ─── BufferPool convenience wrapper ───────────────────────────────────────────

/// Allocate typed regions from a pool, write via the backing buffer, dispatch.
#[test]
fn test_buffer_pool_alloc_and_dispatch() {
    let device = make_device();
    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    const N: usize = 64;
    let pool_size = 256 * 3; // 3 x 256-byte aligned regions
    let mut pool = BufferPool::new(&device, pool_size as u64).expect("create pool");

    let src_view = pool.alloc::<u32>(N as u64).expect("alloc src");
    let dst_view = pool.alloc::<u32>(N as u64).expect("alloc dst");

    // Write source data into the backing buffer at the correct offset
    let src_data: Vec<u32> = (1..=N as u32).collect();
    pool.backing_buffer()
        .write_data(0, &src_data)
        .expect("write src data");

    let src_idx = src_view.bindless_index().expect("src bindless index");
    let dst_idx = dst_view.bindless_index().expect("dst bindless index");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants_raw(&[src_idx, dst_idx]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    // Read back entire pool and verify the destination region
    let mut output = vec![0u8; pool_size];
    pool.backing_buffer()
        .read_to_cpu(&device, &mut output)
        .expect("readback");

    // Destination starts at 256 bytes (first aligned offset after 64*4=256)
    let dst_offset = 256usize;
    let dst_slice: &[u32] = bytemuck::cast_slice(&output[dst_offset..dst_offset + N * 4]);
    for (i, &val) in dst_slice.iter().enumerate() {
        assert_eq!(
            val,
            (i + 1) as u32,
            "pool dst[{}]: expected {}, got {}",
            i,
            i + 1,
            val
        );
    }

    assert!(pool.used() > 0);
    assert!(pool.remaining() < pool.capacity());
}

/// alloc_with_data allocates and uploads in one call; verify via readback.
#[test]
fn test_buffer_pool_alloc_with_data() {
    let device = make_device();
    const N: usize = 64;
    let total = BufferPool::padded_size(&[(N, std::mem::size_of::<u32>())]);
    let mut pool = BufferPool::new(&device, total).expect("create pool");
    let data: Vec<u32> = (1..=N as u32).collect();
    let view = pool.alloc_with_data(&data).expect("alloc_with_data");
    assert_eq!(view.size(), (N * std::mem::size_of::<u32>()) as u64);

    let mut output = vec![0u8; total as usize];
    pool.backing_buffer()
        .read_to_cpu(&device, &mut output)
        .expect("readback");
    let roundtripped: &[u32] = bytemuck::cast_slice(&output[..N * 4]);
    for (i, &val) in roundtripped.iter().enumerate() {
        assert_eq!(val, (i + 1) as u32, "mismatch at index {}", i);
    }
}

/// alloc_with_data with empty slice allocates zero-length view.
#[test]
fn test_buffer_pool_alloc_with_data_empty() {
    let device = make_device();
    let mut pool = BufferPool::new(&device, 1024).expect("create pool");
    let view = pool
        .alloc_with_data::<u32>(&[])
        .expect("alloc_with_data empty");
    assert_eq!(view.size(), 0);
}

// ─── goldy_exp utility correctness ────────────────────────────────────────────

/// `positive_mod(x, m)` must always return a value in `[0, m)`.
///
/// HLSL `fmod` returns negative values when `x < 0`, which breaks UV wrapping.
/// This test verifies the double-fmod formula on the actual GPU path.
#[test]
fn test_positive_mod_correctness() {
    const SHADER: &str = r#"
import goldy_exp;

#define OUT goldy_dyn_scattered<float>(0)

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    // scalar: negative dividend
    OUT[0] = positive_mod(-1.0, 3.0);    // 2.0
    OUT[1] = positive_mod(-3.0, 3.0);    // 0.0
    OUT[2] = positive_mod(-0.5, 1.0);    // 0.5

    // scalar: positive / zero inputs (must be unchanged)
    OUT[3] = positive_mod(2.5, 3.0);     // 2.5
    OUT[4] = positive_mod(0.0, 1.0);     // 0.0

    // float2 overload
    float2 r = positive_mod(float2(-1.0, -0.5), float2(3.0, 1.0));
    OUT[5] = r.x;   // 2.0
    OUT[6] = r.y;   // 0.5
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SHADER).expect("compile positive_mod shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let buf = Buffer::with_data(&device, &[0.0f32; 7], DataAccess::Scattered)
        .expect("create output buffer");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants(&[&buf]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 7 * 4];
    buf.read_to_cpu(&device, &mut raw).expect("read_to_cpu");
    let result: &[f32] = bytemuck::cast_slice(&raw);

    let eps = 1e-5f32;
    let cases: &[(usize, f32, &str)] = &[
        (0, 2.0, "positive_mod(-1, 3)"),
        (1, 0.0, "positive_mod(-3, 3)"),
        (2, 0.5, "positive_mod(-0.5, 1)"),
        (3, 2.5, "positive_mod(2.5, 3)"),
        (4, 0.0, "positive_mod(0, 1)"),
        (5, 2.0, "float2 positive_mod x"),
        (6, 0.5, "float2 positive_mod y"),
    ];
    for &(i, expected, label) in cases {
        assert!(
            (result[i] - expected).abs() < eps,
            "{}: expected {}, got {}",
            label,
            expected,
            result[i]
        );
    }
}

/// `modelview_right` extracts column 0 from a 4×4 matrix, and
/// `billboard_cylindrical_offset` offsets a point along that vector.
#[test]
fn test_billboard_math() {
    const SHADER: &str = r#"
import goldy_exp;

#define OUT goldy_dyn_scattered<float>(0)

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    // Row-major construction. Column 0 = (m[0][0], m[1][0], m[2][0]) = (1, 5, 9).
    float4x4 m = float4x4(
        1, 0, 0, 0,
        5, 1, 0, 0,
        9, 0, 1, 0,
        0, 0, 0, 1
    );
    float3 r = modelview_right(m);
    OUT[0] = r.x;   // 1.0
    OUT[1] = r.y;   // 5.0
    OUT[2] = r.z;   // 9.0

    // center=(1,2,3), cam_right=(1,0,0), offset=5 → (6, 2, 3)
    float3 off = billboard_cylindrical_offset(
        float3(1.0, 2.0, 3.0),
        float3(1.0, 0.0, 0.0),
        5.0
    );
    OUT[3] = off.x;  // 6.0
    OUT[4] = off.y;  // 2.0
    OUT[5] = off.z;  // 3.0

    // Identity matrix: right = (1, 0, 0)
    float4x4 ident = float4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    );
    float3 ident_right = modelview_right(ident);
    OUT[6] = ident_right.x;  // 1.0
    OUT[7] = ident_right.y;  // 0.0
    OUT[8] = ident_right.z;  // 0.0
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SHADER).expect("compile billboard shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let buf = Buffer::with_data(&device, &[0.0f32; 9], DataAccess::Scattered)
        .expect("create output buffer");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants(&[&buf]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 9 * 4];
    buf.read_to_cpu(&device, &mut raw).expect("read_to_cpu");
    let result: &[f32] = bytemuck::cast_slice(&raw);

    let eps = 1e-5f32;
    let cases: &[(usize, f32, &str)] = &[
        (0, 1.0, "modelview_right col0.x"),
        (1, 5.0, "modelview_right col0.y"),
        (2, 9.0, "modelview_right col0.z"),
        (3, 6.0, "cylindrical offset x"),
        (4, 2.0, "cylindrical offset y (unchanged)"),
        (5, 3.0, "cylindrical offset z (unchanged)"),
        (6, 1.0, "identity right.x"),
        (7, 0.0, "identity right.y"),
        (8, 0.0, "identity right.z"),
    ];
    for &(i, expected, label) in cases {
        assert!(
            (result[i] - expected).abs() < eps,
            "{}: expected {}, got {}",
            label,
            expected,
            result[i]
        );
    }
}

// ─── RWStructuredBuffer<T> typed variable assignment ──────────────────────────

/// Verify `goldy_dyn_buf_ro` / `goldy_dyn_scattered` can be assigned to locals and used together.
/// `goldy_dyn_buf_ro` returns `StructuredBuffer<T>` on DX12 (SRV) but `StorageBuffer<T>` on SPIRV/Metal;
/// use `var` so Slang infers the correct type per target. Push constants: slot 0 = read buffer
/// (`bindless_srv_index()` on DX12, same as `bindless_index()` on Vulkan), slot 1 = UAV.
#[test]
fn test_dyn_scattered_typed_variable_assignment() {
    const SHADER: &str = r#"
import goldy_exp;

struct Pair { uint a; uint b; };

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    ReadOnlyBuffer<Pair> input = goldy_dyn_buf_ro<Pair>(0);
    StorageBuffer<Pair> output = goldy_dyn_scattered<Pair>(1);
    uint idx = id.x;
    if (idx >= 8) return;
    Pair p = input[idx];
    output[idx].a = p.a + p.b;
    output[idx].b = p.a * p.b;
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SHADER).expect("compile typed-var shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Pair {
        a: u32,
        b: u32,
    }

    let input_data: Vec<Pair> = (0..8)
        .map(|i| Pair {
            a: i + 1,
            b: i + 10,
        })
        .collect();
    let input_buf =
        Buffer::with_data(&device, &input_data, DataAccess::Scattered).expect("input buffer");
    let output_buf = Buffer::with_data(&device, &[Pair { a: 0, b: 0 }; 8], DataAccess::Scattered)
        .expect("output buffer");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        // goldy_dyn_buf_ro uses SRV on DX12; goldy_dyn_scattered uses UAV
        pass.set_push_constants_raw(&[
            input_buf.bindless_srv_index().expect("srv"),
            output_buf.bindless_index().expect("uav"),
        ]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 8 * std::mem::size_of::<Pair>()];
    output_buf.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[Pair] = bytemuck::cast_slice(&raw);

    for i in 0..8u32 {
        let expected_a = (i + 1) + (i + 10);
        let expected_b = (i + 1) * (i + 10);
        assert_eq!(
            result[i as usize].a, expected_a,
            "output[{}].a: expected {}, got {}",
            i, expected_a, result[i as usize].a
        );
        assert_eq!(
            result[i as usize].b, expected_b,
            "output[{}].b: expected {}, got {}",
            i, expected_b, result[i as usize].b
        );
    }
}

// ─── Heap overflow: allocations exceeding primary heap ────────────────────────

/// Allocate 80 MB across 10 buffers (exceeds the default 64 MB primary heap),
/// copy from the first to the last via a compute shader, and verify correctness.
/// This proves that overflow heap creation and multi-heap `use_heap` work.
#[test]
fn test_heap_overflow_allocation() {
    const LARGE_COPY_SHADER: &str = r#"
import goldy_exp;

#define INPUT  goldy_dyn_scattered<uint>(0)
#define OUTPUT goldy_dyn_scattered<uint>(1)

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    if (idx >= 2097152) return;  // 8 MB / 4 bytes = 2M elements
    OUTPUT[idx] = INPUT[idx];
}
"#;

    let device = make_device();
    let shader =
        ShaderModule::from_slang(&device, LARGE_COPY_SHADER).expect("compile large copy shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    const BUF_SIZE: u64 = 8 * 1024 * 1024; // 8 MB each
    const NUM_BUFFERS: usize = 10; // 80 MB total > 64 MB primary
    const ELEM_COUNT: usize = (BUF_SIZE / 4) as usize;

    let mut buffers = Vec::with_capacity(NUM_BUFFERS);
    for i in 0..NUM_BUFFERS {
        let data: Vec<u32> = if i == 0 {
            (0..ELEM_COUNT as u32).collect()
        } else {
            vec![0u32; ELEM_COUNT]
        };
        buffers.push(
            Buffer::with_data(&device, &data, DataAccess::Scattered)
                .unwrap_or_else(|e| panic!("Failed to create buffer {}: {}", i, e)),
        );
    }

    let workgroups = (ELEM_COUNT as u32).div_ceil(64);
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants(&[&buffers[0], &buffers[NUM_BUFFERS - 1]]);
        pass.dispatch(workgroups, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut output = vec![0u8; BUF_SIZE as usize];
    buffers[NUM_BUFFERS - 1]
        .read_to_cpu(&device, &mut output)
        .expect("read_to_cpu");

    let result: &[u32] = bytemuck::cast_slice(&output);
    for i in (0..ELEM_COUNT).step_by(1024) {
        assert_eq!(
            result[i], i as u32,
            "element {} expected {} got {} — overflow heap copy failed",
            i, i, result[i]
        );
    }
}

#[test]
fn test_compute_write_to_texture() {
    const SHADER: &str = r#"
import goldy_exp;

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    RWTexture2D<float4> output = goldy_dyn_direct_spatial<float4>(0);
    uint2 dims;
    output.GetDimensions(dims.x, dims.y);
    if (id.x < dims.x && id.y < dims.y) {
        output[id.xy] = float4(1.0, 0.0, 0.0, 1.0);
    }
}
"#;

    let instance = Instance::new().expect("instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .or_else(|_| instance.create_device(DeviceType::Other))
        .expect("device");

    let shader = ShaderModule::from_slang(&device, SHADER).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    let width = 16u32;
    let height = 16u32;
    let texture = Texture::new(
        &device,
        width,
        height,
        TextureFormat::Rgba8Unorm,
        SpatialAccess::Direct,
        TextureFlags::COPY_SRC,
    )
    .expect("texture");

    let wg_x = (width + 7) / 8;
    let wg_y = (height + 7) / 8;
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants_raw(&[texture.bindless_index().expect("tex bindless")]);
        pass.dispatch(wg_x, wg_y, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut output = vec![0u8; (width * height * 4) as usize];
    texture.read_to_cpu(&mut output).expect("readback");

    let nonzero = output.iter().filter(|&&b| b != 0).count();
    assert!(
        nonzero > 0,
        "Texture readback is all zeros after compute write ({} bytes)",
        output.len()
    );
    assert_eq!(output[0], 255, "R channel should be 255 (solid red)");
    assert_eq!(output[1], 0, "G channel should be 0");
    assert_eq!(output[2], 0, "B channel should be 0");
    assert_eq!(output[3], 255, "A channel should be 255");
}

// ─── SRV view with FirstElement != 0 (WARP bug target) ───────────────────────

/// Read from a buffer view via SRV (goldy_dyn_buf_ro) where the view has a
/// non-zero offset (FirstElement != 0 in the descriptor). On WARP this has been
/// observed to read from element 0 instead of FirstElement.
#[test]
fn test_srv_view_with_first_element_offset() {
    const SRV_COPY_SHADER: &str = r#"
import goldy_exp;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    ReadOnlyBuffer<uint> input = goldy_dyn_buf_ro<uint>(0);
    StorageBuffer<uint> output = goldy_dyn_scattered<uint>(1);
    output[id.x] = input[id.x];
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SRV_COPY_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    const N: usize = 256;
    let mut data = vec![0u32; N * 2];
    for (i, slot) in data.iter_mut().enumerate() {
        *slot = (i + 1) as u32;
    }

    let pool_buf =
        Buffer::with_data(&device, &data, DataAccess::Scattered).expect("create pool buffer");

    // View B starts at offset N (FirstElement = N in the SRV descriptor)
    let view_b = pool_buf
        .create_view((N * 4) as u64, (N * 4) as u64, Some(4))
        .expect("create view B");

    let output_buf =
        Buffer::with_data(&device, &vec![0u32; N], DataAccess::Scattered).expect("output buffer");

    let srv_idx = view_b.bindless_srv_index().expect("view B SRV index");
    let uav_idx = output_buf.bindless_index().expect("output UAV index");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_push_constants_raw(&[srv_idx, uav_idx]);
        pass.dispatch(N as u32 / 64, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut output = vec![0u8; N * 4];
    output_buf
        .read_to_cpu(&device, &mut output)
        .expect("read_to_cpu");

    let result: &[u32] = bytemuck::cast_slice(&output);
    let mut mismatches = 0;
    for (i, &val) in result.iter().enumerate() {
        let expected = (N + i + 1) as u32;
        if val != expected {
            if mismatches < 5 {
                eprintln!(
                    "MISMATCH output[{}]: got {} expected {} (FirstElement={})",
                    i, val, expected, N
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "SRV view read returned wrong data ({} of {} elements wrong). \
         This indicates the WARP SRV FirstElement bug.",
        mismatches, N
    );
}

/// Kitchen-sink WARP SRV reproducer combining ALL factors from Ekrano:
/// - Varying strides (4, 8, 16, 20, 24, 32, 96) matching Ekrano's buffer layout
/// - Large buffers (N=4096 elements per view)
/// - GPU-written indirect dispatch args (pipeline_setup computes workgroup counts)
/// - All dispatches use dispatch_indirect
/// - Multiple pipeline switches (6 different shader pipelines)
/// - UAV fill → SRV read pattern on clip_inp (stride 8)
/// - Many churn dispatches with mixed SRV/UAV bindings
/// Standalone-buffer SRV test: matches Ekrano's no-pool path where SRV still
/// fails on WARP. Creates many standalone (non-pooled) buffers with various
/// strides, fills via UAV, reads back via SRV through a heavy dispatch chain.
/// Also adds massive descriptor heap pressure via dummy resources and
/// interleaved texture/buffer creation.
#[test]
fn test_warp_srv_standalone_heavy() {
    if cfg!(target_os = "macos") {
        eprintln!("Skipping test_warp_srv_standalone_heavy on macOS");
        return;
    }

    const FILL_U2_SHADER: &str = r#"
import goldy_exp;

struct Pair { uint a; uint b; };

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    StorageBuffer<Pair> dst = goldy_dyn_scattered<Pair>(0);
    ReadOnlyBuffer<uint> params = goldy_dyn_buf_ro<uint>(1);
    uint base = params[0];
    Pair p;
    p.a = base + id.x * 2;
    p.b = base + id.x * 2 + 1;
    dst[id.x] = p;
}
"#;

    const COPY_U2_SHADER: &str = r#"
import goldy_exp;

struct Pair { uint a; uint b; };

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    ReadOnlyBuffer<Pair> src = goldy_dyn_buf_ro<Pair>(0);
    StorageBuffer<Pair> dst = goldy_dyn_scattered<Pair>(1);
    dst[id.x] = src[id.x];
}
"#;

    const FILL_U1_SHADER: &str = r#"
import goldy_exp;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    StorageBuffer<uint> dst = goldy_dyn_scattered<uint>(0);
    ReadOnlyBuffer<uint> params = goldy_dyn_buf_ro<uint>(1);
    dst[id.x] = params[0] + id.x;
}
"#;

    const CHURN_SHADER: &str = r#"
import goldy_exp;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    ReadOnlyBuffer<uint> src0 = goldy_dyn_buf_ro<uint>(0);
    ReadOnlyBuffer<uint> src1 = goldy_dyn_buf_ro<uint>(1);
    ReadOnlyBuffer<uint> src2 = goldy_dyn_buf_ro<uint>(2);
    ReadOnlyBuffer<uint> src3 = goldy_dyn_buf_ro<uint>(3);
    StorageBuffer<uint> dst0 = goldy_dyn_scattered<uint>(4);
    StorageBuffer<uint> dst1 = goldy_dyn_scattered<uint>(5);
    StorageBuffer<uint> dst2 = goldy_dyn_scattered<uint>(6);
    uint v = src0[id.x] + src1[id.x] + src2[id.x] + src3[id.x];
    dst0[id.x] = v;
    dst1[id.x] = v + 100;
    dst2[id.x] = v + 200;
}
"#;

    const CHURN2_SHADER: &str = r#"
import goldy_exp;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    ReadOnlyBuffer<uint> src0 = goldy_dyn_buf_ro<uint>(0);
    ReadOnlyBuffer<uint> src1 = goldy_dyn_buf_ro<uint>(1);
    StorageBuffer<uint> dst0 = goldy_dyn_scattered<uint>(2);
    StorageBuffer<uint> dst1 = goldy_dyn_scattered<uint>(3);
    dst0[id.x] = src0[id.x] + src1[id.x];
    dst1[id.x] = src0[id.x] * 2;
}
"#;

    // GPU-writes indirect dispatch args
    const SETUP_SHADER: &str = r#"
import goldy_exp;

struct IndirectArgs { uint x; uint y; uint z; uint pad; };

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    StorageBuffer<IndirectArgs> indirect = goldy_dyn_scattered<IndirectArgs>(0);
    ReadOnlyBuffer<uint> wg_counts = goldy_dyn_buf_ro<uint>(1);
    uint slot = tid.x;
    IndirectArgs args;
    args.x = wg_counts[slot];
    args.y = 1;
    args.z = 1;
    args.pad = 0;
    indirect[slot] = args;
}
"#;

    let device = make_device();

    let fill_u2_pipe = {
        let s = ShaderModule::from_slang(&device, FILL_U2_SHADER).expect("fill u2");
        ComputePipeline::new(&device, &s).expect("fill u2 pipe")
    };
    let copy_u2_pipe = {
        let s = ShaderModule::from_slang(&device, COPY_U2_SHADER).expect("copy u2");
        ComputePipeline::new(&device, &s).expect("copy u2 pipe")
    };
    let fill_u1_pipe = {
        let s = ShaderModule::from_slang(&device, FILL_U1_SHADER).expect("fill u1");
        ComputePipeline::new(&device, &s).expect("fill u1 pipe")
    };
    let churn_pipe = {
        let s = ShaderModule::from_slang(&device, CHURN_SHADER).expect("churn");
        ComputePipeline::new(&device, &s).expect("churn pipe")
    };
    let churn2_pipe = {
        let s = ShaderModule::from_slang(&device, CHURN2_SHADER).expect("churn2");
        ComputePipeline::new(&device, &s).expect("churn2 pipe")
    };
    let setup_pipe = {
        let s = ShaderModule::from_slang(&device, SETUP_SHADER).expect("setup");
        ComputePipeline::new(&device, &s).expect("setup pipe")
    };

    const N: usize = 4096;
    const N_WG: u32 = N as u32 / 64;

    // ---------------------------------------------------------------
    // Phase A: Create 200 dummy buffers interleaved with textures to
    //          push descriptor heap offsets high and fragment the
    //          heap layout (buffer SRV, texture SRV, buffer UAV, ...).
    // ---------------------------------------------------------------
    let mut _dummy_bufs: Vec<Buffer> = Vec::new();
    let mut _dummy_texs: Vec<Texture> = Vec::new();
    let strides: &[usize] = &[4, 8, 12, 16, 20, 24, 32, 96];
    for i in 0..200 {
        let stride = strides[i % strides.len()];
        let buf = Buffer::new_with_stride(
            &device,
            (N * stride) as u64,
            DataAccess::Scattered,
            Some(stride as u32),
        )
        .expect("dummy buf");
        _dummy_bufs.push(buf);

        if i % 10 == 0 {
            let tex = Texture::new(
                &device,
                64,
                64,
                TextureFormat::Rgba8Unorm,
                SpatialAccess::Interpolated,
                TextureFlags::COPY_DST,
            )
            .expect("dummy tex");
            _dummy_texs.push(tex);
        }
    }

    // ---------------------------------------------------------------
    // Phase B: The actual test buffers — all STANDALONE (non-pooled),
    //          matching Ekrano's path when use_pool() returns false.
    //          Each gets its own D3D12 resource + SRV/UAV descriptors.
    // ---------------------------------------------------------------
    let buf_strides: &[usize] = &[
        96, 4, 20, 24, 16, 4, 8, 8, 32, 16, 16, 32, 8, 32, 8, 8, 24, 4, 24,
    ];
    let mut bufs: Vec<Buffer> = Vec::new();
    for &stride in buf_strides {
        let size = (N * stride) as u64;
        let buf =
            Buffer::new_with_stride(&device, size, DataAccess::Scattered, Some(stride as u32))
                .expect("test buf");
        bufs.push(buf);
    }
    for _ in 0..15 {
        let buf = Buffer::new_with_stride(&device, (N * 4) as u64, DataAccess::Scattered, Some(4))
            .expect("scratch");
        bufs.push(buf);
    }

    // More interleaved textures after the test buffers
    let _tex_gradient = Texture::new(
        &device,
        512,
        512,
        TextureFormat::Rgba8Unorm,
        SpatialAccess::Interpolated,
        TextureFlags::COPY_DST,
    )
    .expect("gradient tex");
    let _tex_atlas = Texture::new(
        &device,
        1024,
        1024,
        TextureFormat::Rgba8Unorm,
        SpatialAccess::Interpolated,
        TextureFlags::COPY_DST,
    )
    .expect("atlas tex");
    let _tex_output = Texture::new(
        &device,
        1920,
        1080,
        TextureFormat::Rgba8Unorm,
        SpatialAccess::Direct,
        TextureFlags::COPY_SRC,
    )
    .expect("output tex");

    let clip_inp_idx = 6; // stride 8
    let output = Buffer::new_with_stride(&device, (N * 8) as u64, DataAccess::Scattered, Some(8))
        .expect("out");

    // Indirect dispatch buffer with some ZERO workgroup counts mixed in
    let num_ind_slots = 50;
    let indirect_buf = Buffer::with_data(
        &device,
        &vec![0u32; num_ind_slots * 4],
        DataAccess::Scattered,
    )
    .expect("indirect buf");

    let mut wg_data = vec![0u32; num_ind_slots];
    for (i, slot) in wg_data.iter_mut().enumerate() {
        if i < bufs.len() {
            *slot = N_WG;
        } else {
            // Some zero slots — WARP might mishandle zero-dispatch + SRV binding
            *slot = 0;
        }
    }
    // Sprinkle in explicit zero-dispatch slots among the real buffers
    for &z in &[2usize, 5, 10, 15] {
        if z < wg_data.len() {
            wg_data[z] = 0;
        }
    }
    // But ensure clip_inp and stride-4 scratch slots are non-zero
    wg_data[clip_inp_idx] = N_WG;
    for i in 19..bufs.len().min(num_ind_slots) {
        wg_data[i] = N_WG;
    }
    let wg_counts_buf =
        Buffer::with_data(&device, &wg_data, DataAccess::Scattered).expect("wg counts");

    let make_param = |base: u32| -> Buffer {
        Buffer::with_data(&device, &[base], DataAccess::Scattered).expect("param")
    };

    // Stride-4 buffer indices for churn dispatches
    let u4: Vec<usize> = (0..bufs.len())
        .filter(|&i| buf_strides.get(i).copied().unwrap_or(4) == 4)
        .chain(19..bufs.len())
        .collect();
    let nu4 = u4.len();

    // ---------------------------------------------------------------
    // Phase C: Multi-frame dispatch loop (10 frames on WARP)
    // ---------------------------------------------------------------
    let num_frames: usize = if device.device_type() == DeviceType::Cpu {
        10
    } else {
        1
    };
    for frame in 0..num_frames {
        let base_val = 50000 + (frame as u32) * 100000;

        let mut encoder = ComputeEncoder::new();
        {
            let mut pass = encoder.begin_compute_pass();

            // GPU-write indirect dispatch args
            pass.set_pipeline(&setup_pipe);
            pass.set_push_constants_raw(&[
                indirect_buf.bindless_index().unwrap(),
                wg_counts_buf.bindless_srv_index().unwrap(),
            ]);
            pass.dispatch(num_ind_slots as u32, 1, 1);

            // Fill stride-4 standalone buffers
            pass.set_pipeline(&fill_u1_pipe);
            for &v in &[1usize, 5, 17] {
                let fp = make_param((v as u32 + 1) * 10000 + frame as u32);
                pass.set_push_constants_raw(&[
                    bufs[v].bindless_index().unwrap(),
                    fp.bindless_srv_index().unwrap(),
                ]);
                pass.dispatch_indirect(&indirect_buf, (v * 16) as u64);
            }
            // Fill scratch buffers
            for v in 19..bufs.len() {
                let fp = make_param((v as u32 + 1) * 10000 + frame as u32);
                pass.set_push_constants_raw(&[
                    bufs[v].bindless_index().unwrap(),
                    fp.bindless_srv_index().unwrap(),
                ]);
                pass.dispatch_indirect(&indirect_buf, (v * 16) as u64);
            }

            // Fill stride-8 standalone buffers (including clip_inp)
            pass.set_pipeline(&fill_u2_pipe);
            for &v in &[6usize, 7, 12, 14, 15] {
                let fp = make_param(if v == clip_inp_idx {
                    base_val
                } else {
                    (v as u32 + 1) * 10000 + frame as u32
                });
                pass.set_push_constants_raw(&[
                    bufs[v].bindless_index().unwrap(),
                    fp.bindless_srv_index().unwrap(),
                ]);
                pass.dispatch_indirect(&indirect_buf, (v * 16) as u64);
            }

            // 30 churn dispatches: alternating pipelines with stride-4 views
            for stage in 0..30 {
                if stage % 3 == 2 {
                    pass.set_pipeline(&churn2_pipe);
                    pass.set_push_constants_raw(&[
                        bufs[u4[stage % nu4]].bindless_srv_index().unwrap(),
                        bufs[u4[(stage + 1) % nu4]].bindless_srv_index().unwrap(),
                        bufs[u4[(stage + 2) % nu4]].bindless_index().unwrap(),
                        bufs[u4[(stage + 3) % nu4]].bindless_index().unwrap(),
                    ]);
                } else {
                    pass.set_pipeline(&churn_pipe);
                    pass.set_push_constants_raw(&[
                        bufs[u4[stage % nu4]].bindless_srv_index().unwrap(),
                        bufs[u4[(stage + 1) % nu4]].bindless_srv_index().unwrap(),
                        bufs[u4[(stage + 2) % nu4]].bindless_srv_index().unwrap(),
                        bufs[u4[(stage + 3) % nu4]].bindless_srv_index().unwrap(),
                        bufs[u4[(stage + 4) % nu4]].bindless_index().unwrap(),
                        bufs[u4[(stage + 5) % nu4]].bindless_index().unwrap(),
                        bufs[u4[(stage + 6) % nu4]].bindless_index().unwrap(),
                    ]);
                }
                // Use a zero-dispatch slot every 5th stage to probe WARP's
                // handling of zero workgroup count + SRV binding
                let ind_slot = if stage % 5 == 4 {
                    2 // zero-dispatch slot
                } else {
                    u4[stage % nu4]
                };
                pass.dispatch_indirect(&indirect_buf, (ind_slot * 16) as u64);
            }

            // Final SRV read: clip_inp (stride 8) → output
            pass.set_pipeline(&copy_u2_pipe);
            pass.set_push_constants_raw(&[
                bufs[clip_inp_idx].bindless_srv_index().unwrap(),
                output.bindless_index().unwrap(),
            ]);
            pass.dispatch_indirect(&indirect_buf, (clip_inp_idx * 16) as u64);
        }
        encoder.dispatch(&device).expect("dispatch");

        if frame == num_frames - 1 {
            let mut out_bytes = vec![0u8; N * 8];
            output
                .read_to_cpu(&device, &mut out_bytes)
                .expect("read output");
            let result: &[u32] = bytemuck::cast_slice(&out_bytes);

            let mut mismatches = 0;
            for i in 0..N {
                let exp_a = base_val + (i as u32) * 2;
                let exp_b = exp_a + 1;
                let got_a = result[i * 2];
                let got_b = result[i * 2 + 1];
                if got_a != exp_a || got_b != exp_b {
                    if mismatches < 10 {
                        eprintln!(
                            "frame {} Pair[{}]: got ({}, {}) expected ({}, {})",
                            frame, i, got_a, got_b, exp_a, exp_b,
                        );
                    }
                    mismatches += 1;
                }
            }

            assert_eq!(
                mismatches, 0,
                "frame {}: SRV read of standalone stride-8 buffer returned wrong data \
                 ({}/{} wrong). WARP SRV bug on structured buffers.",
                frame, mismatches, N
            );
        }
    }
}

/// Pool-view variant: same heavy pipeline but using BufferPool views (non-zero
/// FirstElement). Kept alongside the standalone test to cover both paths.
#[test]
fn test_warp_srv_pool_combined() {
    if cfg!(target_os = "macos") {
        eprintln!("Skipping test_warp_srv_pool_combined on macOS");
        return;
    }

    const SETUP_SHADER: &str = r#"
import goldy_exp;

struct IndirectArgs { uint x; uint y; uint z; uint pad; };

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    StorageBuffer<IndirectArgs> indirect = goldy_dyn_scattered<IndirectArgs>(0);
    ReadOnlyBuffer<uint> wg_counts = goldy_dyn_buf_ro<uint>(1);
    uint slot = tid.x;
    IndirectArgs args;
    args.x = wg_counts[slot];
    args.y = 1;
    args.z = 1;
    args.pad = 0;
    indirect[slot] = args;
}
"#;

    const FILL_U2_SHADER: &str = r#"
import goldy_exp;

struct Pair { uint a; uint b; };

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    StorageBuffer<Pair> dst = goldy_dyn_scattered<Pair>(0);
    ReadOnlyBuffer<uint> params = goldy_dyn_buf_ro<uint>(1);
    uint base = params[0];
    Pair p;
    p.a = base + id.x * 2;
    p.b = base + id.x * 2 + 1;
    dst[id.x] = p;
}
"#;

    const COPY_U2_SHADER: &str = r#"
import goldy_exp;

struct Pair { uint a; uint b; };

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    ReadOnlyBuffer<Pair> src = goldy_dyn_buf_ro<Pair>(0);
    StorageBuffer<Pair> dst = goldy_dyn_scattered<Pair>(1);
    dst[id.x] = src[id.x];
}
"#;

    const FILL_U1_SHADER: &str = r#"
import goldy_exp;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    StorageBuffer<uint> dst = goldy_dyn_scattered<uint>(0);
    ReadOnlyBuffer<uint> params = goldy_dyn_buf_ro<uint>(1);
    dst[id.x] = params[0] + id.x;
}
"#;

    const CHURN_SHADER: &str = r#"
import goldy_exp;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    ReadOnlyBuffer<uint> src0 = goldy_dyn_buf_ro<uint>(0);
    ReadOnlyBuffer<uint> src1 = goldy_dyn_buf_ro<uint>(1);
    ReadOnlyBuffer<uint> src2 = goldy_dyn_buf_ro<uint>(2);
    ReadOnlyBuffer<uint> src3 = goldy_dyn_buf_ro<uint>(3);
    StorageBuffer<uint> dst0 = goldy_dyn_scattered<uint>(4);
    StorageBuffer<uint> dst1 = goldy_dyn_scattered<uint>(5);
    StorageBuffer<uint> dst2 = goldy_dyn_scattered<uint>(6);
    uint v = src0[id.x] + src1[id.x] + src2[id.x] + src3[id.x];
    dst0[id.x] = v;
    dst1[id.x] = v + 100;
    dst2[id.x] = v + 200;
}
"#;

    const CHURN2_SHADER: &str = r#"
import goldy_exp;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    ReadOnlyBuffer<uint> src0 = goldy_dyn_buf_ro<uint>(0);
    ReadOnlyBuffer<uint> src1 = goldy_dyn_buf_ro<uint>(1);
    StorageBuffer<uint> dst0 = goldy_dyn_scattered<uint>(2);
    StorageBuffer<uint> dst1 = goldy_dyn_scattered<uint>(3);
    dst0[id.x] = src0[id.x] + src1[id.x];
    dst1[id.x] = src0[id.x] * 2;
}
"#;

    let device = make_device();

    let setup_pipe = {
        let s = ShaderModule::from_slang(&device, SETUP_SHADER).expect("setup");
        ComputePipeline::new(&device, &s).expect("setup pipe")
    };
    let fill_u2_pipe = {
        let s = ShaderModule::from_slang(&device, FILL_U2_SHADER).expect("fill u2");
        ComputePipeline::new(&device, &s).expect("fill u2 pipe")
    };
    let copy_u2_pipe = {
        let s = ShaderModule::from_slang(&device, COPY_U2_SHADER).expect("copy u2");
        ComputePipeline::new(&device, &s).expect("copy u2 pipe")
    };
    let fill_u1_pipe = {
        let s = ShaderModule::from_slang(&device, FILL_U1_SHADER).expect("fill u1");
        ComputePipeline::new(&device, &s).expect("fill u1 pipe")
    };
    let churn_pipe = {
        let s = ShaderModule::from_slang(&device, CHURN_SHADER).expect("churn");
        ComputePipeline::new(&device, &s).expect("churn pipe")
    };
    let churn2_pipe = {
        let s = ShaderModule::from_slang(&device, CHURN2_SHADER).expect("churn2");
        ComputePipeline::new(&device, &s).expect("churn2 pipe")
    };

    const N: usize = 4096;

    let allocs: &[(usize, usize)] = &[
        (N, 96), // 0  "config"
        (N, 4),  // 1  "scene"
        (N, 20), // 2  "tagmonoid"
        (N, 24), // 3  "path_bbox"
        (N, 16), // 4  "draw_monoid"
        (N, 4),  // 5  "info_bin_data"
        (N, 8),  // 6  "clip_inp" (THE TARGET)
        (N, 8),  // 7  "clip_bic"
        (N, 32), // 8  "clip_el"
        (N, 16), // 9  "clip_bbox"
        (N, 16), // 10 "draw_bbox"
        (N, 32), // 11 "bump"
        (N, 8),  // 12 "bin_header"
        (N, 32), // 13 "path"
        (N, 8),  // 14 "tile"
        (N, 8),  // 15 "seg_counts"
        (N, 24), // 16 "segments"
        (N, 4),  // 17 "ptcl"
        (N, 24), // 18 "lines"
        (N, 4),  // 19 scratch_u32_0
        (N, 4),  // 20 scratch_u32_1
        (N, 4),  // 21 scratch_u32_2
        (N, 4),  // 22 scratch_u32_3
        (N, 4),  // 23 scratch_u32_4
        (N, 4),  // 24 scratch_u32_5
        (N, 4),  // 25 scratch_u32_6
        (N, 4),  // 26 scratch_u32_7
        (N, 4),  // 27 scratch_u32_8
        (N, 4),  // 28 scratch_u32_9
    ];

    // Descriptor heap pressure: 100 dummy buffers created BEFORE the pool
    let mut _dummies: Vec<Buffer> = Vec::new();
    for i in 0..100 {
        let stride = [4, 8, 16, 32][i % 4];
        let buf = Buffer::with_data(
            &device,
            &vec![0u32; 1024 * stride / 4],
            DataAccess::Scattered,
        )
        .expect("dummy");
        _dummies.push(buf);
    }

    let pool_size = BufferPool::padded_size(allocs);
    let mut pool = BufferPool::new(&device, pool_size).expect("pool");

    let clip_inp_idx = 6;
    let clip_n = N;
    let clip_wg = clip_n as u32 / 64;

    let _gradient_tex = Texture::new(
        &device,
        512,
        512,
        TextureFormat::Rgba8Unorm,
        SpatialAccess::Interpolated,
        TextureFlags::COPY_DST,
    )
    .expect("gradient texture");
    let _atlas_tex = Texture::new(
        &device,
        1024,
        1024,
        TextureFormat::Rgba8Unorm,
        SpatialAccess::Interpolated,
        TextureFlags::COPY_DST,
    )
    .expect("atlas texture");
    let _output_tex = Texture::new(
        &device,
        1920,
        1080,
        TextureFormat::Rgba8Unorm,
        SpatialAccess::Direct,
        TextureFlags::COPY_SRC,
    )
    .expect("output texture");

    let num_slots = 30;
    let indirect_buf =
        Buffer::with_data(&device, &vec![0u32; num_slots * 4], DataAccess::Scattered)
            .expect("indirect buf");

    let mut wg_data = vec![0u32; num_slots];
    for (i, slot) in wg_data.iter_mut().enumerate() {
        *slot = if i < allocs.len() {
            allocs[i].0 as u32 / 64
        } else {
            clip_wg
        };
    }
    let wg_counts_buf =
        Buffer::with_data(&device, &wg_data, DataAccess::Scattered).expect("wg counts");

    let make_param = |base: u32| -> Buffer {
        Buffer::with_data(&device, &[base], DataAccess::Scattered).expect("param")
    };

    let output =
        Buffer::with_data(&device, &vec![0u32; clip_n * 2], DataAccess::Scattered).expect("out");

    let num_frames: usize = if device.device_type() == DeviceType::Cpu {
        10
    } else {
        1
    };
    for frame in 0..num_frames {
        pool.reset();
        pool.backing_buffer()
            .clear(&device, 0, pool_size)
            .expect("clear");

        let mut views = Vec::new();
        for &(count, stride) in allocs {
            let view = pool
                .alloc_bytes((count * stride) as u64, Some(stride as u32))
                .expect("alloc");
            views.push(view);
        }

        let base_val = 50000 + (frame as u32) * 100000;

        let mut encoder = ComputeEncoder::new();
        {
            let mut pass = encoder.begin_compute_pass();

            pass.set_pipeline(&setup_pipe);
            pass.set_push_constants_raw(&[
                indirect_buf.bindless_index().unwrap(),
                wg_counts_buf.bindless_srv_index().unwrap(),
            ]);
            pass.dispatch(num_slots as u32, 1, 1);

            pass.set_pipeline(&fill_u1_pipe);
            for &v in &[1usize, 5, 17] {
                let fp = make_param((v as u32 + 1) * 10000 + frame as u32);
                pass.set_push_constants_raw(&[
                    views[v].bindless_index().unwrap(),
                    fp.bindless_srv_index().unwrap(),
                ]);
                pass.dispatch_indirect(&indirect_buf, (v * 16) as u64);
            }

            pass.set_pipeline(&fill_u2_pipe);
            for &v in &[6usize, 7, 12, 14, 15] {
                let fp = make_param(if v == clip_inp_idx {
                    base_val
                } else {
                    (v as u32 + 1) * 10000 + frame as u32
                });
                pass.set_push_constants_raw(&[
                    views[v].bindless_index().unwrap(),
                    fp.bindless_srv_index().unwrap(),
                ]);
                pass.dispatch_indirect(&indirect_buf, (v * 16) as u64);
            }

            let u4: &[usize] = &[1, 5, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28];
            let nu4 = u4.len();
            for stage in 0..20 {
                if stage % 3 == 2 {
                    pass.set_pipeline(&churn2_pipe);
                    pass.set_push_constants_raw(&[
                        views[u4[stage % nu4]].bindless_srv_index().unwrap(),
                        views[u4[(stage + 1) % nu4]].bindless_srv_index().unwrap(),
                        views[u4[(stage + 2) % nu4]].bindless_index().unwrap(),
                        views[u4[(stage + 3) % nu4]].bindless_index().unwrap(),
                    ]);
                } else {
                    pass.set_pipeline(&churn_pipe);
                    pass.set_push_constants_raw(&[
                        views[u4[stage % nu4]].bindless_srv_index().unwrap(),
                        views[u4[(stage + 1) % nu4]].bindless_srv_index().unwrap(),
                        views[u4[(stage + 2) % nu4]].bindless_srv_index().unwrap(),
                        views[u4[(stage + 3) % nu4]].bindless_srv_index().unwrap(),
                        views[u4[(stage + 4) % nu4]].bindless_index().unwrap(),
                        views[u4[(stage + 5) % nu4]].bindless_index().unwrap(),
                        views[u4[(stage + 6) % nu4]].bindless_index().unwrap(),
                    ]);
                }
                pass.dispatch_indirect(&indirect_buf, (u4[stage % nu4] * 16) as u64);
            }

            pass.set_pipeline(&churn2_pipe);
            pass.set_push_constants_raw(&[
                views[u4[0]].bindless_srv_index().unwrap(),
                views[u4[5]].bindless_srv_index().unwrap(),
                views[u4[8]].bindless_index().unwrap(),
                views[u4[10]].bindless_index().unwrap(),
            ]);
            pass.dispatch_indirect(&indirect_buf, (u4[0] * 16) as u64);

            pass.set_pipeline(&copy_u2_pipe);
            pass.set_push_constants_raw(&[
                views[clip_inp_idx].bindless_srv_index().unwrap(),
                output.bindless_index().unwrap(),
            ]);
            pass.dispatch_indirect(&indirect_buf, (clip_inp_idx * 16) as u64);
        }
        encoder.dispatch(&device).expect("pipeline dispatch");

        if frame == num_frames - 1 {
            let mut out_bytes = vec![0u8; clip_n * 8];
            output
                .read_to_cpu(&device, &mut out_bytes)
                .expect("read output");
            let result: &[u32] = bytemuck::cast_slice(&out_bytes);

            let mut mismatches = 0;
            for i in 0..clip_n {
                let exp_a = base_val + (i as u32) * 2;
                let exp_b = exp_a + 1;
                let got_a = result[i * 2];
                let got_b = result[i * 2 + 1];
                if got_a != exp_a || got_b != exp_b {
                    if mismatches < 10 {
                        eprintln!(
                            "frame {} Pair[{}]: got ({}, {}) expected ({}, {}) \
                             (FirstElement={}, offset={})",
                            frame,
                            i,
                            got_a,
                            got_b,
                            exp_a,
                            exp_b,
                            views[clip_inp_idx].offset() / 8,
                            views[clip_inp_idx].offset(),
                        );
                    }
                    mismatches += 1;
                }
            }

            assert_eq!(
                mismatches, 0,
                "frame {}: SRV read of stride-8 pool view returned wrong data ({}/{} wrong). \
                 WARP may be ignoring FirstElement on the SRV descriptor.",
                frame, mismatches, clip_n
            );
        }
    }
}
