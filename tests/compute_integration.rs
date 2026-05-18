//! Compute pipeline integration tests.
//!
//! These tests verify compute pipeline functionality with actual GPU backends.
//! They are only compiled when at least one backend feature is enabled.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

mod common;

use goldy::{
    types::{BackendType, BufferFlags, SpatialAccess, TextureFlags, TextureFormat},
    Buffer, BufferPool, ComputeEncoder, ComputePipeline, DataAccess, DeviceType, Instance,
    ShaderModule, Texture,
};

/// Simple compute shader that doubles each value in a buffer.
const DOUBLE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] * 2;
}
"#;

/// Compute shader that reads from one buffer and writes to another.
const COPY_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x];
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
        // Bind buffer resource slots
        pass.bind_resources(&[&buffer]);
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
        // Bind buffer resource slots
        // Order matches shader slots: [input (slot 0), output (slot 1)]
        pass.bind_resources(&[&input_buffer, &output_buffer]);
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

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + 1;
}
"#;

/// Compute shader that sums six input buffers into an output buffer.
/// Exercises bindless slots 0–5 (slot indices 4+ were broken before the 16-slot fix).
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

/// Helper: create a device (discrete or integrated).
fn make_device() -> goldy::Device {
    let instance = goldy::Instance::new().expect("Failed to create instance");
    instance
        .create_device(goldy::DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(goldy::DeviceType::IntegratedGpu))
        .expect("Failed to create device")
}

/// Minimal compute shader for headless / validation tests (HLSL-style entry point).
#[cfg(feature = "vulkan")]
const MINIMAL_COMPUTE_FOR_VK_VALIDATION: &str = r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
}
"#;

/// True when the default host process is using the Vulkan backend (e.g. Linux CI).  
/// On Windows with DX12 as default, returns false so we do not spawn a Vulkan-only subprocess.
#[cfg(feature = "vulkan")]
fn vk_api_validation_active_backend_is_vulkan() -> bool {
    let Ok(instance) = Instance::new() else {
        return false;
    };
    let Ok(device) = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
    else {
        return false;
    };
    device.backend_type() == BackendType::Vulkan
}

/// Re-run this integration test binary in a subprocess with Vulkan validation enabled.
/// Parent process skips GPU work (avoids validation overhead + layer state on the shared harness).
#[cfg(feature = "vulkan")]
fn run_in_subprocess_with_vk_validation(test_name: &str) {
    let exe = std::env::current_exe().expect("current_exe for subprocess");
    let output = std::process::Command::new(exe)
        .args([test_name, "--exact", "--nocapture"])
        .env("GOLDY_SUBPROC", "1")
        .env("GOLDY_VALIDATION", "api")
        .env("GOLDY_BACKEND", "vk")
        .env_remove("VK_LAYER_PATH")
        .output()
        .unwrap_or_else(|e| panic!("spawn subprocess for {test_name}: {e}"));

    assert!(
        output.status.success(),
        "Vulkan validation subprocess for `{test_name}` failed (exit {:?})\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Vulkan validation layer regression: timeline semaphores require `timelineSemaphore` device feature.
#[test]
#[cfg(feature = "vulkan")]
fn vk_api_validation_timeline_semaphore() {
    if std::env::var("GOLDY_SUBPROC").is_err() {
        if !vk_api_validation_active_backend_is_vulkan() {
            return;
        }
        run_in_subprocess_with_vk_validation("vk_api_validation_timeline_semaphore");
        return;
    }

    let instance = Instance::new().expect("instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("device");
    let shader =
        ShaderModule::from_slang(&device, MINIMAL_COMPUTE_FOR_VK_VALIDATION).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.dispatch(1, 1, 1);
    }
    let tv = encoder.submit(&device).expect("submit");

    let buf = Buffer::new(&device, 256, DataAccess::Scattered).expect("buffer");
    drop(buf);

    device.wait_until(tv).expect("wait_until");
}

/// Vulkan validation layer regression: per-device resource teardown (no cross-device pool key bugs).
#[test]
#[cfg(feature = "vulkan")]
fn vk_api_validation_two_device_teardown() {
    if std::env::var("GOLDY_SUBPROC").is_err() {
        if !vk_api_validation_active_backend_is_vulkan() {
            return;
        }
        run_in_subprocess_with_vk_validation("vk_api_validation_two_device_teardown");
        return;
    }

    let submit_minimal = |device: &goldy::Device| {
        let shader =
            ShaderModule::from_slang(device, MINIMAL_COMPUTE_FOR_VK_VALIDATION).expect("shader");
        let pipeline = ComputePipeline::new(device, &shader).expect("pipeline");
        let mut encoder = ComputeEncoder::new();
        {
            let mut pass = encoder.begin_compute_pass();
            pass.set_pipeline(&pipeline);
            pass.dispatch(1, 1, 1);
        }
        encoder.dispatch(device).expect("dispatch");
    };

    let i1 = Instance::new().expect("i1");
    let d1 = i1
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| i1.create_device(DeviceType::IntegratedGpu))
        .expect("d1");
    let _b1 = Buffer::new(&d1, 256, DataAccess::Scattered).expect("b1");
    submit_minimal(&d1);

    let i2 = Instance::new().expect("i2");
    let d2 = i2
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| i2.create_device(DeviceType::IntegratedGpu))
        .expect("d2");
    let _b2 = Buffer::new(&d2, 256, DataAccess::Scattered).expect("b2");
    submit_minimal(&d2);

    drop(d1);
    drop(i1);
    drop(d2);
    drop(i2);
}

// ─── Buffer resize (Phase 1: stable handles, realloc-copy fallback) ───────────

#[test]
fn resize_preserves_contents() {
    let device = make_device();
    let mut buf = Buffer::with_data(&device, &[1u32, 2, 3, 4], DataAccess::Scattered).expect("buf");
    buf.resize_to(32).expect("resize");
    let mut out = vec![0u8; 32];
    buf.read_to_cpu(&device, &mut out).expect("read");
    let words: &[u32] = bytemuck::cast_slice(&out);
    assert_eq!(&words[..4], &[1u32, 2, 3, 4]);
    for w in &words[4..8] {
        assert_eq!(*w, 0, "new region should be zero");
    }
}

#[test]
fn resize_preserves_bindless_index() {
    let device = make_device();
    let mut buf = Buffer::new(&device, 16, DataAccess::Scattered).expect("buf");
    let idx = buf.bindless_index().expect("bindless");
    buf.resize_to(256).expect("resize");
    assert_eq!(buf.bindless_index(), Some(idx));
}

#[test]
fn resize_down_truncates() {
    let device = make_device();
    let mut buf =
        Buffer::with_data(&device, &[10u32, 20, 30, 40], DataAccess::Scattered).expect("buf");
    buf.resize_to(8).expect("resize down");
    let mut out = vec![0u8; 8];
    buf.read_to_cpu(&device, &mut out).expect("read");
    let words: &[u32] = bytemuck::cast_slice(&out);
    assert_eq!(words, &[10u32, 20]);
}

#[test]
fn resize_uninitialized_skips_copy() {
    let device = make_device();
    let mut buf =
        Buffer::with_data(&device, &[0xABCD_BEEFu32], DataAccess::Scattered).expect("buf");
    buf.resize_to_uninitialized(8).expect("resize uni");
    let mut out = vec![0u8; 8];
    buf.read_to_cpu(&device, &mut out).expect("read");
}

#[test]
fn buffer_pool_resize() {
    let device = make_device();
    let mut pool = BufferPool::new(&device, 1024).expect("pool");
    let v1 = pool.alloc::<u32>(4).expect("v1");
    let v2 = pool.alloc::<u32>(4).expect("v2");
    let i1 = v1.bindless_index().unwrap();
    let i2 = v2.bindless_index().unwrap();
    pool.resize(2048).expect("resize pool");
    let _v3 = pool.alloc::<u32>(8).expect("v3");
    assert_eq!(v1.bindless_index(), Some(i1));
    assert_eq!(v2.bindless_index(), Some(i2));
}

#[test]
fn new_with_capacity_hint_smoke() {
    let device = make_device();
    let b = Buffer::new_with_capacity_hint(&device, 16, 4096, DataAccess::Scattered).expect("b");
    assert_eq!(b.size(), 16);
    assert!(b.allocated_size() >= 4096, "expected oversize allocation");
}

#[test]
fn oversize_resize_within_capacity_preserves_and_zeros_tail() {
    let device = make_device();
    let mut buf =
        Buffer::new_with_capacity_hint(&device, 16, 4096, DataAccess::Scattered).expect("buf");
    let idx = buf.bindless_index().expect("bindless");
    buf.write(0, &[0xabu8; 16]).expect("seed");
    buf.resize_to(256).expect("grow within cap");
    assert_eq!(buf.bindless_index(), Some(idx));
    assert!(buf.size() >= 256);
    let mut got = vec![0u8; 256];
    buf.read_to_cpu(&device, &mut got).expect("read");
    assert_eq!(&got[..16], &[0xabu8; 16]);
    assert!(got[16..].iter().all(|&x| x == 0));
}

#[test]
fn oversize_resize_beyond_capacity_falls_back_and_preserves() {
    let device = make_device();
    let mut buf =
        Buffer::new_with_capacity_hint(&device, 16, 256, DataAccess::Scattered).expect("buf");
    buf.write(0, &[7u8; 16]).expect("w");
    buf.resize_to(512).expect("grow past cap");
    assert!(buf.allocated_size() >= 512);
    let mut got = vec![0u8; 512];
    buf.read_to_cpu(&device, &mut got).expect("r");
    assert_eq!(&got[..16], &[7u8; 16]);
}

#[test]
fn hint_unused_above_does_not_corrupt_prefix() {
    let device = make_device();
    let mut buf =
        Buffer::new_with_capacity_hint(&device, 64, 4096, DataAccess::Scattered).expect("buf");
    buf.write(0, &[0x11u8; 64]).expect("w");
    buf.hint_unused_above(32);
    let mut got = vec![0u8; 32];
    buf.read_to_cpu(&device, &mut got).expect("r");
    assert_eq!(&got[..], &[0x11u8; 32]);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn device_capabilities_metal_reports_constant_resize() {
    use goldy::{types::BufferResizeCost, BackendType, Instance};
    let inst = Instance::new().expect("i");
    let device = inst
        .create_device(goldy::DeviceType::IntegratedGpu)
        .or_else(|_| inst.create_device(goldy::DeviceType::DiscreteGpu))
        .expect("dev");
    assert_eq!(device.backend_type(), BackendType::Metal);
    let caps = device.capabilities();
    assert_eq!(caps.buffer_resize_cost, BufferResizeCost::Constant);
    assert!(caps.buffer_decommit_supported);
}

#[cfg(feature = "vulkan")]
#[test]
fn device_capabilities_vulkan_reports_pagebind_when_sparse() {
    use goldy::{types::BufferResizeCost, BackendType, Instance};
    let inst = Instance::new().expect("i");
    let device = inst
        .create_device(goldy::DeviceType::DiscreteGpu)
        .or_else(|_| inst.create_device(goldy::DeviceType::IntegratedGpu))
        .expect("dev");
    if device.backend_type() != BackendType::Vulkan {
        return;
    }
    let caps = device.capabilities();
    if caps.buffer_resize_cost == BufferResizeCost::PageBind {
        assert_eq!(caps.buffer_page_size, 64 * 1024);
        assert!(caps.buffer_decommit_supported);
    }
}

#[cfg(feature = "dx12")]
#[test]
fn device_capabilities_dx12_reports_pagebind_when_reserved_supported() {
    use goldy::{types::BufferResizeCost, BackendType, Instance};
    // With multiple backends enabled, `GOLDY_BACKEND` may select a non-DX12 API; skip in that case.
    let inst = Instance::new().expect("i");
    let device = inst
        .create_device(goldy::DeviceType::DiscreteGpu)
        .or_else(|_| inst.create_device(goldy::DeviceType::IntegratedGpu))
        .expect("dev");
    if device.backend_type() != BackendType::Dx12 {
        return;
    }
    let caps = device.capabilities();
    if caps.buffer_resize_cost == BufferResizeCost::PageBind {
        assert_eq!(caps.buffer_page_size, 64 * 1024);
        assert!(caps.buffer_decommit_supported);
    }
}

#[cfg(any(feature = "vulkan", feature = "dx12"))]
#[test]
fn sparse_backend_oversize_resize_and_hint_within_capacity() {
    use goldy::{types::BufferResizeCost, Instance};
    let inst = Instance::new().expect("i");
    let device = inst
        .create_device(goldy::DeviceType::DiscreteGpu)
        .or_else(|_| inst.create_device(goldy::DeviceType::IntegratedGpu))
        .expect("dev");
    if device.capabilities().buffer_resize_cost != BufferResizeCost::PageBind {
        return;
    }
    let mut buf =
        Buffer::new_with_capacity_hint(&device, 64, 4096, DataAccess::Scattered).expect("buf");
    buf.write(0, &[0x11u8; 64]).expect("w");
    buf.resize_to(256).expect("grow within cap");
    let mut got = vec![0u8; 256];
    buf.read_to_cpu(&device, &mut got).expect("r");
    assert_eq!(&got[..64], &[0x11u8; 64]);
    assert!(got[64..].iter().all(|&x| x == 0));

    buf.hint_unused_above(32);
    let mut prefix = vec![0u8; 32];
    buf.read_to_cpu(&device, &mut prefix).expect("r2");
    assert_eq!(&prefix[..], &[0x11u8; 32]);
}

/// DX12 reserved-buffer path: cross tile boundaries, `hint_unused_above`, regrowth, stable bindless, compute.
#[cfg(all(feature = "dx12", target_os = "windows"))]
#[test]
fn dx12_reserved_buffer_resize_compute_smoke() {
    use goldy::types::BufferResizeCost;
    const SMOKY_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(16, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] * 2;
}
"#;
    let inst = Instance::new().expect("i");
    let device = inst
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| inst.create_device(DeviceType::IntegratedGpu))
        .expect("dev");
    if device.backend_type() != BackendType::Dx12 {
        return;
    }
    if device.capabilities().buffer_resize_cost != BufferResizeCost::PageBind {
        return;
    }

    let mut buf =
        Buffer::new_with_capacity_hint(&device, 256, 4 * 64 * 1024, DataAccess::Scattered)
            .expect("buf");
    let bindless = buf.bindless_index().expect("bindless");

    let initial: Vec<u32> = (1..=16).collect();
    buf.write(0, bytemuck::cast_slice(&initial)).expect("w");

    buf.resize_to(200 * 1024).expect("grow across tiles");
    assert_eq!(buf.bindless_index(), Some(bindless));

    let mut read = vec![0u32; 16];
    buf.read_to_cpu(&device, bytemuck::cast_slice_mut(&mut read))
        .expect("read");
    assert_eq!(read, initial);

    let shader = ShaderModule::from_slang(&device, SMOKY_SHADER).expect("shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&buf]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    buf.read_to_cpu(&device, bytemuck::cast_slice_mut(&mut read))
        .expect("read2");
    for (i, &v) in read.iter().enumerate() {
        assert_eq!(v, (i as u32 + 1) * 2, "after first dispatch[{i}]");
    }

    // Shrink to one tile, decommit reserved tail with `hint_unused_above`, then grow again.
    buf.resize_to(64 * 1024).expect("shrink to one tile");
    assert_eq!(buf.bindless_index(), Some(bindless));
    buf.hint_unused_above(64 * 1024);
    buf.resize_to(200 * 1024).expect("grow after decommit hint");
    assert_eq!(buf.bindless_index(), Some(bindless));

    let initial2: Vec<u32> = (0..16).collect();
    buf.write(0, bytemuck::cast_slice(&initial2)).expect("w2");
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&buf]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch2");

    buf.read_to_cpu(&device, bytemuck::cast_slice_mut(&mut read))
        .expect("read3");
    for (i, &v) in read.iter().enumerate() {
        assert_eq!(v, (i as u32) * 2, "after second dispatch[{i}]");
    }
}

#[test]
fn hint_unused_above_smoke() {
    let device = make_device();
    let mut buf = Buffer::new(&device, 64, DataAccess::Scattered).expect("buf");
    buf.hint_unused_above(32);
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
        pass.bind_resources(&[&buffer]);
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
        pass.bind_resources(&[&input_buf, &output_buf]);
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
        pass.bind_resources(&[&input_buf, &output_buf]);
        pass.dispatch(1, 1, 1);
        // Clear output — must happen AFTER the copy dispatch.
        pass.clear_buffer(&output_buf, 0, 0);
        // Pass 2: increment output (zeros → 1s).
        pass.set_pipeline(&inc_pipeline);
        pass.bind_resources(&[&output_buf]);
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
        pass.bind_resources(&[&data_buf]);
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
        pass.bind_resources(&[&data_buf]);

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

// ─── Many resource slots (>4, exercises 16-slot expansion) ────────────────────

/// Shader using 6 bindless slots (0–5). Before the 16-slot expansion, slots 4+
/// were mapped to garbage indices and this test would produce wrong results.
#[test]
fn test_compute_many_resource_slots() {
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
        pass.bind_resources(&[&a, &b, &c, &d, &e, &out]);
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
    impl goldy::StructuredBufferElement for Particle {}

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
        pass.bind_resources(&[&buffer]);
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
        pass.bind_resources_raw(&[idx_a, idx_b]);
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
        pass.bind_resources_raw(&[idx]);
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
        pass.bind_resources_raw(&[src_idx, dst_idx]);
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

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<float> out, ThreadId id) {
    // scalar: negative dividend
    out[0] = positive_mod(-1.0, 3.0);    // 2.0
    out[1] = positive_mod(-3.0, 3.0);    // 0.0
    out[2] = positive_mod(-0.5, 1.0);    // 0.5

    // scalar: positive / zero inputs (must be unchanged)
    out[3] = positive_mod(2.5, 3.0);     // 2.5
    out[4] = positive_mod(0.0, 1.0);     // 0.0

    // float2 overload
    float2 r = positive_mod(float2(-1.0, -0.5), float2(3.0, 1.0));
    out[5] = r.x;   // 2.0
    out[6] = r.y;   // 0.5
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
        pass.bind_resources(&[&buf]);
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

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<float> out, ThreadId id) {
    // Row-major construction. Column 0 = (m[0][0], m[1][0], m[2][0]) = (1, 5, 9).
    float4x4 m = float4x4(
        1, 0, 0, 0,
        5, 1, 0, 0,
        9, 0, 1, 0,
        0, 0, 0, 1
    );
    float3 r = modelview_right(m);
    out[0] = r.x;   // 1.0
    out[1] = r.y;   // 5.0
    out[2] = r.z;   // 9.0

    // center=(1,2,3), cam_right=(1,0,0), offset=5 → (6, 2, 3)
    float3 off = billboard_cylindrical_offset(
        float3(1.0, 2.0, 3.0),
        float3(1.0, 0.0, 0.0),
        5.0
    );
    out[3] = off.x;  // 6.0
    out[4] = off.y;  // 2.0
    out[5] = off.z;  // 3.0

    // Identity matrix: right = (1, 0, 0)
    float4x4 ident = float4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    );
    float3 ident_right = modelview_right(ident);
    out[6] = ident_right.x;  // 1.0
    out[7] = ident_right.y;  // 0.0
    out[8] = ident_right.z;  // 0.0
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
        pass.bind_resources(&[&buf]);
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

/// Verify `goldy_buf_ro` / `goldy_scattered` can be assigned to locals and used together.
/// `goldy_buf_ro` returns `ReadOnlyBuffer<T>` (SRV on DX12, StorageBuffer on Vulkan/Metal).
/// Resource slots: slot 0 = read buffer (`bindless_srv_index()` on DX12), slot 1 = UAV.
#[test]
fn test_scattered_typed_variable_assignment() {
    const SHADER: &str = r#"
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

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SHADER).expect("compile typed-var shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Pair {
        a: u32,
        b: u32,
    }
    impl goldy::StructuredBufferElement for Pair {}

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
        pass.bind_resources_raw(&[
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

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    uint idx = id.x;
    if (idx >= 2097152) return;  // 8 MB / 4 bytes = 2M elements
    output[idx] = input[idx];
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
        pass.bind_resources(&[&buffers[0], &buffers[NUM_BUFFERS - 1]]);
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

    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources_raw(&[texture.bindless_index().expect("tex bindless")]);
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

// ─── CPU_READABLE buffer tests ────────────────────────────────────────────────

/// A compute shader that doubles each element (same as `DOUBLE_SHADER`). Used by the
/// coherent tests to avoid a dependency on the module-level constant's exact binding layout.
const DOUBLE_SHADER_COHERENT: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] * 2;
}
"#;

/// Create a [`BufferFlags::CPU_READABLE`] storage buffer, run a GPU compute pass that doubles every
/// element, then read back via [`Buffer::read_to_cpu`] (handles DX12 UAV → READBACK internally).
#[test]
fn test_cpu_readable_compute_write_and_read() {
    let device = make_device();

    let shader = ShaderModule::from_slang(&device, DOUBLE_SHADER_COHERENT).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    const N: usize = 64;
    let initial: Vec<u32> = (0..N as u32).collect();

    let buffer = Buffer::with_bytes_stride_and_flags(
        &device,
        bytemuck::cast_slice(&initial),
        DataAccess::Scattered,
        size_of::<u32>() as u32,
        BufferFlags::CPU_READABLE,
    )
    .expect("create CPU_READABLE buffer");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&buffer]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut out = vec![0u8; N * size_of::<u32>()];
    buffer.read_to_cpu(&device, &mut out).expect("read_to_cpu");

    let result: &[u32] = bytemuck::cast_slice(&out);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val,
            (i as u32) * 2,
            "element {i}: expected {} got {val}",
            i * 2
        );
    }
}

/// Two back-to-back non-blocking submits: `WriteBuffer` into the **same**
/// scattered buffer handle, copying to distinct outputs — exercises per-buffer
/// staging races fixed by the compute staging belt (Vulkan/DX12).
#[test]
fn test_write_buffer_reuse_across_submissions() {
    use goldy::{NodeAccess, TaskGraph};

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    const N: usize = 16;
    let mid = Buffer::new(
        &device,
        (N * core::mem::size_of::<u32>()) as u64,
        DataAccess::Scattered,
    )
    .expect("mid buffer");
    let out_a = Buffer::new(
        &device,
        (N * core::mem::size_of::<u32>()) as u64,
        DataAccess::Scattered,
    )
    .expect("out_a");
    let out_b = Buffer::new(
        &device,
        (N * core::mem::size_of::<u32>()) as u64,
        DataAccess::Scattered,
    )
    .expect("out_b");

    let idx_in = mid.bindless_index().expect("mid bindless");
    let idx_out_a = out_a.bindless_index().expect("out_a bindless");
    let idx_out_b = out_b.bindless_index().expect("out_b bindless");

    let data_a: Vec<u32> = (100..100 + N as u32).collect();
    let data_b: Vec<u32> = (900..900 + N as u32).collect();

    let mut g1 = TaskGraph::new();
    g1.write_buffer(&mid, 0, bytemuck::cast_slice(&data_a).to_vec());
    g1.node("copy_a", &pipeline)
        .bind_buffer(&mid, NodeAccess::Read)
        .bind_buffer(&out_a, NodeAccess::Write)
        .bind_resources_raw_slice(&[idx_in, idx_out_a])
        .dispatch(1, 1, 1);

    let mut g2 = TaskGraph::new();
    g2.write_buffer(&mid, 0, bytemuck::cast_slice(&data_b).to_vec());
    g2.node("copy_b", &pipeline)
        .bind_buffer(&mid, NodeAccess::Read)
        .bind_buffer(&out_b, NodeAccess::Write)
        .bind_resources_raw_slice(&[idx_in, idx_out_b])
        .dispatch(1, 1, 1);

    let tv1 = g1.submit(&device).expect("submit 1");
    let tv2 = g2.submit(&device).expect("submit 2");

    device.wait_until(tv1).expect("wait 1");
    device.wait_until(tv2).expect("wait 2");

    let read_u32 = |buf: &Buffer| -> Vec<u32> {
        let mut raw = vec![0u8; N * core::mem::size_of::<u32>()];
        buf.read_to_cpu(&device, &mut raw).expect("readback");
        bytemuck::cast_slice(&raw).to_vec()
    };

    let got_a = read_u32(&out_a);
    let got_b = read_u32(&out_b);
    assert_eq!(got_a, data_a, "output A corrupted (staging race?)");
    assert_eq!(got_b, data_b, "output B wrong");
}

/// CPU writes to a [`BufferFlags::CPU_READABLE`] buffer are reflected in reads.
///
/// On Vulkan and Metal the buffer lives in host-visible/shared memory, so a plain
/// `buffer.write()` followed immediately by `buffer.read_to_cpu()` round-trips.
///
/// On DX12 the primary resource is a DEFAULT-heap UAV (not host-visible), so
/// `read_to_cpu` copies UAV → READBACK internally.
#[test]
fn test_cpu_readable_cpu_write_read_roundtrip() {
    let device = make_device();

    const N: usize = 16;
    let initial: Vec<u32> = vec![0xABCD_1234u32; N];

    let buffer = Buffer::with_bytes_stride_and_flags(
        &device,
        bytemuck::cast_slice(&initial),
        DataAccess::Scattered,
        size_of::<u32>() as u32,
        BufferFlags::CPU_READABLE,
    )
    .expect("create CPU_READABLE buffer");

    let new_values: Vec<u32> = (100..100 + N as u32).collect();
    buffer
        .write(0, bytemuck::cast_slice(&new_values))
        .expect("write");

    let mut out = vec![0u8; N * size_of::<u32>()];
    buffer.read_to_cpu(&device, &mut out).expect("read_to_cpu");

    let result: &[u32] = bytemuck::cast_slice(&out);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val,
            100 + i as u32,
            "element {i}: expected {} got {val}",
            100 + i
        );
    }
}

// ── uniform entry-point parameter tests ──────────────────────────────────────
//
// `uniform T param` in a `cs_main` signature maps directly to a resource
// slot. The tests below exercise:
//
//  • uint round-trip (basic sanity)
//  • zero value is preserved
//  • maximum u32 (0xFFFF_FFFF) passes through unmodified
//  • float (bit-for-bit round-trip of f32)
//  • two independent scalar params in adjacent slots
//  • scalar param after two buffer slots

/// A `uniform uint` entry-point param round-trips a u32.
#[test]
fn test_uniform_param_uint_roundtrip() {
    const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint value, ThreadId id) {
    out[0] = value;
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
    let out = Buffer::with_data(&device, &[0u32; 1], DataAccess::Scattered).expect("out");

    const EXPECTED: u32 = 42;
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        let heap_idx = out.bindless_index().unwrap();
        pass.bind_resources_raw_with_user(&[heap_idx], &[EXPECTED]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: u32 = bytemuck::pod_read_unaligned(&raw);
    assert_eq!(result, EXPECTED, "uniform uint round-trip failed");
}

/// Zero passes through a `uniform uint` parameter unchanged.
#[test]
fn test_uniform_param_uint_zero() {
    const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint value, ThreadId id) {
    out[0] = value;
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
    let out = Buffer::with_data(&device, &[0xDEAD_BEEFu32; 1], DataAccess::Scattered).expect("out");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        let heap_idx = out.bindless_index().unwrap();
        pass.bind_resources_raw_with_user(&[heap_idx], &[0u32]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: u32 = bytemuck::pod_read_unaligned(&raw);
    assert_eq!(
        result, 0,
        "zero should pass through uniform uint param unchanged"
    );
}

/// Maximum u32 (0xFFFF_FFFF) passes through a `uniform uint` parameter unchanged.
#[test]
fn test_uniform_param_uint_max() {
    const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint value, ThreadId id) {
    out[0] = value;
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
    let out = Buffer::with_data(&device, &[0u32; 1], DataAccess::Scattered).expect("out");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        let heap_idx = out.bindless_index().unwrap();
        pass.bind_resources_raw_with_user(&[heap_idx], &[u32::MAX]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: u32 = bytemuck::pod_read_unaligned(&raw);
    assert_eq!(
        result,
        u32::MAX,
        "u32::MAX should pass through uniform uint param unchanged"
    );
}

/// A `uniform float` entry-point param reinterprets raw bits as a float.
/// We push the bit-pattern of a float and expect the shader to write it back,
/// then reinterpret on the CPU side to confirm identity.
#[test]
fn test_uniform_param_float_reinterpret() {
    const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<float> out, float value, ThreadId id) {
    out[0] = value;
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
    let out = Buffer::with_data(&device, &[0u32; 1], DataAccess::Scattered).expect("out");

    #[allow(clippy::approx_constant)]
    let value: f32 = 3.14159;
    let bits = value.to_bits();

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        let heap_idx = out.bindless_index().unwrap();
        pass.bind_resources_raw_with_user(&[heap_idx], &[bits]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result_bits: u32 = bytemuck::pod_read_unaligned(&raw);
    let result_float = f32::from_bits(result_bits);
    assert_eq!(
        result_bits, bits,
        "float bit pattern should survive uniform float param round-trip (got {result_float}, expected {value})"
    );
}

/// Two independent `uniform uint` params in adjacent slots each retrieve their own value.
#[test]
fn test_uniform_two_independent_scalar_params() {
    const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint a, uint b, ThreadId id) {
    out[0] = a;
    out[1] = b;
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
    let out = Buffer::with_data(&device, &[0u32; 2], DataAccess::Scattered).expect("out");

    const A: u32 = 0xABCD;
    const B: u32 = 0x1234;

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        let heap_idx = out.bindless_index().unwrap();
        pass.bind_resources_raw_with_user(&[heap_idx], &[A, B]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 8];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[u32] = bytemuck::cast_slice(&raw);
    assert_eq!(result[0], A, "param a → out[0] mismatch");
    assert_eq!(result[1], B, "param b → out[1] mismatch");
}

/// A scalar `uniform uint` param after two buffer-slot params correctly
/// retrieves its value from the third resource slot.
#[test]
fn test_uniform_scalar_after_two_buffer_params() {
    const SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> inp, Scattered<uint> out, uint offset, ThreadId id) {
    out[id.x] = inp[id.x] + offset;
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, SHADER).expect("compile");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    const N: usize = 64;
    let input: Vec<u32> = (0..N as u32).collect();
    let inp = Buffer::with_data(&device, &input, DataAccess::Scattered).expect("inp");
    let out = Buffer::with_data(&device, &[0u32; N], DataAccess::Scattered).expect("out");

    const OFFSET: u32 = 100;

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        let inp_idx = inp.bindless_index().unwrap();
        let out_idx = out.bindless_index().unwrap();
        pass.bind_resources_raw_with_user(&[inp_idx, out_idx], &[OFFSET]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; N * 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[u32] = bytemuck::cast_slice(&raw);
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val,
            i as u32 + OFFSET,
            "element {i}: expected {}, got {val}",
            i as u32 + OFFSET
        );
    }
}

/// Headless compute: a buffer dropped after a standalone submit stays in the backend's
/// deferred-destruction queue until `wait_until` sees the matching timeline value.
#[test]
fn headless_deferred_buffer_destroy_drains_after_timeline_wait() {
    const MINIMAL_SHADER: &str = r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
}
"#;

    let device = make_device();
    let shader = ShaderModule::from_slang(&device, MINIMAL_SHADER).expect("compile");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.dispatch(1, 1, 1);
    }
    let tv = encoder.submit(&device).expect("submit");

    let buf = Buffer::new(&device, 256, DataAccess::Scattered).expect("buffer");

    let pending_after_drop = {
        drop(buf);
        device.deferred_deletion_pending_count()
    };
    assert!(
        pending_after_drop > 0,
        "expected deferred deletion queue to retain GPU resources until the timeline catches up"
    );

    device.wait_until(tv).expect("wait_until");

    assert_eq!(
        device.deferred_deletion_pending_count(),
        0,
        "wait_until should drain deferred destruction for completed timeline values"
    );
}

// ── flush_deferred_deletions ───────────────────────────────────────────────────

/// After a submit+wait, `flush_deferred_deletions` reclaims all pending slots
/// without requiring a second `wait_until` call.
#[test]
fn flush_deferred_deletions_reclaims_slots_after_gpu_idle() {
    let device = make_device();

    // Allocate, submit some work, then drop the buffer so its slot enters
    // the pending-free list.
    let buf = Buffer::new(&device, 256, DataAccess::Scattered).expect("buffer");
    let tv = {
        let encoder = ComputeEncoder::new();
        encoder.submit(&device).expect("submit empty work")
    };
    device.wait_until(tv).expect("wait");

    // Drop happens *after* wait_until, so the GPU is idle but the slot may
    // still be in the pending list depending on when the backend last ran
    // its internal cleanup. `flush_deferred_deletions` must drain it.
    drop(buf);
    device.flush_deferred_deletions();

    assert_eq!(
        device.deferred_deletion_pending_count(),
        0,
        "flush_deferred_deletions must reclaim all slots when GPU is idle"
    );
}

/// A buffer dropped *between* submits has its slot reclaimed by
/// `flush_deferred_deletions` once the matching timeline has been signaled.
#[test]
fn flush_deferred_deletions_respects_gpu_progress() {
    let device = make_device();

    // Submit work so the timeline advances past zero.
    let tv = {
        let encoder = ComputeEncoder::new();
        encoder.submit(&device).expect("submit")
    };

    // Drop a buffer while GPU may still be in flight.
    let buf = Buffer::new(&device, 256, DataAccess::Scattered).expect("buffer");
    drop(buf);

    // Without waiting for the GPU, calling flush is safe (no panic/crash).
    // Slots that haven't been signaled yet should remain pending.
    device.flush_deferred_deletions();

    // After waiting, flushing again must fully drain the queue.
    device.wait_until(tv).expect("wait");
    device.flush_deferred_deletions();

    assert_eq!(
        device.deferred_deletion_pending_count(),
        0,
        "pending slots must be zero after wait_until + flush_deferred_deletions"
    );
}

/// Calling `flush_deferred_deletions` on a fresh device with no in-flight work
/// is a no-op and must not panic.
#[test]
fn flush_deferred_deletions_noop_on_idle_device() {
    let device = make_device();
    device.flush_deferred_deletions();
    assert_eq!(device.deferred_deletion_pending_count(), 0);
}

// ============================================================================
// Element-stride validation integration tests
//
// These exercise the full GPU path: compile a [goldy_compute] shader, create
// a pipeline, create a buffer with a deliberately wrong element_stride, bind
// it, and verify that stride validation fires (or that matching strides pass).
// ============================================================================

/// Env-var guard: these tests mutate `GOLDY_VALIDATE_LAYOUTS` which is
/// process-global, so they must not run concurrently.
static STRIDE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Helper: create a device for stride-validation tests.
fn make_device_for_stride_tests() -> goldy::Device {
    let instance = Instance::new().expect("instance");
    instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("device")
}

/// Shader that reads from a `Scattered<uint>` buffer (element stride 4).
const STRIDE_UINT_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + 1;
}
"#;

/// A shader that uses a user-defined struct as a broadcast uniform and a
/// `Scattered<float4>` buffer (element stride 16).
/// The struct uses four floats (16 bytes) so its reflected size is identical
/// under both HLSL packing (DX12) and std140 rules (SPIR-V/Vulkan).
const STRIDE_STRUCT_SHADER: &str = r#"
import goldy_exp;

struct MyParams { float x; float y; float z; float w; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(MyParams cfg, Scattered<float4> data, ThreadId id) {
    data[id.x] = float4(cfg.x, cfg.y, cfg.z, cfg.w);
}
"#;

/// Matching strides: buffer with stride 4 dispatched to a shader
/// expecting `Scattered<uint>` (stride 4). Must pass validation and execute
/// correctly.
#[test]
fn stride_validation_matching_uint_passes() {
    let _lock = STRIDE_ENV_LOCK.lock().unwrap();
    std::env::set_var("GOLDY_VALIDATE_LAYOUTS", "1");

    let device = make_device_for_stride_tests();
    let shader =
        ShaderModule::from_slang(&device, STRIDE_UINT_SHADER).expect("compile stride shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let initial: Vec<u32> = (0..64).collect();
    let buffer =
        Buffer::with_data(&device, &initial, DataAccess::Scattered).expect("create buffer");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&buffer]);
        pass.dispatch(1, 1, 1);
    }
    let result = encoder.dispatch(&device);

    std::env::remove_var("GOLDY_VALIDATE_LAYOUTS");
    result.expect("dispatch with matching stride must succeed");
}

/// Mismatched stride: buffer created with stride 16 bound to a shader
/// expecting `Scattered<uint>` (stride 4). Must produce a stride-mismatch
/// error when layout validation is enabled.
#[test]
fn stride_validation_mismatched_uint_vs_stride16_fails() {
    let _lock = STRIDE_ENV_LOCK.lock().unwrap();
    std::env::set_var("GOLDY_VALIDATE_LAYOUTS", "1");

    let device = make_device_for_stride_tests();
    let shader =
        ShaderModule::from_slang(&device, STRIDE_UINT_SHADER).expect("compile stride shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let data = vec![0u8; 64 * 16];
    let buffer = Buffer::with_bytes_stride(&device, &data, DataAccess::Scattered, 16)
        .expect("create buffer with stride 16");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&buffer]);
        pass.dispatch(1, 1, 1);
    }
    let result = encoder.dispatch(&device);

    std::env::remove_var("GOLDY_VALIDATE_LAYOUTS");
    let err = result.expect_err("dispatch with wrong stride must fail");
    let msg = err.to_string();
    assert!(msg.contains("stride"), "error must mention 'stride': {msg}");
    assert!(
        msg.contains("slot 0"),
        "error must identify the offending slot: {msg}"
    );
}

/// When layout validation is disabled, the same mismatch must NOT produce an
/// error (validation is opt-in for performance).
#[test]
fn stride_validation_disabled_allows_mismatch() {
    let _lock = STRIDE_ENV_LOCK.lock().unwrap();
    std::env::remove_var("GOLDY_VALIDATE_LAYOUTS");
    std::env::remove_var("GOLDY_VALIDATION");

    let device = make_device_for_stride_tests();
    let shader =
        ShaderModule::from_slang(&device, STRIDE_UINT_SHADER).expect("compile stride shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let data = vec![0u8; 64 * 16];
    let buffer = Buffer::with_bytes_stride(&device, &data, DataAccess::Scattered, 16)
        .expect("create buffer with stride 16");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&buffer]);
        pass.dispatch(1, 1, 1);
    }
    encoder
        .dispatch(&device)
        .expect("dispatch must succeed when validation is off");
}

/// Multi-binding mismatch: the shader expects a broadcast struct (reflected stride) plus
/// `Scattered<float4>` (stride 16). Bind the broadcast slot correctly but give the second slot
/// a buffer with stride 4; only slot 1 should be reported.
#[test]
fn stride_validation_multi_binding_detects_second_slot_mismatch() {
    let _lock = STRIDE_ENV_LOCK.lock().unwrap();
    std::env::set_var("GOLDY_VALIDATE_LAYOUTS", "1");

    let device = make_device_for_stride_tests();
    let shader =
        ShaderModule::from_slang(&device, STRIDE_STRUCT_SHADER).expect("compile struct shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let params_data = vec![0u8; 16];
    let params = Buffer::with_bytes_stride(&device, &params_data, DataAccess::Broadcast, 16)
        .expect("create broadcast buffer with stride 16");

    let wrong_data = vec![0u8; 64 * 4];
    let data_buf = Buffer::with_bytes_stride(&device, &wrong_data, DataAccess::Scattered, 4)
        .expect("create data buf with stride 4");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&params, &data_buf]);
        pass.dispatch(1, 1, 1);
    }
    let result = encoder.dispatch(&device);

    std::env::remove_var("GOLDY_VALIDATE_LAYOUTS");
    let err = result.expect_err("dispatch with wrong stride on slot 1 must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("slot 1"),
        "error must identify slot 1 (data buffer): {msg}"
    );
    assert!(msg.contains("stride"), "error must mention 'stride': {msg}");
}

/// Matching strides with two bindings: broadcast + `Scattered<float4>` both
/// have correct strides. Must pass validation and execute correctly.
#[test]
fn stride_validation_multi_binding_all_correct_passes() {
    let _lock = STRIDE_ENV_LOCK.lock().unwrap();
    std::env::set_var("GOLDY_VALIDATE_LAYOUTS", "1");

    let device = make_device_for_stride_tests();
    let shader =
        ShaderModule::from_slang(&device, STRIDE_STRUCT_SHADER).expect("compile struct shader");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let params_data = vec![0u8; 16];
    let params = Buffer::with_bytes_stride(&device, &params_data, DataAccess::Broadcast, 16)
        .expect("create broadcast buffer with stride 16");

    let data = vec![0u8; 64 * 16];
    let data_buf = Buffer::with_bytes_stride(&device, &data, DataAccess::Scattered, 16)
        .expect("create data buf with stride 16");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&params, &data_buf]);
        pass.dispatch(1, 1, 1);
    }
    let result = encoder.dispatch(&device);

    std::env::remove_var("GOLDY_VALIDATE_LAYOUTS");
    result.expect("dispatch with all-correct strides must succeed");
}

// ============================================================================
// TransientAllocator integration tests
//
// Verify the trait, both strategies, and end-to-end frame lifecycle against a
// real device. Each test uses a fresh device so they remain isolated.
// ============================================================================

use goldy::{
    BumpResetAllocator, EpochRegionsAllocator, TransientAllocator, TransientAllocatorConfig,
    TransientAllocatorStrategy,
};

fn small_config() -> TransientAllocatorConfig {
    TransientAllocatorConfig {
        initial_size: 4 * 1024,
        expected_max: 64 * 1024,
        min_region_size: 4 * 1024,
        max_regions: 4,
        alignment: 256,
        flags: BufferFlags::GPU_ONLY,
    }
}

/// Smoke test: each strategy can `create`, then `begin_frame` → `alloc` → `end_frame` with no
/// GPU work and not panic. Covers the lazy-init and zero-work path.
#[test]
fn transient_allocator_smoke_all_strategies() {
    let device = make_device();
    for strategy in [
        TransientAllocatorStrategy::BumpReset,
        TransientAllocatorStrategy::EpochRegions,
        TransientAllocatorStrategy::Heap,
    ] {
        let mut a = strategy
            .create(&device, small_config())
            .expect("create allocator");
        a.begin_frame(&device, 0).expect("begin");
        let view = a.alloc(&device, 256, Some(4)).expect("alloc");
        // Submit empty work to advance the timeline.
        let tv = ComputeEncoder::new().submit(&device).expect("submit");
        drop(view);
        a.end_frame(&device, tv);
        // Next frame: should not panic, and BumpReset should wait for tv internally.
        a.begin_frame(&device, 0).expect("begin frame 2");
        let _v2 = a.alloc(&device, 256, Some(4)).expect("alloc frame 2");
    }
}

/// Strategy::default() is Heap, and parse handles all canonical names.
#[test]
fn transient_allocator_strategy_default_and_parse() {
    assert_eq!(
        TransientAllocatorStrategy::default(),
        TransientAllocatorStrategy::Heap,
    );
    assert_eq!(
        TransientAllocatorStrategy::parse("bump"),
        Some(TransientAllocatorStrategy::BumpReset),
    );
    assert_eq!(
        TransientAllocatorStrategy::parse("epoch"),
        Some(TransientAllocatorStrategy::EpochRegions),
    );
    assert_eq!(
        TransientAllocatorStrategy::parse("heap"),
        Some(TransientAllocatorStrategy::Heap),
    );
    assert_eq!(TransientAllocatorStrategy::parse("garbage"), None);
}

/// Verifies that BumpResetAllocator waits for the previous frame's epoch before resetting.
/// We measure the elapsed time of `begin_frame` after a submit-without-wait — if the
/// wait kicks in, it should be roughly the GPU latency (non-zero); if it doesn't, the test
/// at least proves no deadlock.
#[test]
fn bump_reset_blocks_on_prev_epoch() {
    let device = make_device();
    let mut a = BumpResetAllocator::new(&device, small_config()).expect("create");

    a.begin_frame(&device, 0).expect("begin");
    let _v = a.alloc(&device, 1024, Some(4)).expect("alloc");
    let tv = ComputeEncoder::new().submit(&device).expect("submit");
    a.end_frame(&device, tv);

    // begin_frame should wait for `tv` if it hasn't completed. After it returns,
    // gpu_progress must be at least tv.
    a.begin_frame(&device, 0).expect("begin 2");
    assert!(
        device.gpu_progress() >= tv,
        "BumpReset must not allow reuse until prev epoch has been signaled"
    );
}

/// Allocating more than a region's capacity in one frame causes EpochRegionsAllocator to
/// spill into additional regions, not panic.
#[test]
fn epoch_regions_spills_when_active_full() {
    let device = make_device();
    let cfg = TransientAllocatorConfig {
        min_region_size: 4 * 1024,
        max_regions: 4,
        ..small_config()
    };
    let mut a = EpochRegionsAllocator::new(&device, cfg).expect("create");

    a.begin_frame(&device, 0).expect("begin");
    let before = a.region_count();
    // Allocate 3 regions worth in 3 chunks — must spill twice.
    let _v1 = a.alloc(&device, 3 * 1024, Some(4)).expect("alloc 1");
    let _v2 = a.alloc(&device, 3 * 1024, Some(4)).expect("alloc 2");
    let _v3 = a.alloc(&device, 3 * 1024, Some(4)).expect("alloc 3");
    let after = a.region_count();
    assert!(
        after >= before + 2,
        "expected at least 2 new regions after 3 region-sized allocs, got {} -> {}",
        before,
        after
    );

    let tv = ComputeEncoder::new().submit(&device).expect("submit");
    a.end_frame(&device, tv);
    // After end_frame all active should be retired.
    assert_eq!(a.retired_count(), after, "all active regions must retire");
    assert_eq!(a.empty_count(), 0);
}

/// EpochRegionsAllocator reclaims retired regions opportunistically in `begin_frame` once
/// the GPU has caught up — no explicit waits required on the hot path.
#[test]
fn epoch_regions_reclaims_after_gpu_catches_up() {
    let device = make_device();
    let mut a = EpochRegionsAllocator::new(&device, small_config()).expect("create");

    // Frame 1: allocate, submit, retire.
    a.begin_frame(&device, 0).expect("begin 1");
    let _v = a.alloc(&device, 2048, Some(4)).expect("alloc");
    let tv = ComputeEncoder::new().submit(&device).expect("submit");
    a.end_frame(&device, tv);
    assert!(a.retired_count() >= 1);
    assert_eq!(a.empty_count(), 0);

    // Wait so the next begin_frame can reclaim.
    device.wait_until(tv).expect("wait");

    // Frame 2: begin_frame must promote the retired region back to empty before activating.
    a.begin_frame(&device, 0).expect("begin 2");
    // The previously retired region should now be Active (count == 1). No retired left.
    assert_eq!(a.retired_count(), 0);
}

/// Across many frames, the EpochRegionsAllocator's region count stays bounded by max_regions.
#[test]
fn epoch_regions_respects_max_regions_cap() {
    let device = make_device();
    let cfg = TransientAllocatorConfig {
        min_region_size: 4 * 1024,
        max_regions: 3,
        ..small_config()
    };
    let mut a = EpochRegionsAllocator::new(&device, cfg.clone()).expect("create");

    // Push 10 frames through; each frame is small and submitted+waited so retirees can be
    // reclaimed. Region count should stay ≤ max_regions.
    for _ in 0..10 {
        a.begin_frame(&device, 0).expect("begin");
        let _v = a.alloc(&device, 1024, Some(4)).expect("alloc");
        let tv = ComputeEncoder::new().submit(&device).expect("submit");
        a.end_frame(&device, tv);
        device.wait_until(tv).expect("wait");
        assert!(
            a.region_count() <= cfg.max_regions,
            "region count {} exceeded cap {}",
            a.region_count(),
            cfg.max_regions
        );
    }
}

/// `clear()` on either strategy must release all allocations and put the allocator into a
/// clean state ready for re-use.
#[test]
fn transient_allocator_clear_resets_state() {
    let device = make_device();
    for strategy in [
        TransientAllocatorStrategy::BumpReset,
        TransientAllocatorStrategy::EpochRegions,
    ] {
        let mut a = strategy
            .create(&device, small_config())
            .expect("create allocator");
        a.begin_frame(&device, 0).expect("begin");
        let _v = a.alloc(&device, 1024, Some(4)).expect("alloc");
        let tv = ComputeEncoder::new().submit(&device).expect("submit");
        a.end_frame(&device, tv);
        device.wait_until(tv).expect("wait");
        a.clear();
        // After clear, used_this_frame should be zero and we can begin a new frame.
        assert_eq!(a.used_this_frame(), 0);
        a.begin_frame(&device, 0).expect("begin after clear");
    }
}

/// The surface-rendering path calls begin_frame before end_frame for the previous frame
/// (because the timeline value is not known until after Frame::present). EpochRegions must
/// handle this by parking the previous frame's regions as Pending and promoting them once
/// the deferred end_frame arrives.
#[test]
fn epoch_regions_deferred_end_frame_does_not_leak() {
    let device = make_device();
    let cfg = TransientAllocatorConfig {
        min_region_size: 4 * 1024,
        max_regions: 3,
        ..small_config()
    };
    let mut a = EpochRegionsAllocator::new(&device, cfg).expect("create");

    // Simulate surface-rendering lifecycle: begin → alloc → (no end_frame) → begin → ...
    for i in 0..10u64 {
        a.begin_frame(&device, 0).expect("begin");
        let _v = a.alloc(&device, 1024, Some(4)).expect("alloc");
        let tv = ComputeEncoder::new().submit(&device).expect("submit");
        // Simulate: finish() happened, but timeline not known yet for surface path.
        // The NEXT begin_frame arrives before end_frame.

        // On odd frames, simulate delayed end_frame arriving (from note_frame_presented).
        // On even frames, let it carry over.
        if i % 2 == 1 {
            a.end_frame(&device, tv);
            device.wait_until(tv).expect("wait");
        }
    }

    // Final end_frame so everything can be reclaimed.
    let tv = ComputeEncoder::new().submit(&device).expect("submit");
    a.end_frame(&device, tv);
    device.wait_until(tv).expect("wait");

    // All regions should be reclaimable. Region count must be bounded by max_regions.
    assert!(
        a.region_count() <= 3,
        "region count {} exceeded max_regions 3",
        a.region_count()
    );
}

/// When end_frame is never called between begin_frames (worst case), the allocator must
/// not panic or allocate unbounded memory. Pending regions are force-reclaimed when needed.
#[test]
fn epoch_regions_survives_continuous_begin_without_end() {
    let device = make_device();
    let cfg = TransientAllocatorConfig {
        min_region_size: 4 * 1024,
        max_regions: 3,
        ..small_config()
    };
    let mut a = EpochRegionsAllocator::new(&device, cfg).expect("create");

    // 20 frames of begin_frame + alloc with NO end_frame at all.
    for _ in 0..20 {
        a.begin_frame(&device, 0).expect("begin");
        let _v = a.alloc(&device, 512, Some(4)).expect("alloc");
        let _tv = ComputeEncoder::new().submit(&device).expect("submit");
        // Deliberately skip end_frame — simulates broken caller or surface-path delay.
    }

    assert!(
        a.region_count() <= 3,
        "region count {} exceeded max_regions 3 even without end_frame",
        a.region_count()
    );
    assert_eq!(
        a.active_count(),
        1,
        "only the current frame should be Active"
    );
    // At most one Pending region (from the frame immediately before the current one).
    // The rest must have been force-reclaimed when the safety valve fired.
    assert!(
        a.pending_count() <= 1,
        "at most 1 pending region expected, got {}",
        a.pending_count()
    );
}

/// After a deferred end_frame, the assigned epoch enables proper reclamation of all
/// previously-pending regions on the next begin_frame.
#[test]
fn epoch_regions_pending_promoted_to_retired_on_end_frame() {
    let device = make_device();
    let cfg = TransientAllocatorConfig {
        min_region_size: 4 * 1024,
        max_regions: 3,
        ..small_config()
    };
    let mut a = EpochRegionsAllocator::new(&device, cfg).expect("create");

    // Frame 1: begin, alloc, NO end_frame (surface path).
    a.begin_frame(&device, 0).expect("begin 1");
    let _v1 = a.alloc(&device, 1024, Some(4)).expect("alloc 1");
    let tv1 = ComputeEncoder::new().submit(&device).expect("submit 1");

    // Frame 2: begin_frame moves frame 1's active regions to Pending.
    a.begin_frame(&device, 0).expect("begin 2");
    assert_eq!(a.pending_count(), 1, "frame 1's region should be Pending");
    assert_eq!(a.active_count(), 1, "frame 2 should have one Active region");

    // Deferred end_frame for frame 1 arrives.
    a.end_frame(&device, tv1);
    assert_eq!(
        a.pending_count(),
        0,
        "end_frame should promote Pending to Retired"
    );
    // Both frame 1's (was Pending) and frame 2's (was Active) regions are now Retired.
    assert_eq!(a.retired_count(), 2);

    // Wait and verify reclamation works.
    device.wait_until(tv1).expect("wait");
    a.begin_frame(&device, 0).expect("begin 3");
    assert_eq!(a.retired_count(), 0, "waited regions should be reclaimed");
}

/// BufferPool::would_fit must agree with alloc_bytes for power-of-two and non-power-of-two
/// strides.
#[test]
fn buffer_pool_would_fit_agrees_with_alloc_bytes() {
    let device = make_device();
    let mut pool = BufferPool::new(&device, 4096).expect("pool");

    assert!(pool.would_fit(1024, Some(4)));
    let _v = pool.alloc_bytes(1024, Some(4)).expect("alloc");

    // Non-power-of-two stride: 12 (vec3<f32>). LCM(256, 12) = 768.
    // After consuming 1024 bytes at alignment 256, used == 1024. The next alloc with stride
    // 12 aligns to 768 → aligned_offset = 1536. Byte size must be a multiple of 12 for
    // StructuredBuffer views (e.g. 1020 = 85 × 12, not 1024).
    assert!(pool.would_fit(1020, Some(12)));
    let _v2 = pool.alloc_bytes(1020, Some(12)).expect("alloc np2");

    // Remaining tail after second slot: 4096 - (1536 + 1020). would_fit must reject oversize allocs.
    assert!(pool.would_fit(1024, Some(4)));
    assert!(!pool.would_fit(4096, Some(4)));
}

// ─── Transient buffer (graph-colored) path ─────────────────────────────────────

const WRITE_IOTA_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    if (id.x < 64) data[id.x] = id.x + 1;
}
"#;

/// Exercises the device-owned placement heap transient-buffer path end-to-end:
///
/// 1. Graph has one transient buffer T.
/// 2. Dispatch A writes `id.x + 1` into T.
/// 3. Dispatch B copies T -> output (a regular CPU-readable buffer).
/// 4. `graph.dispatch(&device)` resolves transients via the device-owned
///    placement heap (view creation, bindless patching, submission).
/// 5. Read back output and verify values.
#[test]
fn test_transient_buffer_write_then_copy() {
    use goldy::{NodeAccess, TaskGraph};

    let device = make_device();

    let write_shader =
        ShaderModule::from_slang(&device, WRITE_IOTA_SHADER).expect("compile write shader");
    let write_pipeline =
        ComputePipeline::new(&device, &write_shader).expect("create write pipeline");

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile copy shader");
    let copy_pipeline = ComputePipeline::new(&device, &copy_shader).expect("create copy pipeline");

    const N: usize = 64;
    let byte_size = (N * core::mem::size_of::<u32>()) as u64;

    let output = Buffer::new(&device, byte_size, DataAccess::Scattered).expect("output buffer");
    let output_uav = output.bindless_index().expect("output UAV");

    let mut graph = TaskGraph::new();
    let tid = graph.transient_buffer(byte_size);

    graph
        .node("write_iota", &write_pipeline)
        .bind_transient_buffer(tid, NodeAccess::Write)
        .bind_resources_raw_slice(&[u32::MAX])
        .dispatch(1, 1, 1);

    graph
        .node("copy_out", &copy_pipeline)
        .bind_transient_buffer(tid, NodeAccess::ReadWrite)
        .bind_buffer(&output, NodeAccess::Write)
        .bind_resources_raw_slice(&[u32::MAX, output_uav])
        .dispatch(1, 1, 1);

    graph.dispatch(&device).expect("dispatch transient graph");

    let mut raw = vec![0u8; byte_size as usize];
    output.read_to_cpu(&device, &mut raw).expect("read output");
    let result: &[u32] = bytemuck::cast_slice(&raw);

    let expected: Vec<u32> = (1..=N as u32).collect();
    assert_eq!(
        result,
        &expected[..],
        "transient buffer graph-colored path produced wrong data"
    );
}

/// Control test: same dispatches but with a regular buffer instead of transient.
/// If this passes but the transient version fails, the issue is in the
/// graph-colored infrastructure.
#[test]
fn test_regular_buffer_write_then_copy() {
    use goldy::{NodeAccess, TaskGraph};

    let device = make_device();

    let write_shader =
        ShaderModule::from_slang(&device, WRITE_IOTA_SHADER).expect("compile write shader");
    let write_pipeline =
        ComputePipeline::new(&device, &write_shader).expect("create write pipeline");

    let copy_shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("compile copy shader");
    let copy_pipeline = ComputePipeline::new(&device, &copy_shader).expect("create copy pipeline");

    const N: usize = 64;
    let byte_size = (N * core::mem::size_of::<u32>()) as u64;

    let scratch = Buffer::new(&device, byte_size, DataAccess::Scattered).expect("scratch buffer");
    let scratch_uav = scratch.bindless_index().expect("scratch UAV");

    let output = Buffer::new(&device, byte_size, DataAccess::Scattered).expect("output buffer");
    let output_uav = output.bindless_index().expect("output UAV");

    let mut graph = TaskGraph::new();

    graph.clear_buffer(&scratch, 0, byte_size);

    graph
        .node("write_iota", &write_pipeline)
        .bind_buffer(&scratch, NodeAccess::Write)
        .bind_resources_raw_slice(&[scratch_uav])
        .dispatch(1, 1, 1);

    graph
        .node("copy_out", &copy_pipeline)
        .bind_buffer(&scratch, NodeAccess::Read)
        .bind_buffer(&output, NodeAccess::Write)
        .bind_resources_raw_slice(&[scratch_uav, output_uav])
        .dispatch(1, 1, 1);

    graph.dispatch(&device).expect("dispatch graph");

    let mut raw = vec![0u8; byte_size as usize];
    output.read_to_cpu(&device, &mut raw).expect("read output");
    let result: &[u32] = bytemuck::cast_slice(&raw);

    let expected: Vec<u32> = (1..=N as u32).collect();
    assert_eq!(
        result,
        &expected[..],
        "regular buffer path produced wrong data"
    );
}

// ─── collectives: GPU execution tests ────────────────────────────────────────
//
// Each test runs a small compute shader that exercises one collective algorithm
// from `goldy_exp/collectives.slang` and reads the output back to the CPU to
// verify that the GPU-side arithmetic is correct.  These are distinct from the
// compile-only `test_collectives_compiles` test in `src/shaders.rs`.
//
// ⚠  Cross-module [ForceInline] + groupshared writes — known Slang bugs
//
// Slang issues #10641 and #10642 document that [ForceInline] functions whose
// bodies *write* to a groupshared parameter produce incorrect DXIL when the
// call site is in a different module.  Every collective except
// `workgroup_upper_bound` writes to groupshared internally, so calling them
// via `import goldy_exp` would trigger the bug and produce wrong results.
//
// Workaround (same strategy as ekrano's coarse.slang): each test shader
// inlines the algorithm body directly in the entry point.  `import goldy_exp`
// is kept for the Goldy binding framework ([goldy_compute], Scattered<T>,
// etc.) but the collective bodies are NOT invoked through the module boundary.
//
// `workgroup_upper_bound` is the one exception: it only reads groupshared
// (never writes inside the function), so the cross-module call is safe.

// ── workgroup_inclusive_scan_wave_uint_sum ─────────────────────────────────

/// Inlined body of `workgroup_inclusive_scan_wave_uint_sum<64>`, uniform
/// input (all threads contribute 1).  Expected output[i] = i + 1.
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

/// Same algorithm with ramp input (thread i contributes i+1).
/// Expected output[i] = T(i+1) = (i+1)*(i+2)/2.
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

/// WG_SIZE=256, uniform input — regression test for the production size used
/// in ekrano's coarse shader.  Expected output[i] = i + 1.
///
/// sh_scratch must hold one entry per wave.  SM6.0 mandates WaveGetLaneCount()
/// >= 4, giving at most 256/4 = 64 waves, so [64] is the safe minimum.
/// ([32] overflows on WARP where WaveGetLaneCount() == 4.)
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

// ── workgroup_reduce ────────────────────────────────────────────────────────

/// Inlined right-sweep reduce (uint, N=64), all-ones input.
/// Thread 0 accumulates the total; expected output[0] = 64.
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

// ── workgroup_inclusive_scan ────────────────────────────────────────────────

/// Inlined left-sweep inclusive scan (uint, N=64), all-ones input.
/// Expected output[i] = i + 1.
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

// ── workgroup_broadcast ─────────────────────────────────────────────────────

/// Inlined broadcast of value 42 over 64 threads.  All output[i] must = 42.
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

// ── workgroup_upper_bound ───────────────────────────────────────────────────
// workgroup_upper_bound only *reads* groupshared (never writes inside the
// function), so the cross-module call is safe — this test exercises the
// goldy_exp version directly.

/// `workgroup_upper_bound` with prefix_sums[i] = i+1 over 64 threads.
/// upper_bound(k) in {1,2,...,64} = k for all k in [0, 63].
const UPPER_BOUND_64: &str = r#"
import goldy_exp;
groupshared uint sh_ps[64];
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> OUT, GroupThreadId local_id) {
    uint ix = local_id.x;
    sh_ps[ix] = ix + 1u; // build linear prefix-sum: [1, 2, 3, ..., 64]
    GroupMemoryBarrierWithGroupSync();
    OUT[ix] = workgroup_upper_bound<6>(ix, sh_ps);
}
"#;

#[test]
fn test_wave_inclusive_scan_uniform_64() {
    let device = make_device();
    let shader = ShaderModule::from_slang(&device, WAVE_SCAN_64_UNIFORM)
        .expect("compile WAVE_SCAN_64_UNIFORM");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let out = Buffer::new(&device, 64 * 4, DataAccess::Scattered).expect("output buffer");
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&out]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 64 * 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[u32] = bytemuck::cast_slice(&raw);

    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val,
            i as u32 + 1,
            "wave_scan_uniform_64[{i}]: expected {} got {val}",
            i + 1
        );
    }
}

#[test]
fn test_wave_inclusive_scan_ramp_64() {
    let device = make_device();
    let shader =
        ShaderModule::from_slang(&device, WAVE_SCAN_64_RAMP).expect("compile WAVE_SCAN_64_RAMP");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let out = Buffer::new(&device, 64 * 4, DataAccess::Scattered).expect("output buffer");
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&out]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 64 * 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[u32] = bytemuck::cast_slice(&raw);

    // Expected: triangular numbers T(1)=1, T(2)=3, T(3)=6, ... T(k)=k*(k+1)/2
    for (i, &val) in result.iter().enumerate() {
        let k = (i + 1) as u32;
        let expected = k * (k + 1) / 2;
        assert_eq!(
            val, expected,
            "wave_scan_ramp_64[{i}]: expected {expected} got {val}"
        );
    }
}

#[test]
fn test_wave_inclusive_scan_uniform_256() {
    let device = make_device();
    let shader = ShaderModule::from_slang(&device, WAVE_SCAN_256_UNIFORM)
        .expect("compile WAVE_SCAN_256_UNIFORM");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let out = Buffer::new(&device, 256 * 4, DataAccess::Scattered).expect("output buffer");
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&out]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 256 * 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[u32] = bytemuck::cast_slice(&raw);

    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val,
            i as u32 + 1,
            "wave_scan_uniform_256[{i}]: expected {} got {val}",
            i + 1
        );
    }
}

#[test]
fn test_workgroup_reduce_uint_correct() {
    let device = make_device();
    let shader =
        ShaderModule::from_slang(&device, REDUCE_64_UNIFORM).expect("compile REDUCE_64_UNIFORM");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let out = Buffer::new(&device, 64 * 4, DataAccess::Scattered).expect("output buffer");
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&out]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 64 * 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[u32] = bytemuck::cast_slice(&raw);

    // Thread 0 accumulates the total (64 ones = 64); other threads hold partial sums.
    assert_eq!(
        result[0], 64,
        "workgroup_reduce: thread 0 must hold total 64, got {}",
        result[0]
    );
}

#[test]
fn test_workgroup_inclusive_scan_uint_correct() {
    let device = make_device();
    let shader = ShaderModule::from_slang(&device, INCLUSIVE_SCAN_64_UNIFORM)
        .expect("compile INCLUSIVE_SCAN_64_UNIFORM");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let out = Buffer::new(&device, 64 * 4, DataAccess::Scattered).expect("output buffer");
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&out]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 64 * 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[u32] = bytemuck::cast_slice(&raw);

    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val,
            i as u32 + 1,
            "workgroup_inclusive_scan[{i}]: expected {} got {val}",
            i + 1
        );
    }
}

#[test]
fn test_workgroup_broadcast_correct() {
    let device = make_device();
    let shader = ShaderModule::from_slang(&device, BROADCAST_64).expect("compile BROADCAST_64");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let out = Buffer::new(&device, 64 * 4, DataAccess::Scattered).expect("output buffer");
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&out]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 64 * 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[u32] = bytemuck::cast_slice(&raw);

    for (i, &val) in result.iter().enumerate() {
        assert_eq!(val, 42, "workgroup_broadcast[{i}]: expected 42 got {val}");
    }
}

#[test]
fn test_workgroup_upper_bound_linear() {
    let device = make_device();
    let shader = ShaderModule::from_slang(&device, UPPER_BOUND_64).expect("compile UPPER_BOUND_64");
    let pipeline = ComputePipeline::new(&device, &shader).expect("create pipeline");

    let out = Buffer::new(&device, 64 * 4, DataAccess::Scattered).expect("output buffer");
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&out]);
        pass.dispatch(1, 1, 1);
    }
    encoder.dispatch(&device).expect("dispatch");

    let mut raw = vec![0u8; 64 * 4];
    out.read_to_cpu(&device, &mut raw).expect("readback");
    let result: &[u32] = bytemuck::cast_slice(&raw);

    // prefix_sums = [1, 2, ..., 64]; upper_bound(k) = k for k in [0, 63].
    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val, i as u32,
            "workgroup_upper_bound[{i}]: expected {i} got {val}"
        );
    }
}

/// Round-trip test for `SpatialAccess::DirectInterpolated`:
/// 1. Write a known pattern into the texture via UAV (storage image) in a compute pass.
/// 2. Read it back via SRV (sampled image) using hardware bilinear (at texel centres, so
///    the result should be exact) in a second compute pass.
/// 3. Assert that the round-tripped values match the original.
///
/// This validates that both the storage and sampled bindless slots point to the same
/// underlying resource on all supported backends.
#[test]
fn texture_dual_view_round_trip() {
    const W: u32 = 4;
    const H: u32 = 4;
    const N: usize = (W * H) as usize;

    /// Write pass: fill a DirectSpatial (UAV) texture with known RGBA values.
    const WRITE_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(4, 4, 1)]
void cs_main(DirectSpatial<float4> dst, ThreadId id) {
    uint x = id.x;
    uint y = id.y;
    // Encode pixel coordinates as (x/255, y/255, 0, 1)
    dst[uint2(x, y)] = float4(float(x) / 255.0, float(y) / 255.0, 0.0, 1.0);
}
"#;

    /// Read pass: sample the Interpolated (SRV) view at texel centres and write to a buffer.
    const READ_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(4, 4, 1)]
void cs_main(Interpolated<float4> src, Filter smp, Scattered<uint> out, ThreadId id) {
    uint x = id.x;
    uint y = id.y;
    float2 uv = (float2(x, y) + 0.5) / float2(4.0, 4.0);
    float4 v = src.Sample(smp, uv);
    // Pack 8-bit rgba into a single uint.
    uint r = uint(v.x * 255.0 + 0.5);
    uint g = uint(v.y * 255.0 + 0.5);
    uint b = uint(v.z * 255.0 + 0.5);
    uint a = uint(v.w * 255.0 + 0.5);
    out[y * 4 + x] = r | (g << 8) | (b << 16) | (a << 24);
}
"#;

    let device = make_device();

    // Check that the device supports DirectInterpolated (all backends should).
    let tex = Texture::new(
        &device,
        W,
        H,
        TextureFormat::Rgba8Unorm,
        SpatialAccess::DirectInterpolated,
        TextureFlags::empty(),
    )
    .expect("create DirectInterpolated texture");

    let storage_idx = tex
        .bindless_index()
        .expect("DirectInterpolated must have a storage bindless index");
    let sampled_idx = tex
        .bindless_sampled_index()
        .expect("DirectInterpolated must have a sampled bindless index");
    assert_ne!(
        storage_idx, sampled_idx,
        "storage and sampled slots must be distinct"
    );

    // Write pass (UAV).
    let write_shader = ShaderModule::from_slang(&device, WRITE_SHADER).expect("compile write");
    let write_pipeline = ComputePipeline::new(&device, &write_shader).expect("write pipeline");
    let mut enc = ComputeEncoder::new();
    {
        let mut pass = enc.begin_compute_pass();
        pass.set_pipeline(&write_pipeline);
        pass.bind_resources(&[&tex]);
        pass.dispatch(1, 1, 1);
    }
    enc.dispatch(&device).expect("write dispatch");

    // Read pass (SRV + sampler).
    let sampler = goldy::Sampler::nearest(&device).expect("create sampler");
    let out = Buffer::new(&device, (N * 4) as u64, DataAccess::Scattered).expect("out buffer");
    let read_shader = ShaderModule::from_slang(&device, READ_SHADER).expect("compile read");
    let read_pipeline = ComputePipeline::new(&device, &read_shader).expect("read pipeline");
    let mut enc2 = ComputeEncoder::new();
    {
        let mut pass = enc2.begin_compute_pass();
        pass.set_pipeline(&read_pipeline);
        // Bind sampled view, sampler, output buffer.
        // We use the Texture's borrow as Interpolated view here.
        pass.bind_resources_with_handles(&[
            storage_idx,  // UAV (not read this pass, but keeps slot order)
            sampled_idx,  // Texture2D<float4> SRV
            sampler.bindless_index().unwrap(),
            out.bindless_index().unwrap(),
        ]);
        pass.dispatch(1, 1, 1);
    }
    enc2.dispatch(&device).expect("read dispatch");

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
