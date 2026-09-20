//! Integration tests for `#[goldy::compute]` Rust→Slang kernels (issue #78).

#![cfg(feature = "gpu")]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    compute, BackendType, BufferKind, DepositTarget, DeviceDescriptor, Instance, MemoryExchange,
    RequestAdapterOptions, RetainedPool, Scheme, StructuredBufferElement, TextureFlags, TextureFormat, TextureKind,
};
use std::sync::Arc;

#[compute(workgroup_size = [64, 1, 1])]
fn saxpy(x: &[f32], y: &mut [f32], a: f32) {
    let i = goldy::gpu::global_id().x;
    if i < y.len() {
        y[i] = a * x[i] + y[i];
    }
}

#[compute(workgroup_size = [64, 1, 1])]
fn double_u32(data: &mut [u32]) {
    let i = goldy::gpu::global_id().x;
    if i < data.len() {
        data[i] = data[i] * 2u32;
    }
}

#[compute(workgroup_size = [8, 8, 1])]
fn fill_red(output: goldy::gpu::DirectSpatial<goldy::gpu::Float4>) {
    let tid = goldy::gpu::global_id();
    output[tid.xy] = goldy::gpu::float4(1.0, 0.0, 0.0, 1.0);
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, goldy::GpuType)]
struct PlasmaUniforms {
    width: u32,
    height: u32,
    time: f32,
}

#[compute(workgroup_size = [1, 1, 1])]
fn read_plasma_uniforms(uniforms: &[PlasmaUniforms], out: &mut [f32]) {
    let i = goldy::gpu::global_id().x;
    if i == 0 {
        let u: PlasmaUniforms = uniforms[0];
        out[0] = u.width as f32;
        out[1] = u.height as f32;
        out[2] = u.time;
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, goldy::GpuType)]
struct TightVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

#[compute(workgroup_size = [1, 1, 1])]
fn read_tight_vertex(verts: &[TightVertex], out: &mut [f32]) {
    let i = goldy::gpu::global_id().x;
    if i == 0 {
        let v: TightVertex = verts[0];
        out[0] = v.position.x;
        out[1] = v.position.y;
        out[2] = v.position.z;
        out[3] = v.uv.x;
        out[4] = v.uv.y;
    }
}

fn float4_storage_format(device: &goldy::Device) -> TextureFormat {
    match device.backend_type() {
        BackendType::Cuda | BackendType::WebGpu => TextureFormat::Rgba32Float,
        _ => TextureFormat::Rgba8Unorm,
    }
}

fn main() {
    let mut args = libtest_mimic::Arguments::from_args();
    let instance = Instance::new().expect("instance");
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("device");
    submission::clamp_test_threads(&mut args, &device);
    let device = Arc::new(device);

    let tests = vec![
        libtest_mimic::Trial::test("rust_kernel_canonical_source_shape", || {
            assert!(saxpy::CANONICAL_SOURCE.contains("[goldy_compute]"));
            assert!(saxpy::CANONICAL_SOURCE.contains("BufRO<float> x"));
            assert!(saxpy::CANONICAL_SOURCE.contains("Scattered<float> y"));
            assert!(saxpy::CANONICAL_SOURCE.contains("float a"));
            assert!(saxpy::CANONICAL_SOURCE.contains("ThreadId _goldy_gid"));
            assert!(saxpy::CANONICAL_SOURCE.contains("[numthreads(64, 1, 1)]"));
            assert!(double_u32::CANONICAL_SOURCE.contains("Scattered<uint> data"));
            Ok(())
        }),
        libtest_mimic::Trial::test("rust_kernel_double_u32_gpu", {
            let device = Arc::clone(&device);
            move || {
                let ctx = device.create_context()?;
                let mut pool = RetainedPool::new(Arc::clone(&device));
                let n = 64usize;
                let input: Vec<u32> = (0..n as u32).collect();
                let data = pool.acquire_buffer_with_data(&input, BufferKind::Scattered)?;

                let kernel = double_u32::Kernel::prepare(&device)?;
                let mut scheme = Scheme::new(&ctx);
                kernel.record(&mut scheme, "double", &data).over_1d(n as u32);
                let grant = MemoryExchange::new(scheme.context()).bind_withdraw(&mut scheme, &data)?;
                let mut frame = scheme.submit()?;
                let bytes = grant.claim(&mut frame)?.consume()?;
                let out: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();
                assert_eq!(out.len(), n);
                for i in 0..n {
                    assert_eq!(out[i], (i as u32) * 2, "index {i}");
                }
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("rust_kernel_saxpy_gpu", {
            let device = Arc::clone(&device);
            move || {
                let ctx = device.create_context()?;
                let mut pool = RetainedPool::new(Arc::clone(&device));
                let n = 256usize;
                let a = 2.0f32;
                let x_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let y_data: Vec<f32> = (0..n).map(|i| (i * 3) as f32).collect();
                let expected: Vec<f32> = (0..n).map(|i| a * (i as f32) + (i * 3) as f32).collect();
                let x = pool.acquire_buffer_with_data(&x_data, BufferKind::Scattered)?;
                let y = pool.acquire_buffer_with_data(&y_data, BufferKind::Scattered)?;

                let kernel = saxpy::Kernel::prepare(&device)?;
                let mut scheme = Scheme::new(&ctx);
                kernel
                    .record(&mut scheme, "saxpy", &x, &y, a)
                    .groups([(n as u32).div_ceil(64), 1, 1]);
                let grant = MemoryExchange::new(scheme.context()).bind_withdraw(&mut scheme, &y)?;
                let mut frame = scheme.submit()?;
                let bytes = grant.claim(&mut frame)?.consume()?;
                let out: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
                assert_eq!(out.len(), n);
                for i in 0..n {
                    assert!(
                        (out[i] - expected[i]).abs() < 1e-5,
                        "index {i}: {} vs {}",
                        out[i],
                        expected[i]
                    );
                }
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("kernel_abi_roundtrip_from_canonical", || {
            let def = goldy::slang::try_kernel_def_from_source(saxpy::CANONICAL_SOURCE)
                .expect("parse saxpy canonical source");
            assert_eq!(def.entry, "cs_main");
            assert_eq!(def.workgroup_size, [64, 1, 1]);
            assert_eq!(def.params.len(), 3);
            assert!(def.builtins.global_id);
            let wrapper = goldy::slang::emit_wrapper_from_kernel_def(&def);
            assert!(wrapper.contains("[shader(\"compute\")]"));
            assert!(wrapper.contains("goldy_frame_table_index"));
            assert!(wrapper.contains("_goldy_user_cs_main"));
            Ok(())
        }),
        libtest_mimic::Trial::test("rust_kernel_image_canonical_source", || {
            assert!(fill_red::CANONICAL_SOURCE.contains("DirectSpatial<float4> output"));
            assert!(fill_red::CANONICAL_SOURCE.contains("float4(1.0, 0.0, 0.0, 1.0)"));
            assert!(fill_red::CANONICAL_SOURCE.contains("[numthreads(8, 8, 1)]"));
            let def = goldy::slang::try_kernel_def_from_source(fill_red::CANONICAL_SOURCE)
                .expect("parse fill_red canonical source");
            assert_eq!(def.params.len(), 1);
            assert_eq!(def.params[0].category, goldy::ParamCategory::StorageImage);
            Ok(())
        }),
        libtest_mimic::Trial::test("rust_kernel_fill_red_gpu", {
            let device = Arc::clone(&device);
            move || {
                let ctx = device.create_context()?;
                let mut pool = RetainedPool::new(Arc::clone(&device));
                let format = float4_storage_format(&device);
                let width = 8u32;
                let height = 8u32;
                let texture = pool.acquire_texture(
                    width,
                    height,
                    format,
                    TextureKind::Direct,
                    TextureFlags::COPY_SRC,
                    None,
                )?;

                let kernel = fill_red::Kernel::prepare(&device)?;
                let mut scheme = Scheme::new(&ctx);
                kernel.record(&mut scheme, "fill", &texture).over_2d(width, height);
                let grant = MemoryExchange::new(scheme.context()).bind_withdraw(&mut scheme, &texture)?;
                let mut frame = scheme.submit()?;
                let bytes = grant.claim(&mut frame)?.consume()?;
                assert!(!bytes.iter().all(|&b| b == 0), "texture readback all zeros");
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("gpu_type_uniforms_deposit_without_author_padding", {
            let device = Arc::clone(&device);
            move || {
                assert_eq!(std::mem::size_of::<PlasmaUniforms>(), 12);
                assert_eq!(PlasmaUniforms::gpu_element_stride(), 12);
                let ctx = device.create_context()?;
                let pool = RetainedPool::new(Arc::clone(&device));
                let uniforms = pool.acquire_buffer_with_data(
                    &[PlasmaUniforms {
                        width: 0,
                        height: 0,
                        time: 0.0,
                    }],
                    BufferKind::Scattered,
                )?;
                let out = pool.acquire_buffer_with_data(&[0.0f32; 3], BufferKind::Scattered)?;

                let mut upload = Scheme::new(&ctx);
                let deposit = MemoryExchange::new(&ctx).bind_deposit(
                    &mut upload,
                    DepositTarget::buffer_elements::<PlasmaUniforms>(&uniforms, 1),
                )?;
                deposit.write_data(
                    0,
                    &[PlasmaUniforms {
                        width: 8,
                        height: 4,
                        time: 1.5,
                    }],
                )?;
                upload.submit()?;

                let kernel = read_plasma_uniforms::Kernel::prepare(&device)?;
                let mut scheme = Scheme::new(&ctx);
                kernel.record(&mut scheme, "read", &uniforms, &out).over_1d(1);
                let grant = MemoryExchange::new(scheme.context()).bind_withdraw(&mut scheme, &out)?;
                let mut frame = scheme.submit()?;
                let bytes = grant.claim(&mut frame)?.consume()?;
                let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
                assert_eq!(got.len(), 3);
                assert!((got[0] - 8.0).abs() < 1e-5, "width {}", got[0]);
                assert!((got[1] - 4.0).abs() < 1e-5, "height {}", got[1]);
                assert!((got[2] - 1.5).abs() < 1e-5, "time {}", got[2]);
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("gpu_type_deposit_packs_float3_then_float2", {
            let device = Arc::clone(&device);
            move || {
                assert_eq!(std::mem::size_of::<TightVertex>(), 20);
                assert_eq!(TightVertex::gpu_element_stride(), 32);
                let ctx = device.create_context()?;
                let pool = RetainedPool::new(Arc::clone(&device));
                let verts = pool.acquire_buffer_with_data(
                    &[TightVertex {
                        position: [0.0; 3],
                        uv: [0.0; 2],
                    }],
                    BufferKind::Scattered,
                )?;
                let out = pool.acquire_buffer_with_data(&[0.0f32; 5], BufferKind::Scattered)?;

                let mut upload = Scheme::new(&ctx);
                let deposit = MemoryExchange::new(&ctx).bind_deposit(
                    &mut upload,
                    DepositTarget::buffer_elements::<TightVertex>(&verts, 1),
                )?;
                deposit.write_data(
                    0,
                    &[TightVertex {
                        position: [1.0, 2.0, 3.0],
                        uv: [4.0, 5.0],
                    }],
                )?;
                upload.submit()?;

                let kernel = read_tight_vertex::Kernel::prepare(&device)?;
                let mut scheme = Scheme::new(&ctx);
                kernel.record(&mut scheme, "read", &verts, &out).over_1d(1);
                let grant = MemoryExchange::new(scheme.context()).bind_withdraw(&mut scheme, &out)?;
                let mut frame = scheme.submit()?;
                let bytes = grant.claim(&mut frame)?.consume()?;
                let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
                assert_eq!(got.len(), 5);
                for (i, want) in [1.0, 2.0, 3.0, 4.0, 5.0].iter().enumerate() {
                    assert!((got[i] - want).abs() < 1e-5, "index {i}: {} vs {want}", got[i]);
                }
                Ok(())
            }
        }),
    ];

    libtest_mimic::run(&args, tests).exit();
}
