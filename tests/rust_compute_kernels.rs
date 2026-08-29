//! Integration tests for `#[goldy::compute]` Rust→Slang kernels (issue #78).

#![cfg(feature = "gpu")]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    compute, BufferKind, DeviceDescriptor, Instance, MemoryExchange, RequestAdapterOptions, RetainedPool, Scheme,
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
    ];

    libtest_mimic::run(&args, tests).exit();
}
