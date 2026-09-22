//! Semantic `scheme.matmul` — native (cuBLAS/MPS) or Goldy stdlib fallback.

#![cfg(feature = "gpu")]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    BufferKind, MatMulDesc, MatMulView, MemoryExchange, RequestAdapterOptions, Runtime, RuntimeDescriptor, Scheme,
};
use std::ops::Shr;
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn runtime() -> Runtime {
    goldy::Instance::new()
        .expect("instance")
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_runtime(&RuntimeDescriptor::default())
        .expect("runtime")
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let mut sub = scheme.submit().expect("submit");
    let bytes = (&mut sub >> buf).take::<u8>().expect("host take");
    bytemuck::cast_slice(&bytes).to_vec()
}

#[test]
fn gemv_identity() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let x = device
        .acquire_buffer_with_data(&[1.0f32, 2.0], BufferKind::Scattered)
        .unwrap();
    let w = device
        .acquire_buffer_with_data(&[1.0f32, 0.0, 0.0, 1.0], BufferKind::Scattered)
        .unwrap();
    let out = device
        .acquire_buffer_with_data(&[0.0f32, 0.0], BufferKind::Scattered)
        .unwrap();
    let mut scheme = Scheme::new(&ctx);
    scheme
        .matmul("gemv", MatMulDesc::gemv(2, 2))
        .a(&w, MatMulView::packed())
        .b(&x, MatMulView::packed())
        .out(&out, MatMulView::packed())
        .record();
    let got = read_f32(&mut scheme, &out);
    assert_eq!(got, vec![1.0, 2.0]);
    let _ = scheme.submit().unwrap();
    assert!(
        scheme.replay_stats().records <= 2,
        "matmul should retain after first submit, stats={:?}",
        scheme.replay_stats()
    );
}

#[test]
fn gemm_2x3_times_3x2() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    // A = [[1, 2, 3], [4, 5, 6]]
    let a = device
        .acquire_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], BufferKind::Scattered)
        .unwrap();
    // B = [[7, 8], [9, 10], [11, 12]]
    let b = device
        .acquire_buffer_with_data(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], BufferKind::Scattered)
        .unwrap();
    let c = device
        .acquire_buffer_with_data(&[0.0f32; 4], BufferKind::Scattered)
        .unwrap();
    let mut scheme = Scheme::new(&ctx);
    scheme
        .matmul("gemm", MatMulDesc::gemm(2, 2, 3))
        .a(&a, MatMulView::packed())
        .b(&b, MatMulView::packed())
        .out(&c, MatMulView::packed())
        .record();
    let got = read_f32(&mut scheme, &c);
    // [1*7+2*9+3*11, 1*8+2*10+3*12; 4*7+5*9+6*11, 4*8+5*10+6*12]
    assert_eq!(got, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn gemv_reads_weighted_offset() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    // blob = [junk, junk, 1, 0, 0, 1]
    let w = device
        .acquire_buffer_with_data(&[9.0f32, 8.0, 1.0, 0.0, 0.0, 1.0], BufferKind::Scattered)
        .unwrap();
    let x = device
        .acquire_buffer_with_data(&[3.0f32, 4.0], BufferKind::Scattered)
        .unwrap();
    let out = device
        .acquire_buffer_with_data(&[0.0f32, 0.0], BufferKind::Scattered)
        .unwrap();
    let mut scheme = Scheme::new(&ctx);
    scheme
        .matmul("offset", MatMulDesc::gemv(2, 2))
        .a(&w, MatMulView::offset(2))
        .b(&x, MatMulView::packed())
        .out(&out, MatMulView::packed())
        .record();
    assert_eq!(read_f32(&mut scheme, &out), vec![3.0, 4.0]);
}
