//! Semantic `scheme.matmul` — native (cuBLAS/MPS) or Goldy stdlib fallback.

#![cfg(feature = "gpu")]

#[path = "common/submission.rs"]
mod submission;

use goldy::{BufferKind, MatMulDesc, MatMulView, RequestAdapterOptions, Runtime, RuntimeDescriptor, Scheme};
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

fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 2001) as f32 / 1000.0 - 1.0
        })
        .collect()
}

struct GemvCase {
    m: usize,
    k: usize,
    lda: usize,
    a_off: usize,
    x_stride: usize,
    x_off: usize,
    y_stride: usize,
    y_off: usize,
}

fn assert_gemv_matches_reference(case: GemvCase) {
    let GemvCase {
        m,
        k,
        lda,
        a_off,
        x_stride,
        x_off,
        y_stride,
        y_off,
    } = case;
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let a_host = pseudo_random(a_off + m * lda, 1);
    let x_host = pseudo_random(x_off + k * x_stride, 2);
    let sentinel = 7.5f32;
    let y_len = y_off + m * y_stride;
    let a = device.acquire_buffer_with_data(&a_host, BufferKind::Scattered).unwrap();
    let x = device.acquire_buffer_with_data(&x_host, BufferKind::Scattered).unwrap();
    let y = device
        .acquire_buffer_with_data(&vec![sentinel; y_len], BufferKind::Scattered)
        .unwrap();
    let mut scheme = Scheme::new(&ctx);
    scheme
        .matmul("gemv", MatMulDesc::gemv(m as u32, k as u32))
        .a(&a, MatMulView::strided(a_off as u64, lda as u32))
        .b(&x, MatMulView::strided(x_off as u64, x_stride as u32))
        .out(&y, MatMulView::strided(y_off as u64, y_stride as u32))
        .record();
    let got = read_f32(&mut scheme, &y);
    for (i, value) in got.iter().enumerate() {
        let written = i >= y_off && (i - y_off) % y_stride == 0 && (i - y_off) / y_stride < m;
        if !written {
            assert_eq!(*value, sentinel, "gemv wrote outside y at element {i}");
            continue;
        }
        let row = (i - y_off) / y_stride;
        let expected: f64 = (0..k)
            .map(|j| f64::from(a_host[a_off + row * lda + j]) * f64::from(x_host[x_off + j * x_stride]))
            .sum();
        let tolerance = 1e-5 * k as f64;
        assert!(
            (f64::from(*value) - expected).abs() <= tolerance,
            "row {row}: got {value}, expected {expected} (m={m}, k={k})"
        );
    }
}

#[test]
fn gemv_strided_operands_match_reference() {
    assert_gemv_matches_reference(GemvCase {
        m: 37,
        k: 301,
        lda: 305,
        a_off: 3,
        x_stride: 2,
        x_off: 1,
        y_stride: 3,
        y_off: 2,
    });
}

#[test]
fn gemv_model_shapes_match_reference() {
    for (m, k) in [(288, 288), (768, 288), (288, 768), (4099, 1031), (1, 5)] {
        assert_gemv_matches_reference(GemvCase {
            m,
            k,
            lda: k,
            a_off: 0,
            x_stride: 1,
            x_off: 0,
            y_stride: 1,
            y_off: 0,
        });
    }
}
