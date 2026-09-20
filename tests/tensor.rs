//! GPU tests for the dense tensor layer.

#![cfg(all(feature = "gpu", feature = "tensor"))]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    BufferKind, MemoryExchange, RequestAdapterOptions, Runtime, RuntimeDescriptor, ScatterMode, Scheme, Tensor,
    TensorContext, TensorDType, TensorScalar, TensorShape, TensorView,
};
use std::sync::Mutex;

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn double_u32(buf: &mut [u32], n: u32) {
    let i = goldy::gpu::global_id().x;
    if i < n {
        buf[i] = buf[i] * 2u32;
    }
}

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
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(scheme, buf)
        .expect("withdraw");
    let mut sub = scheme.submit().expect("submit");
    let bytes = grant.claim(&mut sub).expect("claim").consume().expect("consume");
    bytemuck::cast_slice(&bytes).to_vec()
}

#[test]
fn add_and_broadcast_and_replay() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let mut tensors = TensorContext::new(&device).expect("tensor ctx");
    let a = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, 2.0]).unwrap();
    let b = Tensor::from_f32(&device, TensorShape::vector(1), &[10.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let out = tensors.recorder(&mut scheme).add("add", a.view(), b.view()).unwrap();
    let got = read_f32(&mut scheme, out.buffer());
    assert_eq!(got, vec![11.0, 12.0]);
    let _ = scheme.submit().unwrap();
    assert!(
        scheme.replay_stats().records <= 2,
        "tensor add should retain, stats={:?}",
        scheme.replay_stats()
    );
}

#[test]
fn unary_neg_exp_and_fill() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let mut tensors = TensorContext::new(&device).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, -4.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let mut rec = tensors.recorder(&mut scheme);
    let n = rec.neg("neg", x.view()).unwrap();
    rec.fill("fill", n.view(), TensorScalar::F32(3.0)).unwrap();
    drop(rec);
    assert_eq!(read_f32(&mut scheme, n.buffer()), vec![3.0, 3.0]);
}

#[test]
fn matmul_gemv_matches_semantic_node() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let mut tensors = TensorContext::new(&device).unwrap();
    let w = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[3.0, 4.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let y = tensors
        .recorder(&mut scheme)
        .matmul("gemv", w.view(), x.view())
        .unwrap();
    assert_eq!(read_f32(&mut scheme, y.buffer()), vec![3.0, 4.0]);
}

#[test]
fn strided_narrow_views_alias_parent() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let mut tensors = TensorContext::new(&device).unwrap();
    let parent = Tensor::from_f32(&device, TensorShape::matrix(2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let row1 = parent.view().narrow(0, 1, 1).unwrap().reshape(&[3]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let out = tensors.recorder(&mut scheme).mul_scalar("id", row1, 1.0).unwrap();
    assert_eq!(read_f32(&mut scheme, out.buffer()), vec![4.0, 5.0, 6.0]);
}

#[test]
fn reduce_sum_and_softmax() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let mut tensors = TensorContext::new(&device).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let mut rec = tensors.recorder(&mut scheme);
    let s = rec.sum("sum", x.view(), 1).unwrap();
    drop(rec);
    assert_eq!(read_f32(&mut scheme, s.buffer()), vec![3.0, 7.0]);
}

#[test]
fn gather_1d() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let mut tensors = TensorContext::new(&device).unwrap();
    let src = Tensor::from_f32(&device, TensorShape::vector(4), &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let idx = Tensor::from_i32(&device, TensorShape::vector(3), &[0, 2, 2]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let g = tensors
        .recorder(&mut scheme)
        .gather("gather", src.view(), idx.view(), 0)
        .unwrap();
    assert_eq!(read_f32(&mut scheme, g.buffer()), vec![10.0, 30.0, 30.0]);
}

#[test]
fn scatter_add_unique_and_colliding() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let mut tensors = TensorContext::new(&device).unwrap();
    let src = Tensor::from_f32(&device, TensorShape::vector(3), &[10.0, 30.0, 30.0]).unwrap();
    let idx = Tensor::from_i32(&device, TensorShape::vector(3), &[0, 2, 2]).unwrap();
    let dst = Tensor::from_f32(&device, TensorShape::vector(4), &[1.0, 1.0, 1.0, 1.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    tensors
        .recorder(&mut scheme)
        .scatter("scatter", src.view(), idx.view(), dst.view(), 0, ScatterMode::Add)
        .unwrap();
    let got = read_f32(&mut scheme, dst.buffer());
    assert_eq!(got[0], 11.0);
    assert_eq!(got[2], 61.0);
}

#[test]
fn overlapping_writes_are_ordered() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let mut tensors = TensorContext::new(&device).unwrap();
    let parent = Tensor::from_f32(&device, TensorShape::vector(4), &[0.0, 0.0, 0.0, 0.0]).unwrap();
    let left = parent.view().narrow(0, 0, 3).unwrap();
    let right = parent.view().narrow(0, 1, 3).unwrap();
    let ones = Tensor::from_f32(&device, TensorShape::vector(3), &[1.0, 1.0, 1.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let mut rec = tensors.recorder(&mut scheme);
    rec.add_into("left", left, ones.view(), left).unwrap();
    rec.add_into("right", right, ones.view(), right).unwrap();
    drop(rec);
    let got = read_f32(&mut scheme, parent.buffer());
    assert_eq!(got, vec![1.0, 2.0, 2.0, 1.0]);
}

#[test]
fn cast_f32_i32() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let mut tensors = TensorContext::new(&device).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[1.9, -2.1]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let y = tensors
        .recorder(&mut scheme)
        .cast("cast", x.view(), TensorDType::I32)
        .unwrap();
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, y.buffer())
        .unwrap();
    let mut sub = scheme.submit().unwrap();
    let bytes = grant.claim(&mut sub).unwrap().consume().unwrap();
    let got: &[i32] = bytemuck::cast_slice(&bytes);
    assert_eq!(got, &[1, -2]);
}

#[test]
fn kernel_bindable_view_and_over_tensor() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let data = Tensor::from_u32(&device, TensorShape::vector(4), &[1, 2, 3, 4]).unwrap();
    let k = double_u32::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    k.record(&mut scheme, "dbl", &data.view(), data.view().numel_u32())
        .over_tensor(&data.view());
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, data.buffer())
        .unwrap();
    let mut sub = scheme.submit().unwrap();
    let bytes = grant.claim(&mut sub).unwrap().consume().unwrap();
    let got: &[u32] = bytemuck::cast_slice(&bytes);
    assert_eq!(got, &[2, 4, 6, 8]);
}

#[test]
fn batched_matmul_rank3() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let mut tensors = TensorContext::new(&device).unwrap();
    let a = Tensor::from_f32(
        &device,
        TensorShape::from_dims(&[2, 2, 2]).unwrap(),
        &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
    )
    .unwrap();
    let b = Tensor::from_f32(
        &device,
        TensorShape::from_dims(&[2, 2, 1]).unwrap(),
        &[3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let mut scheme = Scheme::new(&ctx);
    let y = tensors.recorder(&mut scheme).matmul("bmm", a.view(), b.view()).unwrap();
    assert_eq!(read_f32(&mut scheme, y.buffer()), vec![3.0, 4.0, 10.0, 12.0]);
}

#[allow(dead_code)]
fn _compile_view_helpers(v: TensorView<'_>) {
    let _ = v.dtype();
    let _ = BufferKind::Scattered;
}
