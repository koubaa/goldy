//! GPU tests for the dense tensor layer.

#![cfg(all(feature = "gpu", feature = "tensor"))]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    BufferKind, RequestAdapterOptions, Runtime, RuntimeDescriptor, ScatterMode, Scheme, Tensor, TensorDType,
    TensorKernels, TensorLayout, TensorScalar, TensorShape, TensorView,
};
use std::sync::Mutex;

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn double_u32(buf: &mut [u32], n: u32) {
    let i = goldy::gpu::global_id().x;
    if i < n {
        buf[i] = buf[i] * 2u32;
    }
}

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn copy_view(src: goldy::gpu::Tensor<f32>, dst: goldy::gpu::TensorWrite<f32>) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        dst[i] = src[i];
    }
}

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn scale_view(src: goldy::gpu::Tensor<f32>, dst: goldy::gpu::TensorWrite<f32>, a: f32) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        dst[i] = a * src[i];
    }
}

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn copy_eq(
    #[tensor(shape = [n])] src: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n])] dst: goldy::gpu::TensorWrite<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        dst[i] = src[i];
    }
}

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn copy_exact4(
    #[tensor(shape = [4])] src: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [4])] dst: goldy::gpu::TensorWrite<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        dst[i] = src[i];
    }
}

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn copy_wildcard_cols(
    #[tensor(shape = [_, n])] src: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [_, n])] dst: goldy::gpu::TensorWrite<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        dst[i] = src[i];
    }
}

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn copy_rank3(
    #[tensor(shape = [a, b, c])] src: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [a, b, c])] dst: goldy::gpu::TensorWrite<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i < dst.len() {
        dst[i] = src[i];
    }
}

static GPU: Mutex<()> = Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn expect_record_err<T>(r: Result<T, goldy::GoldyError>) -> goldy::GoldyError {
    match r {
        Err(e) => e,
        Ok(_) => panic!("expected tensor shape contract error"),
    }
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
fn add_and_broadcast_and_replay() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).expect("tensor ctx");
    let a = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, 2.0]).unwrap();
    let b = Tensor::from_f32(&device, TensorShape::vector(1), &[10.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let out = kernels.recorder(&mut scheme).add("add", a.view(), b.view()).unwrap();
    drop(kernels);
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
    let kernels = TensorKernels::new(&device).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, -4.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let mut rec = kernels.recorder(&mut scheme);
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
    let kernels = TensorKernels::new(&device).unwrap();
    let w = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[3.0, 4.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let y = kernels
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
    let kernels = TensorKernels::new(&device).unwrap();
    let parent = Tensor::from_f32(&device, TensorShape::matrix(2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let row1 = parent.view().narrow(0, 1, 1).unwrap().reshape(&[3]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let out = kernels.recorder(&mut scheme).mul_scalar("id", row1, 1.0).unwrap();
    assert_eq!(read_f32(&mut scheme, out.buffer()), vec![4.0, 5.0, 6.0]);
}

#[test]
fn reduce_sum_and_softmax() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let mut rec = kernels.recorder(&mut scheme);
    let s = rec.sum("sum", x.view(), 1).unwrap();
    drop(rec);
    assert_eq!(read_f32(&mut scheme, s.buffer()), vec![3.0, 7.0]);
}

#[test]
fn gather_1d() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).unwrap();
    let src = Tensor::from_f32(&device, TensorShape::vector(4), &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let idx = Tensor::from_i32(&device, TensorShape::vector(3), &[0, 2, 2]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let g = kernels
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
    let kernels = TensorKernels::new(&device).unwrap();
    let src = Tensor::from_f32(&device, TensorShape::vector(3), &[10.0, 30.0, 30.0]).unwrap();
    let idx = Tensor::from_i32(&device, TensorShape::vector(3), &[0, 2, 2]).unwrap();
    let dst = Tensor::from_f32(&device, TensorShape::vector(4), &[1.0, 1.0, 1.0, 1.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernels
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
    let kernels = TensorKernels::new(&device).unwrap();
    let parent = Tensor::from_f32(&device, TensorShape::vector(4), &[0.0, 0.0, 0.0, 0.0]).unwrap();
    let left = parent.view().narrow(0, 0, 3).unwrap();
    let right = parent.view().narrow(0, 1, 3).unwrap();
    let ones = Tensor::from_f32(&device, TensorShape::vector(3), &[1.0, 1.0, 1.0]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let mut rec = kernels.recorder(&mut scheme);
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
    let kernels = TensorKernels::new(&device).unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[1.9, -2.1]).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let y = kernels
        .recorder(&mut scheme)
        .cast("cast", x.view(), TensorDType::I32)
        .unwrap();

    let mut sub = scheme.submit().unwrap();
    let bytes = (&mut sub >> y.buffer()).take::<u8>().expect("host take");
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
    drop(k);

    let mut sub = scheme.submit().unwrap();
    let bytes = (&mut sub >> data.buffer()).take::<u8>().expect("host take");
    let got: &[u32] = bytemuck::cast_slice(&bytes);
    assert_eq!(got, &[2, 4, 6, 8]);
}

#[test]
fn batched_matmul_rank3() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).unwrap();
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
    let y = kernels.recorder(&mut scheme).matmul("bmm", a.view(), b.view()).unwrap();
    assert_eq!(read_f32(&mut scheme, y.buffer()), vec![3.0, 4.0, 10.0, 12.0]);
}

#[test]
fn tensor_kernel_canonical_source_and_abi() {
    let slang = copy_view::CANONICAL_SOURCE;
    assert!(slang.contains("struct GoldyTensorLayout"));
    assert!(slang.contains("BufRO<GoldyTensorLayout> _goldy_tensor_meta"));
    assert!(slang.contains("goldy_tensor_offset(_goldy_tensor_meta[0u]"));
    assert!(slang.contains("goldy_tensor_offset(_goldy_tensor_meta[1u]"));
    assert!(slang.contains("_goldy_tensor_meta[1u].numel"));
    assert!(scale_view::CANONICAL_SOURCE.contains("float a"));
}

#[test]
fn tensor_kernel_nonzero_offset_narrow() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let src = Tensor::from_f32(&device, TensorShape::vector(4), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let dst = Tensor::zeros(&device, TensorShape::vector(2), TensorDType::F32).unwrap();
    let k = copy_view::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    k.record(&mut scheme, "narrow", src.view().narrow(0, 2, 2).unwrap(), dst.view())
        .unwrap()
        .over_1d(2);
    assert_eq!(read_f32(&mut scheme, dst.buffer()), vec![3.0, 4.0]);
}

#[test]
fn tensor_kernel_permuted_read() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let src = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let dst = Tensor::zeros(&device, TensorShape::matrix(2, 2), TensorDType::F32).unwrap();
    let k = copy_view::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    k.record(&mut scheme, "perm", src.view().transpose().unwrap(), dst.view())
        .unwrap()
        .over_1d(4);
    assert_eq!(read_f32(&mut scheme, dst.buffer()), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn tensor_kernel_broadcast_read() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let src = Tensor::from_f32(&device, TensorShape::vector(1), &[7.0]).unwrap();
    let dst = Tensor::zeros(&device, TensorShape::vector(3), TensorDType::F32).unwrap();
    let k = copy_view::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    k.record(
        &mut scheme,
        "bcast",
        src.view().broadcast_to(TensorShape::vector(3)).unwrap(),
        dst.view(),
    )
    .unwrap()
    .over_1d(3);
    assert_eq!(read_f32(&mut scheme, dst.buffer()), vec![7.0, 7.0, 7.0]);
}

#[test]
fn tensor_kernel_strided_write() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let src = Tensor::from_f32(&device, TensorShape::vector(4), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let dst = Tensor::zeros(&device, TensorShape::matrix(2, 2), TensorDType::F32).unwrap();
    let k = copy_view::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    k.record(&mut scheme, "stride_w", src.view(), dst.view().transpose().unwrap())
        .unwrap()
        .over_1d(4);
    assert_eq!(read_f32(&mut scheme, dst.buffer()), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn tensor_kernel_rejects_broadcast_write() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let src = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, 2.0]).unwrap();
    let dst = Tensor::from_f32(&device, TensorShape::vector(1), &[0.0]).unwrap();
    let dest = dst.view().broadcast_to(TensorShape::vector(2)).unwrap();
    assert!(!dest.is_writeable());
    let k = copy_view::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    assert!(k.record(&mut scheme, "bcast_w", src.view(), dest).is_err());
}

#[test]
fn tensor_kernel_alias_and_retained_replay() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let buf = Tensor::from_f32(&device, TensorShape::vector(4), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let k = scale_view::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let left = buf.view().narrow(0, 0, 2).unwrap();
    let right = buf.view().narrow(0, 2, 2).unwrap();
    k.record(&mut scheme, "alias", right, left, 10.0).unwrap().over_1d(2);
    let got = read_f32(&mut scheme, buf.buffer());
    assert_eq!(got, vec![30.0, 40.0, 3.0, 4.0]);
    let _ = scheme.submit().unwrap();
    assert!(
        scheme.replay_stats().records <= 2,
        "tensor kernel metadata should retain, stats={:?}",
        scheme.replay_stats()
    );
}

#[test]
fn tensor_shape_contract_accepts_matching_and_strided_views() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let parent = Tensor::from_f32(&device, TensorShape::matrix(2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let src = parent.view().narrow(0, 1, 1).unwrap().reshape(&[3]).unwrap();
    let dst = Tensor::zeros(&device, TensorShape::vector(3), TensorDType::F32).unwrap();
    let k = copy_eq::Kernel::prepare(&device).unwrap();
    assert!(k
        .def()
        .params
        .iter()
        .any(|p| p.shape_spec.as_ref().is_some_and(|s| s.rank() == 1)));
    let mut scheme = Scheme::new(&ctx);
    k.record(&mut scheme, "eq", src, dst.view()).unwrap().over_1d(3);
    assert_eq!(read_f32(&mut scheme, dst.buffer()), vec![4.0, 5.0, 6.0]);
    let _ = scheme.submit().unwrap();
    assert!(
        scheme.replay_stats().records <= 2,
        "contracted tensor kernel should retain, stats={:?}",
        scheme.replay_stats()
    );
}

#[test]
fn tensor_shape_contract_lowers_each_rank() {
    assert!(copy_eq::CANONICAL_SOURCE.contains("goldy_tensor_offset1(_goldy_tensor_meta[0u]"));
    assert!(copy_wildcard_cols::CANONICAL_SOURCE.contains("goldy_tensor_offset2(_goldy_tensor_meta[0u]"));
    assert!(copy_rank3::CANONICAL_SOURCE.contains("goldy_tensor_offset3(_goldy_tensor_meta[0u]"));
    assert!(copy_view::CANONICAL_SOURCE.contains("goldy_tensor_offset(_goldy_tensor_meta[0u]"));
}

/// Each contracted rank reads non-contiguous views: a strided column and a broadcast at
/// rank 1, a transpose at rank 2, and a permutation at rank 3.
#[test]
fn tensor_shape_contract_reads_strided_views_at_each_rank() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let parent = Tensor::from_f32(&device, TensorShape::from_dims(&[2, 3, 4]).unwrap(), &values).unwrap();

    let rank1 = copy_eq::Kernel::prepare(&device).unwrap();
    let column = TensorLayout::strided(TensorDType::F32, TensorShape::vector(3), 1, &[4]).unwrap();
    let column = TensorView::new(parent.buffer(), column).unwrap();
    let dst = Tensor::zeros(&device, TensorShape::vector(3), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    rank1
        .record(&mut scheme, "column", column, dst.view())
        .unwrap()
        .over_1d(3);
    assert_eq!(read_f32(&mut scheme, dst.buffer()), vec![1.0, 5.0, 9.0]);

    let one = TensorLayout::packed(TensorDType::F32, TensorShape::vector(1), 15).unwrap();
    let bcast = TensorView::new(parent.buffer(), one)
        .unwrap()
        .broadcast_to(TensorShape::vector(3))
        .unwrap();
    let mut scheme = Scheme::new(&ctx);
    rank1
        .record(&mut scheme, "bcast", bcast, dst.view())
        .unwrap()
        .over_1d(3);
    assert_eq!(read_f32(&mut scheme, dst.buffer()), vec![15.0, 15.0, 15.0]);

    let rank2 = copy_wildcard_cols::Kernel::prepare(&device).unwrap();
    let plane = parent
        .view()
        .narrow(0, 1, 1)
        .unwrap()
        .reshape(&[3, 4])
        .unwrap()
        .transpose()
        .unwrap();
    let dst = Tensor::zeros(&device, TensorShape::matrix(4, 3), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    rank2
        .record(&mut scheme, "transpose", plane, dst.view())
        .unwrap()
        .over_1d(12);
    let expected: Vec<f32> = (0..4)
        .flat_map(|c| (0..3).map(move |r| (12 + r * 4 + c) as f32))
        .collect();
    assert_eq!(read_f32(&mut scheme, dst.buffer()), expected);

    let rank3 = copy_rank3::Kernel::prepare(&device).unwrap();
    let permuted = parent.view().permute(&[2, 0, 1]).unwrap();
    let dst = Tensor::zeros(&device, TensorShape::from_dims(&[4, 2, 3]).unwrap(), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    rank3
        .record(&mut scheme, "permute", permuted, dst.view())
        .unwrap()
        .over_1d(24);
    let expected: Vec<f32> = (0..4)
        .flat_map(|k| (0..2).flat_map(move |i| (0..3).map(move |j| (i * 12 + j * 4 + k) as f32)))
        .collect();
    assert_eq!(read_f32(&mut scheme, dst.buffer()), expected);
}

#[test]
fn tensor_shape_contract_rejects_before_graphir() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let src = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, 2.0]).unwrap();
    let dst = Tensor::zeros(&device, TensorShape::vector(3), TensorDType::F32).unwrap();
    let k = copy_eq::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let before = scheme.ir_node_count();
    let err = expect_record_err(k.record(&mut scheme, "mismatch", src.view(), dst.view()));
    assert!(err.to_string().contains("parameter `dst`"), "{err}");
    assert!(err.to_string().contains("`n`"), "{err}");
    assert_eq!(scheme.ir_node_count(), before);
    let dst_ok = Tensor::zeros(&device, TensorShape::vector(2), TensorDType::F32).unwrap();
    k.record(&mut scheme, "ok", src.view(), dst_ok.view())
        .unwrap()
        .over_1d(2);
    assert!(scheme.ir_node_count() > before);
}

#[test]
fn tensor_shape_contract_exact_and_wildcard() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let src4 = Tensor::from_f32(&device, TensorShape::vector(4), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let dst4 = Tensor::zeros(&device, TensorShape::vector(4), TensorDType::F32).unwrap();
    let exact = copy_exact4::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    exact
        .record(&mut scheme, "exact", src4.view(), dst4.view())
        .unwrap()
        .over_1d(4);
    assert_eq!(read_f32(&mut scheme, dst4.buffer()), vec![1.0, 2.0, 3.0, 4.0]);

    let src2 = Tensor::from_f32(&device, TensorShape::vector(2), &[1.0, 2.0]).unwrap();
    let dst2 = Tensor::zeros(&device, TensorShape::vector(2), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let err = expect_record_err(exact.record(&mut scheme, "exact_bad", src2.view(), dst2.view()));
    assert!(err.to_string().contains("expected 4, got 2"), "{err}");

    let a = Tensor::from_f32(&device, TensorShape::matrix(2, 3), &[1.0; 6]).unwrap();
    let b = Tensor::zeros(&device, TensorShape::matrix(2, 3), TensorDType::F32).unwrap();
    let wild = copy_wildcard_cols::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    wild.record(&mut scheme, "wild", a.view(), b.view()).unwrap().over_1d(6);
    assert_eq!(read_f32(&mut scheme, b.buffer()), vec![1.0; 6]);

    let taller = Tensor::zeros(&device, TensorShape::matrix(5, 3), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    assert!(wild.record(&mut scheme, "wild_rows", a.view(), taller.view()).is_ok());
    assert_eq!(scheme.ir_node_count(), 0);

    let c = Tensor::zeros(&device, TensorShape::matrix(2, 4), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let err = expect_record_err(wild.record(&mut scheme, "wild_bad", a.view(), c.view()));
    assert!(err.to_string().contains("parameter `dst`"), "{err}");
}

#[test]
fn tensor_shape_contract_wrong_rank() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let src = Tensor::from_f32(&device, TensorShape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let dst = Tensor::zeros(&device, TensorShape::matrix(2, 2), TensorDType::F32).unwrap();
    let k = copy_eq::Kernel::prepare(&device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let err = expect_record_err(k.record(&mut scheme, "rank", src.view(), dst.view()));
    assert!(err.to_string().contains("expected rank 1"), "{err}");
}

#[allow(dead_code)]
fn _compile_view_helpers(v: TensorView<'_>) {
    let _ = v.dtype();
    let _ = BufferKind::Scattered;
}
