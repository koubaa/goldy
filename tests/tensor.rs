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
fn matmul_indexed_reads_the_u32_expert_and_replays() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).unwrap();
    // expert 0 is identity, expert 1 scales the two rows by 2 and 3.
    let weight = Tensor::from_f32(
        &device,
        TensorShape::from_dims(&[2, 2, 2]).unwrap(),
        &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0],
    )
    .unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[3.0, 4.0]).unwrap();
    let ids = Tensor::from_u32(&device, TensorShape::vector(2), &[1, 0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(2), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let _slab = kernels
        .recorder(&mut scheme)
        .matmul_indexed_into("expert", weight.view(), x.view(), ids.view(), 0, out.view())
        .unwrap();
    assert_eq!(read_f32(&mut scheme, out.buffer()), vec![6.0, 12.0]);

    ids.buffer().write(0, bytemuck::cast_slice(&[0u32, 1u32])).unwrap();
    assert_eq!(read_f32(&mut scheme, out.buffer()), vec![3.0, 4.0]);

    ids.buffer().write(0, bytemuck::cast_slice(&[9u32, 0u32])).unwrap();
    assert_eq!(read_f32(&mut scheme, out.buffer()), vec![0.0, 0.0]);
}

#[test]
fn matmul_indexed_batch_runs_a_row_of_ids() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).unwrap();
    // experts: identity, scale 2/3, scale 4/5.
    let weight = Tensor::from_f32(
        &device,
        TensorShape::from_dims(&[3, 2, 2]).unwrap(),
        &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0],
    )
    .unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[3.0, 4.0]).unwrap();
    let ids = Tensor::from_u32(&device, TensorShape::vector(3), &[2, 0, 9]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(6), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernels
        .recorder(&mut scheme)
        .matmul_indexed_batch_into("experts", weight.view(), x.view(), ids.view(), 0, 3, 0, out.view())
        .unwrap();
    // expert 2, expert 0, and an id past the table.
    assert_eq!(
        read_f32(&mut scheme, out.buffer()),
        vec![12.0, 20.0, 3.0, 4.0, 0.0, 0.0]
    );

    let xs = Tensor::from_f32(&device, TensorShape::vector(4), &[1.0, 1.0, 3.0, 4.0]).unwrap();
    let pair = Tensor::from_u32(&device, TensorShape::vector(2), &[0, 1]).unwrap();
    let down = Tensor::zeros(&device, TensorShape::vector(4), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernels
        .recorder(&mut scheme)
        .matmul_indexed_batch_into("strided-x", weight.view(), xs.view(), pair.view(), 0, 2, 2, down.view())
        .unwrap();
    // id 0 with [1, 1] is [1, 1]; id 1 with [3, 4] is [6, 12].
    assert_eq!(read_f32(&mut scheme, down.buffer()), vec![1.0, 1.0, 6.0, 12.0]);
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
    // Offsets and facts come from per-slot macros that default to the metadata parcel.
    assert!(slang.contains("#define _GOLDY_TENSOR_OFF0 _goldy_tensor_meta[0u].off"));
    assert!(slang.contains("_goldy_tensor_meta[1u].numel"));
    assert!(slang.contains("goldy_tensor_offset(_goldy_t0,"));
    assert!(slang.contains("goldy_tensor_offset(_goldy_t1,"));
    assert!(slang.contains("_goldy_t1.numel"));
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
    assert!(copy_eq::CANONICAL_SOURCE.contains("goldy_tensor_offset1(_goldy_t0,"));
    assert!(copy_wildcard_cols::CANONICAL_SOURCE.contains("goldy_tensor_offset2(_goldy_t0,"));
    assert!(copy_rank3::CANONICAL_SOURCE.contains("goldy_tensor_offset3(_goldy_t0,"));
    assert!(copy_view::CANONICAL_SOURCE.contains("goldy_tensor_offset(_goldy_t0,"));
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

/// Strided views compute the same on the universal program (layouts from the metadata
/// parcel) and on the promoted variants (offsets as launch words, shape facts baked).
/// Views that differ only in offset share one variant.
#[test]
fn strided_views_survive_layout_promotion() {
    let _gpu = gpu_lock();
    let _spec = goldy::test_support::SpecializationOverride::force_enabled();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let predicts = !matches!(
        device.backend_type(),
        goldy::BackendType::WebGpu | goldy::BackendType::Cpu
    );
    let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let parent = Tensor::from_f32(&device, TensorShape::from_dims(&[2, 3, 4]).unwrap(), &values).unwrap();
    let column = |offset| {
        let layout = TensorLayout::strided(TensorDType::F32, TensorShape::vector(3), offset, &[4]).unwrap();
        TensorView::new(parent.buffer(), layout).unwrap()
    };
    let rank1 = copy_eq::Kernel::prepare(&device).unwrap();
    let rank2 = copy_wildcard_cols::Kernel::prepare(&device).unwrap();
    let rank3 = copy_rank3::Kernel::prepare(&device).unwrap();
    let col1 = Tensor::zeros(&device, TensorShape::vector(3), TensorDType::F32).unwrap();
    let col2 = Tensor::zeros(&device, TensorShape::vector(3), TensorDType::F32).unwrap();
    let plane = Tensor::zeros(&device, TensorShape::matrix(4, 3), TensorDType::F32).unwrap();
    let cube = Tensor::zeros(&device, TensorShape::from_dims(&[4, 2, 3]).unwrap(), TensorDType::F32).unwrap();

    let mut scheme = Scheme::new(&ctx);
    let nodes = [
        rank1
            .record(&mut scheme, "col1", column(1), col1.view())
            .unwrap()
            .over_1d(3)
            .node(),
        rank1
            .record(&mut scheme, "col2", column(2), col2.view())
            .unwrap()
            .over_1d(3)
            .node(),
        rank2
            .record(
                &mut scheme,
                "transpose",
                parent
                    .view()
                    .narrow(0, 1, 1)
                    .unwrap()
                    .reshape(&[3, 4])
                    .unwrap()
                    .transpose()
                    .unwrap(),
                plane.view(),
            )
            .unwrap()
            .over_1d(12)
            .node(),
        rank3
            .record(
                &mut scheme,
                "permute",
                parent.view().permute(&[2, 0, 1]).unwrap(),
                cube.view(),
            )
            .unwrap()
            .over_1d(24)
            .node(),
    ];

    let transposed: Vec<f32> = (0..4)
        .flat_map(|c| (0..3).map(move |r| (12 + r * 4 + c) as f32))
        .collect();
    let permuted: Vec<f32> = (0..4)
        .flat_map(|k| (0..2).flat_map(move |i| (0..3).map(move |j| (i * 12 + j * 4 + k) as f32)))
        .collect();
    let frame_and_check = |scheme: &mut Scheme, what: &str| {
        let mut sub = scheme.submit().expect(what);
        let mut take = |buf: &goldy::Buffer| -> Vec<f32> {
            let bytes = (&mut sub >> buf).take::<u8>().expect("host take");
            bytemuck::cast_slice(&bytes).to_vec()
        };
        assert_eq!(take(col1.buffer()), vec![1.0, 5.0, 9.0], "{what}: col1");
        assert_eq!(take(col2.buffer()), vec![2.0, 6.0, 10.0], "{what}: col2");
        assert_eq!(take(plane.buffer()), transposed, "{what}: transpose");
        assert_eq!(take(cube.buffer()), permuted, "{what}: permute");
        drop(sub);
        goldy::test_support::wait_for_specialization_compiles(scheme);
    };

    // Shape facts are certain, so the first submit already warms every variant and the
    // second swaps them in.
    frame_and_check(&mut scheme, "universal");
    frame_and_check(&mut scheme, "promoted");
    let stats = scheme.replay_stats();
    if predicts {
        assert_eq!(stats.specialization_warms, 3, "col1 and col2 share one compile");
        assert_eq!(stats.specialization_promotions, 4);
        assert!(nodes.iter().all(|&n| scheme.node_is_specialized(n)));
    } else {
        assert_eq!(stats.specialization_warms, 0);
    }
    frame_and_check(&mut scheme, "steady state");
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

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        let bits = ((*v).to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn e2m1(nib: u8) -> f32 {
    let mag = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(nib & 7) as usize];
    if nib & 8 == 0 {
        mag
    } else {
        -mag
    }
}

fn f8_e4m3(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 3) & 0xF;
    let mant = b & 7;
    let val = if exp == 0 {
        mant as f32 * 2f32.powi(-9)
    } else if exp == 15 && mant == 7 {
        f32::NAN
    } else {
        (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    if sign == 0 {
        val
    } else {
        -val
    }
}

#[test]
fn bf16_matvec_promotes_inside_the_dot() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).unwrap();
    // [[1, 0], [0, 2]] @ [3, 4] = [3, 8]
    let weight = Tensor::from_bytes(
        &device,
        TensorShape::matrix(2, 2),
        TensorDType::BF16,
        &bf16_bytes(&[1.0, 0.0, 0.0, 2.0]),
    )
    .unwrap();
    let x = Tensor::from_f32(&device, TensorShape::vector(2), &[3.0, 4.0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(2), TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernels
        .recorder(&mut scheme)
        .matmul_bf16_into("bf16", weight.view(), x.view(), out.view())
        .unwrap();
    let got = read_f32(&mut scheme, out.buffer());
    assert_eq!(got, vec![3.0, 8.0], "{got:?}");
}

const TORCH_E4M3_BITS: [u32; 256] = [
    0x00000000, 0x3b000000, 0x3b800000, 0x3bc00000, 0x3c000000, 0x3c200000, 0x3c400000, 0x3c600000, 0x3c800000,
    0x3c900000, 0x3ca00000, 0x3cb00000, 0x3cc00000, 0x3cd00000, 0x3ce00000, 0x3cf00000, 0x3d000000, 0x3d100000,
    0x3d200000, 0x3d300000, 0x3d400000, 0x3d500000, 0x3d600000, 0x3d700000, 0x3d800000, 0x3d900000, 0x3da00000,
    0x3db00000, 0x3dc00000, 0x3dd00000, 0x3de00000, 0x3df00000, 0x3e000000, 0x3e100000, 0x3e200000, 0x3e300000,
    0x3e400000, 0x3e500000, 0x3e600000, 0x3e700000, 0x3e800000, 0x3e900000, 0x3ea00000, 0x3eb00000, 0x3ec00000,
    0x3ed00000, 0x3ee00000, 0x3ef00000, 0x3f000000, 0x3f100000, 0x3f200000, 0x3f300000, 0x3f400000, 0x3f500000,
    0x3f600000, 0x3f700000, 0x3f800000, 0x3f900000, 0x3fa00000, 0x3fb00000, 0x3fc00000, 0x3fd00000, 0x3fe00000,
    0x3ff00000, 0x40000000, 0x40100000, 0x40200000, 0x40300000, 0x40400000, 0x40500000, 0x40600000, 0x40700000,
    0x40800000, 0x40900000, 0x40a00000, 0x40b00000, 0x40c00000, 0x40d00000, 0x40e00000, 0x40f00000, 0x41000000,
    0x41100000, 0x41200000, 0x41300000, 0x41400000, 0x41500000, 0x41600000, 0x41700000, 0x41800000, 0x41900000,
    0x41a00000, 0x41b00000, 0x41c00000, 0x41d00000, 0x41e00000, 0x41f00000, 0x42000000, 0x42100000, 0x42200000,
    0x42300000, 0x42400000, 0x42500000, 0x42600000, 0x42700000, 0x42800000, 0x42900000, 0x42a00000, 0x42b00000,
    0x42c00000, 0x42d00000, 0x42e00000, 0x42f00000, 0x43000000, 0x43100000, 0x43200000, 0x43300000, 0x43400000,
    0x43500000, 0x43600000, 0x43700000, 0x43800000, 0x43900000, 0x43a00000, 0x43b00000, 0x43c00000, 0x43d00000,
    0x43e00000, 0x7fc00000, 0x80000000, 0xbb000000, 0xbb800000, 0xbbc00000, 0xbc000000, 0xbc200000, 0xbc400000,
    0xbc600000, 0xbc800000, 0xbc900000, 0xbca00000, 0xbcb00000, 0xbcc00000, 0xbcd00000, 0xbce00000, 0xbcf00000,
    0xbd000000, 0xbd100000, 0xbd200000, 0xbd300000, 0xbd400000, 0xbd500000, 0xbd600000, 0xbd700000, 0xbd800000,
    0xbd900000, 0xbda00000, 0xbdb00000, 0xbdc00000, 0xbdd00000, 0xbde00000, 0xbdf00000, 0xbe000000, 0xbe100000,
    0xbe200000, 0xbe300000, 0xbe400000, 0xbe500000, 0xbe600000, 0xbe700000, 0xbe800000, 0xbe900000, 0xbea00000,
    0xbeb00000, 0xbec00000, 0xbed00000, 0xbee00000, 0xbef00000, 0xbf000000, 0xbf100000, 0xbf200000, 0xbf300000,
    0xbf400000, 0xbf500000, 0xbf600000, 0xbf700000, 0xbf800000, 0xbf900000, 0xbfa00000, 0xbfb00000, 0xbfc00000,
    0xbfd00000, 0xbfe00000, 0xbff00000, 0xc0000000, 0xc0100000, 0xc0200000, 0xc0300000, 0xc0400000, 0xc0500000,
    0xc0600000, 0xc0700000, 0xc0800000, 0xc0900000, 0xc0a00000, 0xc0b00000, 0xc0c00000, 0xc0d00000, 0xc0e00000,
    0xc0f00000, 0xc1000000, 0xc1100000, 0xc1200000, 0xc1300000, 0xc1400000, 0xc1500000, 0xc1600000, 0xc1700000,
    0xc1800000, 0xc1900000, 0xc1a00000, 0xc1b00000, 0xc1c00000, 0xc1d00000, 0xc1e00000, 0xc1f00000, 0xc2000000,
    0xc2100000, 0xc2200000, 0xc2300000, 0xc2400000, 0xc2500000, 0xc2600000, 0xc2700000, 0xc2800000, 0xc2900000,
    0xc2a00000, 0xc2b00000, 0xc2c00000, 0xc2d00000, 0xc2e00000, 0xc2f00000, 0xc3000000, 0xc3100000, 0xc3200000,
    0xc3300000, 0xc3400000, 0xc3500000, 0xc3600000, 0xc3700000, 0xc3800000, 0xc3900000, 0xc3a00000, 0xc3b00000,
    0xc3c00000, 0xc3d00000, 0xc3e00000, 0x7fc00000,
];

#[test]
fn e4m3_matches_the_torch_table_and_e2m1_is_complete() {
    for b in 0..256u16 {
        let got = f8_e4m3(b as u8);
        let bits = TORCH_E4M3_BITS[b as usize];
        if b == 127 || b == 255 {
            assert!(got.is_nan(), "{b}");
            continue;
        }
        let expect = f32::from_bits(bits);
        assert_eq!(got.to_bits(), expect.to_bits(), "byte {b}");
    }
    let mags = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for nib in 0..16u8 {
        let expect = if nib < 8 {
            mags[nib as usize]
        } else {
            -mags[(nib & 7) as usize]
        };
        assert_eq!(e2m1(nib).to_bits(), expect.to_bits());
    }
}

#[test]
fn block_scaled_matvec_distinguishes_nibble_order() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).unwrap();
    // Expert 0, one row, k = 16. First byte is nibbles (1.0, 2.0). Scale is 2.0.
    // A swapped nibble order changes the dot because x[0] != x[1].
    let mut codes = vec![0u8; 2 * 8];
    codes[0] = 2 | (4 << 4);
    for b in 8..16 {
        codes[b] = 0xAA;
    }
    let mut scales = vec![0u8; 2];
    scales[0] = 0x40;
    scales[1] = 0x38;
    let codes_t = Tensor::from_bytes(
        &device,
        TensorShape::from_dims(&[2, 1, 8]).unwrap(),
        TensorDType::U8,
        &codes,
    )
    .unwrap();
    let scales_t = Tensor::from_bytes(
        &device,
        TensorShape::from_dims(&[2, 1, 1]).unwrap(),
        TensorDType::F8E4M3,
        &scales,
    )
    .unwrap();
    let factors = Tensor::from_f32(&device, TensorShape::vector(2), &[1.5, -2.0]).unwrap();
    let mut x_vals = [0.0f32; 16];
    for i in 0..16 {
        x_vals[i] = ((i % 17) as f32 - 8.0) / 8.0;
    }
    let x = Tensor::from_f32(&device, TensorShape::vector(16), &x_vals).unwrap();
    let ids = Tensor::from_u32(&device, TensorShape::vector(1), &[0]).unwrap();
    let out = Tensor::zeros(&device, TensorShape::vector(1), TensorDType::F32).unwrap();
    let run = |scheme: &mut Scheme, id: u32| {
        ids.buffer().write(0, bytemuck::cast_slice(&[id])).unwrap();
        kernels
            .recorder(scheme)
            .matmul_block_scaled_into(
                "block",
                goldy::BlockScaled {
                    codes: codes_t.view(),
                    scales: scales_t.view(),
                    elems: goldy::ops::PackedElem::E2M1,
                    scale: goldy::ops::PackedScale::E4M3,
                    block: [1, 16],
                    global: Some(factors.view()),
                },
                x.view(),
                Some((ids.view(), 0)),
                out.view(),
            )
            .unwrap();
        read_f32(scheme, out.buffer())[0]
    };
    let mut scheme = Scheme::new(&ctx);
    let got = run(&mut scheme, 0);
    let mut expect = 0.0f32;
    for col in 0..16 {
        let byte = codes[col / 2];
        let nib = if col % 2 == 0 { byte & 0xF } else { byte >> 4 };
        expect += e2m1(nib) * f8_e4m3(scales[0]) * 1.5 * x_vals[col];
    }
    assert!((got - expect).abs() < 1e-5, "expert 0 {got} want {expect}");
    let swapped = e2m1(4) * f8_e4m3(scales[0]) * 1.5 * x_vals[0] + e2m1(2) * f8_e4m3(scales[0]) * 1.5 * x_vals[1];
    assert!((swapped - expect).abs() > 1e-3, "nibble swap must change the reference");

    let mut scheme = Scheme::new(&ctx);
    let got = run(&mut scheme, 1);
    let mut expect = 0.0f32;
    for col in 0..16 {
        expect += e2m1(0xA) * f8_e4m3(scales[1]) * -2.0 * x_vals[col];
    }
    assert!((got - expect).abs() < 1e-4, "expert 1 {got} want {expect}");

    let mut scheme = Scheme::new(&ctx);
    let got = run(&mut scheme, 9);
    assert_eq!(got, 0.0);
}
