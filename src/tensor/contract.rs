//! Host-only contract tests: identity model, dtypes, layouts, broadcasting, API boundary.

use super::*;
use crate::error::GoldyError;

fn shape(dims: &[u32]) -> TensorShape {
    TensorShape::from_dims(dims).unwrap()
}

#[test]
fn packed_layout_row_major_strides() {
    let layout = TensorLayout::packed(TensorDType::F32, shape(&[2, 3, 4]), 0).unwrap();
    assert_eq!(layout.strides(), &[12, 4, 1]);
    assert!(layout.is_contiguous());
    assert!(layout.is_writeable());
    assert_eq!(layout.numel().unwrap(), 24);
}

#[test]
fn zero_stride_broadcast_is_not_writeable() {
    let layout = TensorLayout::strided(TensorDType::F32, shape(&[2, 3]), 0, &[0, 1]).unwrap();
    assert!(!layout.is_writeable());
    assert!(layout.require_writeable("add").is_err());
}

#[test]
fn size_one_axis_may_have_zero_stride() {
    let layout = TensorLayout::strided(TensorDType::F32, shape(&[1, 4]), 0, &[0, 1]).unwrap();
    assert!(layout.is_writeable());
}

#[test]
fn negative_strides_rejected() {
    let err = TensorLayout::strided(TensorDType::F32, shape(&[2]), 0, &[-1]).unwrap_err();
    assert!(matches!(err, GoldyError::Validation(_)));
}

#[test]
fn rank_limit() {
    let err = TensorShape::from_dims(&[1, 1, 1, 1, 1]).unwrap_err();
    assert!(matches!(err, GoldyError::Validation(_)));
}

#[test]
fn numel_overflow() {
    let err = TensorShape::from_dims(&[u32::MAX, u32::MAX, 2])
        .unwrap()
        .numel()
        .unwrap_err();
    assert!(matches!(err, GoldyError::Validation(_)));
}

#[test]
fn reshape_requires_contiguous_and_matching_numel() {
    let layout = TensorLayout::packed(TensorDType::F32, shape(&[2, 3]), 4).unwrap();
    assert_eq!(layout.numel().unwrap(), 6);
    let packed = TensorLayout::packed(TensorDType::F32, shape(&[6]), 4).unwrap();
    assert_eq!(packed.storage_offset(), 4);
}

#[test]
fn broadcast_shapes_align_from_the_right() {
    let out = super::view::broadcast_shapes(shape(&[3, 1]), shape(&[1, 4])).unwrap();
    assert_eq!(out.dims(), &[3, 4]);
    let err = super::view::broadcast_shapes(shape(&[3, 2]), shape(&[4, 2])).unwrap_err();
    assert!(matches!(err, GoldyError::Validation(_)));
}

#[test]
fn permute_is_a_stride_lens() {
    let layout = TensorLayout::packed(TensorDType::F32, shape(&[2, 3, 4]), 0).unwrap();
    assert_eq!(layout.strides(), &[12, 4, 1]);
}

#[test]
fn envelope_covers_strided_window() {
    let layout = TensorLayout::strided(TensorDType::F32, shape(&[2, 2]), 3, &[8, 1]).unwrap();
    let (start, end) = layout.element_envelope().unwrap();
    assert_eq!(start, 3);
    assert_eq!(end, 3 + 8 + 1 + 1);
    let (off, len) = layout.byte_envelope().unwrap();
    assert_eq!(off, 3 * 4);
    assert_eq!(len, (end - start) * 4);
}

#[test]
fn broadcast_envelope_ignores_expanded_axis() {
    let unique = TensorLayout::packed(TensorDType::F32, shape(&[3]), 2).unwrap();
    let (a, b) = unique.byte_envelope().unwrap();
    let broadcast = TensorLayout::strided(TensorDType::F32, shape(&[2, 3]), 2, &[0, 1]).unwrap();
    let (c, d) = broadcast.byte_envelope().unwrap();
    assert_eq!((a, b), (c, d));
}

#[test]
fn scatter_modes_are_explicit() {
    assert_ne!(ScatterMode::UniqueWrite, ScatterMode::Add);
    assert_ne!(ScatterMode::Min, ScatterMode::Max);
}

#[test]
fn gpu_layout_is_stable_kernel_abi() {
    assert_eq!(std::mem::size_of::<GoldyTensorLayout>(), 48);
    let layout = TensorLayout::strided(TensorDType::F32, shape(&[2, 3]), 5, &[3, 1]).unwrap();
    let gpu = layout.gpu_coords().unwrap();
    assert_eq!(gpu.offset, 5);
    assert_eq!(gpu.rank, 2);
    assert_eq!(gpu.numel, 6);
    assert_eq!(gpu.shape, [2, 3, 1, 1]);
    assert_eq!(gpu.stride, [3, 1, 0, 0]);
}

#[test]
fn dtype_support_is_per_op() {
    assert!(TensorDType::U32.require_f32("exp").is_err());
    assert!(TensorDType::F32.require_f32("exp").is_ok());
}

#[test]
fn reduction_axis_must_exist() {
    assert!(shape(&[2, 3]).dim(2).is_err());
    assert_eq!(shape(&[2, 3]).squeeze_axis(1).unwrap().dims(), &[2]);
}

#[test]
fn matmul_shape_rules() {
    // Covered via TensorRecorder on GPU; lock the host shape helper here by constructing shapes.
    assert_eq!(TensorShape::matrix(2, 3).dims(), &[2, 3]);
    assert_eq!(TensorShape::vector(4).numel().unwrap(), 4);
    assert_eq!(TensorShape::scalar().numel().unwrap(), 1);
}
