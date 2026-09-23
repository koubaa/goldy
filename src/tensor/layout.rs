//! Storage layout: dtype, shape, element offset, and per-axis strides.

use super::dtype::TensorDType;
use super::shape::TensorShape;
use super::MAX_TENSOR_RANK;
use crate::error::GoldyError;

/// Packed or strided layout over a parent parcel. Negative strides are out of scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorLayout {
    dtype: TensorDType,
    shape: TensorShape,
    /// Element offset from the start of the parent buffer.
    storage_offset: u64,
    strides: [i64; MAX_TENSOR_RANK],
}

impl TensorLayout {
    /// Packed row-major layout at `storage_offset`.
    pub fn packed(dtype: TensorDType, shape: TensorShape, storage_offset: u64) -> Result<Self, GoldyError> {
        Ok(Self {
            dtype,
            shape,
            storage_offset,
            strides: packed_stride_array(shape)?,
        })
    }

    /// Explicit strides (length must equal rank). Zero strides are allowed (broadcast).
    pub fn strided(
        dtype: TensorDType,
        shape: TensorShape,
        storage_offset: u64,
        strides: &[i64],
    ) -> Result<Self, GoldyError> {
        if strides.len() != shape.rank() {
            return Err(GoldyError::Validation(format!(
                "tensor layout: {} strides for rank {}",
                strides.len(),
                shape.rank()
            )));
        }
        if strides.iter().any(|&s| s < 0) {
            return Err(GoldyError::Validation(
                "tensor layout: negative strides are not supported".into(),
            ));
        }
        let mut packed = [0i64; MAX_TENSOR_RANK];
        packed[..strides.len()].copy_from_slice(strides);
        Ok(Self {
            dtype,
            shape,
            storage_offset,
            strides: packed,
        })
    }

    pub fn dtype(self) -> TensorDType {
        self.dtype
    }

    pub fn shape(self) -> TensorShape {
        self.shape
    }

    pub fn storage_offset(self) -> u64 {
        self.storage_offset
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides[..self.shape.rank()]
    }

    pub fn numel(self) -> Result<u64, GoldyError> {
        self.shape.numel()
    }

    pub fn is_contiguous(self) -> bool {
        match packed_stride_array(self.shape) {
            Ok(expected) => self.strides() == &expected[..self.shape.rank()],
            Err(_) => false,
        }
    }

    /// True when every axis with `shape > 1` has a strictly positive stride.
    pub fn is_writeable(self) -> bool {
        self.strides()
            .iter()
            .zip(self.shape.dims())
            .all(|(&stride, &dim)| dim <= 1 || stride > 0)
    }

    /// Inclusive element envelope `[min, max]` relative to storage offset 0.
    pub fn element_envelope(self) -> Result<(u64, u64), GoldyError> {
        let mut min_e = self.storage_offset as i128;
        let mut max_e = self.storage_offset as i128;
        if self.shape.is_empty() {
            let start = u64::try_from(min_e.max(0)).unwrap_or(0);
            return Ok((start, start));
        }
        if self.shape.rank() == 0 {
            let end = min_e
                .checked_add(1)
                .ok_or_else(|| GoldyError::Validation("tensor layout: envelope overflow".into()))?;
            return Ok((self.storage_offset, u64::try_from(end).unwrap_or(u64::MAX)));
        }
        for (&dim, &stride) in self.shape.dims().iter().zip(self.strides()) {
            if dim == 0 {
                let start = u64::try_from(min_e.max(0)).unwrap_or(0);
                return Ok((start, start));
            }
            let span = (i128::from(dim) - 1).saturating_mul(i128::from(stride));
            if span >= 0 {
                max_e = max_e
                    .checked_add(span)
                    .ok_or_else(|| GoldyError::Validation("tensor layout: envelope overflow".into()))?;
            } else {
                min_e = min_e
                    .checked_add(span)
                    .ok_or_else(|| GoldyError::Validation("tensor layout: envelope overflow".into()))?;
            }
        }
        max_e = max_e
            .checked_add(1)
            .ok_or_else(|| GoldyError::Validation("tensor layout: envelope overflow".into()))?;
        if min_e < 0 {
            return Err(GoldyError::Validation(
                "tensor layout: envelope starts before the parent buffer".into(),
            ));
        }
        Ok((
            u64::try_from(min_e).unwrap_or(u64::MAX),
            u64::try_from(max_e).unwrap_or(u64::MAX),
        ))
    }

    /// Byte envelope `[offset, offset+len)` covered by this layout.
    pub fn byte_envelope(self) -> Result<(u64, u64), GoldyError> {
        let elem = self.dtype.size_bytes() as u64;
        let (start, end) = self.element_envelope()?;
        let start_b = start
            .checked_mul(elem)
            .ok_or_else(|| GoldyError::Validation("tensor layout: byte envelope overflow".into()))?;
        let end_b = end
            .checked_mul(elem)
            .ok_or_else(|| GoldyError::Validation("tensor layout: byte envelope overflow".into()))?;
        Ok((start_b, end_b.saturating_sub(start_b)))
    }

    pub fn require_dtype(self, expected: TensorDType, op: &str) -> Result<(), GoldyError> {
        if self.dtype == expected {
            Ok(())
        } else {
            Err(GoldyError::Validation(format!(
                "tensor {op}: expected {} got {}",
                expected.name(),
                self.dtype.name()
            )))
        }
    }

    pub fn require_writeable(self, op: &str) -> Result<(), GoldyError> {
        if self.is_writeable() {
            Ok(())
        } else {
            Err(GoldyError::Validation(format!(
                "tensor {op}: destination view is not a legal write layout (zero stride on an expanded axis)"
            )))
        }
    }

    pub fn gpu_coords(self) -> Result<GoldyTensorLayout, GoldyError> {
        let numel = self.numel()?;
        let numel = u32::try_from(numel)
            .map_err(|_| GoldyError::Validation("tensor layout: numel does not fit in u32".into()))?;
        let offset = u32::try_from(self.storage_offset)
            .map_err(|_| GoldyError::Validation("tensor layout: storage offset does not fit in u32".into()))?;
        let mut shape = [1u32; MAX_TENSOR_RANK];
        let mut stride = [0u32; MAX_TENSOR_RANK];
        for (i, (&d, &s)) in self.shape.dims().iter().zip(self.strides()).enumerate() {
            shape[i] = d.max(1);
            stride[i] = u32::try_from(s)
                .map_err(|_| GoldyError::Validation("tensor layout: stride does not fit in u32".into()))?;
        }
        // Rank-0 scalar: one element at `offset`.
        if self.shape.rank() == 0 {
            shape = [1; MAX_TENSOR_RANK];
            stride = [0; MAX_TENSOR_RANK];
        }
        Ok(GoldyTensorLayout {
            offset,
            rank: self.shape.rank() as u32,
            numel,
            shape,
            stride,
            pad: 0,
        })
    }
}

/// Packed row-major strides (`stride[k] = product(shape[k+1..])`).
fn packed_stride_array(shape: TensorShape) -> Result<[i64; MAX_TENSOR_RANK], GoldyError> {
    let mut strides = [0i64; MAX_TENSOR_RANK];
    let mut acc = 1i64;
    for i in (0..shape.rank()).rev() {
        strides[i] = acc;
        acc = acc
            .checked_mul(i64::from(shape.dims()[i]))
            .ok_or_else(|| GoldyError::Validation("tensor layout: packed stride overflow".into()))?;
    }
    Ok(strides)
}

impl TensorLayout {
    pub(crate) fn with_shape_strides(
        self,
        shape: TensorShape,
        strides: [i64; MAX_TENSOR_RANK],
        storage_offset: u64,
    ) -> Self {
        Self {
            dtype: self.dtype,
            shape,
            storage_offset,
            strides,
        }
    }
}

/// Packed kernel-ABI layout: parent element offset, rank, numel, four extents, four strides.
///
/// Unused axes are stored as extent `1` and stride `0` so a shader can delinearize a
/// logical 1D index through all four axes. Matches the `GoldyTensorLayout` Slang struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GoldyTensorLayout {
    pub offset: u32,
    pub rank: u32,
    pub numel: u32,
    pub shape: [u32; MAX_TENSOR_RANK],
    pub stride: [u32; MAX_TENSOR_RANK],
    pub pad: u32,
}

impl crate::buffer::StructuredBufferElement for GoldyTensorLayout {}

const _: () = assert!(std::mem::size_of::<GoldyTensorLayout>() == 48);

#[allow(dead_code)]
pub(crate) type GpuCoords = GoldyTensorLayout;
