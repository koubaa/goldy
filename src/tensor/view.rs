//! Owned tensors and metadata-only views over parent parcels.

use super::dtype::TensorDType;
use super::layout::TensorLayout;
use super::shape::TensorShape;
use super::MAX_TENSOR_RANK;
use crate::error::GoldyError;
use crate::parcel::Buffer;
use crate::runtime::Runtime;
use crate::types::BufferKind;

/// Owned dense tensor: a buffer parcel plus shape/dtype/layout metadata.
///
/// The buffer is the ownership identity. Views derived from this tensor never
/// mint a second parcel; they are lenses over the same backing.
pub struct Tensor {
    buffer: Buffer,
    layout: TensorLayout,
}

impl Tensor {
    pub fn from_buffer(buffer: Buffer, layout: TensorLayout) -> Result<Self, GoldyError> {
        validate_fits(&buffer, layout)?;
        Ok(Self { buffer, layout })
    }

    pub fn zeros(runtime: &Runtime, shape: TensorShape, dtype: TensorDType) -> Result<Self, GoldyError> {
        let layout = TensorLayout::packed(dtype, shape, 0)?;
        let n = layout.numel()?;
        let bytes = n
            .checked_mul(dtype.size_bytes() as u64)
            .ok_or_else(|| GoldyError::Validation("tensor zeros: byte size overflow".into()))?;
        let buffer = runtime
            .acquire_buffer(
                bytes.max(dtype.size_bytes() as u64),
                BufferKind::Scattered,
                Some(dtype.size_bytes() as u32),
                crate::types::BufferFlags::empty(),
                None,
            )
            .map_err(GoldyError::from)?;
        Self::from_buffer(buffer, layout)
    }

    pub fn from_f32(runtime: &Runtime, shape: TensorShape, data: &[f32]) -> Result<Self, GoldyError> {
        from_slice(runtime, shape, TensorDType::F32, data)
    }

    pub fn from_u32(runtime: &Runtime, shape: TensorShape, data: &[u32]) -> Result<Self, GoldyError> {
        from_slice(runtime, shape, TensorDType::U32, data)
    }

    pub fn from_i32(runtime: &Runtime, shape: TensorShape, data: &[i32]) -> Result<Self, GoldyError> {
        from_slice(runtime, shape, TensorDType::I32, data)
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn layout(&self) -> TensorLayout {
        self.layout
    }

    pub fn dtype(&self) -> TensorDType {
        self.layout.dtype()
    }

    pub fn shape(&self) -> TensorShape {
        self.layout.shape()
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            buffer: &self.buffer,
            layout: self.layout,
        }
    }

    pub fn as_view(&self) -> TensorView<'_> {
        self.view()
    }
}

fn from_slice<T: crate::buffer::StructuredBufferElement>(
    runtime: &Runtime,
    shape: TensorShape,
    dtype: TensorDType,
    data: &[T],
) -> Result<Tensor, GoldyError> {
    let layout = TensorLayout::packed(dtype, shape, 0)?;
    let n = layout.numel()?;
    if data.len() as u64 != n {
        return Err(GoldyError::Validation(format!(
            "tensor from_slice: data len {} != numel {n}",
            data.len()
        )));
    }
    let buffer = runtime
        .acquire_buffer_with_data(data, BufferKind::Scattered)
        .map_err(GoldyError::from)?;
    Tensor::from_buffer(buffer, layout)
}

fn validate_fits(buffer: &Buffer, layout: TensorLayout) -> Result<(), GoldyError> {
    if buffer.is_partitioned() {
        return Err(GoldyError::Validation(
            "tensor: partitioned buffers cannot back a tensor; bind individual parcels".into(),
        ));
    }
    let (byte_off, byte_len) = layout.byte_envelope()?;
    let end = byte_off
        .checked_add(byte_len)
        .ok_or_else(|| GoldyError::Validation("tensor: view envelope overflow".into()))?;
    if end > buffer.byte_size() {
        return Err(GoldyError::Validation(format!(
            "tensor: view envelope [{byte_off}, {end}) exceeds buffer size {}",
            buffer.byte_size()
        )));
    }
    if byte_off % layout.dtype().size_bytes() as u64 != 0 {
        return Err(GoldyError::Validation(
            "tensor: view byte offset is not aligned to the element size".into(),
        ));
    }
    Ok(())
}

/// Metadata lens over a parent [`Buffer`]. Copying a view does not copy data or identity.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    buffer: &'a Buffer,
    layout: TensorLayout,
}

impl std::fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorView")
            .field("dtype", &self.layout.dtype())
            .field("shape", &self.layout.shape())
            .field("layout", &self.layout)
            .field("buffer_bytes", &self.buffer.byte_size())
            .finish()
    }
}

impl<'a> TensorView<'a> {
    pub fn new(buffer: &'a Buffer, layout: TensorLayout) -> Result<Self, GoldyError> {
        validate_fits(buffer, layout)?;
        Ok(Self { buffer, layout })
    }

    /// Packed window of `shape` starting at element `storage_offset` in `buffer`.
    pub fn packed_at(
        buffer: &'a Buffer,
        dtype: TensorDType,
        storage_offset: u64,
        shape: TensorShape,
    ) -> Result<Self, GoldyError> {
        Self::new(buffer, TensorLayout::packed(dtype, shape, storage_offset)?)
    }

    pub fn buffer(self) -> &'a Buffer {
        self.buffer
    }

    pub fn layout(self) -> TensorLayout {
        self.layout
    }

    pub fn dtype(self) -> TensorDType {
        self.layout.dtype()
    }

    pub fn shape(self) -> TensorShape {
        self.layout.shape()
    }

    pub fn storage_offset(self) -> u64 {
        self.layout.storage_offset()
    }

    pub fn numel(self) -> Result<u64, GoldyError> {
        self.layout.numel()
    }

    pub fn numel_u32(self) -> u32 {
        self.layout
            .numel()
            .ok()
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0)
    }

    pub fn is_contiguous(self) -> bool {
        self.layout.is_contiguous()
    }

    pub fn is_writeable(self) -> bool {
        self.layout.is_writeable()
    }

    pub fn byte_envelope(self) -> Result<(u64, u64), GoldyError> {
        self.layout.byte_envelope()
    }

    /// Slice `axis` to `[start, start+length)`.
    pub fn narrow(self, axis: usize, start: u32, length: u32) -> Result<Self, GoldyError> {
        let dim = self.shape().dim(axis)?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| GoldyError::Validation("tensor narrow: start+length overflow".into()))?;
        if end > dim {
            return Err(GoldyError::Validation(format!(
                "tensor narrow: [{start}, {end}) exceeds dim {dim} on axis {axis}"
            )));
        }
        let stride = self.layout.strides()[axis];
        let extra = i64::from(start)
            .checked_mul(stride)
            .ok_or_else(|| GoldyError::Validation("tensor narrow: offset overflow".into()))?;
        let new_off = (self.layout.storage_offset() as i64)
            .checked_add(extra)
            .ok_or_else(|| GoldyError::Validation("tensor narrow: offset overflow".into()))?;
        if new_off < 0 {
            return Err(GoldyError::Validation("tensor narrow: offset would be negative".into()));
        }
        let shape = self.shape();
        let mut dims: Vec<u32> = shape.dims().to_vec();
        dims[axis] = length;
        let shape = TensorShape::from_dims(&dims)?;
        let mut strides = [0i64; MAX_TENSOR_RANK];
        strides[..shape.rank()].copy_from_slice(self.layout.strides());
        let layout = self.layout.with_shape_strides(shape, strides, new_off as u64);
        Self::new(self.buffer, layout)
    }

    /// Packed-only reshape. Element count must match; the result is contiguous row-major.
    pub fn reshape(self, dims: &[u32]) -> Result<Self, GoldyError> {
        if !self.is_contiguous() {
            return Err(GoldyError::Validation(
                "tensor reshape: view is not contiguous; materialize with `contiguous` first".into(),
            ));
        }
        let shape = TensorShape::from_dims(dims)?;
        if shape.numel()? != self.numel()? {
            return Err(GoldyError::Validation(format!(
                "tensor reshape: numel {} != {}",
                shape.numel()?,
                self.numel()?
            )));
        }
        let layout = TensorLayout::packed(self.dtype(), shape, self.storage_offset())?;
        Self::new(self.buffer, layout)
    }

    /// Permute axes. `axes` is a permutation of `0..rank`.
    pub fn permute(self, axes: &[usize]) -> Result<Self, GoldyError> {
        let shape = self.shape();
        let rank = shape.rank();
        if axes.len() != rank {
            return Err(GoldyError::Validation(format!(
                "tensor permute: {} axes for rank {rank}",
                axes.len()
            )));
        }
        let mut seen = [false; MAX_TENSOR_RANK];
        let mut dims = Vec::with_capacity(rank);
        let mut strides = [0i64; MAX_TENSOR_RANK];
        for (out_i, &axis) in axes.iter().enumerate() {
            if axis >= rank || seen[axis] {
                return Err(GoldyError::Validation(format!(
                    "tensor permute: invalid permutation {axes:?}"
                )));
            }
            seen[axis] = true;
            dims.push(shape.dims()[axis]);
            strides[out_i] = self.layout.strides()[axis];
        }
        let shape = TensorShape::from_dims(&dims)?;
        let layout = self.layout.with_shape_strides(shape, strides, self.storage_offset());
        Self::new(self.buffer, layout)
    }

    /// Swap the last two axes (no-op for rank < 2).
    pub fn transpose(self) -> Result<Self, GoldyError> {
        let rank = self.shape().rank();
        if rank < 2 {
            return Ok(self);
        }
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(rank - 2, rank - 1);
        self.permute(&axes)
    }

    /// Broadcast to `target`. Existing extents must be 1 or already equal.
    pub fn broadcast_to(self, target: TensorShape) -> Result<Self, GoldyError> {
        let src = self.shape();
        if target.rank() < src.rank() {
            return Err(GoldyError::Validation("tensor broadcast_to: cannot reduce rank".into()));
        }
        let pad = target.rank() - src.rank();
        let mut dims = Vec::with_capacity(target.rank());
        let mut strides = [0i64; MAX_TENSOR_RANK];
        for (i, &t) in target.dims().iter().enumerate() {
            if i < pad {
                dims.push(t);
                strides[i] = 0;
                continue;
            }
            let s = src.dims()[i - pad];
            let st = self.layout.strides()[i - pad];
            if s == t {
                dims.push(s);
                strides[i] = st;
            } else if s == 1 {
                dims.push(t);
                strides[i] = 0;
            } else {
                return Err(GoldyError::Validation(format!(
                    "tensor broadcast_to: dim {s} cannot broadcast to {t}"
                )));
            }
        }
        let shape = TensorShape::from_dims(&dims)?;
        let layout = self.layout.with_shape_strides(shape, strides, self.storage_offset());
        Self::new(self.buffer, layout)
    }

    /// Flatten to a rank-1 packed view of the same elements (contiguous only).
    pub fn flatten(self) -> Result<Self, GoldyError> {
        let n = u32::try_from(self.numel()?)
            .map_err(|_| GoldyError::Validation("tensor flatten: numel does not fit in u32".into()))?;
        self.reshape(&[n])
    }
}

pub(crate) fn broadcast_shapes(a: TensorShape, b: TensorShape) -> Result<TensorShape, GoldyError> {
    let ra = a.rank();
    let rb = b.rank();
    let rank = ra.max(rb);
    let mut dims = vec![0u32; rank];
    for (i, dim) in dims.iter_mut().enumerate() {
        let da = if i < rank - ra { 1 } else { a.dims()[i - (rank - ra)] };
        let db = if i < rank - rb { 1 } else { b.dims()[i - (rank - rb)] };
        if da == db {
            *dim = da;
        } else if da == 1 {
            *dim = db;
        } else if db == 1 {
            *dim = da;
        } else {
            return Err(GoldyError::Validation(format!(
                "tensor broadcast: incompatible dims {da} and {db}"
            )));
        }
    }
    TensorShape::from_dims(&dims)
}
