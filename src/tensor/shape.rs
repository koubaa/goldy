//! Concrete tensor shapes (rank 0..=[`super::MAX_TENSOR_RANK`]).

use super::MAX_TENSOR_RANK;
use crate::error::GoldyError;

/// Concrete shape with packed dimensions. Symbolic extents are out of scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorShape {
    rank: u32,
    dims: [u32; MAX_TENSOR_RANK],
}

impl TensorShape {
    /// Rank-0 scalar.
    pub const fn scalar() -> Self {
        Self {
            rank: 0,
            dims: [0; MAX_TENSOR_RANK],
        }
    }

    /// Rank-1 vector.
    pub const fn vector(n: u32) -> Self {
        Self {
            rank: 1,
            dims: [n, 0, 0, 0],
        }
    }

    /// Rank-2 matrix `[rows, cols]`.
    pub const fn matrix(rows: u32, cols: u32) -> Self {
        Self {
            rank: 2,
            dims: [rows, cols, 0, 0],
        }
    }

    /// Construct from a slice of dimensions.
    pub fn from_dims(dims: &[u32]) -> Result<Self, GoldyError> {
        if dims.len() > MAX_TENSOR_RANK {
            return Err(GoldyError::Validation(format!(
                "tensor shape: rank {} exceeds MAX_TENSOR_RANK {MAX_TENSOR_RANK}",
                dims.len()
            )));
        }
        let mut out = [0u32; MAX_TENSOR_RANK];
        for (i, &d) in dims.iter().enumerate() {
            out[i] = d;
        }
        Ok(Self {
            rank: dims.len() as u32,
            dims: out,
        })
    }

    pub const fn rank(self) -> usize {
        self.rank as usize
    }

    pub fn dims(&self) -> &[u32] {
        &self.dims[..self.rank as usize]
    }

    pub fn dim(self, axis: usize) -> Result<u32, GoldyError> {
        if axis >= self.rank() {
            return Err(GoldyError::Validation(format!(
                "tensor shape: axis {axis} out of rank {}",
                self.rank()
            )));
        }
        Ok(self.dims[axis])
    }

    /// Product of extents. Empty tensors (any zero dim) have numel 0. Rank-0 is 1.
    pub fn numel(self) -> Result<u64, GoldyError> {
        if self.rank == 0 {
            return Ok(1);
        }
        let mut n = 1u64;
        for &d in self.dims() {
            n = n
                .checked_mul(u64::from(d))
                .ok_or_else(|| GoldyError::Validation("tensor shape: numel overflowed u64".into()))?;
        }
        Ok(n)
    }

    pub fn is_empty(self) -> bool {
        self.rank > 0 && self.dims().iter().any(|&d| d == 0)
    }

    /// Drop `axis` (must be in range).
    pub fn squeeze_axis(self, axis: usize) -> Result<Self, GoldyError> {
        if axis >= self.rank() {
            return Err(GoldyError::Validation(format!(
                "tensor shape: squeeze axis {axis} out of rank {}",
                self.rank()
            )));
        }
        let mut dims = Vec::with_capacity(self.rank().saturating_sub(1));
        for (i, &d) in self.dims().iter().enumerate() {
            if i != axis {
                dims.push(d);
            }
        }
        Self::from_dims(&dims)
    }

    /// Insert `size` at `axis` (0..=rank).
    pub fn insert_axis(self, axis: usize, size: u32) -> Result<Self, GoldyError> {
        if axis > self.rank() {
            return Err(GoldyError::Validation(format!(
                "tensor shape: insert axis {axis} out of rank {}",
                self.rank()
            )));
        }
        if self.rank() + 1 > MAX_TENSOR_RANK {
            return Err(GoldyError::Validation(
                "tensor shape: inserting an axis would exceed MAX_TENSOR_RANK".into(),
            ));
        }
        let mut dims = Vec::with_capacity(self.rank() + 1);
        dims.extend_from_slice(&self.dims()[..axis]);
        dims.push(size);
        dims.extend_from_slice(&self.dims()[axis..]);
        Self::from_dims(&dims)
    }
}

impl TryFrom<&[u32]> for TensorShape {
    type Error = GoldyError;

    fn try_from(dims: &[u32]) -> Result<Self, Self::Error> {
        Self::from_dims(dims)
    }
}
