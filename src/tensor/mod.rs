//! Dense tensor algebra over Goldy parcels.
//!
//! This feature owns concrete shapes, dtypes, strides, views, broadcasting, and
//! general tensor operations. It records into the existing [`crate::Scheme`] —
//! there is no second scheduler. Autograd, modules, optimizers, model formats,
//! RMSNorm, RoPE, attention, and KV-cache policy are **out of scope**.
//!
//! Views are metadata lenses: they never mint a new ownership identity. Binding a
//! view claims a conservative buffer-range envelope (or the parent buffer) so
//! aliases remain visible to Goldy, while the shader bindless slot is the parent
//! parcel.

mod bind;
mod dtype;
mod kernels;
mod layout;
mod matmul;
mod ops;
mod semantic;
mod shape;
mod view;

#[cfg(test)]
mod contract;

pub use dtype::TensorDType;
pub use layout::{GoldyTensorLayout, TensorLayout};
pub use ops::{ScatterMode, TensorKernels, TensorRecorder, TensorScalar};
pub use shape::TensorShape;
pub use view::{Tensor, TensorView};

/// Maximum supported tensor rank (concrete extents only).
pub const MAX_TENSOR_RANK: usize = 4;
