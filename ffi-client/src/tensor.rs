//! RAII tensor algebra over the Goldy C ABI (`tensor` feature).

use crate::error::{check, non_null, Result};
use crate::runtime::Runtime;
use crate::scheme::Scheme;
use crate::sys::{self, GoldyTensor, GoldyTensorDType, GoldyTensorKernels, GoldyTensorShape};
use std::ffi::CString;

/// Dense tensor element type. Operation support is per-op, not universal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDType {
    F32,
    U32,
    I32,
}

impl From<TensorDType> for GoldyTensorDType {
    fn from(v: TensorDType) -> Self {
        match v {
            TensorDType::F32 => GoldyTensorDType::GOLDY_TENSOR_D_TYPE_F32,
            TensorDType::U32 => GoldyTensorDType::GOLDY_TENSOR_D_TYPE_U32,
            TensorDType::I32 => GoldyTensorDType::GOLDY_TENSOR_D_TYPE_I32,
        }
    }
}

impl From<GoldyTensorDType> for TensorDType {
    fn from(v: GoldyTensorDType) -> Self {
        match v {
            GoldyTensorDType::GOLDY_TENSOR_D_TYPE_F32 => TensorDType::F32,
            GoldyTensorDType::GOLDY_TENSOR_D_TYPE_U32 => TensorDType::U32,
            GoldyTensorDType::GOLDY_TENSOR_D_TYPE_I32 => TensorDType::I32,
        }
    }
}

/// Concrete shape (`rank` leading entries of `dims` are live).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    pub rank: u32,
    pub dims: [u32; 4],
}

impl TensorShape {
    pub fn from_dims(dims: &[u32]) -> Self {
        let mut out = [0u32; 4];
        let n = dims.len().min(4);
        out[..n].copy_from_slice(&dims[..n]);
        Self {
            rank: n as u32,
            dims: out,
        }
    }

    pub fn dims(&self) -> &[u32] {
        &self.dims[..self.rank as usize]
    }
}

impl From<GoldyTensorShape> for TensorShape {
    fn from(s: GoldyTensorShape) -> Self {
        Self {
            rank: s.rank,
            dims: s.dims,
        }
    }
}

/// Owned dense tensor (C ABI handle).
pub struct Tensor {
    ptr: *mut GoldyTensor,
}

impl Tensor {
    pub(crate) fn from_ptr(ptr: *mut GoldyTensor) -> Result<Self> {
        Ok(Self { ptr: non_null(ptr)? })
    }

    pub(crate) fn as_ptr(&self) -> *const GoldyTensor {
        self.ptr
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut GoldyTensor {
        self.ptr
    }

    pub fn dtype(&self) -> TensorDType {
        unsafe { sys::goldy_tensor_dtype(self.ptr) }.into()
    }

    pub fn shape(&self) -> Result<TensorShape> {
        let mut out = GoldyTensorShape { rank: 0, dims: [0; 4] };
        check(unsafe { sys::goldy_tensor_shape(self.ptr, &mut out) })?;
        Ok(out.into())
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_tensor_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Prepared tensor kernels. Layout parcels intern onto recorded schemes.
pub struct TensorKernels {
    ptr: *mut GoldyTensorKernels,
}

impl TensorKernels {
    pub fn new(runtime: &Runtime) -> Result<Self> {
        Ok(Self {
            ptr: non_null(unsafe { sys::goldy_tensor_kernels_create(runtime.ptr) })?,
        })
    }

    pub fn add(&mut self, scheme: &mut Scheme, label: &str, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let label = CString::new(label).map_err(|e| crate::error::GoldyError::from_message(e.to_string()))?;
        Tensor::from_ptr(unsafe {
            sys::goldy_tensor_add(self.ptr, scheme.as_ptr(), label.as_ptr(), a.as_ptr(), b.as_ptr())
        })
    }

    pub fn matmul(&mut self, scheme: &mut Scheme, label: &str, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let label = CString::new(label).map_err(|e| crate::error::GoldyError::from_message(e.to_string()))?;
        Tensor::from_ptr(unsafe {
            sys::goldy_tensor_matmul(self.ptr, scheme.as_ptr(), label.as_ptr(), a.as_ptr(), b.as_ptr())
        })
    }

    pub fn fill_f32(&mut self, scheme: &mut Scheme, label: &str, tensor: &mut Tensor, value: f32) -> Result<()> {
        let label = CString::new(label).map_err(|e| crate::error::GoldyError::from_message(e.to_string()))?;
        check(unsafe {
            sys::goldy_tensor_fill_f32(self.ptr, scheme.as_ptr(), label.as_ptr(), tensor.as_mut_ptr(), value)
        })
    }
}

impl Drop for TensorKernels {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_tensor_kernels_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
