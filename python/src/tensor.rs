//! Python wrappers for Goldy's dense tensor algebra (`tensor` feature).
//!
//! NumPy conversion copies through acquire / [`MemoryExchange`] rather than promising
//! arbitrary zero-copy host views of GPU storage.

use crate::error::IntoPyResult;
use crate::parcel::{parcel_from_cloned, PyParcel};
use crate::runtime::PyRuntime;
use crate::scheme::PyScheme;
use goldy::{Tensor, TensorContext, TensorDType, TensorScalar, TensorShape};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::cell::RefCell;

/// Dense tensor element type. Operation support is per-op, not universal.
#[pyclass(name = "TensorDType", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyTensorDType {
    F32 = 0,
    U32 = 1,
    I32 = 2,
}

impl From<PyTensorDType> for TensorDType {
    fn from(v: PyTensorDType) -> Self {
        match v {
            PyTensorDType::F32 => TensorDType::F32,
            PyTensorDType::U32 => TensorDType::U32,
            PyTensorDType::I32 => TensorDType::I32,
        }
    }
}

impl From<TensorDType> for PyTensorDType {
    fn from(v: TensorDType) -> Self {
        match v {
            TensorDType::F32 => PyTensorDType::F32,
            TensorDType::U32 => PyTensorDType::U32,
            TensorDType::I32 => PyTensorDType::I32,
        }
    }
}

fn parse_shape(shape: Vec<u32>) -> PyResult<TensorShape> {
    TensorShape::from_dims(&shape).into_py_result()
}

/// Owned dense tensor: a parcel plus shape/dtype/layout metadata.
#[pyclass(name = "Tensor", module = "goldy", unsendable)]
pub struct PyTensor {
    pub(crate) inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn dtype(&self) -> PyTensorDType {
        self.inner.dtype().into()
    }

    #[getter]
    fn shape(&self) -> Vec<u32> {
        self.inner.shape().dims().to_vec()
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.shape().rank()
    }

    /// Borrow the backing parcel for [`MemoryExchange.bind_withdraw`] / deposit.
    fn parcel(&self) -> PyParcel {
        parcel_from_cloned(self.inner.buffer().whole().clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(dtype={:?}, shape={:?})",
            self.inner.dtype(),
            self.inner.shape().dims()
        )
    }
}

pub(crate) fn tensor_from_owned(inner: Tensor) -> PyTensor {
    PyTensor { inner }
}

/// Prepared tensor kernels plus layout keepalive. Keep alive while recorded schemes exist.
#[pyclass(name = "TensorContext", module = "goldy", unsendable)]
pub struct PyTensorContext {
    inner: RefCell<TensorContext>,
}

#[pymethods]
impl PyTensorContext {
    #[new]
    fn new(runtime: &PyRuntime) -> PyResult<Self> {
        Ok(Self {
            inner: RefCell::new(TensorContext::new(&runtime.inner).into_py_result()?),
        })
    }

    fn add(&self, scheme: &PyScheme, label: &str, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
        scheme.ensure_no_active_recorder()?;
        let mut ctx = self.inner.borrow_mut();
        let mut scheme = scheme.inner.borrow_mut();
        let mut rec = ctx.recorder(&mut scheme);
        rec.add(label, a.inner.view(), b.inner.view())
            .into_py_result()
            .map(tensor_from_owned)
    }

    fn matmul(&self, scheme: &PyScheme, label: &str, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
        scheme.ensure_no_active_recorder()?;
        let mut ctx = self.inner.borrow_mut();
        let mut scheme = scheme.inner.borrow_mut();
        let mut rec = ctx.recorder(&mut scheme);
        rec.matmul(label, a.inner.view(), b.inner.view())
            .into_py_result()
            .map(tensor_from_owned)
    }

    fn fill_f32(&self, scheme: &PyScheme, label: &str, tensor: &PyTensor, value: f32) -> PyResult<()> {
        scheme.ensure_no_active_recorder()?;
        let mut ctx = self.inner.borrow_mut();
        let mut scheme = scheme.inner.borrow_mut();
        let mut rec = ctx.recorder(&mut scheme);
        rec.fill(label, tensor.inner.view(), TensorScalar::F32(value))
            .into_py_result()
    }

    fn __repr__(&self) -> String {
        "TensorContext()".to_string()
    }
}

/// Build a packed tensor from a 1-D NumPy array or bytes plus an explicit shape.
pub(crate) fn acquire_tensor(
    runtime: &goldy::Runtime,
    data: &Bound<'_, PyAny>,
    shape: Vec<u32>,
    dtype: PyTensorDType,
) -> PyResult<PyTensor> {
    let shape = parse_shape(shape)?;
    let dt: TensorDType = dtype.into();
    let (bytes, stride) = crate::bytes_util::extract_bytes_with_stride(data)?;
    if stride != 4 {
        return Err(PyValueError::new_err(
            "tensor acquire expects 4-byte elements (float32 / uint32 / int32)",
        ));
    }
    match dt {
        TensorDType::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(PyValueError::new_err("F32 tensor data is not a multiple of 4 bytes"));
            }
            let vals: &[f32] = bytemuck::cast_slice(&bytes);
            Tensor::from_f32(runtime, shape, vals)
                .into_py_result()
                .map(tensor_from_owned)
        }
        TensorDType::U32 => {
            if bytes.len() % 4 != 0 {
                return Err(PyValueError::new_err("U32 tensor data is not a multiple of 4 bytes"));
            }
            let vals: &[u32] = bytemuck::cast_slice(&bytes);
            Tensor::from_u32(runtime, shape, vals)
                .into_py_result()
                .map(tensor_from_owned)
        }
        TensorDType::I32 => {
            if bytes.len() % 4 != 0 {
                return Err(PyValueError::new_err("I32 tensor data is not a multiple of 4 bytes"));
            }
            let vals: &[i32] = bytemuck::cast_slice(&bytes);
            Tensor::from_i32(runtime, shape, vals)
                .into_py_result()
                .map(tensor_from_owned)
        }
    }
}

pub(crate) fn zeros_tensor(runtime: &goldy::Runtime, shape: Vec<u32>, dtype: PyTensorDType) -> PyResult<PyTensor> {
    let shape = parse_shape(shape)?;
    Tensor::zeros(runtime, shape, dtype.into())
        .into_py_result()
        .map(tensor_from_owned)
}
