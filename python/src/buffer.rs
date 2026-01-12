//! Python wrapper for Buffer with NumPy support.

use crate::device::PyDevice;
use crate::error::IntoPyResult;
use crate::types::PyBufferUsage;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use std::sync::Arc;

/// A GPU buffer.
///
/// Buffers hold vertex data, index data, uniforms, or storage data on the GPU.
#[pyclass(name = "Buffer", module = "goldy")]
pub struct PyBuffer {
    pub(crate) inner: Arc<goldy::Buffer>,
}

#[pymethods]
impl PyBuffer {
    /// Create a new buffer from data.
    ///
    /// Args:
    ///     device: The GPU device.
    ///     data: Buffer data as a numpy array (any numeric dtype) or bytes.
    ///     usage: Buffer usage flags (e.g., BufferUsage.VERTEX).
    ///
    /// Returns:
    ///     A new Buffer instance.
    ///
    /// Example:
    ///     >>> import numpy as np
    ///     >>> vertices = np.array([0.0, -0.5, 0.5, 0.5, -0.5, 0.5], dtype=np.float32)
    ///     >>> buffer = goldy.Buffer(device, vertices, goldy.BufferUsage.VERTEX)
    #[new]
    fn new(device: &PyDevice, data: &Bound<'_, PyAny>, usage: PyBufferUsage) -> PyResult<Self> {
        let bytes = extract_bytes(data)?;
        let buffer =
            goldy::Buffer::with_bytes(&device.inner, &bytes, usage.into()).into_py_result()?;

        Ok(PyBuffer {
            inner: Arc::new(buffer),
        })
    }

    /// Create an empty buffer of a given size.
    ///
    /// Args:
    ///     device: The GPU device.
    ///     size: Size in bytes.
    ///     usage: Buffer usage flags.
    ///
    /// Returns:
    ///     A new empty Buffer instance.
    #[staticmethod]
    fn empty(device: &PyDevice, size: u64, usage: PyBufferUsage) -> PyResult<Self> {
        let buffer = goldy::Buffer::new(&device.inner, size, usage.into()).into_py_result()?;
        Ok(PyBuffer {
            inner: Arc::new(buffer),
        })
    }

    /// Write data to the buffer at an offset.
    ///
    /// Args:
    ///     offset: Byte offset to write at.
    ///     data: Data to write as numpy array or bytes.
    fn write(&self, offset: u64, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = extract_bytes(data)?;
        self.inner.write(offset, &bytes).into_py_result()
    }

    /// Get the buffer size in bytes.
    #[getter]
    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn __repr__(&self) -> String {
        format!("Buffer(size={})", self.inner.size())
    }
}

/// Extract bytes from a Python object (numpy array or bytes).
fn extract_bytes(data: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    // Try numpy array first (most common case)
    if let Ok(arr) = data.downcast::<PyArray1<f32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok(bytemuck::cast_slice(slice).to_vec());
    }

    if let Ok(arr) = data.downcast::<PyArray1<f64>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok(bytemuck::cast_slice(slice).to_vec());
    }

    if let Ok(arr) = data.downcast::<PyArray1<i32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok(bytemuck::cast_slice(slice).to_vec());
    }

    if let Ok(arr) = data.downcast::<PyArray1<u32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok(bytemuck::cast_slice(slice).to_vec());
    }

    if let Ok(arr) = data.downcast::<PyArray1<i16>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok(bytemuck::cast_slice(slice).to_vec());
    }

    if let Ok(arr) = data.downcast::<PyArray1<u16>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok(bytemuck::cast_slice(slice).to_vec());
    }

    if let Ok(arr) = data.downcast::<PyArray1<i8>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok(bytemuck::cast_slice(slice).to_vec());
    }

    if let Ok(arr) = data.downcast::<PyArray1<u8>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok(slice.to_vec());
    }

    // Try bytes
    if let Ok(bytes) = data.extract::<Vec<u8>>() {
        return Ok(bytes);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected numpy array or bytes",
    ))
}
