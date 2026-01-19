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
        let (bytes, element_stride) = extract_bytes_with_stride(data)?;
        // Use the correct element stride for StructuredBuffer views on DX12
        let buffer =
            goldy::Buffer::with_bytes_stride(&device.inner, &bytes, usage.into(), element_stride)
                .into_py_result()?;

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
        let (bytes, _stride) = extract_bytes_with_stride(data)?;
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

/// Extract bytes and element stride from a Python object (numpy array or bytes).
/// Returns (bytes, element_stride) where element_stride is the size of each element in bytes.
fn extract_bytes_with_stride(data: &Bound<'_, PyAny>) -> PyResult<(Vec<u8>, u32)> {
    // Try numpy array first (most common case)
    if let Ok(arr) = data.downcast::<PyArray1<f32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 4)); // f32 = 4 bytes
    }

    if let Ok(arr) = data.downcast::<PyArray1<f64>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 8)); // f64 = 8 bytes
    }

    if let Ok(arr) = data.downcast::<PyArray1<i32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 4)); // i32 = 4 bytes
    }

    if let Ok(arr) = data.downcast::<PyArray1<u32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 4)); // u32 = 4 bytes
    }

    if let Ok(arr) = data.downcast::<PyArray1<i16>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 2)); // i16 = 2 bytes
    }

    if let Ok(arr) = data.downcast::<PyArray1<u16>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 2)); // u16 = 2 bytes
    }

    if let Ok(arr) = data.downcast::<PyArray1<i8>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((bytemuck::cast_slice(slice).to_vec(), 1)); // i8 = 1 byte
    }

    if let Ok(arr) = data.downcast::<PyArray1<u8>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        return Ok((slice.to_vec(), 1)); // u8 = 1 byte
    }

    // Try bytes
    if let Ok(bytes) = data.extract::<Vec<u8>>() {
        return Ok((bytes, 1)); // Raw bytes = 1 byte stride
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected numpy array or bytes",
    ))
}
