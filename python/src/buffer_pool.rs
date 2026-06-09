//! Python wrappers for BufferPool and BufferView.

use crate::device::PyDevice;
use crate::error::{GoldyError, IntoPyResult};
use crate::types::PyResourceAccess;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Bump allocator over a single GPU buffer.
#[pyclass(name = "BufferPool", module = "goldy", unsendable)]
pub struct PyBufferPool {
    inner: goldy::BufferPool,
}

#[pymethods]
impl PyBufferPool {
    #[new]
    fn new(device: &PyDevice, capacity: u64) -> PyResult<Self> {
        let pool = goldy::BufferPool::new(&device.inner, capacity).into_py_result()?;
        Ok(PyBufferPool { inner: pool })
    }

    /// Allocate `count` u32 elements.
    fn alloc_u32(&mut self, count: u64) -> PyResult<PyBufferView> {
        let view = self.inner.alloc::<u32>(count).into_py_result()?;
        Ok(PyBufferView { inner: view })
    }

    /// Write bytes into the backing buffer at a byte offset.
    fn write_backing(&self, byte_offset: u64, data: &[u8]) -> PyResult<()> {
        self.inner.backing_buffer().write(byte_offset, data).into_py_result()
    }

    fn __repr__(&self) -> String {
        "BufferPool()".to_string()
    }
}

/// Sub-range view into a buffer pool allocation.
#[pyclass(name = "BufferView", module = "goldy", unsendable)]
pub struct PyBufferView {
    pub(crate) inner: goldy::BufferView,
}

#[pymethods]
impl PyBufferView {
    #[getter]
    fn size(&self) -> u64 {
        self.inner.size()
    }

    #[getter]
    fn offset(&self) -> u64 {
        self.inner.offset()
    }

    fn resource_index(&self, access: PyResourceAccess) -> PyResult<u32> {
        self.inner
            .resource_index(access.into())
            .ok_or_else(|| GoldyError::new_err("bindless resource index unavailable"))
    }

    fn write_u32<'py>(&self, _py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<()> {
        let arr = data.extract::<Bound<'py, PyArray1<u32>>>()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        self.inner.write_data(slice).into_py_result()
    }

    fn __repr__(&self) -> String {
        format!("BufferView(size={}, offset={})", self.inner.size(), self.inner.offset())
    }
}
