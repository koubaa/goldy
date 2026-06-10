//! Python wrapper for [`goldy::RetainedPool`].

use crate::device::PyDevice;
use crate::error::IntoPyResult;
use crate::parcel::PyParcel;
use crate::types::PyBufferKind;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::cell::RefCell;
use std::sync::Arc;

/// Deed-governed pool for retained GPU parcels.
#[pyclass(name = "RetainedPool", module = "goldy", unsendable)]
pub struct PyRetainedPool {
    pub(crate) inner: RefCell<goldy::RetainedPool>,
}

#[pymethods]
impl PyRetainedPool {
    #[new]
    fn new(device: &PyDevice) -> Self {
        Self {
            inner: RefCell::new(goldy::RetainedPool::new(device.inner.clone())),
        }
    }

    /// Acquire a retained buffer parcel from numpy array or bytes.
    fn acquire_buffer(&self, data: &Bound<'_, PyAny>, access: PyBufferKind) -> PyResult<PyParcel> {
        let (bytes, element_stride) = crate::buffer::extract_bytes_with_stride(data)?;
        let parcel = self
            .inner
            .borrow_mut()
            .acquire_buffer(
                bytes.len() as u64,
                access.into(),
                Some(element_stride),
                goldy::BufferFlags::empty(),
                Some(&bytes),
            )
            .into_py_result()?;
        Ok(PyParcel {
            inner: Arc::new(parcel),
        })
    }

    fn __repr__(&self) -> String {
        "RetainedPool()".to_string()
    }
}
