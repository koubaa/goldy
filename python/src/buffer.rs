//! Python wrapper for retained [`goldy::Buffer`].

use crate::error::GoldyError;
use crate::parcel::{parcel_from_buffer_unit, PyParcel};
use pyo3::prelude::*;
use std::sync::Arc;

/// An acquired GPU buffer — possibly partitioned into independently bindable parcels.
#[pyclass(name = "Buffer", module = "goldy", unsendable)]
pub struct PyBuffer {
    pub(crate) inner: Arc<goldy::Buffer>,
}

#[pymethods]
impl PyBuffer {
    #[getter]
    fn byte_size(&self) -> u64 {
        self.inner.byte_size()
    }

    #[getter]
    fn unit_count(&self) -> usize {
        self.inner.unit_count()
    }

    #[getter]
    fn is_partitioned(&self) -> bool {
        self.inner.is_partitioned()
    }

    /// Obtain a bindable parcel by ordinal index.
    fn unit(&self, index: usize) -> PyResult<PyParcel> {
        if index >= self.inner.unit_count() {
            return Err(GoldyError::new_err(format!(
                "buffer unit index {index} out of range (unit_count={})",
                self.inner.unit_count()
            )));
        }
        Ok(parcel_from_buffer_unit(Arc::clone(&self.inner), index))
    }

    /// Obtain a bindable parcel by field name (named partitioned fields only).
    fn field(&self, name: &str) -> PyResult<PyParcel> {
        let parcel_ptr = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.field(name) as *const goldy::Parcel
        })) {
            Ok(p) => p,
            Err(_) => return Err(GoldyError::new_err(format!("unknown buffer field {name:?}"))),
        };
        for i in 0..self.inner.unit_count() {
            if std::ptr::eq(self.inner.unit(i), unsafe { &*parcel_ptr }) {
                return Ok(parcel_from_buffer_unit(Arc::clone(&self.inner), i));
            }
        }
        Err(GoldyError::new_err(format!("unknown buffer field {name:?}")))
    }

    fn __getitem__(&self, index: usize) -> PyResult<PyParcel> {
        self.unit(index)
    }

    fn __repr__(&self) -> String {
        format!(
            "Buffer(byte_size={}, unit_count={})",
            self.inner.byte_size(),
            self.inner.unit_count()
        )
    }
}

pub(crate) fn buffer_from_owned(buffer: goldy::Buffer) -> PyBuffer {
    PyBuffer {
        inner: Arc::new(buffer),
    }
}
