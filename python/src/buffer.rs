//! Python wrapper for retained [`goldy::Buffer`].

use crate::device::PyDevice;
use crate::error::{GoldyError, IntoPyResult};
use crate::parcel::{parcel_from_buffer_unit, PyParcel};
use crate::types::PyResourceAccess;
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

    /// Bindless resource index for one buffer unit.
    fn unit_resource_index(&self, unit: u32, access: PyResourceAccess) -> PyResult<u32> {
        let idx = unit as usize;
        if idx >= self.inner.unit_count() {
            return Err(GoldyError::new_err(format!(
                "buffer unit index {unit} out of range (unit_count={})",
                self.inner.unit_count()
            )));
        }
        self.inner
            .unit(idx)
            .resource_index(access.into())
            .ok_or_else(|| GoldyError::new_err("buffer unit resource index unavailable"))
    }

    /// Read one buffer unit back to CPU memory.
    fn unit_read_to_cpu<'py>(&self, py: Python<'py>, unit: u32, device: &PyDevice) -> PyResult<Bound<'py, PyAny>> {
        let idx = unit as usize;
        if idx >= self.inner.unit_count() {
            return Err(GoldyError::new_err(format!(
                "buffer unit index {unit} out of range (unit_count={})",
                self.inner.unit_count()
            )));
        }
        let size = self.inner.unit(idx).byte_size() as usize;
        let mut output = vec![0u8; size];
        self.inner
            .unit(idx)
            .read_to_cpu(&device.inner, &mut output)
            .into_py_result()?;
        Ok(pyo3::types::PyBytes::new(py, &output).into_any())
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
