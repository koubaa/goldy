//! Python wrapper for retained [`goldy::Parcel`].

use crate::device::PyDevice;
use crate::error::{GoldyError, IntoPyResult};
use crate::types::PyResourceAccess;
use pyo3::prelude::*;
use std::sync::Arc;

/// Opaque retained GPU parcel (buffer or texture).
#[pyclass(name = "Parcel", module = "goldy", unsendable)]
pub struct PyParcel {
    pub(crate) inner: Arc<goldy::Parcel>,
}

#[pymethods]
impl PyParcel {
    #[getter]
    fn byte_size(&self) -> u64 {
        self.inner.byte_size()
    }

    fn resource_index(&self, access: PyResourceAccess) -> PyResult<u32> {
        self.inner
            .resource_index(access.into())
            .ok_or_else(|| GoldyError::new_err("bindless resource index unavailable"))
    }

    /// Bindless resource index for one mosaic sub-view.
    fn mosaic_view_resource_index(&self, slot: u32, access: PyResourceAccess) -> PyResult<u32> {
        self.inner
            .mosaic_view_resource_index(goldy::MosaicSlot(slot), access.into())
            .ok_or_else(|| GoldyError::new_err("mosaic view resource index unavailable"))
    }

    /// Read one mosaic sub-view back to CPU memory.
    fn mosaic_view_read_to_cpu<'py>(
        &self,
        py: Python<'py>,
        slot: u32,
        device: &PyDevice,
    ) -> PyResult<Bound<'py, PyAny>> {
        let size = self.inner.view(goldy::MosaicSlot(slot)).size() as usize;
        let mut output = vec![0u8; size];
        self.inner
            .mosaic_view_read_to_cpu(&device.inner, goldy::MosaicSlot(slot), &mut output)
            .into_py_result()?;
        Ok(pyo3::types::PyBytes::new(py, &output).into_any())
    }

    fn read_to_cpu<'py>(&self, py: Python<'py>, device: &PyDevice) -> PyResult<Bound<'py, PyAny>> {
        let mut output = vec![0u8; self.inner.byte_size() as usize];
        self.inner.read_to_cpu(&device.inner, &mut output).into_py_result()?;
        Ok(pyo3::types::PyBytes::new(py, &output).into_any())
    }

    fn __repr__(&self) -> String {
        format!("Parcel(byte_size={})", self.inner.byte_size())
    }
}
