//! Python wrapper for retained [`goldy::Parcel`].

use pyo3::prelude::*;
use std::sync::Arc;

/// How a Python [`PyParcel`] resolves to a live [`goldy::Parcel`].
pub(crate) enum PyParcelInner {
    /// One unit of an acquired [`goldy::Buffer`].
    BufferUnit { buffer: Arc<goldy::Buffer>, unit: usize },
}

impl PyParcelInner {
    pub(crate) fn as_parcel(&self) -> &goldy::Parcel {
        match self {
            Self::BufferUnit { buffer, unit } => buffer.unit(*unit),
        }
    }
}

/// Opaque retained GPU parcel (one buffer unit).
#[pyclass(name = "Parcel", module = "goldy", unsendable)]
pub struct PyParcel {
    pub(crate) inner: PyParcelInner,
}

pub(crate) fn parcel_from_buffer_unit(buffer: Arc<goldy::Buffer>, unit: usize) -> PyParcel {
    PyParcel {
        inner: PyParcelInner::BufferUnit { buffer, unit },
    }
}

#[pymethods]
impl PyParcel {
    #[getter]
    fn byte_size(&self) -> u64 {
        self.inner.as_parcel().byte_size()
    }

    fn __repr__(&self) -> String {
        format!("Parcel(byte_size={})", self.inner.as_parcel().byte_size())
    }
}
