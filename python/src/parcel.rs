//! Python wrapper for retained [`goldy::Parcel`].

use crate::error::GoldyError;
use crate::types::PyResourceAccess;
use pyo3::prelude::*;
use std::sync::Arc;

/// How a Python [`PyParcel`] resolves to a live [`goldy::Parcel`].
pub(crate) enum PyParcelInner {
    /// Texture parcel acquired directly from the pool.
    Owned(Arc<goldy::Parcel>),
    /// One unit of an acquired [`goldy::Buffer`].
    BufferUnit { buffer: Arc<goldy::Buffer>, unit: usize },
}

impl PyParcelInner {
    pub(crate) fn as_parcel(&self) -> &goldy::Parcel {
        match self {
            Self::Owned(p) => p,
            Self::BufferUnit { buffer, unit } => buffer.unit(*unit),
        }
    }
}

/// Opaque retained GPU parcel (texture or one buffer unit).
#[pyclass(name = "Parcel", module = "goldy", unsendable)]
pub struct PyParcel {
    pub(crate) inner: PyParcelInner,
}

pub(crate) fn parcel_from_texture(parcel: goldy::Parcel) -> PyParcel {
    PyParcel {
        inner: PyParcelInner::Owned(Arc::new(parcel)),
    }
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

    fn resource_index(&self, access: PyResourceAccess) -> PyResult<u32> {
        self.inner
            .as_parcel()
            .resource_index(access.into())
            .ok_or_else(|| GoldyError::new_err("bindless resource index unavailable"))
    }

    fn __repr__(&self) -> String {
        format!("Parcel(byte_size={})", self.inner.as_parcel().byte_size())
    }
}
