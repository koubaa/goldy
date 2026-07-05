//! Python wrapper for retained [`goldy::Texture`].

use pyo3::prelude::*;
use std::sync::Arc;

/// Acquired retained GPU texture.
#[pyclass(name = "Texture", module = "goldy", unsendable)]
pub struct PyTexture {
    pub(crate) inner: Arc<goldy::Texture>,
}

pub(crate) fn texture_from_owned(texture: goldy::Texture) -> PyTexture {
    PyTexture {
        inner: Arc::new(texture),
    }
}

#[pymethods]
impl PyTexture {
    #[getter]
    fn byte_size(&self) -> u64 {
        self.inner.byte_size()
    }

    fn __repr__(&self) -> String {
        format!("Texture(byte_size={})", self.inner.byte_size())
    }
}
