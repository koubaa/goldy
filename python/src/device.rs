//! Python wrapper for Device.

use crate::error::IntoPyResult;
use pyo3::prelude::*;
use std::sync::Arc;

/// A GPU device - used to create resources and render.
///
/// The Device is the primary interface for GPU operations.
#[pyclass(name = "Device", module = "goldy")]
#[derive(Clone)]
pub struct PyDevice {
    pub inner: Arc<goldy::Device>,
}

#[pymethods]
impl PyDevice {
    /// Get the adapter ID this device was created on.
    #[getter]
    fn adapter_id(&self) -> u32 {
        self.inner.adapter_id()
    }

    /// Check if the device is still valid.
    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    /// Check if a shader library is registered.
    fn has_library(&self, name: &str) -> bool {
        self.inner.has_library(name)
    }

    /// List all registered shader libraries.
    fn list_libraries(&self) -> Vec<String> {
        self.inner.list_libraries()
    }

    /// Register a shader library from source.
    ///
    /// Args:
    ///     name: The library name (used in `import` statements).
    ///     source: The Slang source code for the library.
    ///
    /// Raises:
    ///     GoldyError: If a library with the same name is already registered.
    fn register_library(&self, name: &str, source: &str) -> PyResult<()> {
        let library = goldy::ShaderLibrary::from_source(name, source);
        self.inner.register_library(library).into_py_result()
    }

    /// Unregister a shader library.
    ///
    /// Returns:
    ///     True if the library was found and removed, False otherwise.
    fn unregister_library(&self, name: &str) -> bool {
        self.inner.unregister_library(name)
    }

    fn __repr__(&self) -> String {
        format!(
            "Device(adapter_id={}, valid={})",
            self.inner.adapter_id(),
            self.inner.is_valid()
        )
    }
}

