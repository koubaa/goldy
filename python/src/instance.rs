//! Python wrappers for Instance and Adapter.

use crate::error::IntoPyResult;
use crate::runtime::PyRuntime;
use crate::types::{PyBackendType, PyDeviceType};
use pyo3::prelude::*;
use std::sync::Arc;

/// GPU instance - entry point for Goldy.
///
/// Create an instance to enumerate adapters and create devices.
#[pyclass(name = "Instance", module = "goldy")]
pub struct PyInstance {
    inner: goldy::Instance,
}

#[pymethods]
impl PyInstance {
    /// Create a new Goldy instance.
    #[new]
    fn new() -> PyResult<Self> {
        let inner = goldy::Instance::new().into_py_result()?;
        Ok(PyInstance { inner })
    }

    /// Enumerate available GPU adapters.
    fn enumerate_adapters(&self) -> Vec<PyAdapter> {
        self.inner
            .enumerate_adapters()
            .into_iter()
            .map(|a| PyAdapter { inner: a })
            .collect()
    }

    /// Request the best available GPU adapter (highest performance by default).
    ///
    /// Returns:
    ///     An Adapter instance.
    ///
    /// Raises:
    ///     GoldyError: If no suitable adapter is found.
    fn request_adapter(&self) -> PyResult<PyAdapter> {
        let adapter = self
            .inner
            .request_adapter(&goldy::RequestAdapterOptions::default())
            .into_py_result()?;
        Ok(PyAdapter { inner: adapter })
    }

    /// Get the backend type (Vulkan, Metal, DX12).
    #[getter]
    fn backend_type(&self) -> PyBackendType {
        self.inner.backend_type().into()
    }

    fn __repr__(&self) -> String {
        format!("Instance(backend={:?})", self.inner.backend_type())
    }
}

/// Information about a GPU adapter.
#[pyclass(name = "Adapter", module = "goldy")]
pub struct PyAdapter {
    inner: goldy::Adapter,
}

#[pymethods]
impl PyAdapter {
    /// Get the adapter ID.
    #[getter]
    fn id(&self) -> u32 {
        self.inner.id()
    }

    /// Get the adapter name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get the device type.
    #[getter]
    fn device_type(&self) -> PyDeviceType {
        self.inner.device_type().into()
    }

    /// Get the vendor name.
    #[getter]
    fn vendor(&self) -> String {
        self.inner.vendor().to_string()
    }

    /// Create a device on this adapter.
    ///
    /// Returns:
    ///     A new Runtime instance.
    ///
    /// Raises:
    ///     GoldyError: If device creation fails.
    fn request_runtime(&self) -> PyResult<PyRuntime> {
        let device = self
            .inner
            .request_runtime(&goldy::RuntimeDescriptor::default())
            .into_py_result()?;
        Ok(PyRuntime {
            inner: Arc::new(device),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Adapter(id={}, name='{}', vendor='{}')",
            self.inner.id(),
            self.inner.name(),
            self.inner.vendor()
        )
    }
}
