//! Python wrappers for Instance and Adapter.

use crate::device::PyDevice;
use crate::error::IntoPyResult;
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

    /// Create a device on the first adapter matching the given type.
    ///
    /// Args:
    ///     preferred_type: The preferred device type (e.g., DeviceType.DISCRETE_GPU).
    ///
    /// Returns:
    ///     A new Device instance.
    ///
    /// Raises:
    ///     GoldyError: If no suitable adapter is found or device creation fails.
    fn create_device(&self, preferred_type: PyDeviceType) -> PyResult<PyDevice> {
        let device = self
            .inner
            .create_device(preferred_type.into())
            .into_py_result()?;
        Ok(PyDevice {
            inner: Arc::new(device),
        })
    }

    /// Create a device on a specific adapter by ID.
    ///
    /// Args:
    ///     adapter_id: The adapter ID (from Adapter.id).
    ///
    /// Returns:
    ///     A new Device instance.
    fn create_device_for_adapter(&self, adapter_id: u32) -> PyResult<PyDevice> {
        let device = self
            .inner
            .create_device_for_adapter(adapter_id)
            .into_py_result()?;
        Ok(PyDevice {
            inner: Arc::new(device),
        })
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

    fn __repr__(&self) -> String {
        format!(
            "Adapter(id={}, name='{}', vendor='{}')",
            self.inner.id(),
            self.inner.name(),
            self.inner.vendor()
        )
    }
}

