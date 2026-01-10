//! GPU device management.

use crate::backend::{self, AdapterInfo, DeviceHandle, GpuBackend};
use crate::types::*;
use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};

/// GPU instance - entry point for RAG.
///
/// Create an instance to enumerate adapters and create devices.
pub struct Instance {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
}

impl Instance {
    /// Create a new RAG instance.
    pub fn new() -> Result<Self> {
        let backend = backend::create_default_backend()?;
        Ok(Self {
            backend: Arc::new(Mutex::new(backend)),
        })
    }

    /// Enumerate available GPU adapters.
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        let backend = self.backend.lock().unwrap();
        backend
            .enumerate_adapters()
            .into_iter()
            .map(|info| Adapter { info })
            .collect()
    }

    /// Create a device on the first adapter matching the given type.
    pub fn create_device(&self, preferred_type: DeviceType) -> Result<Device> {
        let adapters = self.enumerate_adapters();
        
        // Find preferred adapter
        let adapter = adapters
            .iter()
            .find(|a| a.info.device_type == preferred_type)
            .or_else(|| adapters.first())
            .context("No GPU adapters available")?;

        self.create_device_for_adapter(adapter.info.id)
    }

    /// Create a device on a specific adapter by ID.
    pub fn create_device_for_adapter(&self, adapter_id: u32) -> Result<Device> {
        let mut backend = self.backend.lock().unwrap();
        let handle = backend.create_device(adapter_id)?;
        
        Ok(Device {
            backend: Arc::clone(&self.backend),
            handle,
            adapter_id,
        })
    }

    /// Get the backend type (Vulkan, Metal, DX12).
    pub fn backend_type(&self) -> BackendType {
        self.backend.lock().unwrap().backend_type()
    }
}

/// Information about a GPU adapter.
#[derive(Debug, Clone)]
pub struct Adapter {
    pub info: AdapterInfo,
}

impl Adapter {
    /// Get the adapter ID.
    pub fn id(&self) -> u32 {
        self.info.id
    }

    /// Get the adapter name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Get the device type.
    pub fn device_type(&self) -> DeviceType {
        self.info.device_type
    }

    /// Get the vendor name.
    pub fn vendor(&self) -> &str {
        &self.info.vendor
    }
}

/// A GPU device - used to create resources and render.
pub struct Device {
    pub(crate) backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: DeviceHandle,
    adapter_id: u32,
}

impl Device {
    /// Get the adapter ID this device was created on.
    pub fn adapter_id(&self) -> u32 {
        self.adapter_id
    }

    /// Check if the device is still valid.
    pub fn is_valid(&self) -> bool {
        self.backend.lock().unwrap().is_device_valid(self.handle)
    }

    /// Get the device handle (internal use).
    pub(crate) fn handle(&self) -> DeviceHandle {
        self.handle
    }

    /// Get the backend (internal use).
    pub(crate) fn backend(&self) -> &Arc<Mutex<Box<dyn GpuBackend>>> {
        &self.backend
    }


    /// Create a device from a backend for testing purposes.
    #[cfg(test)]
    pub(crate) fn from_backend(backend: Box<dyn GpuBackend>) -> anyhow::Result<Self> {
        let backend = Arc::new(Mutex::new(backend));
        let handle = {
            let mut b = backend.lock().unwrap();
            b.create_device(0)?
        };
        Ok(Self {
            backend,
            handle,
            adapter_id: 0,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_device(self.handle);
    }
}

