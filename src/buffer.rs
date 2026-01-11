//! GPU buffer management.

use crate::backend::{BufferHandle, GpuBackend};
use crate::device::Device;
use crate::types::BufferUsage;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A GPU buffer.
pub struct Buffer {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: BufferHandle,
    size: u64,
    usage: BufferUsage,
}

impl Buffer {
    /// Create a new buffer.
    pub fn new(device: &Device, size: u64, usage: BufferUsage) -> Result<Self> {
        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_buffer(device.handle, size, usage)?;

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
            size,
            usage,
        })
    }

    /// Create a buffer initialized with data.
    pub fn with_data<T: bytemuck::Pod>(
        device: &Device,
        data: &[T],
        usage: BufferUsage,
    ) -> Result<Self> {
        let bytes = bytemuck::cast_slice(data);
        let buffer = Self::new(device, bytes.len() as u64, usage)?;
        buffer.write(0, bytes)?;
        Ok(buffer)
    }

    /// Create a buffer initialized with raw bytes.
    pub fn with_bytes(device: &Device, data: &[u8], usage: BufferUsage) -> Result<Self> {
        let buffer = Self::new(device, data.len() as u64, usage)?;
        buffer.write(0, data)?;
        Ok(buffer)
    }

    /// Write data to the buffer.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.write_buffer(self.handle, offset, data)
    }

    /// Write typed data to the buffer.
    pub fn write_data<T: bytemuck::Pod>(&self, offset: u64, data: &[T]) -> Result<()> {
        self.write(offset, bytemuck::cast_slice(data))
    }

    /// Get the buffer size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the buffer usage flags.
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_buffer(self.handle);
    }
}
