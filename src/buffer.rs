//! GPU buffer management.

use crate::backend::{BufferHandle, GpuBackend};
use crate::device::Device;
use crate::types::DataAccess;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A GPU buffer.
pub struct Buffer {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: BufferHandle,
    size: u64,
    access: DataAccess,
}

impl Buffer {
    /// Create a new buffer with the specified access pattern.
    ///
    /// # Access Patterns
    ///
    /// - `DataAccess::Scattered`: Any thread can access any address (read/write).
    ///   Use for general-purpose data (StructuredBuffer, RWStructuredBuffer).
    ///
    /// - `DataAccess::Broadcast`: All threads read the same address.
    ///   Hardware optimizes for wave-wide broadcast (ConstantBuffer).
    pub fn new(device: &Device, size: u64, access: DataAccess) -> Result<Self> {
        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_buffer(device.handle, size, access, None)?;

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
            size,
            access,
        })
    }

    /// Create a buffer initialized with data.
    ///
    /// See [`Buffer::new`] for access pattern documentation.
    pub fn with_data<T: bytemuck::Pod>(
        device: &Device,
        data: &[T],
        access: DataAccess,
    ) -> Result<Self> {
        let bytes = bytemuck::cast_slice(data);
        let element_stride = std::mem::size_of::<T>() as u32;
        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_buffer(
            device.handle,
            bytes.len() as u64,
            access,
            Some(element_stride),
        )?;
        drop(backend);

        let buffer = Self {
            backend: Arc::clone(&device.backend),
            handle,
            size: bytes.len() as u64,
            access,
        };
        buffer.write(0, bytes)?;
        Ok(buffer)
    }

    /// Create a buffer initialized with raw bytes.
    ///
    /// See [`Buffer::new`] for access pattern documentation.
    pub fn with_bytes(device: &Device, data: &[u8], access: DataAccess) -> Result<Self> {
        // For raw bytes, use stride of 1 (byte-addressable)
        Self::with_bytes_stride(device, data, access, 1)
    }

    /// Create a buffer initialized with raw bytes and a custom element stride.
    ///
    /// The stride is used for creating StructuredBuffer views on DX12. For example,
    /// if the data contains u32 values, use stride=4 so the GPU can correctly
    /// interpret the buffer as `StructuredBuffer<uint>`.
    ///
    /// See [`Buffer::new`] for access pattern documentation.
    pub fn with_bytes_stride(
        device: &Device,
        data: &[u8],
        access: DataAccess,
        element_stride: u32,
    ) -> Result<Self> {
        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_buffer(
            device.handle,
            data.len() as u64,
            access,
            Some(element_stride),
        )?;
        drop(backend);

        let buffer = Self {
            backend: Arc::clone(&device.backend),
            handle,
            size: data.len() as u64,
            access,
        };
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

    /// Get the buffer's access pattern.
    pub fn access(&self) -> DataAccess {
        self.access
    }

    /// Get the buffer's index in the global bindless descriptor set.
    ///
    /// Returns `Some(index)` if this buffer is registered in the global descriptor set.
    /// All buffers with Scattered or Broadcast access are registered.
    pub fn bindless_index(&self) -> Option<u32> {
        let backend = self.backend.lock().unwrap();
        backend.buffer_bindless_index(self.handle)
    }

    /// Read buffer contents back to CPU memory.
    ///
    /// The `output` slice must be at least `size` bytes. Reads from offset 0.
    pub fn read_to_cpu(&self, device: &Device, output: &mut [u8]) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.read_buffer_to_cpu(device.handle, self.handle, output)
    }

    /// Clear the buffer (fill with zeros) from offset for size bytes.
    pub fn clear(&self, device: &Device, offset: u64, size: u64) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.clear_buffer(device.handle, self.handle, offset, size)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_buffer(self.handle);
    }
}
