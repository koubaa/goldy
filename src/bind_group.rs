//! Bind group management for uniform buffers and other shader resources.
//!
//! Bind groups are the mechanism to pass data to shaders beyond vertex data.
//! They contain references to buffers (uniform, storage) and other resources.

use crate::backend::{
    BindGroupHandle, BindGroupLayoutHandle, BindGroupLayoutEntry, BindGroupEntry,
    BindingResource, GpuBackend,
};
use crate::buffer::Buffer;
use crate::device::Device;
use anyhow::Result;
use std::sync::{Arc, Mutex};

// Re-export backend types for convenience
pub use crate::backend::{BindingType, ShaderStages};

/// Description of a binding in a bind group layout.
#[derive(Debug, Clone)]
pub struct BindGroupLayoutBinding {
    /// Binding index (matches shader's `[[vk::binding(N, 0)]]`).
    pub binding: u32,
    /// Which shader stages can access this binding.
    pub visibility: ShaderStages,
    /// Type of resource at this binding.
    pub ty: BindingType,
}

impl BindGroupLayoutBinding {
    /// Create a uniform buffer binding visible to all graphics stages.
    pub fn uniform(binding: u32) -> Self {
        Self {
            binding,
            visibility: ShaderStages::ALL,
            ty: BindingType::UniformBuffer,
        }
    }

    /// Create a uniform buffer binding visible only to the vertex stage.
    pub fn uniform_vertex(binding: u32) -> Self {
        Self {
            binding,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::UniformBuffer,
        }
    }

    /// Create a uniform buffer binding visible only to the fragment stage.
    pub fn uniform_fragment(binding: u32) -> Self {
        Self {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::UniformBuffer,
        }
    }

    /// Create a storage buffer binding.
    pub fn storage(binding: u32, read_only: bool) -> Self {
        Self {
            binding,
            visibility: ShaderStages::ALL,
            ty: BindingType::StorageBuffer { read_only },
        }
    }
}

/// A bind group layout defines the structure of a bind group.
///
/// This describes what types of resources will be bound and at which binding indices.
/// The layout must be created before creating bind groups or pipelines that use them.
pub struct BindGroupLayout {
    #[allow(dead_code)]
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: BindGroupLayoutHandle,
}

impl std::fmt::Debug for BindGroupLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindGroupLayout")
            .field("handle", &self.handle)
            .finish()
    }
}

impl BindGroupLayout {
    /// Create a new bind group layout.
    pub fn new(device: &Device, bindings: &[BindGroupLayoutBinding]) -> Result<Self> {
        let entries: Vec<BindGroupLayoutEntry> = bindings
            .iter()
            .map(|b| BindGroupLayoutEntry {
                binding: b.binding,
                visibility: b.visibility,
                ty: b.ty.clone(),
            })
            .collect();

        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_bind_group_layout(device.handle, &entries)?;

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
        })
    }
}

impl Drop for BindGroupLayout {
    fn drop(&mut self) {
        // Bind group layouts are typically long-lived and destroyed with the device
        // No explicit destruction needed as the device cleanup handles this
    }
}

/// Description of a buffer binding in a bind group.
#[derive(Clone)]
pub struct BufferBinding<'a> {
    /// Binding index (must match the layout).
    pub binding: u32,
    /// The buffer to bind.
    pub buffer: &'a Buffer,
    /// Offset into the buffer (usually 0).
    pub offset: u64,
    /// Size of the binding (None = entire buffer from offset).
    pub size: Option<u64>,
}

impl<'a> std::fmt::Debug for BufferBinding<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferBinding")
            .field("binding", &self.binding)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .finish()
    }
}

impl<'a> BufferBinding<'a> {
    /// Create a buffer binding for the entire buffer.
    pub fn new(binding: u32, buffer: &'a Buffer) -> Self {
        Self {
            binding,
            buffer,
            offset: 0,
            size: None,
        }
    }

    /// Create a buffer binding with offset and size.
    pub fn with_range(binding: u32, buffer: &'a Buffer, offset: u64, size: u64) -> Self {
        Self {
            binding,
            buffer,
            offset,
            size: Some(size),
        }
    }
}

/// A bind group contains actual resource bindings matching a layout.
///
/// Bind groups are created from a layout and contain references to buffers
/// and other resources that shaders can access.
pub struct BindGroup {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: BindGroupHandle,
}

impl BindGroup {
    /// Create a new bind group from a layout and buffer bindings.
    pub fn new(
        device: &Device,
        layout: &BindGroupLayout,
        bindings: &[BufferBinding],
    ) -> Result<Self> {
        let entries: Vec<BindGroupEntry> = bindings
            .iter()
            .map(|b| BindGroupEntry {
                binding: b.binding,
                resource: BindingResource::Buffer {
                    buffer: b.buffer.handle,
                    offset: b.offset,
                    size: b.size.unwrap_or(b.buffer.size()),
                },
            })
            .collect();

        let mut backend = device.backend.lock().unwrap();
        let handle = backend.create_bind_group(device.handle, layout.handle, &entries)?;

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
        })
    }
}

impl Drop for BindGroup {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_bind_group(self.handle);
    }
}

