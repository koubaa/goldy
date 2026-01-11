//! Bind group management for uniform buffers and other shader resources.
//!
//! Bind groups are the mechanism to pass data to shaders beyond vertex data.
//! They contain references to buffers (uniform, storage), textures, and samplers.

use crate::backend::{
    BindGroupEntry, BindGroupHandle, BindGroupLayoutEntry, BindGroupLayoutHandle, BindingResource,
    GpuBackend,
};
use crate::buffer::Buffer;
use crate::device::Device;
use crate::sampler::Sampler;
use crate::texture::Texture;
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

    /// Create a sampled texture binding visible to the fragment stage.
    pub fn texture(binding: u32) -> Self {
        Self {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture,
        }
    }

    /// Create a sampled texture binding visible to all graphics stages.
    pub fn texture_all(binding: u32) -> Self {
        Self {
            binding,
            visibility: ShaderStages::ALL,
            ty: BindingType::Texture,
        }
    }

    /// Create a sampler binding visible to the fragment stage.
    pub fn sampler(binding: u32) -> Self {
        Self {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler,
        }
    }

    /// Create a sampler binding visible to all graphics stages.
    pub fn sampler_all(binding: u32) -> Self {
        Self {
            binding,
            visibility: ShaderStages::ALL,
            ty: BindingType::Sampler,
        }
    }

    /// Create a storage texture binding (read-write in shader).
    pub fn storage_texture(binding: u32) -> Self {
        Self {
            binding,
            visibility: ShaderStages::ALL,
            ty: BindingType::StorageTexture,
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

    /// Get the backend handle for this layout.
    ///
    /// This is used when creating pipelines that use this layout.
    pub fn handle(&self) -> BindGroupLayoutHandle {
        self.handle
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

/// Description of a texture binding in a bind group.
#[derive(Clone)]
pub struct TextureBinding<'a> {
    /// Binding index (must match the layout).
    pub binding: u32,
    /// The texture to bind.
    pub texture: &'a Texture,
}

impl<'a> std::fmt::Debug for TextureBinding<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureBinding")
            .field("binding", &self.binding)
            .finish()
    }
}

impl<'a> TextureBinding<'a> {
    /// Create a texture binding.
    pub fn new(binding: u32, texture: &'a Texture) -> Self {
        Self { binding, texture }
    }
}

/// Description of a sampler binding in a bind group.
#[derive(Clone)]
pub struct SamplerBinding<'a> {
    /// Binding index (must match the layout).
    pub binding: u32,
    /// The sampler to bind.
    pub sampler: &'a Sampler,
}

impl<'a> std::fmt::Debug for SamplerBinding<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerBinding")
            .field("binding", &self.binding)
            .finish()
    }
}

impl<'a> SamplerBinding<'a> {
    /// Create a sampler binding.
    pub fn new(binding: u32, sampler: &'a Sampler) -> Self {
        Self { binding, sampler }
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
    ///
    /// Use this when your bind group only contains buffer bindings (uniform, storage).
    /// For bind groups with textures and samplers, use `with_resources()`.
    pub fn new(
        device: &Device,
        layout: &BindGroupLayout,
        bindings: &[BufferBinding],
    ) -> Result<Self> {
        Self::with_resources(device, layout, bindings, &[], &[])
    }

    /// Create a new bind group with buffers, textures, and samplers.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use goldy::{Device, BindGroup, BindGroupLayout, BindGroupLayoutBinding, BufferBinding, TextureBinding, SamplerBinding};
    ///
    /// fn create_textured_bind_group(
    ///     device: &Device,
    ///     layout: &BindGroupLayout,
    ///     uniform_buffer: &goldy::Buffer,
    ///     texture: &goldy::Texture,
    ///     sampler: &goldy::Sampler,
    /// ) -> anyhow::Result<BindGroup> {
    ///     BindGroup::with_resources(
    ///         device,
    ///         layout,
    ///         &[BufferBinding::new(0, uniform_buffer)],
    ///         &[TextureBinding::new(1, texture)],
    ///         &[SamplerBinding::new(2, sampler)],
    ///     )
    /// }
    /// ```
    pub fn with_resources(
        device: &Device,
        layout: &BindGroupLayout,
        buffer_bindings: &[BufferBinding],
        texture_bindings: &[TextureBinding],
        sampler_bindings: &[SamplerBinding],
    ) -> Result<Self> {
        let mut entries: Vec<BindGroupEntry> = Vec::new();

        // Add buffer bindings
        for b in buffer_bindings {
            entries.push(BindGroupEntry {
                binding: b.binding,
                resource: BindingResource::Buffer {
                    buffer: b.buffer.handle,
                    offset: b.offset,
                    size: b.size.unwrap_or(b.buffer.size()),
                },
            });
        }

        // Add texture bindings
        for t in texture_bindings {
            entries.push(BindGroupEntry {
                binding: t.binding,
                resource: BindingResource::Texture(t.texture.handle),
            });
        }

        // Add sampler bindings
        for s in sampler_bindings {
            entries.push(BindGroupEntry {
                binding: s.binding,
                resource: BindingResource::Sampler(s.sampler.handle),
            });
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;

    fn create_test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    // BindGroupLayoutBinding tests

    #[test]
    fn test_uniform_binding() {
        let binding = BindGroupLayoutBinding::uniform(0);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.visibility, ShaderStages::ALL);
        assert!(matches!(binding.ty, BindingType::UniformBuffer));
    }

    #[test]
    fn test_uniform_vertex_binding() {
        let binding = BindGroupLayoutBinding::uniform_vertex(1);
        assert_eq!(binding.binding, 1);
        assert_eq!(binding.visibility, ShaderStages::VERTEX);
        assert!(matches!(binding.ty, BindingType::UniformBuffer));
    }

    #[test]
    fn test_uniform_fragment_binding() {
        let binding = BindGroupLayoutBinding::uniform_fragment(2);
        assert_eq!(binding.binding, 2);
        assert_eq!(binding.visibility, ShaderStages::FRAGMENT);
        assert!(matches!(binding.ty, BindingType::UniformBuffer));
    }

    #[test]
    fn test_storage_binding_read_write() {
        let binding = BindGroupLayoutBinding::storage(0, false);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.visibility, ShaderStages::ALL);
        match binding.ty {
            BindingType::StorageBuffer { read_only } => assert!(!read_only),
            _ => panic!("Expected StorageBuffer"),
        }
    }

    #[test]
    fn test_storage_binding_read_only() {
        let binding = BindGroupLayoutBinding::storage(3, true);
        assert_eq!(binding.binding, 3);
        match binding.ty {
            BindingType::StorageBuffer { read_only } => assert!(read_only),
            _ => panic!("Expected StorageBuffer"),
        }
    }

    #[test]
    fn test_texture_binding() {
        let binding = BindGroupLayoutBinding::texture(0);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.visibility, ShaderStages::FRAGMENT);
        assert!(matches!(binding.ty, BindingType::Texture));
    }

    #[test]
    fn test_texture_all_binding() {
        let binding = BindGroupLayoutBinding::texture_all(1);
        assert_eq!(binding.binding, 1);
        assert_eq!(binding.visibility, ShaderStages::ALL);
        assert!(matches!(binding.ty, BindingType::Texture));
    }

    #[test]
    fn test_sampler_binding() {
        let binding = BindGroupLayoutBinding::sampler(0);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.visibility, ShaderStages::FRAGMENT);
        assert!(matches!(binding.ty, BindingType::Sampler));
    }

    #[test]
    fn test_sampler_all_binding() {
        let binding = BindGroupLayoutBinding::sampler_all(1);
        assert_eq!(binding.binding, 1);
        assert_eq!(binding.visibility, ShaderStages::ALL);
        assert!(matches!(binding.ty, BindingType::Sampler));
    }

    #[test]
    fn test_storage_texture_binding() {
        let binding = BindGroupLayoutBinding::storage_texture(0);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.visibility, ShaderStages::ALL);
        assert!(matches!(binding.ty, BindingType::StorageTexture));
    }

    // BindGroupLayout tests

    #[test]
    fn test_bind_group_layout_creation() {
        let device = create_test_device();

        let layout = BindGroupLayout::new(&device, &[BindGroupLayoutBinding::uniform(0)]).unwrap();

        // Layout handle should be non-zero
        assert!(layout.handle() > 0);
    }

    #[test]
    fn test_bind_group_layout_multiple_bindings() {
        let device = create_test_device();

        let layout = BindGroupLayout::new(
            &device,
            &[
                BindGroupLayoutBinding::uniform(0),
                BindGroupLayoutBinding::storage(1, false),
                BindGroupLayoutBinding::texture(2),
                BindGroupLayoutBinding::sampler(3),
            ],
        )
        .unwrap();

        assert!(layout.handle() > 0);
    }

    #[test]
    fn test_bind_group_layout_empty() {
        let device = create_test_device();

        // Empty layout should work
        let layout = BindGroupLayout::new(&device, &[]).unwrap();
        assert!(layout.handle() > 0);
    }

    // BufferBinding tests

    #[test]
    fn test_buffer_binding_new() {
        let device = create_test_device();
        let buffer = Buffer::new(&device, 256, crate::types::BufferUsage::UNIFORM).unwrap();

        let binding = BufferBinding::new(0, &buffer);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.offset, 0);
        assert!(binding.size.is_none());
    }

    #[test]
    fn test_buffer_binding_with_range() {
        let device = create_test_device();
        let buffer = Buffer::new(&device, 1024, crate::types::BufferUsage::UNIFORM).unwrap();

        let binding = BufferBinding::with_range(0, &buffer, 256, 512);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.offset, 256);
        assert_eq!(binding.size, Some(512));
    }

    // BindGroup tests

    #[test]
    fn test_bind_group_creation() {
        let device = create_test_device();

        let layout = BindGroupLayout::new(&device, &[BindGroupLayoutBinding::uniform(0)]).unwrap();

        let buffer = Buffer::new(&device, 256, crate::types::BufferUsage::UNIFORM).unwrap();

        let bind_group =
            BindGroup::new(&device, &layout, &[BufferBinding::new(0, &buffer)]).unwrap();

        // Just verify creation succeeded
        assert!(bind_group.handle > 0);
    }

    #[test]
    fn test_bind_group_multiple_buffers() {
        let device = create_test_device();

        let layout = BindGroupLayout::new(
            &device,
            &[
                BindGroupLayoutBinding::uniform(0),
                BindGroupLayoutBinding::uniform(1),
                BindGroupLayoutBinding::storage(2, false),
            ],
        )
        .unwrap();

        let uniform1 = Buffer::new(&device, 256, crate::types::BufferUsage::UNIFORM).unwrap();
        let uniform2 = Buffer::new(&device, 128, crate::types::BufferUsage::UNIFORM).unwrap();
        let storage = Buffer::new(&device, 1024, crate::types::BufferUsage::STORAGE).unwrap();

        let bind_group = BindGroup::new(
            &device,
            &layout,
            &[
                BufferBinding::new(0, &uniform1),
                BufferBinding::new(1, &uniform2),
                BufferBinding::new(2, &storage),
            ],
        )
        .unwrap();

        assert!(bind_group.handle > 0);
    }

    #[test]
    fn test_bind_group_with_resources() {
        let device = create_test_device();

        // Layout with buffer, texture, and sampler
        let layout = BindGroupLayout::new(
            &device,
            &[
                BindGroupLayoutBinding::uniform(0),
                BindGroupLayoutBinding::texture(1),
                BindGroupLayoutBinding::sampler(2),
            ],
        )
        .unwrap();

        let buffer = Buffer::new(&device, 256, crate::types::BufferUsage::UNIFORM).unwrap();
        let texture = crate::texture::Texture::new(
            &device,
            64,
            64,
            crate::types::TextureFormat::Rgba8Unorm,
            crate::types::TextureUsage::SAMPLED,
        )
        .unwrap();
        let sampler =
            crate::sampler::Sampler::new(&device, &crate::types::SamplerDesc::default()).unwrap();

        let bind_group = BindGroup::with_resources(
            &device,
            &layout,
            &[BufferBinding::new(0, &buffer)],
            &[TextureBinding::new(1, &texture)],
            &[SamplerBinding::new(2, &sampler)],
        )
        .unwrap();

        assert!(bind_group.handle > 0);
    }

    #[test]
    fn test_bind_group_empty_bindings() {
        let device = create_test_device();

        let layout = BindGroupLayout::new(&device, &[]).unwrap();

        // Empty bind group should work
        let bind_group = BindGroup::new(&device, &layout, &[]).unwrap();
        assert!(bind_group.handle > 0);
    }

    #[test]
    fn test_texture_binding_struct() {
        let device = create_test_device();
        let texture = crate::texture::Texture::new(
            &device,
            64,
            64,
            crate::types::TextureFormat::Rgba8Unorm,
            crate::types::TextureUsage::SAMPLED,
        )
        .unwrap();

        let binding = TextureBinding::new(0, &texture);
        assert_eq!(binding.binding, 0);
    }

    #[test]
    fn test_sampler_binding_struct() {
        let device = create_test_device();
        let sampler =
            crate::sampler::Sampler::new(&device, &crate::types::SamplerDesc::default()).unwrap();

        let binding = SamplerBinding::new(0, &sampler);
        assert_eq!(binding.binding, 0);
    }

    // Debug trait tests

    #[test]
    fn test_bind_group_layout_debug() {
        let device = create_test_device();
        let layout = BindGroupLayout::new(&device, &[BindGroupLayoutBinding::uniform(0)]).unwrap();

        let debug_str = format!("{:?}", layout);
        assert!(debug_str.contains("BindGroupLayout"));
    }

    #[test]
    fn test_buffer_binding_debug() {
        let device = create_test_device();
        let buffer = Buffer::new(&device, 256, crate::types::BufferUsage::UNIFORM).unwrap();
        let binding = BufferBinding::new(0, &buffer);

        let debug_str = format!("{:?}", binding);
        assert!(debug_str.contains("BufferBinding"));
        assert!(debug_str.contains("binding"));
    }

    #[test]
    fn test_texture_binding_debug() {
        let device = create_test_device();
        let texture = crate::texture::Texture::new(
            &device,
            64,
            64,
            crate::types::TextureFormat::Rgba8Unorm,
            crate::types::TextureUsage::SAMPLED,
        )
        .unwrap();
        let binding = TextureBinding::new(0, &texture);

        let debug_str = format!("{:?}", binding);
        assert!(debug_str.contains("TextureBinding"));
    }

    #[test]
    fn test_sampler_binding_debug() {
        let device = create_test_device();
        let sampler =
            crate::sampler::Sampler::new(&device, &crate::types::SamplerDesc::default()).unwrap();
        let binding = SamplerBinding::new(0, &sampler);

        let debug_str = format!("{:?}", binding);
        assert!(debug_str.contains("SamplerBinding"));
    }

    #[test]
    fn test_bind_group_layout_binding_clone() {
        let binding = BindGroupLayoutBinding::uniform(5);
        let cloned = binding.clone();

        assert_eq!(cloned.binding, 5);
        assert_eq!(cloned.visibility, ShaderStages::ALL);
    }
}
