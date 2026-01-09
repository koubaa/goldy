//! GPU backend abstraction.
//!
//! This module defines the `GpuBackend` trait that each graphics API
//! (Vulkan, Metal, DX12) must implement.

#[cfg(feature = "vulkan")]
pub mod vulkan;

use crate::types::*;
use anyhow::Result;

/// Information about a GPU adapter (physical device).
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    /// Adapter index.
    pub id: u32,
    /// Device name.
    pub name: String,
    /// Vendor name.
    pub vendor: String,
    /// Backend type.
    pub backend: BackendType,
    /// Device type (discrete, integrated, etc.).
    pub device_type: DeviceType,
}

/// Opaque handle types.
pub type DeviceHandle = u64;
pub type BufferHandle = u64;
pub type ShaderHandle = u64;
pub type PipelineHandle = u64;

/// Render command for command buffer recording.
#[derive(Debug, Clone)]
pub enum RenderCommand {
    /// Clear the render target.
    Clear(Color),
    /// Set the active pipeline.
    SetPipeline(PipelineHandle),
    /// Set a vertex buffer.
    SetVertexBuffer { slot: u32, buffer: BufferHandle, offset: u64 },
    /// Draw primitives.
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
}

/// GPU backend trait - implemented by Vulkan, Metal, DX12.
pub trait GpuBackend: Send + Sync {
    /// Get the backend type.
    fn backend_type(&self) -> BackendType;

    /// Enumerate available adapters.
    fn enumerate_adapters(&self) -> Vec<AdapterInfo>;

    // Device management
    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle>;
    fn destroy_device(&mut self, device: DeviceHandle);
    fn is_device_valid(&self, device: DeviceHandle) -> bool;

    // Buffer management
    fn create_buffer(&mut self, device: DeviceHandle, size: u64, usage: BufferUsage) -> Result<BufferHandle>;
    fn destroy_buffer(&mut self, buffer: BufferHandle);
    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;
    fn buffer_size(&self, buffer: BufferHandle) -> u64;

    // Shader management
    fn create_shader(&mut self, device: DeviceHandle, wgsl_source: &str) -> Result<ShaderHandle>;
    fn destroy_shader(&mut self, shader: ShaderHandle);

    // Pipeline management
    fn create_pipeline(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
    ) -> Result<PipelineHandle>;
    fn destroy_pipeline(&mut self, pipeline: PipelineHandle);

    // Rendering
    fn begin_frame(&mut self, device: DeviceHandle, width: u32, height: u32, format: TextureFormat) -> Result<()>;
    fn execute_commands(&mut self, device: DeviceHandle, commands: &[RenderCommand]) -> Result<()>;
    fn end_frame(&mut self, device: DeviceHandle, output: &mut [u8]) -> Result<()>;
}

/// Create the default backend for the current platform.
pub fn create_default_backend() -> Result<Box<dyn GpuBackend>> {
    #[cfg(feature = "vulkan")]
    {
        tracing::info!("Creating Vulkan backend");
        Ok(Box::new(vulkan::VulkanBackend::new()?))
    }

    #[cfg(not(feature = "vulkan"))]
    {
        anyhow::bail!("No GPU backend available - enable 'vulkan' feature")
    }
}

