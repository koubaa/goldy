//! GPU backend abstraction.
//!
//! This module defines the `GpuBackend` trait that each graphics API
//! (Vulkan, Metal, DX12, WebGPU) must implement.

#[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
pub mod vulkan;

// Mock backend for testing (always available)
pub mod mock;

// WebGPU backend is currently native-only (uses native Slang compiler)
// For browser WASM builds, use rag-web which uses wgpu directly with slang-wasm
// #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
// pub mod webgpu;

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
pub type BindGroupHandle = u64;
pub type BindGroupLayoutHandle = u64;
pub type RenderTargetHandle = u64;

/// Render command for command buffer recording.
#[derive(Debug, Clone)]
pub enum RenderCommand {
    /// Clear the render target.
    Clear(Color),
    /// Set the active pipeline.
    SetPipeline(PipelineHandle),
    /// Set a vertex buffer.
    SetVertexBuffer { slot: u32, buffer: BufferHandle, offset: u64 },
    /// Set a bind group.
    SetBindGroup { index: u32, bind_group: BindGroupHandle },
    /// Draw primitives.
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
}

/// GPU backend trait - implemented by Vulkan, Metal, DX12, WebGPU.
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
    fn create_shader(&mut self, device: DeviceHandle, slang_source: &str) -> Result<ShaderHandle>;
    fn destroy_shader(&mut self, shader: ShaderHandle);

    // Bind group management
    fn create_bind_group_layout(&mut self, device: DeviceHandle, entries: &[BindGroupLayoutEntry]) -> Result<BindGroupLayoutHandle>;
    fn create_bind_group(&mut self, device: DeviceHandle, layout: BindGroupLayoutHandle, entries: &[BindGroupEntry]) -> Result<BindGroupHandle>;
    fn destroy_bind_group(&mut self, bind_group: BindGroupHandle);

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
    fn create_pipeline_with_layout(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        bind_group_layouts: &[BindGroupLayoutHandle],
    ) -> Result<PipelineHandle>;
    fn destroy_pipeline(&mut self, pipeline: PipelineHandle);

    // Rendering (legacy - use RenderTarget API instead)
    fn begin_frame(&mut self, device: DeviceHandle, width: u32, height: u32, format: TextureFormat) -> Result<()>;
    fn execute_commands(&mut self, device: DeviceHandle, commands: &[RenderCommand]) -> Result<()>;
    fn end_frame(&mut self, device: DeviceHandle, output: &mut [u8]) -> Result<()>;

    // RenderTarget API - GPU buffer stays on GPU, readback is optional
    fn create_render_target(&mut self, device: DeviceHandle, width: u32, height: u32, format: TextureFormat) -> Result<RenderTargetHandle>;
    fn destroy_render_target(&mut self, target: RenderTargetHandle);
    fn render_to_target(&mut self, device: DeviceHandle, target: RenderTargetHandle, commands: &[RenderCommand]) -> Result<()>;
    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()>;
}

/// Trait for consuming render output.
/// 
/// Implementations can consume render targets in different ways:
/// - CPU readback for encoding/streaming
/// - Present to window surface
/// - Pass to video encoder
pub trait RenderConsumer {
    /// Consume a render target.
    fn consume(&mut self, backend: &mut dyn GpuBackend, target: RenderTargetHandle) -> Result<()>;
}

/// Bind group layout entry.
#[derive(Debug, Clone)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: ShaderStages,
    pub ty: BindingType,
}

/// Shader stage visibility flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderStages(pub u32);

impl ShaderStages {
    pub const VERTEX: ShaderStages = ShaderStages(1);
    pub const FRAGMENT: ShaderStages = ShaderStages(2);
    pub const ALL: ShaderStages = ShaderStages(3);
}

/// Binding type for bind groups.
#[derive(Debug, Clone)]
pub enum BindingType {
    UniformBuffer,
    StorageBuffer { read_only: bool },
}

/// Bind group entry.
#[derive(Debug, Clone)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource: BindingResource,
}

/// Resource for a bind group entry.
#[derive(Debug, Clone)]
pub enum BindingResource {
    Buffer { buffer: BufferHandle, offset: u64, size: u64 },
}

/// Create the default backend for the current platform.
pub fn create_default_backend() -> Result<Box<dyn GpuBackend>> {
    #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
    {
        tracing::info!("Creating Vulkan backend");
        Ok(Box::new(vulkan::VulkanBackend::new()?))
    }

    // WebGPU backend is disabled - for browser WASM builds, use rag-web
    // which uses wgpu directly with slang-wasm for shader compilation

    #[cfg(not(all(feature = "vulkan", not(target_arch = "wasm32"))))]
    {
        anyhow::bail!("No GPU backend available - enable 'vulkan' feature (native only)")
    }
}

