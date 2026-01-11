//! GPU backend abstraction.
//!
//! This module defines the `GpuBackend` trait that each graphics API
//! (Vulkan, Metal, DX12, WebGPU) must implement.

#[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
pub mod vulkan;

// DX12 backend for Windows
#[cfg(all(feature = "dx12", target_os = "windows"))]
pub mod dx12;

// Mock backend for testing (always available)
pub mod mock;

// Metal backend for macOS (native Metal, not MoltenVK)
#[cfg(target_os = "macos")]
pub mod metal;

// WebGPU backend is currently native-only (uses native Slang compiler)
// For browser WASM builds, use goldy-web which uses wgpu directly with slang-wasm
// #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
// pub mod webgpu;

use crate::types::{
    BackendType, BufferUsage, Color, DepthFormat, DepthStencilState, DeviceType, IndexFormat,
    PrimitiveTopology, SamplerDesc, TextureFormat, TextureUsage, VertexBufferLayout,
};
use anyhow::Result;

// Re-export raw_window_handle for Surface API users
pub use raw_window_handle;

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
pub type ComputePipelineHandle = u64;
pub type BindGroupHandle = u64;
pub type BindGroupLayoutHandle = u64;
pub type RenderTargetHandle = u64;
pub type SurfaceHandle = u64;
pub type SwapchainImageHandle = u64;
pub type TextureHandle = u64;
pub type SamplerHandle = u64;

/// Render command for command buffer recording.
#[derive(Debug, Clone)]
pub enum RenderCommand {
    /// Clear the color render target.
    Clear(Color),
    /// Clear the depth buffer.
    ClearDepth(f32),
    /// Set the active pipeline.
    SetPipeline(PipelineHandle),
    /// Set a vertex buffer.
    SetVertexBuffer {
        slot: u32,
        buffer: BufferHandle,
        offset: u64,
    },
    /// Set an index buffer.
    SetIndexBuffer {
        buffer: BufferHandle,
        offset: u64,
        format: IndexFormat,
    },
    /// Set a bind group.
    SetBindGroup {
        index: u32,
        bind_group: BindGroupHandle,
    },
    /// Draw primitives (non-indexed).
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    /// Draw indexed primitives.
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        /// Offset added to each index value before fetching the vertex.
        base_vertex: i32,
        first_instance: u32,
    },
}

/// Compute command for compute pass recording.
#[derive(Debug, Clone)]
pub enum ComputeCommand {
    /// Set the active compute pipeline.
    SetPipeline(ComputePipelineHandle),
    /// Set a bind group.
    SetBindGroup {
        index: u32,
        bind_group: BindGroupHandle,
    },
    /// Dispatch compute workgroups.
    Dispatch {
        workgroups_x: u32,
        workgroups_y: u32,
        workgroups_z: u32,
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
    fn create_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        usage: BufferUsage,
    ) -> Result<BufferHandle>;
    fn destroy_buffer(&mut self, buffer: BufferHandle);
    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()>;
    fn buffer_size(&self, buffer: BufferHandle) -> u64;

    // Shader management
    fn create_shader(&mut self, device: DeviceHandle, slang_source: &str) -> Result<ShaderHandle>;
    fn create_shader_with_paths(
        &mut self,
        device: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
    ) -> Result<ShaderHandle>;
    fn destroy_shader(&mut self, shader: ShaderHandle);

    // Bind group management
    fn create_bind_group_layout(
        &mut self,
        device: DeviceHandle,
        entries: &[BindGroupLayoutEntry],
    ) -> Result<BindGroupLayoutHandle>;
    fn create_bind_group(
        &mut self,
        device: DeviceHandle,
        layout: BindGroupLayoutHandle,
        entries: &[BindGroupEntry],
    ) -> Result<BindGroupHandle>;
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

    // Pipeline with depth stencil state
    fn create_pipeline_with_depth(
        &mut self,
        device: DeviceHandle,
        vertex_shader: ShaderHandle,
        fragment_shader: ShaderHandle,
        vertex_layout: &VertexBufferLayout,
        topology: PrimitiveTopology,
        target_format: TextureFormat,
        bind_group_layouts: &[BindGroupLayoutHandle],
        depth_stencil: Option<&DepthStencilState>,
    ) -> Result<PipelineHandle>;

    // RenderTarget API - GPU buffer stays on GPU, readback is optional
    fn create_render_target(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<RenderTargetHandle>;
    /// Create a render target with an optional depth buffer.
    fn create_render_target_with_depth(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        color_format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle>;
    fn destroy_render_target(&mut self, target: RenderTargetHandle);
    fn render_to_target(
        &mut self,
        device: DeviceHandle,
        target: RenderTargetHandle,
        commands: &[RenderCommand],
    ) -> Result<()>;
    fn read_target_to_cpu(&mut self, target: RenderTargetHandle, output: &mut [u8]) -> Result<()>;

    // Texture management
    fn create_texture(
        &mut self,
        device: DeviceHandle,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Result<TextureHandle>;
    fn write_texture(
        &mut self,
        texture: TextureHandle,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<()>;
    fn destroy_texture(&mut self, texture: TextureHandle);

    // Sampler management
    fn create_sampler(&mut self, device: DeviceHandle, desc: &SamplerDesc)
        -> Result<SamplerHandle>;
    fn destroy_sampler(&mut self, sampler: SamplerHandle);

    // Surface API - zero-copy presentation to window
    /// Create a surface for presenting to a window.
    /// The window handle is platform-specific (HWND on Windows, wl_surface on Wayland, NSView on macOS).
    fn create_surface(
        &mut self,
        device: DeviceHandle,
        window: &dyn raw_window_handle::HasWindowHandle,
        display: &dyn raw_window_handle::HasDisplayHandle,
    ) -> Result<SurfaceHandle>;

    /// Destroy a surface.
    fn destroy_surface(&mut self, surface: SurfaceHandle);

    /// Acquire the next swapchain image to render to.
    fn surface_acquire(&mut self, surface: SurfaceHandle) -> Result<SwapchainImageHandle>;

    /// Render commands to a swapchain image.
    fn surface_render(
        &mut self,
        surface: SurfaceHandle,
        image: SwapchainImageHandle,
        commands: &[RenderCommand],
    ) -> Result<()>;

    /// Present a swapchain image to the screen.
    fn surface_present(
        &mut self,
        surface: SurfaceHandle,
        image: SwapchainImageHandle,
    ) -> Result<()>;

    /// Resize the surface (recreates swapchain).
    fn surface_resize(&mut self, surface: SurfaceHandle, width: u32, height: u32) -> Result<()>;

    /// Get the current surface dimensions.
    fn surface_size(&self, surface: SurfaceHandle) -> (u32, u32);

    /// Get the texture format used by a surface's swapchain.
    /// Use this to ensure your render pipeline matches the surface format.
    fn surface_format(&self, surface: SurfaceHandle) -> TextureFormat;

    // Compute pipeline management
    /// Create a compute pipeline from a compute shader.
    fn create_compute_pipeline(
        &mut self,
        device: DeviceHandle,
        compute_shader: ShaderHandle,
        bind_group_layouts: &[BindGroupLayoutHandle],
    ) -> Result<ComputePipelineHandle>;

    /// Destroy a compute pipeline.
    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle);

    /// Execute compute commands.
    /// This submits compute work to the GPU and waits for completion.
    fn dispatch_compute(&mut self, device: DeviceHandle, commands: &[ComputeCommand])
        -> Result<()>;
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
    pub const COMPUTE: ShaderStages = ShaderStages(4);
    pub const ALL: ShaderStages = ShaderStages(7); // VERTEX | FRAGMENT | COMPUTE
}

/// Binding type for bind groups.
#[derive(Debug, Clone)]
pub enum BindingType {
    UniformBuffer,
    StorageBuffer {
        read_only: bool,
    },
    /// Sampled texture (read-only in shader).
    Texture,
    /// Sampler for texture sampling.
    Sampler,
    /// Storage texture (read-write in shader).
    StorageTexture,
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
    Buffer {
        buffer: BufferHandle,
        offset: u64,
        size: u64,
    },
    Texture(TextureHandle),
    Sampler(SamplerHandle),
}

/// Create the default backend for the current platform.
pub fn create_default_backend() -> Result<Box<dyn GpuBackend>> {
    // On Windows with dx12 feature, prefer DX12
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    {
        tracing::info!("Creating DX12 backend");
        Ok(Box::new(dx12::Dx12Backend::new()?))
    }

    // Vulkan fallback on non-DX12 platforms
    #[cfg(all(
        feature = "vulkan",
        not(target_arch = "wasm32"),
        not(all(feature = "dx12", target_os = "windows"))
    ))]
    {
        tracing::info!("Creating Vulkan backend");
        Ok(Box::new(vulkan::VulkanBackend::new()?))
    }

    // No backend available
    #[cfg(not(any(
        all(feature = "dx12", target_os = "windows"),
        all(feature = "vulkan", not(target_arch = "wasm32"))
    )))]
    {
        anyhow::bail!("No GPU backend available - enable 'vulkan' or 'dx12' feature")
    }
}

/// Create a specific backend by type.
pub fn create_backend(backend_type: BackendType) -> Result<Box<dyn GpuBackend>> {
    match backend_type {
        #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
        BackendType::Vulkan => {
            tracing::info!("Creating Vulkan backend");
            Ok(Box::new(vulkan::VulkanBackend::new()?))
        }
        #[cfg(all(feature = "dx12", target_os = "windows"))]
        BackendType::Dx12 => {
            tracing::info!("Creating DX12 backend");
            Ok(Box::new(dx12::Dx12Backend::new()?))
        }
        _ => anyhow::bail!("Backend {:?} not available on this platform", backend_type),
    }
}
