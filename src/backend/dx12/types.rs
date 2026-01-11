//! DX12 backend internal types.
//!
//! This module contains all the state structs used by the DX12 backend.

use super::super::{BufferHandle, DeviceHandle};
use crate::types::{DepthFormat, SamplerDesc, TextureFormat};
use windows::Win32::Graphics::{Direct3D12, Dxgi};

/// Information about a physical DXGI adapter.
/// Named DxgiAdapterInfo to avoid conflict with super::AdapterInfo.
pub(crate) struct DxgiAdapterInfo {
    pub adapter: Dxgi::IDXGIAdapter1,
    pub desc: Dxgi::DXGI_ADAPTER_DESC1,
    pub adapter_id: u32,
}

/// A logical D3D12 device with associated resources.
pub(crate) struct LogicalDevice {
    pub device: Direct3D12::ID3D12Device,
    pub adapter_id: u32,
    pub command_queue: Direct3D12::ID3D12CommandQueue,
    pub command_allocator: Direct3D12::ID3D12CommandAllocator,
    pub rtv_heap: Direct3D12::ID3D12DescriptorHeap,
    pub rtv_descriptor_size: u32,
    pub dsv_heap: Direct3D12::ID3D12DescriptorHeap,
    pub dsv_descriptor_size: u32,
    pub cbv_srv_uav_heap: Direct3D12::ID3D12DescriptorHeap,
    pub cbv_srv_uav_descriptor_size: u32,
    pub sampler_heap: Direct3D12::ID3D12DescriptorHeap,
    pub sampler_descriptor_size: u32,
    pub fence: Direct3D12::ID3D12Fence,
    pub fence_value: u64,
}

/// GPU buffer state.
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub resource: Direct3D12::ID3D12Resource,
    pub size: u64,
}

/// Shader module state with cached compiled bytecode.
pub(crate) struct ShaderState {
    pub device_handle: DeviceHandle,
    pub slang_source: String,
    /// Cached compiled vertex shader bytecode
    pub vertex_bytecode: Option<Vec<u8>>,
    /// Cached compiled fragment shader bytecode
    pub fragment_bytecode: Option<Vec<u8>>,
    /// Cached compiled compute shader bytecode
    pub compute_bytecode: Option<Vec<u8>>,
}

/// Graphics pipeline state.
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline_state: Direct3D12::ID3D12PipelineState,
    pub root_signature: Direct3D12::ID3D12RootSignature,
    /// Vertex buffer stride from vertex layout
    pub vertex_stride: u32,
}

/// Compute pipeline state.
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline_state: Direct3D12::ID3D12PipelineState,
    pub root_signature: Direct3D12::ID3D12RootSignature,
    /// Bind group layout handles for looking up binding types during dispatch.
    pub bind_group_layouts: Vec<super::super::BindGroupLayoutHandle>,
}

/// Bind group layout (root signature descriptor table layout) state.
pub(crate) struct BindGroupLayoutState {
    pub device_handle: DeviceHandle,
    pub entries: Vec<super::super::BindGroupLayoutEntry>,
}

/// Bind group state.
pub(crate) struct BindGroupState {
    pub device_handle: DeviceHandle,
    pub layout_handle: super::super::BindGroupLayoutHandle,
    pub buffer_bindings: Vec<(u32, BufferHandle, u64, u64)>, // binding, buffer, offset, size
}

/// GPU render target state with optional staging for CPU readback.
pub(crate) struct RenderTargetState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// GPU-only render target texture
    pub texture: Direct3D12::ID3D12Resource,
    /// RTV descriptor handle offset
    pub rtv_offset: u32,
    /// Depth buffer (optional)
    pub depth_format: Option<DepthFormat>,
    pub depth_texture: Option<Direct3D12::ID3D12Resource>,
    pub dsv_offset: Option<u32>,
    /// Staging buffer for CPU readback (lazy-created on first read)
    pub staging_buffer: Option<Direct3D12::ID3D12Resource>,
    /// Command list for rendering
    pub command_list: Direct3D12::ID3D12GraphicsCommandList,
    /// Track if we've rendered (for readback validation)
    pub has_rendered: bool,
}

/// GPU texture state.
pub(crate) struct TextureState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub resource: Direct3D12::ID3D12Resource,
    /// SRV descriptor offset in CBV/SRV/UAV heap
    pub srv_offset: u32,
}

/// GPU sampler state.
pub(crate) struct SamplerState {
    pub device_handle: DeviceHandle,
    /// Sampler descriptor offset in sampler heap
    pub sampler_offset: u32,
    #[allow(dead_code)]
    pub desc: SamplerDesc,
}

/// Maximum number of frames that can be in-flight at once.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Per-frame synchronization resources for proper swapchain pipelining.
pub(crate) struct FrameSync {
    pub command_list: Direct3D12::ID3D12GraphicsCommandList,
    pub command_allocator: Direct3D12::ID3D12CommandAllocator,
    pub fence_value: u64,
}

/// Surface (swapchain) state for window presentation.
pub(crate) struct SurfaceState {
    pub device_handle: DeviceHandle,
    pub swapchain: Dxgi::IDXGISwapChain3,
    pub render_targets: Vec<Direct3D12::ID3D12Resource>,
    pub rtv_offsets: Vec<u32>,
    pub width: u32,
    pub height: u32,
    pub format: Dxgi::Common::DXGI_FORMAT,
    /// Current frame index (0..MAX_FRAMES_IN_FLIGHT)
    pub current_frame: usize,
    /// Currently acquired swapchain image index
    pub current_image_index: Option<u32>,
    /// Per-frame synchronization resources
    pub frame_sync: Vec<FrameSync>,
}
