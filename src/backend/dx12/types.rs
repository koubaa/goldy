//! DX12 backend internal types.
//!
//! This module contains all the state structs used by the DX12 backend.
//!
//! ## Bindless Architecture
//!
//! The DX12 backend uses shader-visible descriptor heaps for bindless resource access:
//! - A large CBV/SRV/UAV heap contains all buffer and texture descriptors
//! - A large sampler heap contains all sampler descriptors
//! - Resources are registered at creation time and assigned heap offsets
//! - Shaders access resources by indexing into the descriptor heaps

use super::super::{BufferHandle, DeviceHandle, SamplerHandle, TextureHandle};
use crate::types::{DepthFormat, SamplerDesc, TextureFormat};
use std::collections::HashMap;
use windows::Win32::Graphics::{Direct3D12, Dxgi};

/// Maximum number of descriptors in the CBV/SRV/UAV heap for bindless rendering
pub const MAX_BINDLESS_CBV_SRV_UAV: u32 = 16384;

/// Maximum number of descriptors in the sampler heap for bindless rendering
pub const MAX_BINDLESS_SAMPLERS: u32 = 2048;

/// Maximum number of resource indices in root constants
pub const MAX_ROOT_CONSTANT_INDICES: usize = 16;

/// Root constants for passing bindless resource indices to shaders.
/// This is used to tell shaders which indices in the descriptor heaps to access.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct BindlessIndices {
    /// Resource indices (buffers, textures, samplers packed sequentially)
    pub indices: [u32; MAX_ROOT_CONSTANT_INDICES],
}

/// Registry for tracking bindless resource descriptor heap offsets.
/// 
/// IMPORTANT: All CBV, SRV, and UAV descriptors share the same heap (cbv_srv_uav_heap),
/// so we use a unified offset counter to avoid collisions.
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    /// Unified offset counter for CBV/SRV/UAV heap (they all share the same heap!)
    next_cbv_srv_uav_offset: u32,
    next_sampler_offset: u32,
    /// Maps buffer handle to its primary descriptor offset
    pub buffer_offsets: HashMap<BufferHandle, u32>,
    /// Maps buffer handle to its secondary SRV offset (for storage buffers that need read access)
    pub buffer_srv_offsets: HashMap<BufferHandle, u32>,
    pub texture_offsets: HashMap<TextureHandle, u32>,
    pub sampler_offsets: HashMap<SamplerHandle, u32>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            // All CBV/SRV/UAV descriptors use a single unified counter
            // to avoid descriptor heap collisions
            next_cbv_srv_uav_offset: 0,
            next_sampler_offset: 0,
            buffer_offsets: HashMap::new(),
            buffer_srv_offsets: HashMap::new(),
            texture_offsets: HashMap::new(),
            sampler_offsets: HashMap::new(),
        }
    }

    pub fn register_buffer_cbv(&mut self, handle: BufferHandle) -> u32 {
        let offset = self.next_cbv_srv_uav_offset;
        self.next_cbv_srv_uav_offset += 1;
        self.buffer_offsets.insert(handle, offset);
        offset
    }

    pub fn register_buffer_srv(&mut self, handle: BufferHandle) -> u32 {
        let offset = self.next_cbv_srv_uav_offset;
        self.next_cbv_srv_uav_offset += 1;
        // Store in secondary map since buffer may already have a UAV offset
        self.buffer_srv_offsets.insert(handle, offset);
        offset
    }

    pub fn register_buffer_uav(&mut self, handle: BufferHandle) -> u32 {
        let offset = self.next_cbv_srv_uav_offset;
        self.next_cbv_srv_uav_offset += 1;
        self.buffer_offsets.insert(handle, offset);
        offset
    }

    pub fn register_texture(&mut self, handle: TextureHandle) -> u32 {
        let offset = self.next_cbv_srv_uav_offset;
        self.next_cbv_srv_uav_offset += 1;
        self.texture_offsets.insert(handle, offset);
        offset
    }

    pub fn register_sampler(&mut self, handle: SamplerHandle) -> u32 {
        let offset = self.next_sampler_offset;
        self.next_sampler_offset += 1;
        self.sampler_offsets.insert(handle, offset);
        offset
    }

    /// Get the SRV offset for a buffer (for read-only access to storage buffers)
    pub fn get_buffer_srv_offset(&self, handle: BufferHandle) -> Option<u32> {
        self.buffer_srv_offsets.get(&handle).copied()
    }

    pub fn unregister_buffer(&mut self, handle: BufferHandle) {
        self.buffer_offsets.remove(&handle);
        self.buffer_srv_offsets.remove(&handle);
    }

    pub fn unregister_texture(&mut self, handle: TextureHandle) {
        self.texture_offsets.remove(&handle);
    }

    pub fn unregister_sampler(&mut self, handle: SamplerHandle) {
        self.sampler_offsets.remove(&handle);
    }
}

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

    // Bindless infrastructure
    /// Whether bindless descriptor heap indexing is enabled
    pub bindless_enabled: bool,
    /// Registry tracking resource offsets in descriptor heaps
    pub resource_registry: ResourceRegistry,
}

/// GPU buffer state.
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub resource: Direct3D12::ID3D12Resource,
    pub size: u64,
    /// Primary descriptor heap offset for bindless access (UAV for storage, CBV for uniform)
    pub bindless_offset: Option<u32>,
    /// Secondary SRV descriptor offset for storage buffers (for read-only graphics access)
    pub bindless_srv_offset: Option<u32>,
    /// Whether this is a storage buffer (uses UAV instead of CBV/SRV)
    pub is_storage: bool,
    /// Upload buffer for DEFAULT heap resources (needed for CPU writes)
    pub upload_buffer: Option<Direct3D12::ID3D12Resource>,
}

/// Shader module state with cached compiled bytecode.
pub(crate) struct ShaderState {
    pub device_handle: DeviceHandle,
    pub slang_source: String,
    /// Search paths for Slang module resolution
    pub search_paths: Vec<String>,
    /// Cached compiled vertex shader bytecode
    pub vertex_bytecode: Option<Vec<u8>>,
    /// Cached compiled fragment shader bytecode
    pub fragment_bytecode: Option<Vec<u8>>,
    /// Cached compiled compute shader bytecode
    pub compute_bytecode: Option<Vec<u8>>,
    /// Reflection data for bindless rendering (ParameterBlock layouts)
    pub reflection: Option<crate::slang::ShaderReflection>,
}

/// Graphics pipeline state.
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline_state: Direct3D12::ID3D12PipelineState,
    pub root_signature: Direct3D12::ID3D12RootSignature,
    /// Vertex buffer stride from vertex layout
    pub vertex_stride: u32,
    /// Primitive topology for IASetPrimitiveTopology
    pub topology: crate::types::PrimitiveTopology,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
}

/// Compute pipeline state.
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline_state: Direct3D12::ID3D12PipelineState,
    pub root_signature: Direct3D12::ID3D12RootSignature,
    /// Bind group layout handles for looking up binding types during dispatch.
    pub bind_group_layouts: Vec<super::super::BindGroupLayoutHandle>,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
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
    pub texture_bindings: Vec<(u32, TextureHandle)>,          // binding, texture
    pub sampler_bindings: Vec<(u32, SamplerHandle)>,          // binding, sampler
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
    /// Bindless descriptor heap offset (same as srv_offset when bindless is enabled)
    pub bindless_offset: Option<u32>,
}

/// GPU sampler state.
pub(crate) struct SamplerState {
    pub device_handle: DeviceHandle,
    /// Sampler descriptor offset in sampler heap
    pub sampler_offset: u32,
    #[allow(dead_code)]
    pub desc: SamplerDesc,
    /// Bindless descriptor heap offset (same as sampler_offset when bindless is enabled)
    pub bindless_offset: Option<u32>,
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
