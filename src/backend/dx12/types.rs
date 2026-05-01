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

use super::super::{
    BufferHandle, ComputePipelineHandle, DeviceHandle, PipelineHandle, RenderTargetHandle,
    SamplerHandle, ShaderHandle, SurfaceHandle, TextureHandle,
};
use crate::types::{DepthFormat, SamplerDesc, TextureFormat};
use std::collections::HashMap;
use windows::Win32::Graphics::{Direct3D12, Dxgi};

/// Maximum number of descriptors in the CBV/SRV/UAV heap for bindless rendering
#[allow(dead_code)]
pub const MAX_BINDLESS_CBV_SRV_UAV: u32 = 16384;

/// Maximum number of descriptors in the sampler heap for bindless rendering
#[allow(dead_code)]
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
///
/// Each `register_*` call pops a slot from the appropriate free list before minting a
/// new one, preventing monotonic counter exhaustion when transient resources (e.g.
/// per-frame pool views or swapchain-back-buffer UAVs) are created and destroyed every
/// frame. Without this recycling the `next_cbv_srv_uav_offset` would hit
/// `MAX_BINDLESS_CBV_SRV_UAV` (16 384) and subsequent descriptor writes would go
/// out-of-bounds, corrupting the heap and causing GPU hangs / device loss.
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    /// Monotonic fallback counter: only advanced when the free list is empty.
    next_cbv_srv_uav_offset: u32,
    next_sampler_offset: u32,
    /// Recycled CBV/SRV/UAV heap slots returned by `unregister_buffer` /
    /// `unregister_texture`. Popped first by every `register_*` call.
    free_cbv_srv_uav_slots: Vec<u32>,
    /// Recycled sampler heap slots returned by `unregister_sampler`.
    free_sampler_slots: Vec<u32>,
    /// Maps buffer handle to its primary descriptor offset (UAV for storage, CBV for uniform)
    pub buffer_offsets: HashMap<BufferHandle, u32>,
    /// Maps buffer handle to its secondary SRV offset (for storage buffers that need read access)
    pub buffer_srv_offsets: HashMap<BufferHandle, u32>,
    pub texture_offsets: HashMap<TextureHandle, u32>,
    /// Maps texture handle to UAV offset (for storage textures / SpatialAccess::Direct)
    pub texture_uav_offsets: HashMap<TextureHandle, u32>,
    pub sampler_offsets: HashMap<SamplerHandle, u32>,
}

#[allow(dead_code)]
impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            next_cbv_srv_uav_offset: 0,
            next_sampler_offset: 0,
            free_cbv_srv_uav_slots: Vec::new(),
            free_sampler_slots: Vec::new(),
            buffer_offsets: HashMap::new(),
            buffer_srv_offsets: HashMap::new(),
            texture_offsets: HashMap::new(),
            texture_uav_offsets: HashMap::new(),
            sampler_offsets: HashMap::new(),
        }
    }

    /// Pop a recycled CBV/SRV/UAV slot or mint a fresh one.
    fn alloc_cbv_srv_uav(&mut self) -> u32 {
        self.free_cbv_srv_uav_slots.pop().unwrap_or_else(|| {
            let offset = self.next_cbv_srv_uav_offset;
            self.next_cbv_srv_uav_offset += 1;
            offset
        })
    }

    /// Pop a recycled sampler slot or mint a fresh one.
    fn alloc_sampler(&mut self) -> u32 {
        self.free_sampler_slots.pop().unwrap_or_else(|| {
            let offset = self.next_sampler_offset;
            self.next_sampler_offset += 1;
            offset
        })
    }

    pub fn register_buffer_cbv(&mut self, handle: BufferHandle) -> u32 {
        let offset = self.alloc_cbv_srv_uav();
        self.buffer_offsets.insert(handle, offset);
        offset
    }

    pub fn register_buffer_srv(&mut self, handle: BufferHandle) -> u32 {
        let offset = self.alloc_cbv_srv_uav();
        // Store in secondary map since buffer may already have a UAV offset
        self.buffer_srv_offsets.insert(handle, offset);
        offset
    }

    pub fn register_buffer_uav(&mut self, handle: BufferHandle) -> u32 {
        let offset = self.alloc_cbv_srv_uav();
        self.buffer_offsets.insert(handle, offset);
        offset
    }

    pub fn register_texture(&mut self, handle: TextureHandle) -> u32 {
        let offset = self.alloc_cbv_srv_uav();
        self.texture_offsets.insert(handle, offset);
        offset
    }

    /// Register a UAV descriptor for a texture (e.g. storage image / SpatialAccess::Direct).
    pub fn register_texture_uav(&mut self, handle: TextureHandle) -> u32 {
        let offset = self.alloc_cbv_srv_uav();
        self.texture_uav_offsets.insert(handle, offset);
        offset
    }

    pub fn register_sampler(&mut self, handle: SamplerHandle) -> u32 {
        let offset = self.alloc_sampler();
        self.sampler_offsets.insert(handle, offset);
        offset
    }

    /// Get the SRV offset for a buffer (for read-only access to storage buffers)
    pub fn get_buffer_srv_offset(&self, handle: BufferHandle) -> Option<u32> {
        self.buffer_srv_offsets.get(&handle).copied()
    }

    /// Allocate a raw descriptor slot not tied to any resource handle.
    /// Used for permanent device-lifetime slots (e.g. `scratch_clear_uav_offset`).
    pub fn alloc_cbv_srv_uav_slot(&mut self) -> u32 {
        self.alloc_cbv_srv_uav()
    }

    /// Unregister a buffer, returning all of its descriptor slots to the free list.
    ///
    /// Storage buffers may occupy two slots (UAV primary + SRV secondary); both are
    /// recycled here. Without this, per-frame buffer churn exhausts the 16 384-slot
    /// heap in seconds and subsequent descriptor writes corrupt adjacent entries.
    pub fn unregister_buffer(&mut self, handle: BufferHandle) {
        if let Some(offset) = self.buffer_offsets.remove(&handle) {
            self.free_cbv_srv_uav_slots.push(offset);
        }
        if let Some(offset) = self.buffer_srv_offsets.remove(&handle) {
            self.free_cbv_srv_uav_slots.push(offset);
        }
    }

    /// Unregister a texture, returning all of its descriptor slots to the free list.
    ///
    /// Storage textures may occupy two slots (SRV + UAV); both are recycled here.
    pub fn unregister_texture(&mut self, handle: TextureHandle) {
        if let Some(offset) = self.texture_offsets.remove(&handle) {
            self.free_cbv_srv_uav_slots.push(offset);
        }
        if let Some(offset) = self.texture_uav_offsets.remove(&handle) {
            self.free_cbv_srv_uav_slots.push(offset);
        }
    }

    pub fn unregister_sampler(&mut self, handle: SamplerHandle) {
        if let Some(offset) = self.sampler_offsets.remove(&handle) {
            self.free_sampler_slots.push(offset);
        }
    }
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    /// Simulate the per-frame create/destroy churn that ekrano generates for transient
    /// pool-view buffers. The counter must stay bounded — well below MAX_BINDLESS_CBV_SRV_UAV
    /// — even after far more iterations than the heap limit.
    #[test]
    fn buffer_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as BufferHandle;
            reg.register_buffer_uav(handle);
            reg.unregister_buffer(handle);
        }
        // Only one slot should ever have been minted (slot 0), now sitting in the free list.
        assert_eq!(
            reg.next_cbv_srv_uav_offset, 1,
            "UAV counter grew; slot recycling not working"
        );
        assert_eq!(reg.free_cbv_srv_uav_slots.len(), 1);
    }

    /// Storage buffers register two slots (UAV primary + SRV secondary).
    /// Both must be returned to the free list on unregister.
    #[test]
    fn storage_buffer_dual_slot_recycled() {
        let mut reg = ResourceRegistry::new();
        for i in 0..1_000u64 {
            let handle = i as BufferHandle;
            reg.register_buffer_uav(handle);
            reg.register_buffer_srv(handle);
            reg.unregister_buffer(handle);
        }
        assert_eq!(
            reg.next_cbv_srv_uav_offset, 2,
            "counter should only have advanced twice (one UAV + one SRV slot ever minted)"
        );
        assert_eq!(
            reg.free_cbv_srv_uav_slots.len(),
            2,
            "both slots must be in the free list"
        );
    }

    /// Textures with both SRV and UAV views (storage textures) must recycle both slots.
    #[test]
    fn texture_dual_slot_recycled() {
        let mut reg = ResourceRegistry::new();
        for i in 0..1_000u64 {
            let handle = i as TextureHandle;
            reg.register_texture(handle);
            reg.register_texture_uav(handle);
            reg.unregister_texture(handle);
        }
        assert_eq!(reg.next_cbv_srv_uav_offset, 2);
        assert_eq!(reg.free_cbv_srv_uav_slots.len(), 2);
    }

    /// Sampler slots are in a separate heap; verify they recycle independently.
    #[test]
    fn sampler_slots_recycled() {
        let mut reg = ResourceRegistry::new();
        for i in 0..5_000u64 {
            let handle = i as SamplerHandle;
            reg.register_sampler(handle);
            reg.unregister_sampler(handle);
        }
        assert_eq!(reg.next_sampler_offset, 1);
        assert_eq!(reg.free_sampler_slots.len(), 1);
    }

    /// Simultaneously-live resources must receive distinct slots.
    #[test]
    fn live_resources_get_distinct_slots() {
        let mut reg = ResourceRegistry::new();
        const N: u64 = 64;
        let mut slots: Vec<u32> = (0..N)
            .map(|i| reg.register_buffer_uav(i as BufferHandle))
            .collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(
            slots.len(),
            N as usize,
            "duplicate slots assigned to live resources"
        );
    }

    /// Slots freed by destroyed resources must be reused before the counter advances,
    /// keeping the high-water mark at or below the number of concurrently-live resources.
    #[test]
    fn high_water_mark_bounded_by_live_count() {
        let mut reg = ResourceRegistry::new();
        const LIVE: u64 = 8;
        const ROUNDS: u64 = 10_000;

        // Prime with LIVE simultaneous resources.
        for i in 0..LIVE {
            reg.register_buffer_uav(i as BufferHandle);
        }
        // Repeatedly destroy the oldest and create a new one.
        for i in LIVE..LIVE + ROUNDS {
            reg.unregister_buffer((i - LIVE) as BufferHandle);
            reg.register_buffer_uav(i as BufferHandle);
        }
        assert!(
            reg.next_cbv_srv_uav_offset <= LIVE as u32,
            "counter ({}) exceeded live count ({LIVE}); slot recycling broken",
            reg.next_cbv_srv_uav_offset
        );
    }
}

/// Information about a physical DXGI adapter.
/// Named DxgiAdapterInfo to avoid conflict with super::AdapterInfo.
#[allow(dead_code)]
pub(crate) struct DxgiAdapterInfo {
    pub adapter: Dxgi::IDXGIAdapter1,
    pub desc: Dxgi::DXGI_ADAPTER_DESC1,
    pub adapter_id: u32,
}

/// A slot in the compute command allocator pool.
/// An allocator can only be reset after its associated GPU work has completed.
#[allow(dead_code)]
pub(crate) struct ComputeAllocatorSlot {
    pub allocator: Direct3D12::ID3D12CommandAllocator,
    /// Fence value when this slot was last used (for reuse detection)
    pub fence_value: u64,
}

/// A logical D3D12 device with associated resources.
#[allow(dead_code)]
pub(crate) struct LogicalDevice {
    pub device: Direct3D12::ID3D12Device10,
    pub adapter_id: u32,
    pub command_queue: Direct3D12::ID3D12CommandQueue,
    /// Legacy single allocator for non-compute paths (e.g. render target). Compute uses the pool.
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
    /// Shared root signature for all bindless pipelines (graphics and compute)
    pub bindless_root_signature: Option<Direct3D12::ID3D12RootSignature>,
    /// Command signature for indirect compute dispatch (ExecuteIndirect)
    pub compute_dispatch_indirect_signature: Option<Direct3D12::ID3D12CommandSignature>,
    /// Registry tracking resource offsets in descriptor heaps
    pub resource_registry: ResourceRegistry,
    /// Pool of command allocators for non-blocking compute submission.
    /// Slots can be reused when fence signals completion.
    pub compute_allocator_pool: Vec<ComputeAllocatorSlot>,
    /// Non-shader-visible CBV/SRV/UAV heap for ClearUnorderedAccessViewUint.
    /// DX12 requires a CPU descriptor from a non-shader-visible heap for UAV clears.
    pub cpu_clear_heap: Direct3D12::ID3D12DescriptorHeap,
    /// Reserved shader-visible descriptor slot for structured buffer clears.
    /// Used to hold a temporary R32_UINT UAV so the GPU-side descriptor matches
    /// the clear format at execution time (not just at recording time).
    pub scratch_clear_uav_offset: u32,
}

/// GPU buffer state.
#[allow(dead_code)]
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
    /// Upload buffer for DEFAULT heap resources (lazy-created on first CPU write)
    pub upload_buffer: Option<Direct3D12::ID3D12Resource>,
    /// StructuredBuffer element stride (for UAV clear rect calculations)
    pub element_stride: Option<u32>,
    /// If true, this is a view into another buffer — don't free the resource on destroy.
    pub is_view: bool,
    /// Direct3D 12: paired READBACK resource for [`crate::types::BufferFlags::CPU_COHERENT`]
    /// storage buffers. Copied UAV → READBACK by [`super::buffer::read_to_cpu`].
    pub coherent_readback: Option<Direct3D12::ID3D12Resource>,
    /// Persistent map of `coherent_readback` (see above).
    /// Persistent `Map` result address for the readback resource (`usize` for `Send`/`Sync`).
    pub coherent_readback_mapped: Option<usize>,
    /// Creation-time flags.
    pub flags: crate::types::BufferFlags,
}

/// Shader module state with cached compiled bytecode.
pub(crate) struct ShaderState {
    pub device_handle: DeviceHandle,
    pub slang_source: String,
    /// Search paths for Slang module resolution
    pub search_paths: Vec<String>,
    /// Extra preprocessor defines (e.g. msaa, msaa8)
    pub defines: Vec<(String, String)>,
    /// Per-shader Slang optimization level
    pub optimization_level: crate::types::OptimizationLevel,
    /// Cached compiled vertex shader bytecode
    pub vertex_bytecode: Option<Vec<u8>>,
    /// Cached compiled fragment shader bytecode
    pub fragment_bytecode: Option<Vec<u8>>,
    /// Cached compiled compute shader bytecode
    pub compute_bytecode: Option<Vec<u8>>,
    /// Reflection data for bindless rendering (ParameterBlock layouts)
    pub reflection: Option<crate::slang::ShaderReflection>,
    /// Pending struct layout validation on first stage compile; cleared after success.
    pub layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
}

/// Graphics pipeline state.
#[allow(dead_code)]
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
    /// Per-push-constant-slot category inferred from `goldy_dyn_*(N)` literal
    /// calls in the bound shader(s). Empty disables validation.
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
    /// Per-slot structured element stride from shader reflection (bytes), when resolved.
    pub push_constant_buffer_strides: Vec<Option<u32>>,
    /// Human-readable identifier used in category-mismatch error messages.
    pub shader_debug_name: String,
}

/// Compute pipeline state.
#[allow(dead_code)]
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline_state: Direct3D12::ID3D12PipelineState,
    pub root_signature: Direct3D12::ID3D12RootSignature,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
    /// Per-push-constant-slot category inferred from `goldy_dyn_*(N)` literal
    /// calls in the bound compute shader. Empty disables validation.
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
    /// Per-slot structured element stride from shader reflection (bytes), when resolved.
    pub push_constant_buffer_strides: Vec<Option<u32>>,
    /// Human-readable identifier used in category-mismatch error messages.
    pub shader_debug_name: String,
}

/// GPU render target state with optional staging for CPU readback.
#[allow(dead_code)]
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
    pub command_list: Direct3D12::ID3D12GraphicsCommandList7,
    /// Track if we've rendered (for readback validation)
    pub has_rendered: bool,
}

/// GPU texture state.
#[allow(dead_code)]
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
    /// Last known layout for enhanced texture barriers (replaces legacy `current_state`).
    pub last_layout: Direct3D12::D3D12_BARRIER_LAYOUT,
    /// Whether this texture was created with UAV access (SpatialAccess::Direct).
    pub is_storage: bool,
}

/// GPU sampler state.
#[allow(dead_code)]
pub(crate) struct SamplerState {
    pub device_handle: DeviceHandle,
    /// Sampler descriptor offset in sampler heap
    pub sampler_offset: u32,
    pub desc: SamplerDesc,
    /// Bindless descriptor heap offset (same as sampler_offset when bindless is enabled)
    pub bindless_offset: Option<u32>,
}

/// Maximum number of frames that can be in-flight at once.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Per-frame synchronization resources for proper swapchain pipelining.
#[allow(dead_code)]
pub(crate) struct FrameSync {
    pub command_list: Direct3D12::ID3D12GraphicsCommandList7,
    pub command_allocator: Direct3D12::ID3D12CommandAllocator,
    pub fence_value: u64,
    /// Set after `surface::render` submits. When false, `present` copies the compute scratch texture.
    pub render_pass_submitted: bool,
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
    /// Depth buffer (optional)
    pub depth_format: Option<DepthFormat>,
    #[allow(dead_code)] // Held for ownership; dropped when surface is destroyed
    pub depth_texture: Option<Direct3D12::ID3D12Resource>,
    pub dsv_offset: Option<u32>,
    /// Current frame index (0..MAX_FRAMES_IN_FLIGHT)
    pub current_frame: usize,
    /// Currently acquired swapchain image index
    pub current_image_index: Option<u32>,
    /// Per-frame synchronization resources
    pub frame_sync: Vec<FrameSync>,
    /// Transient texture handle for the currently acquired back buffer,
    /// registered in the bindless descriptor heap as a UAV so compute shaders
    /// can write directly to the swapchain image.
    pub current_texture_handle: Option<super::TextureHandle>,
    /// Per swapchain buffer index: persistent UAV texture for compute; results are
    /// copied to the real back buffer in `present` (swapchain images cannot be UAVs).
    pub compute_scratch_textures: Vec<Option<super::TextureHandle>>,
}

/// Consolidated DX12 backend state.
/// This holds all the resources and state for the DX12 backend.
pub(super) struct Dx12State {
    pub factory: Dxgi::IDXGIFactory4,
    pub adapters: Vec<DxgiAdapterInfo>,
    pub devices: HashMap<DeviceHandle, LogicalDevice>,
    pub next_device_handle: DeviceHandle,
    pub buffers: HashMap<BufferHandle, BufferState>,
    pub next_buffer_handle: BufferHandle,
    pub shaders: HashMap<ShaderHandle, ShaderState>,
    pub next_shader_handle: ShaderHandle,
    pub pipelines: HashMap<PipelineHandle, PipelineState>,
    pub next_pipeline_handle: PipelineHandle,
    pub compute_pipelines: HashMap<ComputePipelineHandle, ComputePipelineState>,
    pub next_compute_pipeline_handle: ComputePipelineHandle,
    pub render_targets: HashMap<RenderTargetHandle, RenderTargetState>,
    pub next_render_target_handle: RenderTargetHandle,
    pub surfaces: HashMap<SurfaceHandle, SurfaceState>,
    pub next_surface_handle: SurfaceHandle,
    pub textures: HashMap<TextureHandle, TextureState>,
    pub next_texture_handle: TextureHandle,
    pub samplers: HashMap<SamplerHandle, SamplerState>,
    pub next_sampler_handle: SamplerHandle,
    /// Next RTV descriptor offset
    pub next_rtv_offset: u32,
    /// Next DSV descriptor offset
    pub next_dsv_offset: u32,
    /// Per-backend Slang compiler instance
    pub slang_compiler: crate::slang::SlangCompiler,
    /// Per-device upload belts for `ComputeCommand::WriteBuffer`.
    pub(super) staging_belts: HashMap<DeviceHandle, super::staging::StagingBelt>,
    /// Pending texture copies awaiting batch submission via [`super::texture::flush_pending_copies`].
    pub(super) pending_texture_copies: Vec<super::texture::PendingTextureCopy>,
    /// Free-list of `ID3D12Resource` objects for reuse, keyed by (width, height, format, is_storage).
    /// Avoids progressive `CreateCommittedResource` slowdown from GPU heap fragmentation.
    pub(super) texture_cache: HashMap<super::texture::TextureCacheKey, Vec<super::texture::CachedTextureResource>>,
}
