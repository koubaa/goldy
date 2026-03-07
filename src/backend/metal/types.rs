//! Metal backend internal types.
//!
//! This module contains all the state structs used by the Metal backend.
//!
//! ## Bindless Architecture
//!
//! The Metal backend uses Argument Buffers Tier 2 for bindless resource access:
//! - Resources are allocated from MTLHeap for efficient residency management
//! - A global argument buffer stores GPU resource IDs (gpuResourceID)
//! - At render time, useHeap() declares residency for all resources
//! - Shaders access resources by index into the argument buffer

use super::super::{
    BufferHandle, ComputePipelineHandle, DeviceHandle, PipelineHandle, RenderTargetHandle,
    SamplerHandle, ShaderHandle, SurfaceHandle, TextureHandle,
};
use crate::types::{DepthFormat, TextureFormat};
use std::collections::HashMap;
// Use explicit crate path to avoid collision with our module name
use ::metal as mtl;
use mtl::{
    ArgumentEncoder, Buffer as MTLBuffer, CommandQueue,
    ComputePipelineState as MTLComputePipelineState, DepthStencilState as MTLDepthStencilState,
    Device as MTLDevice, Heap, Library, MTLPrimitiveType, RenderPipelineState, SamplerState,
    Texture as MTLTexture,
};

/// Maximum size of the argument buffer (supports up to 16K resources)
pub const ARGUMENT_BUFFER_SIZE: u64 = 16 * 1024 * 8; // 8 bytes per resource ID

/// Buffer slot for push constants (resource indices) in shaders.
/// Slang assigns gGoldyDynamic to [[buffer(1)]] (gGoldy ParameterBlock takes [[buffer(0)]]).
pub const PUSH_CONSTANTS_SLOT: u64 = 1;

/// Maximum number of resource indices in push constants
pub const MAX_PUSH_CONSTANT_INDICES: usize = 16;

/// Push constants structure for passing bindless resource indices to shaders.
///
/// Matches the flat layout used by DX12 (`SetGraphicsRoot32BitConstants`) and
/// Vulkan (`vkCmdPushConstants`). Indices are packed sequentially: the caller
/// decides what each slot means (buffer, texture, or sampler index).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BindlessIndices {
    pub indices: [u32; MAX_PUSH_CONSTANT_INDICES],
}

/// A logical Metal device with associated resources.
///
/// Requires Argument Buffers Tier 2 (Apple Silicon, Intel 2017+, AMD 2015+).
pub(crate) struct LogicalDevice {
    pub device: MTLDevice,
    pub command_queue: CommandQueue,
    pub adapter_id: u32,

    // Bindless infrastructure (always present — Tier 2 required)
    /// Heap for buffer allocations
    pub buffer_heap: Heap,
    /// Heap for texture allocations
    pub texture_heap: Heap,
    /// Global argument buffer containing resource IDs
    pub argument_buffer: MTLBuffer,
    /// Encoder for writing buffers to the argument buffer
    pub argument_encoder: ArgumentEncoder,
    /// Encoder for writing textures to the argument buffer
    pub texture_encoder: ArgumentEncoder,
    /// Registry tracking resource indices in the argument buffer
    pub resource_registry: ResourceRegistry,
    /// Count of buffers allocated from heap (for use_heap_at safety)
    pub heap_buffer_count: u32,
    /// Count of textures allocated from heap (for use_heap_at safety)
    pub heap_texture_count: u32,
}

/// Maximum resources per access pattern category (must match GOLDY_MAX_RESOURCES in shaders)
pub const MAX_RESOURCES_PER_CATEGORY: u32 = 64;

/// Registry for tracking bindless resource indices
///
/// The layout matches GoldyBindlessResources in bindless_resources.slang:
/// - storageBuffers[64] at indices 0-63   (Scattered access)
/// - uniformBuffers[64] at indices 64-127 (Broadcast access)
/// - textures, storageImages, samplers at higher offsets
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    next_storage_buffer_index: u32, // Scattered: 0-63
    next_uniform_buffer_index: u32, // Broadcast: 64-127
    next_texture_index: u32,
    next_sampler_index: u32,
    pub buffer_indices: HashMap<BufferHandle, u32>,
    pub texture_indices: HashMap<TextureHandle, u32>,
    pub sampler_indices: HashMap<SamplerHandle, u32>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            // Storage buffers (Scattered) at indices 0-63, bytes 0-511
            next_storage_buffer_index: 0,
            // Uniform buffers (Broadcast) at indices 64-127, bytes 512-1023
            next_uniform_buffer_index: MAX_RESOURCES_PER_CATEGORY,
            // Textures at indices 128-191, bytes 1024-1535 (after storageBuffers[64]+uniformBuffers[64])
            next_texture_index: 2 * MAX_RESOURCES_PER_CATEGORY,
            // Samplers at indices 256-319, bytes 2048-2559 (after textures[64]+storageImages[64])
            next_sampler_index: 4 * MAX_RESOURCES_PER_CATEGORY,
            buffer_indices: HashMap::new(),
            texture_indices: HashMap::new(),
            sampler_indices: HashMap::new(),
        }
    }

    /// Register a storage buffer (Scattered access) - indices 0-63
    pub fn register_storage_buffer(&mut self, handle: BufferHandle) -> u32 {
        let index = self.next_storage_buffer_index;
        self.next_storage_buffer_index += 1;
        self.buffer_indices.insert(handle, index);
        index
    }

    /// Register a uniform buffer (Broadcast access) - indices 64-127
    pub fn register_uniform_buffer(&mut self, handle: BufferHandle) -> u32 {
        let index = self.next_uniform_buffer_index;
        self.next_uniform_buffer_index += 1;
        self.buffer_indices.insert(handle, index);
        index
    }

    pub fn register_texture(&mut self, handle: TextureHandle) -> u32 {
        let index = self.next_texture_index;
        self.next_texture_index += 1;
        self.texture_indices.insert(handle, index);
        index
    }

    pub fn register_sampler(&mut self, handle: SamplerHandle) -> u32 {
        let index = self.next_sampler_index;
        self.next_sampler_index += 1;
        self.sampler_indices.insert(handle, index);
        index
    }

    pub fn unregister_buffer(&mut self, handle: BufferHandle) {
        self.buffer_indices.remove(&handle);
    }

    pub fn unregister_texture(&mut self, handle: TextureHandle) {
        self.texture_indices.remove(&handle);
    }

    pub fn unregister_sampler(&mut self, handle: SamplerHandle) {
        self.sampler_indices.remove(&handle);
    }
}

/// GPU buffer state.
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub buffer: MTLBuffer,
    pub size: u64,
    /// Index in the global argument buffer (always present — heap required).
    pub arg_buffer_index: u32,
}

/// Shader module state with cached compiled stages.
pub(crate) struct ShaderState {
    pub device_handle: DeviceHandle,
    pub slang_source: String,
    /// Search paths for Slang module resolution
    pub search_paths: Vec<String>,
    /// Compiled vertex shader library
    pub vertex_library: Option<Library>,
    /// Compiled fragment shader library
    pub fragment_library: Option<Library>,
    /// Compiled compute shader library
    pub compute_library: Option<Library>,
    /// Reflection data for bindless rendering (ParameterBlock layouts)
    pub reflection: Option<crate::slang::ShaderReflection>,
}

/// Graphics pipeline state.
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: RenderPipelineState,
    pub depth_stencil: Option<MTLDepthStencilState>,
    pub primitive_type: MTLPrimitiveType,
}

/// Compute pipeline state.
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: MTLComputePipelineState,
    /// Thread group size from [numthreads(x, y, z)] attribute
    pub workgroup_size: [u32; 3],
}

/// GPU render target state with optional staging for CPU readback.
pub(crate) struct RenderTargetState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// GPU render target texture
    pub texture: MTLTexture,
    /// Depth buffer (optional)
    pub depth_format: Option<DepthFormat>,
    pub depth_texture: Option<MTLTexture>,
    /// Track if we've rendered (for readback validation)
    pub has_rendered: bool,
}

/// GPU texture state.
pub(crate) struct TextureState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub texture: MTLTexture,
    /// Index in the global argument buffer (always present — heap required).
    pub arg_buffer_index: u32,
}

/// GPU sampler state.
pub(crate) struct SamplerState_ {
    pub device_handle: DeviceHandle,
    pub sampler: SamplerState,
    /// Index in the global argument buffer (always present).
    pub arg_buffer_index: u32,
}

/// Maximum number of frames that can be in-flight at once.
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// Surface (CAMetalLayer) state for window presentation.
pub(crate) struct SurfaceState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// Current frame index (0..MAX_FRAMES_IN_FLIGHT)
    pub current_frame: usize,
    /// The CAMetalLayer (stored as raw pointer for objc interop)
    pub layer: *mut std::ffi::c_void,
}

// Safety: Metal objects are thread-safe when properly synchronized
unsafe impl Send for SurfaceState {}
unsafe impl Sync for SurfaceState {}

/// Consolidated Metal backend state.
/// Holds all resources and state for the Metal backend.
pub(super) struct MetalState {
    pub devices: std::collections::HashMap<DeviceHandle, LogicalDevice>,
    pub next_device_handle: DeviceHandle,
    pub buffers: std::collections::HashMap<BufferHandle, BufferState>,
    pub next_buffer_handle: BufferHandle,
    pub shaders: std::collections::HashMap<ShaderHandle, ShaderState>,
    pub next_shader_handle: ShaderHandle,
    pub pipelines: std::collections::HashMap<PipelineHandle, PipelineState>,
    pub next_pipeline_handle: PipelineHandle,
    pub compute_pipelines: std::collections::HashMap<ComputePipelineHandle, ComputePipelineState>,
    pub next_compute_pipeline_handle: ComputePipelineHandle,
    pub render_targets: std::collections::HashMap<RenderTargetHandle, RenderTargetState>,
    pub next_render_target_handle: RenderTargetHandle,
    pub surfaces: std::collections::HashMap<SurfaceHandle, SurfaceState>,
    pub next_surface_handle: SurfaceHandle,
    pub textures: std::collections::HashMap<TextureHandle, TextureState>,
    pub next_texture_handle: TextureHandle,
    pub samplers: std::collections::HashMap<SamplerHandle, SamplerState_>,
    pub next_sampler_handle: SamplerHandle,
    pub slang_compiler: crate::slang::SlangCompiler,
}
