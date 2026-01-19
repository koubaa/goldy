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

#![allow(dead_code)] // Some fields are for future use

use super::super::{BufferHandle, DeviceHandle, SamplerHandle, TextureHandle};
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

/// Buffer slot for the global argument buffer in shaders
pub const ARGUMENT_BUFFER_SLOT: u64 = 30;

/// Buffer slot for push constants (resource indices) in shaders
pub const PUSH_CONSTANTS_SLOT: u64 = 29;

/// Maximum number of resource indices in push constants
pub const MAX_PUSH_CONSTANT_INDICES: usize = 16;

/// Push constants structure for passing bindless resource indices to shaders
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BindlessIndices {
    /// Buffer indices into the argument buffer (slot 30)
    pub buffer_indices: [u32; MAX_PUSH_CONSTANT_INDICES],
    /// Texture indices into the argument buffer
    pub texture_indices: [u32; MAX_PUSH_CONSTANT_INDICES],
    /// Sampler indices into the argument buffer
    pub sampler_indices: [u32; MAX_PUSH_CONSTANT_INDICES],
}

/// A logical Metal device with associated resources.
pub(crate) struct LogicalDevice {
    pub device: MTLDevice,
    pub command_queue: CommandQueue,
    pub adapter_id: u32,

    // Bindless infrastructure
    /// Heap for buffer allocations (bindless)
    pub buffer_heap: Option<Heap>,
    /// Heap for texture allocations (bindless)
    pub texture_heap: Option<Heap>,
    /// Global argument buffer containing resource IDs
    pub argument_buffer: Option<MTLBuffer>,
    /// Encoder for writing buffers to the argument buffer
    pub argument_encoder: Option<ArgumentEncoder>,
    /// Encoder for writing textures to the argument buffer
    pub texture_encoder: Option<ArgumentEncoder>,
    /// Registry tracking resource indices in the argument buffer
    pub resource_registry: ResourceRegistry,
    /// Whether bindless is enabled (Argument Buffers Tier 2)
    pub bindless_enabled: bool,
    /// Count of buffers allocated from heap (for use_heap_at safety)
    pub heap_buffer_count: u32,
    /// Count of textures allocated from heap (for use_heap_at safety)
    pub heap_texture_count: u32,
}

/// Registry for tracking bindless resource indices
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    next_buffer_index: u32,
    next_texture_index: u32,
    next_sampler_index: u32,
    pub buffer_indices: HashMap<BufferHandle, u32>,
    pub texture_indices: HashMap<TextureHandle, u32>,
    pub sampler_indices: HashMap<SamplerHandle, u32>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            // Start indices at different offsets to avoid collisions
            next_buffer_index: 0,
            next_texture_index: 4096,
            next_sampler_index: 8192,
            buffer_indices: HashMap::new(),
            texture_indices: HashMap::new(),
            sampler_indices: HashMap::new(),
        }
    }

    pub fn register_buffer(&mut self, handle: BufferHandle) -> u32 {
        let index = self.next_buffer_index;
        self.next_buffer_index += 1;
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

    /// Check if an index is in the texture range (4096-8191)
    pub fn is_texture_index(&self, index: u32) -> bool {
        (4096..8192).contains(&index)
    }

    /// Check if an index is in the sampler range (8192+)
    pub fn is_sampler_index(&self, index: u32) -> bool {
        index >= 8192
    }

    /// Reverse lookup: find texture handle by its bindless index
    pub fn texture_handle_by_index(&self, index: u32) -> Option<TextureHandle> {
        self.texture_indices
            .iter()
            .find(|(_, &idx)| idx == index)
            .map(|(&handle, _)| handle)
    }

    /// Reverse lookup: find sampler handle by its bindless index
    pub fn sampler_handle_by_index(&self, index: u32) -> Option<SamplerHandle> {
        self.sampler_indices
            .iter()
            .find(|(_, &idx)| idx == index)
            .map(|(&handle, _)| handle)
    }
}

/// GPU buffer state.
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    /// The actual GPU buffer (may be heap-allocated with Private storage)
    pub buffer: MTLBuffer,
    /// Staging buffer for CPU writes (only used for heap-allocated buffers)
    pub staging_buffer: Option<MTLBuffer>,
    pub size: u64,
    /// Index in the global argument buffer (bindless)
    pub arg_buffer_index: Option<u32>,
    /// Whether this buffer was allocated from a heap (requires staging for writes)
    pub is_heap_allocated: bool,
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
    /// Argument buffer for ParameterBlock bindless rendering
    pub bindless_arg_buffer: Option<MTLBuffer>,
    /// ParameterBlock layouts from shader reflection (for filling arg buffer)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
}

/// Compute pipeline state.
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: MTLComputePipelineState,
    /// Thread group size from [numthreads(x, y, z)] attribute
    pub workgroup_size: [u32; 3],
    /// Argument buffer for ParameterBlock bindless rendering
    pub bindless_arg_buffer: Option<MTLBuffer>,
    /// ParameterBlock layouts from shader reflection (for filling arg buffer)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
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
    /// The actual GPU texture (may be heap-allocated with Private storage)
    pub texture: MTLTexture,
    /// Index in the global argument buffer (bindless)
    pub arg_buffer_index: Option<u32>,
    /// Whether this texture was allocated from a heap
    pub is_heap_allocated: bool,
}

/// GPU sampler state.
pub(crate) struct SamplerState_ {
    pub device_handle: DeviceHandle,
    pub sampler: SamplerState,
    /// Index in the global argument buffer (bindless)
    pub arg_buffer_index: Option<u32>,
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
