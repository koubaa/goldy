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
use crate::backend::FenceToken;
use crate::types::{DepthFormat, TextureFormat};
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Mutex;
// Use explicit crate path to avoid collision with our module name
use ::metal as mtl;
use mtl::{
    ArgumentEncoder, Buffer as MTLBuffer, CommandQueue,
    ComputePipelineState as MTLComputePipelineState, DepthStencilState as MTLDepthStencilState,
    Device as MTLDevice, Heap, Library, MTLPrimitiveType, MTLResourceOptions, RenderPipelineState,
    SamplerState, Texture as MTLTexture,
};

/// Maximum size of the argument buffer (supports up to 16K resources)
pub const ARGUMENT_BUFFER_SIZE: u64 = 16 * 1024 * 8; // 8 bytes per resource ID

/// Buffer slot for push constants (resource indices) in shaders.
/// Slang assigns gGoldyDynamic to [[buffer(1)]] (gGoldy ParameterBlock takes [[buffer(0)]]).
pub const PUSH_CONSTANTS_SLOT: u64 = 1;

/// Starting Metal buffer index for vertex attributes.
/// Slots 0 and 1 are reserved for the argument buffer (gGoldy) and push constants
/// (gGoldyDynamic). Vertex data must use higher indices to avoid collisions.
pub const VERTEX_BUFFER_START_SLOT: u64 = 2;

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

/// Minimum primary heap size (64 MB).
const MIN_HEAP_SIZE: u64 = 64 * 1024 * 1024;

/// Minimum overflow heap size (16 MB).
const MIN_OVERFLOW_HEAP_SIZE: u64 = 16 * 1024 * 1024;

/// Multi-heap allocator for Metal buffer allocations.
///
/// Uses a long-lived primary heap that is right-sized between frames, plus
/// ephemeral overflow heaps created on demand when the primary fills up.
/// Fragmentation is not a concern within a frame because ekrano's allocation
/// pattern is strictly monotonic (all frees are deferred until after recording).
pub(crate) struct HeapAllocator {
    device: MTLDevice,
    primary: Heap,
    overflow: Vec<Heap>,
    high_water_mark: u64,
    primary_size: u64,
    buffer_count: u32,
}

impl HeapAllocator {
    pub fn new(device: MTLDevice, primary: Heap, primary_size: u64) -> Self {
        Self {
            device,
            primary,
            overflow: Vec::new(),
            high_water_mark: 0,
            primary_size,
            buffer_count: 0,
        }
    }

    /// Allocate a buffer from the heap hierarchy.
    ///
    /// Tries primary, then the last overflow heap, then creates a new overflow.
    pub fn allocate(&mut self, size: u64, options: MTLResourceOptions) -> Option<MTLBuffer> {
        if let Some(buf) = self.primary.new_buffer(size, options) {
            self.buffer_count += 1;
            self.update_high_water_mark();
            return Some(buf);
        }

        if let Some(last) = self.overflow.last() {
            if let Some(buf) = last.new_buffer(size, options) {
                self.buffer_count += 1;
                self.update_high_water_mark();
                return Some(buf);
            }
        }

        let overflow_size = (size * 2).max(MIN_OVERFLOW_HEAP_SIZE);
        let new_heap = self.create_heap(overflow_size);
        tracing::info!(
            "Created overflow buffer heap (size={}MB, overflow_count={})",
            overflow_size / 1024 / 1024,
            self.overflow.len() + 1
        );
        let buf = new_heap.new_buffer(size, options);
        self.overflow.push(new_heap);

        if buf.is_some() {
            self.buffer_count += 1;
            self.update_high_water_mark();
        }
        buf
    }

    pub fn has_buffers(&self) -> bool {
        self.buffer_count > 0
    }

    /// Declare all buffer heaps resident for a compute encoder.
    pub fn use_heaps_for_compute(&self, encoder: &mtl::ComputeCommandEncoderRef) {
        if !self.has_buffers() {
            return;
        }
        encoder.use_heap(&self.primary);
        for heap in &self.overflow {
            encoder.use_heap(heap);
        }
    }

    /// Declare all buffer heaps resident for a render encoder at the given stages.
    pub fn use_heaps_for_render(
        &self,
        encoder: &mtl::RenderCommandEncoderRef,
        stages: mtl::MTLRenderStages,
    ) {
        if !self.has_buffers() {
            return;
        }
        encoder.use_heap_at(&self.primary, stages);
        for heap in &self.overflow {
            encoder.use_heap_at(heap, stages);
        }
    }

    /// Right-size the primary heap after a frame completes.
    ///
    /// If overflow heaps were used, the primary is replaced with a larger one
    /// sized to 1.5x the peak usage (rounded up to a power of two). Overflow
    /// heaps are dropped since all buffers have been freed by this point.
    pub fn reset_for_frame(&mut self) {
        if !self.overflow.is_empty() {
            let recommended_max = self.device.recommended_max_working_set_size();
            let new_size = (self.high_water_mark * 3 / 2)
                .next_power_of_two()
                .max(MIN_HEAP_SIZE)
                .min(recommended_max / 2);

            if new_size > self.primary_size {
                let new_primary = self.create_heap(new_size);
                tracing::info!(
                    "Resized primary buffer heap: {}MB -> {}MB (high_water_mark={}MB)",
                    self.primary_size / 1024 / 1024,
                    new_size / 1024 / 1024,
                    self.high_water_mark / 1024 / 1024,
                );
                self.primary = new_primary;
                self.primary_size = new_size;
            }

            let overflow_count = self.overflow.len();
            self.overflow.clear();
            tracing::debug!("Cleared {} overflow buffer heaps", overflow_count);
        }
        self.high_water_mark = 0;
    }

    fn update_high_water_mark(&mut self) {
        let mut total = self.primary.used_size();
        for heap in &self.overflow {
            total += heap.used_size();
        }
        self.high_water_mark = self.high_water_mark.max(total);
    }

    fn create_heap(&self, size: u64) -> Heap {
        let desc = mtl::HeapDescriptor::new();
        desc.set_size(size);
        desc.set_storage_mode(mtl::MTLStorageMode::Shared);
        desc.set_cpu_cache_mode(mtl::MTLCPUCacheMode::DefaultCache);
        desc.set_heap_type(mtl::MTLHeapType::Automatic);
        self.device.new_heap(&desc)
    }
}

/// A logical Metal device with associated resources.
///
/// Requires Argument Buffers Tier 2 (Apple Silicon, Intel 2017+, AMD 2015+).
pub(crate) struct LogicalDevice {
    pub device: MTLDevice,
    pub command_queue: CommandQueue,

    // Bindless infrastructure (always present — Tier 2 required)
    /// Multi-heap allocator for buffer allocations (grows on demand)
    pub heap_allocator: HeapAllocator,
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

    /// Register a uniform buffer (Broadcast access) - local indices 0-63 (shader slot),
    /// global indices 64-127 (argument buffer encoding offset).
    /// Returns the LOCAL index so push constants pass 0-63 to the shader
    /// (which indexes into uniformBuffers[0..63]).
    pub fn register_uniform_buffer(&mut self, handle: BufferHandle) -> u32 {
        let global_index = self.next_uniform_buffer_index;
        let local_index = global_index - MAX_RESOURCES_PER_CATEGORY;
        self.next_uniform_buffer_index += 1;
        self.buffer_indices.insert(handle, local_index);
        local_index
    }

    /// Returns the global argument buffer index for a uniform buffer
    /// (local + MAX_RESOURCES_PER_CATEGORY), needed for encoding offsets.
    pub fn uniform_global_index(local_index: u32) -> u32 {
        local_index + MAX_RESOURCES_PER_CATEGORY
    }

    /// Register a texture — returns the LOCAL index (0-63) for push constants.
    /// Use `texture_global_index()` to get the argument buffer encoding offset.
    pub fn register_texture(&mut self, handle: TextureHandle) -> u32 {
        let global_index = self.next_texture_index;
        let local_index = global_index - 2 * MAX_RESOURCES_PER_CATEGORY;
        self.next_texture_index += 1;
        self.texture_indices.insert(handle, local_index);
        local_index
    }

    /// Returns the global argument buffer index for a texture,
    /// needed for encoding offsets.
    pub fn texture_global_index(local_index: u32) -> u32 {
        local_index + 2 * MAX_RESOURCES_PER_CATEGORY
    }

    /// Register a sampler — returns the LOCAL index (0-63) for push constants.
    /// Use `sampler_global_index()` to get the argument buffer encoding offset.
    pub fn register_sampler(&mut self, handle: SamplerHandle) -> u32 {
        let global_index = self.next_sampler_index;
        let local_index = global_index - 4 * MAX_RESOURCES_PER_CATEGORY;
        self.next_sampler_index += 1;
        self.sampler_indices.insert(handle, local_index);
        local_index
    }

    /// Returns the global argument buffer index for a sampler,
    /// needed for encoding offsets.
    pub fn sampler_global_index(local_index: u32) -> u32 {
        local_index + 4 * MAX_RESOURCES_PER_CATEGORY
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
    /// Extra preprocessor defines (e.g. msaa, msaa8)
    pub defines: Vec<(String, String)>,
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
    /// Held so the GPU sampler stays resident while its ID is in the argument buffer.
    #[allow(dead_code)]
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
    /// Depth buffer (optional)
    pub depth_format: Option<DepthFormat>,
    pub depth_texture: Option<MTLTexture>,
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
    /// Pool of in-flight compute command buffers for non-blocking submit.
    /// Key: FenceToken. Removed when wait completes.
    pub compute_fence_pool: Mutex<HashMap<FenceToken, mtl::CommandBuffer>>,
    pub next_compute_fence_token: AtomicU64,
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
