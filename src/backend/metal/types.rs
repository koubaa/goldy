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
        desc.set_hazard_tracking_mode(mtl::MTLHazardTrackingMode::Tracked);
        self.device.new_heap(&desc)
    }
}

/// Multi-heap allocator for Metal texture allocations.
///
/// Mirrors `HeapAllocator` for buffers: a long-lived primary heap plus
/// ephemeral overflow heaps created on demand when the primary fills up.
pub(crate) struct TextureHeapAllocator {
    device: MTLDevice,
    primary: Heap,
    overflow: Vec<Heap>,
    #[allow(dead_code)]
    primary_size: u64,
    texture_count: u32,
}

impl TextureHeapAllocator {
    pub fn new(device: MTLDevice, primary: Heap, primary_size: u64) -> Self {
        Self {
            device,
            primary,
            overflow: Vec::new(),
            primary_size,
            texture_count: 0,
        }
    }

    /// Allocate a texture from the heap hierarchy.
    ///
    /// Tries primary, then the last overflow heap, then creates a new overflow.
    pub fn allocate(&mut self, descriptor: &mtl::TextureDescriptorRef) -> Option<MTLTexture> {
        if let Some(tex) = self.primary.new_texture(descriptor) {
            self.texture_count += 1;
            return Some(tex);
        }

        if let Some(last) = self.overflow.last() {
            if let Some(tex) = last.new_texture(descriptor) {
                self.texture_count += 1;
                return Some(tex);
            }
        }

        let alloc_size = self.device.heap_texture_size_and_align(descriptor).size;
        let overflow_size = (alloc_size * 2).max(MIN_OVERFLOW_HEAP_SIZE);
        let new_heap = self.create_heap(overflow_size);
        tracing::info!(
            "Created overflow texture heap (size={}MB, overflow_count={})",
            overflow_size / 1024 / 1024,
            self.overflow.len() + 1
        );
        let tex = new_heap.new_texture(descriptor);
        self.overflow.push(new_heap);

        if tex.is_some() {
            self.texture_count += 1;
        }
        tex
    }

    pub fn has_textures(&self) -> bool {
        self.texture_count > 0
    }

    /// Declare all texture heaps resident for a compute encoder.
    pub fn use_heaps_for_compute(&self, encoder: &mtl::ComputeCommandEncoderRef) {
        if !self.has_textures() {
            return;
        }
        encoder.use_heap(&self.primary);
        for heap in &self.overflow {
            encoder.use_heap(heap);
        }
    }

    /// Declare all texture heaps resident for a render encoder at the given stages.
    pub fn use_heaps_for_render(
        &self,
        encoder: &mtl::RenderCommandEncoderRef,
        stages: mtl::MTLRenderStages,
    ) {
        if !self.has_textures() {
            return;
        }
        encoder.use_heap_at(&self.primary, stages);
        for heap in &self.overflow {
            encoder.use_heap_at(heap, stages);
        }
    }

    fn create_heap(&self, size: u64) -> Heap {
        let desc = mtl::HeapDescriptor::new();
        desc.set_size(size);
        desc.set_storage_mode(mtl::MTLStorageMode::Shared);
        desc.set_cpu_cache_mode(mtl::MTLCPUCacheMode::DefaultCache);
        desc.set_heap_type(mtl::MTLHeapType::Automatic);
        desc.set_hazard_tracking_mode(mtl::MTLHazardTrackingMode::Tracked);
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
    /// Multi-heap allocator for texture allocations (grows on demand)
    pub texture_heap: TextureHeapAllocator,
    /// Global argument buffer containing resource IDs
    pub argument_buffer: MTLBuffer,
    /// Encoder for writing buffers to the argument buffer
    pub argument_encoder: ArgumentEncoder,
    /// Encoder for writing textures to the argument buffer
    pub texture_encoder: ArgumentEncoder,
    /// Registry tracking resource indices in the argument buffer
    pub resource_registry: ResourceRegistry,
}

/// Maximum resources per access pattern category (must match GOLDY_MAX_RESOURCES in shaders)
pub const MAX_RESOURCES_PER_CATEGORY: u32 = 64;

/// Registry for tracking bindless resource indices
///
/// The layout matches GoldyBindlessResources in bindless_resources.slang:
/// - storageBuffers[64] at indices 0-63   (Scattered access)
/// - uniformBuffers[64] at indices 64-127 (Broadcast access)
/// - textures[64] at indices 128-191      (Interpolated / Texture2D)
/// - storageImages[64] at indices 192-255 (Direct / RWTexture2D)
/// - samplers[64] at indices 256-319      (Filter config)
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    next_storage_buffer_index: u32,
    next_uniform_buffer_index: u32,
    next_texture_index: u32,
    next_storage_image_index: u32,
    next_sampler_index: u32,
    /// Free list of previously-released storage-image LOCAL indices.
    ///
    /// Populated by `release_storage_image_slot()` (used when a Surface is
    /// destroyed). `register_storage_image()` / `reserve_storage_image_slot()`
    /// pop from this list before bumping `next_storage_image_index`, so
    /// transient bindless slots (per-frame swapchain drawables) don't leak
    /// across the 64-slot storage-image window.
    free_storage_image_slots: Vec<u32>,
    /// Symmetric free list for sampled (Interpolated / `Texture2D`) slots.
    ///
    /// Populated by `release_texture_slot()` when a regular texture is
    /// destroyed; `register_texture()` consults it before bumping
    /// `next_texture_index`. Prevents slot exhaustion when textures are
    /// created and destroyed repeatedly at runtime.
    free_texture_slots: Vec<u32>,
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
            // Textures (Interpolated) at indices 128-191, bytes 1024-1535
            next_texture_index: 2 * MAX_RESOURCES_PER_CATEGORY,
            // Storage images (Direct) at indices 192-255, bytes 1536-2047
            next_storage_image_index: 3 * MAX_RESOURCES_PER_CATEGORY,
            // Samplers at indices 256-319, bytes 2048-2559
            next_sampler_index: 4 * MAX_RESOURCES_PER_CATEGORY,
            free_storage_image_slots: Vec::new(),
            free_texture_slots: Vec::new(),
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

    /// Register a sampled texture (Interpolated / Texture2D) — returns the LOCAL index (0-63).
    /// Use `texture_global_index()` to get the argument buffer encoding offset.
    ///
    /// Reuses a freed slot if available (see `release_texture_slot`).
    pub fn register_texture(&mut self, handle: TextureHandle) -> u32 {
        let local_index = if let Some(free) = self.free_texture_slots.pop() {
            free
        } else {
            let global_index = self.next_texture_index;
            let local = global_index - 2 * MAX_RESOURCES_PER_CATEGORY;
            self.next_texture_index += 1;
            local
        };
        self.texture_indices.insert(handle, local_index);
        local_index
    }

    /// Return a sampled-texture LOCAL index to the free list so it can be
    /// reused by a subsequent `register_texture`.
    pub fn release_texture_slot(&mut self, local_index: u32) {
        self.free_texture_slots.push(local_index);
    }

    /// Returns the global argument buffer index for a sampled texture.
    pub fn texture_global_index(local_index: u32) -> u32 {
        local_index + 2 * MAX_RESOURCES_PER_CATEGORY
    }

    /// Register a storage image (Direct / RWTexture2D) — returns the LOCAL index (0-63).
    /// Use `storage_image_global_index()` to get the argument buffer encoding offset.
    ///
    /// Reuses a freed slot if available (see `release_storage_image_slot`).
    pub fn register_storage_image(&mut self, handle: TextureHandle) -> u32 {
        let local_index = self.allocate_storage_image_local();
        self.texture_indices.insert(handle, local_index);
        local_index
    }

    /// Reserve a storage-image LOCAL index without binding it to a TextureHandle.
    ///
    /// Used for transient bindless slots that outlive any single `TextureHandle`
    /// but belong to a long-lived owner (e.g. a `Surface` that re-encodes a
    /// fresh drawable into the same slot every frame). The owner must release
    /// the slot via [`Self::release_storage_image_slot`] when destroyed.
    pub fn reserve_storage_image_slot(&mut self) -> u32 {
        self.allocate_storage_image_local()
    }

    /// Associate a TextureHandle with a previously-reserved storage-image
    /// LOCAL index so `texture_bindless_index()` / `Texture::bindless_index()`
    /// resolves to the right slot. Does not bump any counters.
    pub fn bind_storage_image_slot(&mut self, handle: TextureHandle, local_index: u32) {
        self.texture_indices.insert(handle, local_index);
    }

    /// Return a storage-image LOCAL index to the free list so it can be
    /// reused by a subsequent `register_storage_image` / `reserve_storage_image_slot`.
    pub fn release_storage_image_slot(&mut self, local_index: u32) {
        self.free_storage_image_slots.push(local_index);
    }

    /// Pop a free slot if any, otherwise bump the monotonic counter.
    fn allocate_storage_image_local(&mut self) -> u32 {
        if let Some(local) = self.free_storage_image_slots.pop() {
            return local;
        }
        let global_index = self.next_storage_image_index;
        let local_index = global_index - 3 * MAX_RESOURCES_PER_CATEGORY;
        self.next_storage_image_index += 1;
        local_index
    }

    /// Returns the global argument buffer index for a storage image.
    pub fn storage_image_global_index(local_index: u32) -> u32 {
        local_index + 3 * MAX_RESOURCES_PER_CATEGORY
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
    pub access: crate::types::DataAccess,
    /// Structured-buffer / uniform element stride from buffer creation.
    pub element_stride: Option<u32>,
}

/// Shader module state with cached compiled stages.
pub(crate) struct ShaderState {
    pub device_handle: DeviceHandle,
    pub slang_source: String,
    /// Search paths for Slang module resolution
    pub search_paths: Vec<String>,
    /// Extra preprocessor defines (e.g. msaa, msaa8)
    pub defines: Vec<(String, String)>,
    /// Per-shader Slang optimization level
    pub optimization_level: crate::types::OptimizationLevel,
    /// Compiled vertex shader library
    pub vertex_library: Option<Library>,
    /// Compiled fragment shader library
    pub fragment_library: Option<Library>,
    /// Compiled compute shader library
    pub compute_library: Option<Library>,
    /// Reflection data for bindless rendering (ParameterBlock layouts)
    pub reflection: Option<crate::slang::ShaderReflection>,
    /// Pending struct layout validation on first stage compile; cleared after success.
    pub layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
}

/// Graphics pipeline state.
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: RenderPipelineState,
    pub depth_stencil: Option<MTLDepthStencilState>,
    pub primitive_type: MTLPrimitiveType,
    /// Per-push-constant-slot category expected by the fragment/vertex shader,
    /// inferred from `goldy_dyn_*(N)` calls. Empty disables validation.
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
    /// Per-slot structured element stride from shader reflection (bytes), when resolved.
    pub push_constant_buffer_strides: Vec<Option<u32>>,
    /// Human-readable identifier used in category-mismatch error messages.
    pub shader_debug_name: String,
}

/// Compute pipeline state.
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: MTLComputePipelineState,
    /// Thread group size from [numthreads(x, y, z)] attribute
    pub workgroup_size: [u32; 3],
    /// Per-push-constant-slot category expected by the compute shader, inferred
    /// from the shader's `goldy_dyn_*(N)` calls during Slang compile. Empty or
    /// all-`None` disables validation. See
    /// [`crate::slang::ShaderReflection::push_constant_categories`].
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
    /// Per-slot structured element stride from shader reflection (bytes), when resolved.
    pub push_constant_buffer_strides: Vec<Option<u32>>,
    /// Human-readable identifier used in category-mismatch error messages.
    /// Defaults to `"cs_main"` for compute pipelines.
    pub shader_debug_name: String,
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
    /// LOCAL index (0..MAX_RESOURCES_PER_CATEGORY) in the texture category this
    /// texture was registered in (`storageImages[]` when `is_storage_image`,
    /// otherwise `textures[]`).
    pub arg_buffer_index: u32,
    /// Which bindless region the `arg_buffer_index` belongs to; needed at
    /// destroy time to release the slot back to the correct free list.
    pub is_storage_image: bool,
    /// When true, the slot is owned by a long-lived entity (e.g. a `Surface`
    /// that re-encodes its drawable each frame) and should NOT be released
    /// when this `TextureState` is dropped. The owner manages slot lifetime.
    pub slot_owned_externally: bool,
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
    /// The currently acquired CAMetalDrawable (set during acquire, cleared on present)
    pub current_drawable: Option<*mut std::ffi::c_void>,
    /// Texture handle for the current drawable's texture (registered for bindless access)
    pub current_texture_handle: Option<TextureHandle>,
    /// Persistent bindless storage-image LOCAL index reserved at surface create
    /// and re-encoded with the current drawable's `MTLTexture` on every `acquire`.
    ///
    /// This avoids leaking a fresh slot per frame (the storage-image window is
    /// only `MAX_RESOURCES_PER_CATEGORY` = 64 slots, so a per-frame allocation
    /// would exhaust it in ~1 second at 60 fps). Released back to the device's
    /// `ResourceRegistry` free list when the surface is destroyed.
    pub bindless_storage_slot: u32,
    /// Current present mode
    pub present_mode: crate::types::PresentMode,
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
