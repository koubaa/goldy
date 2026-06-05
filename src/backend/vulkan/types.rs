//! Vulkan backend internal types.
//!
//! This module contains all the state structs used by the Vulkan backend.
//!
//! ## Bindless Architecture
//!
//! The Vulkan backend uses descriptor indexing for bindless resource access (requires Vulkan 1.4):
//! - A global descriptor set contains arrays of all resource types
//! - Resources are registered at creation time and assigned indices
//! - Shaders access resources by index using nonuniformEXT qualifier
//! - Update-after-bind allows descriptor updates without pipeline barriers

use super::super::{
    BufferHandle, ComputePipelineHandle, DeviceHandle, PipelineHandle, RenderTargetHandle,
    SamplerHandle, ShaderHandle, SurfaceHandle, TextureHandle,
};
use crate::timeline::TimelineValue;
use crate::types::{DepthFormat, TextureFormat};
use ash::vk;
use std::collections::HashMap;

/// Maximum number of descriptors per resource type in the global bindless set
pub const MAX_BINDLESS_RESOURCES: u32 = 16384;

/// Binding indices within the global bindless descriptor set
/// Organized by ACCESS PATTERN:
///
///   0: SCATTERED - Any thread reads/writes any address. No coherence assumptions.
///   1: BROADCAST - All threads read same address. Hardware optimizes for this.
///   2: INTERPOLATED - Hardware filtering between neighboring elements (texture units).
///   3: DIRECT_SPATIAL - 2D/3D indexing without filtering. Read/write.
///   4: FILTER_CONFIG - Not data. Configuration for interpolated access.
///
/// These map to Vulkan descriptor types, but the access pattern is what matters.
pub mod bindless_bindings {
    /// Scattered access: any thread, any address, read/write
    /// Maps to: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    /// Slang: `StructuredBuffer<T>`, `RWStructuredBuffer<T>`
    pub const SCATTERED: u32 = 0;

    /// Broadcast access: all threads same address, read-only (enables cache optimization)
    /// Maps to: VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    /// Slang: `ConstantBuffer<T>`
    pub const BROADCAST: u32 = 1;

    /// Interpolated access: hardware filtering between neighbors (texture units)
    /// Maps to: VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
    /// Slang: `Texture2D<T>` (read with sampler)
    pub const INTERPOLATED: u32 = 2;

    /// Direct spatial access: 2D/3D indexing without filtering, read/write
    /// Maps to: VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
    /// Slang: `RWTexture2D<T>`
    pub const DIRECT_SPATIAL: u32 = 3;

    /// Filter configuration: settings for interpolated access (not data)
    /// Maps to: VK_DESCRIPTOR_TYPE_SAMPLER
    /// Slang: SamplerState
    pub const FILTER_CONFIG: u32 = 4;

    // Legacy aliases for legitibility to graphics programmers
    pub const STORAGE_BUFFERS: u32 = SCATTERED;
    pub const UNIFORM_BUFFERS: u32 = BROADCAST;
    pub const SAMPLED_IMAGES: u32 = INTERPOLATED;
    pub const STORAGE_IMAGES: u32 = DIRECT_SPATIAL;
    pub const SAMPLERS: u32 = FILTER_CONFIG;
}

// PushLayout and its constants live in the shared module so all three backends
// use one definition. Re-export them here so internal code keeps using the
// same unqualified names as before.
pub use super::super::shared::{PushLayout, TOTAL_PUSH_BYTES};

use super::super::shared::SlotAllocator;

/// Registry for tracking bindless resource indices.
///
/// Each resource type has a [`SlotAllocator`] that prefers recycled free-list
/// slots before minting new indices, keeping the live counter bounded by
/// `MAX_BINDLESS_RESOURCES` under create/destroy churn.
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    storage_buffer: SlotAllocator,
    uniform_buffer: SlotAllocator,
    sampled_texture: SlotAllocator,
    storage_image: SlotAllocator,
    sampler: SlotAllocator,
    /// Map BufferHandle -> (bindless_index, is_storage)
    pub buffer_indices: HashMap<BufferHandle, (u32, bool)>,
    /// Map TextureHandle -> (bindless_index, is_storage_image)
    pub texture_indices: HashMap<TextureHandle, (u32, bool)>,
    pub sampler_indices: HashMap<SamplerHandle, u32>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_buffer(&mut self, handle: BufferHandle, is_storage: bool) -> u32 {
        let index = if is_storage {
            self.storage_buffer.alloc()
        } else {
            self.uniform_buffer.alloc()
        };
        self.buffer_indices.insert(handle, (index, is_storage));
        index
    }

    pub fn register_texture(&mut self, handle: TextureHandle, is_storage_image: bool) -> u32 {
        let index = if is_storage_image {
            self.storage_image.alloc()
        } else {
            self.sampled_texture.alloc()
        };
        self.texture_indices
            .insert(handle, (index, is_storage_image));
        index
    }

    pub fn register_sampler(&mut self, handle: SamplerHandle) -> u32 {
        let index = self.sampler.alloc();
        self.sampler_indices.insert(handle, index);
        index
    }

    pub fn unregister_buffer(&mut self, handle: BufferHandle) {
        if let Some((index, is_storage)) = self.buffer_indices.remove(&handle) {
            if is_storage {
                self.storage_buffer.free(index);
            } else {
                self.uniform_buffer.free(index);
            }
        }
    }

    pub fn unregister_texture(&mut self, handle: TextureHandle) {
        if let Some((index, is_storage_image)) = self.texture_indices.remove(&handle) {
            if is_storage_image {
                self.storage_image.free(index);
            } else {
                self.sampled_texture.free(index);
            }
        }
    }

    pub fn unregister_sampler(&mut self, handle: SamplerHandle) {
        if let Some(index) = self.sampler_indices.remove(&handle) {
            self.sampler.free(index);
        }
    }

    /// Number of available (allocatable) slots in the given category.
    pub fn available_slots(&self, category: crate::types::ResourceCategory) -> u32 {
        let allocator = match category {
            crate::types::ResourceCategory::Scattered => &self.storage_buffer,
            crate::types::ResourceCategory::Broadcast => &self.uniform_buffer,
            crate::types::ResourceCategory::Texture => &self.sampled_texture,
            crate::types::ResourceCategory::StorageImage => &self.storage_image,
            crate::types::ResourceCategory::Sampler => &self.sampler,
        };
        MAX_BINDLESS_RESOURCES.saturating_sub(allocator.live_count())
    }
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    /// Simulate the per-frame create/destroy churn that ekrano generates for transient
    /// pool-view storage buffers. The counter must stay bounded — well below
    /// MAX_BINDLESS_RESOURCES — even after far more iterations than the heap limit.
    #[test]
    fn storage_buffer_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as BufferHandle;
            reg.register_buffer(handle, true);
            reg.unregister_buffer(handle);
        }
        assert_eq!(
            reg.storage_buffer.next_fresh(),
            1,
            "storage buffer counter grew; slot recycling not working"
        );
        assert_eq!(reg.storage_buffer.free_count(), 1);
    }

    /// Uniform buffer (non-storage) slots must recycle independently.
    #[test]
    fn uniform_buffer_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as BufferHandle;
            reg.register_buffer(handle, false);
            reg.unregister_buffer(handle);
        }
        assert_eq!(
            reg.uniform_buffer.next_fresh(),
            1,
            "uniform buffer counter grew; slot recycling not working"
        );
    }

    /// Sampled textures (non-storage) must recycle their indices.
    #[test]
    fn sampled_texture_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as TextureHandle;
            reg.register_texture(handle, false);
            reg.unregister_texture(handle);
        }
        assert_eq!(reg.sampled_texture.next_fresh(), 1);
        assert_eq!(reg.sampled_texture.free_count(), 1);
    }

    /// Storage images (RWTexture2D) must recycle their indices.
    #[test]
    fn storage_image_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..50_000u64 {
            let handle = i as TextureHandle;
            reg.register_texture(handle, true);
            reg.unregister_texture(handle);
        }
        assert_eq!(reg.storage_image.next_fresh(), 1);
        assert_eq!(reg.storage_image.free_count(), 1);
    }

    /// Sampler slots must recycle independently.
    #[test]
    fn sampler_slots_recycled_under_churn() {
        let mut reg = ResourceRegistry::new();
        for i in 0..5_000u64 {
            let handle = i as SamplerHandle;
            reg.register_sampler(handle);
            reg.unregister_sampler(handle);
        }
        assert_eq!(reg.sampler.next_fresh(), 1);
        assert_eq!(reg.sampler.free_count(), 1);
    }

    /// Simultaneously-live resources must receive distinct indices.
    #[test]
    fn live_resources_get_distinct_indices() {
        let mut reg = ResourceRegistry::new();
        const N: u64 = 64;
        let mut indices: Vec<u32> = (0..N)
            .map(|i| reg.register_buffer(i as BufferHandle, true))
            .collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(
            indices.len(),
            N as usize,
            "duplicate indices assigned to live resources"
        );
    }

    /// The high-water mark of the monotonic counter must never exceed the number of
    /// concurrently-live resources, i.e. freed slots are reused before fresh ones are minted.
    #[test]
    fn high_water_mark_bounded_by_live_count() {
        let mut reg = ResourceRegistry::new();
        const LIVE: u64 = 8;
        const ROUNDS: u64 = 10_000;

        for i in 0..LIVE {
            reg.register_buffer(i as BufferHandle, true);
        }
        for i in LIVE..LIVE + ROUNDS {
            reg.unregister_buffer((i - LIVE) as BufferHandle);
            reg.register_buffer(i as BufferHandle, true);
        }
        assert!(
            reg.storage_buffer.next_fresh() <= LIVE as u32,
            "counter ({}) exceeded live count ({LIVE}); slot recycling broken",
            reg.storage_buffer.next_fresh()
        );
    }
}

/// Information about a physical Vulkan device.
pub(crate) struct PhysicalDeviceInfo {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub adapter_id: u32,
    /// From physical-device features at enumeration (immutable for this adapter).
    pub supports_sparse_buffer: bool,
    /// From [`vk::PhysicalDeviceLimits::timestamp_compute_and_graphics`].
    pub vk_timestamp_compute_and_graphics: bool,
    pub vk_timestamp_period_ns: f32,
}

/// Per-context async submission stream (timeline, poller, command pool).
pub(crate) struct SubmissionContext {
    pub device: super::DeviceHandle,
    pub timeline_semaphore: vk::Semaphore,
    /// Last device-global seq value submitted on this context.
    pub last_submitted_seq: u64,
    pub signal_queue: std::sync::Arc<crate::signal::SignalQueue>,
    pub fence_shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub fence_thread: Option<std::thread::JoinHandle<()>>,
    pub command_pool: vk::CommandPool,
    pub free_cmd_buffers: Vec<vk::CommandBuffer>,
    pub retained_compute_cb: Option<RetainedVkCb>,
    /// Command buffers to free once this context's timeline reaches the key.
    pub timeline_cmd_buffers: std::collections::HashMap<u64, Vec<vk::CommandBuffer>>,
}

/// A logical Vulkan device with associated resources.
pub(crate) struct LogicalDevice {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    #[allow(dead_code)]
    pub adapter_id: u32,
    pub queue: vk::Queue,
    #[allow(dead_code)]
    pub queue_family: u32,
    /// Queue used for [`vk::Device::queue_bind_sparse`] (often same as graphics).
    pub sparse_binding_queue: vk::Queue,
    pub command_pool: vk::CommandPool,

    /// Host-visible oversize pools use dense allocations; device-local oversize may use sparse binding.
    pub supports_sparse_buffer: bool,
    /// Sparse **buffer** binding alignment from [`vk::MemoryRequirements::alignment`] (typically 64 KiB).
    pub sparse_buffer_block_size: u64,
    #[allow(dead_code)] // captured at device init for diagnostics / future use
    pub sparse_memory_type_index: u32,
    /// Sub-allocated DEVICE_LOCAL pages for [`super::sparse::SparsePagePool`].
    pub sparse_page_pool: Option<super::sparse::SparsePagePool>,

    // Vulkan 1.4 core via KHR extension loaders (ash 0.38 doesn't have core 1.4 wrappers yet)
    pub map_memory2: ash::khr::map_memory2::Device,

    // Bindless infrastructure
    /// Global descriptor pool for bindless resources
    pub bindless_descriptor_pool: Option<vk::DescriptorPool>,
    /// Global descriptor set layout for bindless resources
    pub bindless_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    /// Global descriptor set containing all bindless resources
    pub bindless_descriptor_set: Option<vk::DescriptorSet>,
    /// Pipeline layout for bindless rendering (includes the global set)
    pub bindless_pipeline_layout: Option<vk::PipelineLayout>,
    /// Registry tracking resource indices in the global descriptor set
    pub resource_registry: ResourceRegistry,
    /// Deferred deletion queue for resources that are still in-flight
    pub deletion_queue: DeletionQueue,
    /// Device-global submission sequence (shared value space; contexts signal their own semaphores).
    pub timeline_next: u64,
    /// Minimum completed horizon after a context is destroyed (never lowers `device_retired`).
    pub retired_floor: u64,

    /// Optional driver pipeline cache persisted to disk (`~/.cache/goldy/pipeline_cache_<adapter>.bin`).
    pub pipeline_cache: vk::PipelineCache,

    /// Timestamp query support (`VkPhysicalDeviceLimits::timestamp_compute_and_graphics`).
    pub vk_timestamp_compute_and_graphics: bool,
    pub vk_timestamp_period_ns: f32,
}

/// A Vulkan command buffer retained for resubmission.
pub(crate) struct RetainedVkCb {
    /// Opaque key used to detect staleness (binding fingerprint).
    pub fingerprint: u64,
    /// The retained `VkCommandBuffer` (in executable state when GPU has completed).
    pub command_buffer: vk::CommandBuffer,
}

impl LogicalDevice {
    /// `vkMapMemory2KHR` — core in Vulkan 1.4. Struct-based API that replaces `vkMapMemory`.
    pub unsafe fn map_memory2(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> ash::prelude::VkResult<*mut core::ffi::c_void> {
        let info = vk::MemoryMapInfoKHR::default()
            .memory(memory)
            .offset(offset)
            .size(size);
        let mut ptr = core::ptr::null_mut();
        (self.map_memory2.fp().map_memory2_khr)(self.device.handle(), &info, &mut ptr)
            .result_with_success(ptr)
    }

    /// `vkUnmapMemory2KHR` — core in Vulkan 1.4. Returns `VkResult` (unlike legacy `vkUnmapMemory`).
    pub unsafe fn unmap_memory2(&self, memory: vk::DeviceMemory) -> ash::prelude::VkResult<()> {
        let info = vk::MemoryUnmapInfoKHR::default().memory(memory);
        (self.map_memory2.fp().unmap_memory2_khr)(self.device.handle(), &info).result()
    }
}

/// GPU buffer state.
#[derive(Clone)]
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// Logical byte size (descriptor range, bounds checks).
    pub size: u64,
    /// Bytes reserved in the VkBuffer (`>= size` when oversize allocations are used).
    pub allocation_size: u64,
    /// Index in the global bindless descriptor set (if bindless enabled)
    pub bindless_index: Option<u32>,
    /// Whether this is a storage buffer (vs uniform buffer)
    #[allow(dead_code)]
    pub is_storage: bool,
    /// Structured-buffer / uniform element stride from buffer creation (retained for future validation).
    #[allow(dead_code)]
    pub element_stride: Option<u32>,
    /// HOST_VISIBLE staging buffer for DEVICE_LOCAL storage buffers (CPU upload/readback)
    pub staging_buffer: Option<vk::Buffer>,
    pub staging_memory: Option<vk::DeviceMemory>,
    /// If true, this is a view into another buffer — don't free the VkBuffer/memory on destroy.
    pub is_view: bool,
    /// Set when the buffer was created with [`crate::types::BufferFlags::CPU_READABLE`]:
    /// persistent host mapping of the entire buffer for CPU read/write.
    /// Opaque address of the persistent `map_memory2` region for `CPU_READABLE` storage.
    /// Stored as [`usize`] so `BufferState` is `Send`/`Sync` for `GpuBackend` (raw pointers and
    /// `NonNull` are not in this environment).
    pub host_mapped: Option<usize>,
    /// Mirror of create-time flags.
    pub flags: crate::types::BufferFlags,
    /// Sub-allocated from [`crate::backend::GpuBackend::create_transient_heap`]; `memory` is shared.
    pub transient_heap_suballoc: bool,
    /// Byte offset in parent for buffer views; [`None`] for root buffers.
    pub view_byte_offset: Option<u64>,
    /// `true` for device-local storage using sparse binding (`VkBuffer` sparse flags; no single `memory`).
    pub is_sparse: bool,
    /// Alignment from [`vk::MemoryRequirements::alignment`] when [`Self::is_sparse`]; `0` otherwise.
    pub sparse_block_size: u64,
    /// Per sparse page: physical backing from the page pool (`None` = unbound tile).
    pub sparse_pages: Vec<Option<(vk::DeviceMemory, vk::DeviceSize)>>,
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
    /// Cached compiled vertex shader module
    pub vertex_module: Option<vk::ShaderModule>,
    /// Cached compiled fragment shader module
    pub fragment_module: Option<vk::ShaderModule>,
    /// Cached compiled compute shader module
    pub compute_module: Option<vk::ShaderModule>,
    /// Reflection data for bindless rendering (ParameterBlock layouts)
    pub reflection: Option<crate::slang::ShaderReflection>,
    /// Pending struct layout validation on first stage compile; cleared after success.
    pub layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
}

/// Graphics pipeline state.
#[allow(dead_code)]
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    /// Whether this pipeline owns its layout (false when using bindless_pipeline_layout)
    pub owns_layout: bool,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
    /// Per push-constant slot category expectations from shader analysis.
    pub push_constant_categories: Vec<Option<crate::types::ResourceCategory>>,
    /// Per push-constant slot expected element stride (bytes) from reflection.
    pub binding_element_strides: Vec<Option<u32>>,
    /// Human-readable identifier for debugging.
    pub shader_debug_name: String,
}

/// Compute pipeline state.
#[allow(dead_code)]
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    /// Whether this pipeline owns its layout (false when using bindless_pipeline_layout)
    pub owns_layout: bool,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
    /// Per push-constant slot category expectations from shader analysis.
    pub push_constant_categories: Vec<Option<crate::types::ResourceCategory>>,
    /// Per push-constant slot expected element stride (bytes) from reflection.
    pub binding_element_strides: Vec<Option<u32>>,
    /// Human-readable identifier for debugging.
    pub shader_debug_name: String,
}

/// GPU render target state with optional staging for CPU readback.
pub(crate) struct RenderTargetState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// GPU-only render target image
    pub image: vk::Image,
    pub image_memory: vk::DeviceMemory,
    pub image_view: vk::ImageView,
    /// Depth buffer (optional)
    pub depth_format: Option<DepthFormat>,
    pub depth_image: Option<vk::Image>,
    pub depth_memory: Option<vk::DeviceMemory>,
    pub depth_view: Option<vk::ImageView>,
    /// Staging buffer for CPU readback (lazy-created on first read)
    pub staging_buffer: Option<vk::Buffer>,
    pub staging_memory: Option<vk::DeviceMemory>,
    /// Command buffer for rendering
    pub command_buffer: vk::CommandBuffer,
    /// Track if we've rendered (for readback validation)
    pub has_rendered: bool,
}

/// GPU texture state.
pub(crate) struct TextureState {
    pub device_handle: DeviceHandle,
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub format: TextureFormat,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    /// Staging buffer for texture uploads
    pub staging_buffer: Option<vk::Buffer>,
    pub staging_memory: Option<vk::DeviceMemory>,
    /// Index in the global bindless descriptor set (if bindless enabled).
    /// For `DirectInterpolated` textures this is the storage-image (UAV) slot.
    pub bindless_index: Option<u32>,
    /// For `TextureKind::DirectInterpolated` textures, the sampled-texture (SRV) slot.
    pub sampled_bindless_index: Option<u32>,
    /// Current image layout (for subregion writes / transitions)
    pub current_layout: vk::ImageLayout,
    /// Sub-allocated from a transient heap; `memory` is shared with the heap.
    pub transient_heap_suballoc: bool,
}

/// GPU sampler state.
pub(crate) struct SamplerState {
    pub device_handle: DeviceHandle,
    pub sampler: vk::Sampler,
    /// Index in the global bindless descriptor set (if bindless enabled)
    pub bindless_index: Option<u32>,
}

/// Maximum number of frames that can be in-flight at once.
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// Per-frame synchronization resources for proper swapchain pipelining.
pub(crate) struct FrameSync {
    pub command_buffer: vk::CommandBuffer,
    /// Recorded fresh each present with ONE_TIME_SUBMIT to copy the per-slot
    /// scratch texture into the acquired swapchain image (scratch→swapchain
    /// blit path). This is WSI work, separate from runtime-owned compute
    /// submissions.
    pub copy_command_buffer: vk::CommandBuffer,
    pub image_available_semaphore: vk::Semaphore,
    /// Signaled by Submit 1 (render work) and consumed by Submit 2 (present
    /// barrier) in the *graphics* (render-pass) present path only.
    pub work_done_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub in_flight_fence: vk::Fence,
    /// True when `in_flight_fence` has been submitted to the queue and not yet waited on.
    /// Only the render (graphics) path submits this fence; the compute path does not.
    /// `acquire()` must check this before calling `wait_for_fences` to avoid hanging
    /// when the slot was last used via the compute path.
    pub fence_pending: bool,
    /// Set after `surface_render` submits the graphics command buffer. Compute-only
    /// presentation uses the scratch-texture copy path in `present` instead (see
    /// `surface::present`).
    pub render_pass_submitted: bool,
    /// Device timeline value signaled for this frame slot's final frame work.
    /// Consumed when presenting.
    pub frame_timeline_value: Option<u64>,
    /// Persistent cache of the last compute timeline value signaled for this frame slot.
    /// Unlike `frame_timeline_value`, this is not consumed when presenting.
    pub last_compute_timeline_value: u64,
    /// Timeline value signaled by the WSI copy submit.  Used by `acquire()` to
    /// ensure the scratch texture's copy-read has completed before the slot is
    /// reused for new compute writes.  Higher than `frame_timeline_value` when
    /// the scratch-texture path is active.
    pub copy_timeline_value: Option<u64>,
}

/// A device-local scratch texture used as the compute render target for one
/// frame slot.  Compute shaders write to this image each frame; at present
/// time it is copied into the acquired swapchain image, decoupling the frame's
/// render phase from WSI image availability entirely.
pub(crate) struct ScratchTextureSlot {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    /// Bindless storage-image handle.  Equal to the index in `TextureState`
    /// so compute shaders can write to this slot without knowing about WSI.
    pub texture_handle: super::TextureHandle,
}

/// Surface (swapchain) state for window presentation.
pub(crate) struct SurfaceState {
    pub device_handle: DeviceHandle,
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    /// One pre-recorded command buffer per swapchain image. Each records a
    /// single `UNDEFINED → GENERAL` barrier for that image. Always using
    /// `UNDEFINED` as `old_layout` lets the driver discard prior contents and
    /// avoids per-frame re-recording. Rebuilt on swapchain recreation.
    pub swapchain_prep_command_buffers: Vec<vk::CommandBuffer>,
    /// Pre-recorded `GENERAL → PRESENT_SRC_KHR` barrier, one per swapchain image.
    /// Used as the second submit in the compute path so the fence can fire
    /// before the present barrier executes.
    pub swapchain_compute_present_command_buffers: Vec<vk::CommandBuffer>,
    /// Pre-recorded `COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR` barrier, one per
    /// swapchain image.  Used as the second submit in the graphics (render) path.
    pub swapchain_render_present_command_buffers: Vec<vk::CommandBuffer>,
    /// Persistent bindless texture handles, one per swapchain image, registered
    /// at swapchain creation. Avoids per-frame `vkCreateImageView` /
    /// `vkUpdateDescriptorSets` / `vkDestroyImageView`. Rebuilt on resize.
    pub swapchain_texture_handles: Vec<super::TextureHandle>,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    /// Desired present mode. When changed by `set_present_mode`, `present_mode_dirty`
    /// is set so `resize()` knows to recreate the swapchain even if dimensions match.
    pub present_mode: vk::PresentModeKHR,
    /// True when `present_mode` was updated but the swapchain has not yet been
    /// recreated with the new mode.
    pub present_mode_dirty: bool,
    /// Depth buffer (when depth_format is Some)
    pub depth_format: Option<DepthFormat>,
    pub depth_image: Option<vk::Image>,
    pub depth_memory: Option<vk::DeviceMemory>,
    pub depth_view: Option<vk::ImageView>,
    /// Current frame index (0..MAX_FRAMES_IN_FLIGHT)
    pub current_frame: usize,
    /// Currently acquired swapchain image index
    pub current_image_index: Option<u32>,
    /// Per-frame synchronization resources
    pub frame_sync: Vec<FrameSync>,
    /// One scratch texture slot per frame slot (`MAX_FRAMES_IN_FLIGHT` entries).
    /// Each slot is lazily created on first acquire and reused every frame.
    /// Compute shaders write to the slot for `current_frame`; at present time
    /// that image is copied into the acquired swapchain image, completely
    /// decoupling GPU rendering from WSI image availability.
    ///
    /// NOTE: making this opt-in / configurable (e.g. for a latency-sensitive
    /// mode that trades throughput for lower frame latency) is future work.
    /// The current design is a max-throughput strategy.
    pub scratch_texture_slots: Vec<Option<ScratchTextureSlot>>,
    /// Handle of the scratch texture for the current frame slot — what compute
    /// shaders write to.  Cleared at present; the underlying slot persists.
    pub current_texture_handle: Option<super::TextureHandle>,
    /// Compute commands accumulated for the active frame ([`GpuBackend::record_gpu_work`](crate::backend::GpuBackend::record_gpu_work)).
    pub frame_pending_gpu_commands: Vec<super::GpuCommand>,
    /// Drawables acquired or presented but not yet returned to the swapchain pool.
    pub pending_acquire_count: u32,
    /// `(image_index, timeline)` pairs waiting for GPU completion before `SwapchainReturned`.
    pub pending_swapchain_returns: Vec<(u32, crate::timeline::TimelineValue)>,
}

/// Pending buffer operations for command recording.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct PendingBuffer {
    pub buffer: BufferHandle,
    pub slot: u32,
    pub offset: u64,
}

/// Deferred sparse teardown (after timeline barrier): unbind + pool recycle + buffer destroy.
#[allow(dead_code)]
pub(crate) struct SparseBufferTeardown {
    pub allocation_size: u64,
    pub block_size: u64,
    pub binds: Vec<(u64, vk::DeviceMemory, vk::DeviceSize)>,
}

/// Resource pending deferred deletion.
/// Resources are kept alive until the device timeline reaches the queued barrier
/// ([`DeletionQueue::queue`]) — see [`crate::timeline`].
#[allow(dead_code)]
pub(crate) enum PendingDeletion {
    Buffer {
        buffer_handle: BufferHandle,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        staging_buffer: Option<vk::Buffer>,
        staging_memory: Option<vk::DeviceMemory>,
        /// When set, unbind sparse tiles and recycle before destroying `buffer`.
        sparse_teardown: Option<SparseBufferTeardown>,
    },
    /// A buffer view whose Vk resources belong to the parent buffer; only the
    /// bindless descriptor index needs deregistration.
    BufferView {
        buffer_handle: BufferHandle,
    },
    /// Previous Vk allocation after [`super::buffer::resize`]; the logical
    /// [`BufferHandle`] stays registered — destroy GPU objects only.
    ReplacedBufferGpu {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        staging_buffer: Option<vk::Buffer>,
        staging_memory: Option<vk::DeviceMemory>,
    },
    /// Previous sparse buffer after resize / migration: unbind pages, recycle pool memory, destroy.
    ReplacedSparseBufferGpu {
        buffer: vk::Buffer,
        allocation_size: u64,
        block_size: u64,
        binds: Vec<(u64, vk::DeviceMemory, vk::DeviceSize)>,
        staging_buffer: Option<vk::Buffer>,
        staging_memory: Option<vk::DeviceMemory>,
    },
    Texture {
        texture_handle: TextureHandle,
        image: vk::Image,
        view: vk::ImageView,
        memory: vk::DeviceMemory,
        staging_buffer: Option<vk::Buffer>,
        staging_memory: Option<vk::DeviceMemory>,
    },
    Sampler {
        sampler: vk::Sampler,
    },
}

/// Deferred deletion queue for a device.
/// Tracks resources waiting to be deleted after GPU work completes.
pub(crate) struct DeletionQueue {
    inner: super::super::shared::DeferredQueue<TimelineValue, PendingDeletion>,
}

impl DeletionQueue {
    pub fn new() -> Self {
        Self {
            inner: super::super::shared::DeferredQueue::new(),
        }
    }

    /// Queue a resource for deferred deletion once the device timeline reaches `barrier`.
    pub fn queue(&mut self, barrier: TimelineValue, resource: PendingDeletion) {
        self.inner.push(barrier, resource);
    }

    pub fn drain_up_to(&mut self, completed: TimelineValue) -> Vec<PendingDeletion> {
        self.inner.drain_up_to(completed)
    }

    /// Drain all pending deletions (device teardown).
    pub fn flush_all_drain(&mut self) -> Vec<PendingDeletion> {
        self.inner.flush_all().collect()
    }

    /// Number of resources currently pending deletion.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Deferred-delete one entry ([`SparseBufferTeardown`] needs [`LogicalDevice`] for the page pool).
pub(crate) fn destroy_pending_deletion(ld: &mut LogicalDevice, resource: PendingDeletion) {
    let device = &ld.device;
    let registry = &mut ld.resource_registry;
    let bind_queue = ld.sparse_binding_queue;

    unsafe {
        match resource {
            PendingDeletion::Buffer {
                buffer_handle,
                buffer,
                memory,
                staging_buffer,
                staging_memory,
                sparse_teardown,
            } => {
                registry.unregister_buffer(buffer_handle);
                if let Some(td) = sparse_teardown {
                    if !td.binds.is_empty() {
                        let mut sparse_binds = Vec::with_capacity(td.binds.len());
                        for (res_off, _mem, _mem_off) in &td.binds {
                            sparse_binds.push(
                                vk::SparseMemoryBind::default()
                                    .resource_offset(*res_off)
                                    .size(td.block_size)
                                    .memory(vk::DeviceMemory::default())
                                    .memory_offset(0)
                                    .flags(vk::SparseMemoryBindFlags::empty()),
                            );
                        }
                        if let Err(e) = super::sparse::queue_bind_sparse_sync(
                            device,
                            bind_queue,
                            buffer,
                            &sparse_binds,
                        ) {
                            tracing::warn!(?e, "sparse unbind on buffer destroy failed");
                        }
                        for (_res_off, mem, mem_off) in &td.binds {
                            if let Some(pool) = ld.sparse_page_pool.as_mut() {
                                pool.free_page(*mem, *mem_off);
                            }
                        }
                    }
                    device.destroy_buffer(buffer, None);
                } else {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                if let Some(buf) = staging_buffer {
                    device.destroy_buffer(buf, None);
                }
                if let Some(mem) = staging_memory {
                    device.free_memory(mem, None);
                }
            }
            PendingDeletion::BufferView { buffer_handle } => {
                registry.unregister_buffer(buffer_handle);
            }
            PendingDeletion::ReplacedBufferGpu {
                buffer,
                memory,
                staging_buffer,
                staging_memory,
            } => {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                if let Some(buf) = staging_buffer {
                    device.destroy_buffer(buf, None);
                }
                if let Some(mem) = staging_memory {
                    device.free_memory(mem, None);
                }
            }
            PendingDeletion::ReplacedSparseBufferGpu {
                buffer,
                allocation_size: _,
                block_size,
                binds,
                staging_buffer,
                staging_memory,
            } => {
                if !binds.is_empty() {
                    let mut sparse_binds = Vec::with_capacity(binds.len());
                    for (res_off, _mem, _mem_off) in &binds {
                        sparse_binds.push(
                            vk::SparseMemoryBind::default()
                                .resource_offset(*res_off)
                                .size(block_size)
                                .memory(vk::DeviceMemory::default())
                                .memory_offset(0)
                                .flags(vk::SparseMemoryBindFlags::empty()),
                        );
                    }
                    if let Err(e) = super::sparse::queue_bind_sparse_sync(
                        device,
                        bind_queue,
                        buffer,
                        &sparse_binds,
                    ) {
                        tracing::warn!(?e, "sparse unbind on replaced buffer failed");
                    }
                    for (_res_off, mem, mem_off) in &binds {
                        if let Some(pool) = ld.sparse_page_pool.as_mut() {
                            pool.free_page(*mem, *mem_off);
                        }
                    }
                }
                device.destroy_buffer(buffer, None);
                if let Some(buf) = staging_buffer {
                    device.destroy_buffer(buf, None);
                }
                if let Some(mem) = staging_memory {
                    device.free_memory(mem, None);
                }
            }
            PendingDeletion::Texture {
                texture_handle,
                image,
                view,
                memory,
                staging_buffer,
                staging_memory,
            } => {
                registry.unregister_texture(texture_handle);
                device.destroy_image_view(view, None);
                device.destroy_image(image, None);
                device.free_memory(memory, None);
                if let Some(buf) = staging_buffer {
                    device.destroy_buffer(buf, None);
                }
                if let Some(mem) = staging_memory {
                    device.free_memory(mem, None);
                }
            }
            PendingDeletion::Sampler { sampler } => {
                device.destroy_sampler(sampler, None);
            }
        }
    }
}

impl LogicalDevice {
    /// Drop deferred resources whose barrier is `<= completed` (device-global retirement horizon).
    pub(crate) fn process_deletion_queue_up_to(&mut self, completed: u64) {
        let drained = self.deletion_queue.drain_up_to(completed);
        for r in drained {
            destroy_pending_deletion(self, r);
        }
    }
}

/// Consolidated Vulkan backend state.
/// This holds all the resources and state for the Vulkan backend.
pub(super) struct VulkanState {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_devices: Vec<PhysicalDeviceInfo>,
    pub devices: HashMap<DeviceHandle, LogicalDevice>,
    pub next_device_handle: DeviceHandle,
    pub contexts: HashMap<super::ContextHandle, SubmissionContext>,
    pub next_context_id: super::ContextHandle,
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
    /// Per-backend Slang compiler instance (avoids global state issues in tests)
    pub slang_compiler: crate::slang::SlangCompiler,
    /// Per-submission fences for non-blocking compute; token -> (device, `VkFence`, `Option<VkCommandBuffer>`).
    /// The command buffer is kept alive until the fence signals (Vulkan spec: must not free a pending CB).
    pub compute_fence_pool: HashMap<u64, (DeviceHandle, vk::Fence, Option<vk::CommandBuffer>)>,
    /// Per-device pools that recycle texture-upload staging buffers across frames.
    /// Entries are released with a GPU timeline value and reclaimed once that
    /// timeline completes, avoiding per-frame vkAllocateMemory / vkFreeMemory.
    pub texture_staging_pools:
        HashMap<DeviceHandle, crate::backend::vulkan::staging::TextureStagingPool>,
    /// Per-device staging belts for batched WriteBuffer uploads.
    pub(super) staging_belts: HashMap<DeviceHandle, crate::backend::vulkan::staging::StagingBelt>,
    /// Set to `true` when any Vulkan call returns `VK_ERROR_DEVICE_LOST`.
    /// Polled by [`GpuBackend::is_device_lost`] without holding any lock.
    pub device_lost: std::sync::atomic::AtomicBool,
}
