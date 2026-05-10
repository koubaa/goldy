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
    SamplerHandle, ShaderHandle, SurfaceHandle, TextureHandle, TransientHeapHandle,
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
    /// Slang: StructuredBuffer<T>, RWStructuredBuffer<T>
    pub const SCATTERED: u32 = 0;

    /// Broadcast access: all threads same address, read-only (enables cache optimization)
    /// Maps to: VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    /// Slang: ConstantBuffer<T>
    pub const BROADCAST: u32 = 1;

    /// Interpolated access: hardware filtering between neighbors (texture units)
    /// Maps to: VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
    /// Slang: Texture2D<T> (read with sampler)
    pub const INTERPOLATED: u32 = 2;

    /// Direct spatial access: 2D/3D indexing without filtering, read/write
    /// Maps to: VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
    /// Slang: RWTexture2D<T>
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

/// Maximum number of bindless resource indices in region A.
pub const MAX_BINDLESS_SLOTS: usize = 16;
/// Maximum number of u32 user parameters in region B.
pub const MAX_USER_SLOTS: usize = 8;
/// Total push constant size in bytes.
pub const TOTAL_PUSH_BYTES: usize = 128;

/// Packed 128-byte push constant layout.
///
/// ```text
/// Bytes  0–31:  16 × u16  bindless resource indices  (region A)
/// Bytes 32–63:  8  × u32  user parameters            (region B)
/// Bytes 64–127: 64 × u8   reserved / future           (region C)
/// ```
///
/// - Region A: bindless heap indices for `Scattered<T>`, `BufRO<T>`, textures, samplers, etc.
/// - Region B: per-dispatch scalar user params (uint, float, int …).
/// - Region C: zero-filled, reserved for future extension.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct PushLayout {
    pub bindless: [u16; MAX_BINDLESS_SLOTS],
    pub user: [u32; MAX_USER_SLOTS],
    pub _reserved: [u32; 16],
}

const _: () = assert!(std::mem::size_of::<PushLayout>() == TOTAL_PUSH_BYTES);

// Safety: PushLayout is a POD type with known layout
unsafe impl bytemuck::Pod for PushLayout {}
unsafe impl bytemuck::Zeroable for PushLayout {}

/// Registry for tracking bindless resource indices.
///
/// Each resource type has a monotonically-allocated counter plus a free list of
/// slots reclaimed by `unregister_*`. Registration prefers a recycled slot so the
/// total live + free count stays bounded by `MAX_BINDLESS_RESOURCES` rather than
/// growing unbounded with creation churn.
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    next_storage_buffer_index: u32,
    next_uniform_buffer_index: u32,
    next_sampled_texture_index: u32,
    next_storage_image_index: u32,
    next_sampler_index: u32,
    free_storage_buffer_indices: Vec<u32>,
    free_uniform_buffer_indices: Vec<u32>,
    free_sampled_texture_indices: Vec<u32>,
    free_storage_image_indices: Vec<u32>,
    free_sampler_indices: Vec<u32>,
    /// Map BufferHandle -> (bindless_index, is_storage)
    pub buffer_indices: HashMap<BufferHandle, (u32, bool)>,
    /// Map TextureHandle -> (bindless_index, is_storage_image)
    pub texture_indices: HashMap<TextureHandle, (u32, bool)>,
    pub sampler_indices: HashMap<SamplerHandle, u32>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            next_storage_buffer_index: 0,
            next_uniform_buffer_index: 0,
            next_sampled_texture_index: 0,
            next_storage_image_index: 0,
            next_sampler_index: 0,
            free_storage_buffer_indices: Vec::new(),
            free_uniform_buffer_indices: Vec::new(),
            free_sampled_texture_indices: Vec::new(),
            free_storage_image_indices: Vec::new(),
            free_sampler_indices: Vec::new(),
            buffer_indices: HashMap::new(),
            texture_indices: HashMap::new(),
            sampler_indices: HashMap::new(),
        }
    }

    pub fn register_buffer(&mut self, handle: BufferHandle, is_storage: bool) -> u32 {
        let index = if is_storage {
            self.free_storage_buffer_indices.pop().unwrap_or_else(|| {
                let idx = self.next_storage_buffer_index;
                self.next_storage_buffer_index += 1;
                idx
            })
        } else {
            self.free_uniform_buffer_indices.pop().unwrap_or_else(|| {
                let idx = self.next_uniform_buffer_index;
                self.next_uniform_buffer_index += 1;
                idx
            })
        };
        self.buffer_indices.insert(handle, (index, is_storage));
        index
    }

    pub fn register_texture(&mut self, handle: TextureHandle, is_storage_image: bool) -> u32 {
        let index = if is_storage_image {
            self.free_storage_image_indices.pop().unwrap_or_else(|| {
                let idx = self.next_storage_image_index;
                self.next_storage_image_index += 1;
                idx
            })
        } else {
            self.free_sampled_texture_indices.pop().unwrap_or_else(|| {
                let idx = self.next_sampled_texture_index;
                self.next_sampled_texture_index += 1;
                idx
            })
        };
        self.texture_indices
            .insert(handle, (index, is_storage_image));
        index
    }

    pub fn register_sampler(&mut self, handle: SamplerHandle) -> u32 {
        let index = self.free_sampler_indices.pop().unwrap_or_else(|| {
            let idx = self.next_sampler_index;
            self.next_sampler_index += 1;
            idx
        });
        self.sampler_indices.insert(handle, index);
        index
    }

    pub fn unregister_buffer(&mut self, handle: BufferHandle) {
        if let Some((index, is_storage)) = self.buffer_indices.remove(&handle) {
            if is_storage {
                self.free_storage_buffer_indices.push(index);
            } else {
                self.free_uniform_buffer_indices.push(index);
            }
        }
    }

    pub fn unregister_texture(&mut self, handle: TextureHandle) {
        if let Some((index, is_storage_image)) = self.texture_indices.remove(&handle) {
            if is_storage_image {
                self.free_storage_image_indices.push(index);
            } else {
                self.free_sampled_texture_indices.push(index);
            }
        }
    }

    pub fn unregister_sampler(&mut self, handle: SamplerHandle) {
        if let Some(index) = self.sampler_indices.remove(&handle) {
            self.free_sampler_indices.push(index);
        }
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
            reg.next_storage_buffer_index, 1,
            "storage buffer counter grew; slot recycling not working"
        );
        assert_eq!(reg.free_storage_buffer_indices.len(), 1);
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
            reg.next_uniform_buffer_index, 1,
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
        assert_eq!(reg.next_sampled_texture_index, 1);
        assert_eq!(reg.free_sampled_texture_indices.len(), 1);
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
        assert_eq!(reg.next_storage_image_index, 1);
        assert_eq!(reg.free_storage_image_indices.len(), 1);
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
        assert_eq!(reg.next_sampler_index, 1);
        assert_eq!(reg.free_sampler_indices.len(), 1);
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
            reg.next_storage_buffer_index <= LIVE as u32,
            "counter ({}) exceeded live count ({LIVE}); slot recycling broken",
            reg.next_storage_buffer_index
        );
    }
}

/// Information about a physical Vulkan device.
pub(crate) struct PhysicalDeviceInfo {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub adapter_id: u32,
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
    pub command_pool: vk::CommandPool,

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
    /// Device timeline semaphore for monotonic GPU progress ([`GpuBackend::gpu_progress`]).
    pub timeline_semaphore: vk::Semaphore,
    /// Next timeline value to signal on `timeline_semaphore`.
    pub timeline_next: u64,
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
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
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
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
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
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
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
    /// Index in the global bindless descriptor set (if bindless enabled)
    pub bindless_index: Option<u32>,
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
    /// Dedicated command buffer for the acquire-time "prep" submit that transitions
    /// the newly-acquired swapchain image from `UNDEFINED` to `GENERAL` so that compute
    /// shaders can write it via `RWTexture2D`. Consumes `image_available_semaphore`
    /// and signals `image_ready_semaphore`; later render/present submits wait on the
    /// latter.
    pub prep_command_buffer: vk::CommandBuffer,
    pub image_available_semaphore: vk::Semaphore,
    /// Signaled by the acquire-time prep submit once the swapchain image is in
    /// `GENERAL` layout. The graphics render path and the compute-only flush
    /// submit wait on this so the swapchain image is always in a
    /// compute-writable layout by the time any downstream submit touches it.
    pub image_ready_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub in_flight_fence: vk::Fence,
    /// Set after `surface_render` submits the graphics command buffer. Compute-only
    /// presentation uses a barrier submit in `present` instead (see `surface::present`).
    pub render_pass_submitted: bool,
    /// Device timeline value signaled for this frame slot's last queue submission
    /// (render or compute+present batch). Consumed when presenting.
    pub frame_timeline_value: Option<u64>,
    /// Compute command buffers recorded in [`GpuBackend::end_frame`] and submitted
    /// with the present-barrier batch in [`super::surface::present`].
    pub deferred_compute_cbs: Vec<vk::CommandBuffer>,
    /// Texture upload staging for [`deferred_compute_cbs`], merged into
    /// [`VulkanState::compute_texture_staging_pool`] under the frame's timeline
    /// signal value at present time.
    pub pending_compute_texture_staging: Vec<(vk::Buffer, vk::DeviceMemory)>,
    /// Surface texture handle whose VkImageView + bindless descriptor must stay
    /// alive until the GPU finishes this frame slot's work.  Unregistered in
    /// `acquire()` after `in_flight_fence` has been waited on.
    pub pending_surface_texture: Option<super::TextureHandle>,
}

/// Surface (swapchain) state for window presentation.
pub(crate) struct SurfaceState {
    pub device_handle: DeviceHandle,
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    /// Desired present mode (may differ from `swapchain_present_mode` until
    /// the swapchain is recreated).
    pub present_mode: vk::PresentModeKHR,
    /// Present mode the live swapchain was actually created with.  Used by
    /// `resize()` to detect when a mode change requires swapchain recreation.
    pub swapchain_present_mode: vk::PresentModeKHR,
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
    /// Transient texture handle for the currently acquired swapchain image,
    /// registered in the bindless descriptor set as a storage image so compute
    /// shaders can write directly to the swapchain image.
    pub current_texture_handle: Option<super::TextureHandle>,
    /// Compute commands accumulated for the active frame ([`GpuBackend::record_gpu_work`]).
    pub frame_pending_gpu_commands: Vec<super::GpuCommand>,
}

/// Pending buffer operations for command recording.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) struct PendingBuffer {
    pub buffer: BufferHandle,
    pub slot: u32,
    pub offset: u64,
}

/// Resource pending deferred deletion.
/// Resources are kept alive until the device timeline reaches the queued barrier
/// ([`DeletionQueue::queue`]) — see [`crate::timeline`].
#[allow(dead_code)]
pub(crate) enum PendingDeletion {
    Buffer {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        staging_buffer: Option<vk::Buffer>,
        staging_memory: Option<vk::DeviceMemory>,
    },
    Texture {
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
    /// Resources pending deletion, tagged with a [`TimelineValue`] barrier:
    /// safe to destroy once `gpu_progress >= barrier`.
    pub pending: Vec<(TimelineValue, PendingDeletion)>,
}

impl DeletionQueue {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Queue a resource for deferred deletion once the device timeline reaches `barrier`.
    pub fn queue(&mut self, barrier: TimelineValue, resource: PendingDeletion) {
        self.pending.push((barrier, resource));
    }

    /// Drop resources whose barrier has been reached (`completed` is latest GPU timeline counter).
    pub fn process_up_to(&mut self, device: &ash::Device, completed: TimelineValue) {
        let (to_delete, to_keep): (Vec<_>, Vec<_>) = self
            .pending
            .drain(..)
            .partition(|(barrier, _)| *barrier <= completed);

        self.pending = to_keep;

        for (_, resource) in to_delete {
            unsafe {
                match resource {
                    PendingDeletion::Buffer {
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
                    PendingDeletion::Texture {
                        image,
                        view,
                        memory,
                        staging_buffer,
                        staging_memory,
                    } => {
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
    }

    /// Flush all pending deletions immediately (used when destroying the device)
    pub fn flush_all(&mut self, device: &ash::Device) {
        for (_, resource) in self.pending.drain(..) {
            unsafe {
                match resource {
                    PendingDeletion::Buffer {
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
                    PendingDeletion::Texture {
                        image,
                        view,
                        memory,
                        staging_buffer,
                        staging_memory,
                    } => {
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
    }
}

impl LogicalDevice {
    /// Drop deferred resources for which the device timeline counter has caught up (non-blocking).
    pub(crate) fn process_deletion_queue_up_to_gpu_progress(&mut self) {
        let completed = unsafe {
            self.device
                .get_semaphore_counter_value(self.timeline_semaphore)
                .unwrap_or(0)
        };
        self.deletion_queue.process_up_to(&self.device, completed);
    }
}

/// Single [`vk::DeviceMemory`] block sub-allocated for transient buffers/textures.
pub(crate) struct TransientHeapEntry {
    pub device_handle: DeviceHandle,
    pub memory: vk::DeviceMemory,
    #[allow(dead_code)]
    pub size: u64,
    pub buffers: Vec<BufferHandle>,
    pub textures: Vec<TextureHandle>,
}

/// Consolidated Vulkan backend state.
/// This holds all the resources and state for the Vulkan backend.
pub(super) struct VulkanState {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_devices: Vec<PhysicalDeviceInfo>,
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
    /// Per-backend Slang compiler instance (avoids global state issues in tests)
    pub slang_compiler: crate::slang::SlangCompiler,
    /// Per-submission fences for non-blocking compute; token -> (device, VkFence, Option<VkCommandBuffer>).
    /// The command buffer is kept alive until the fence signals (Vulkan spec: must not free a pending CB).
    pub compute_fence_pool: HashMap<u64, (DeviceHandle, vk::Fence, Option<vk::CommandBuffer>)>,
    /// Texture upload staging (VkBuffer/VkDeviceMemory) freed when the matching compute fence
    /// or timeline reap runs. Keyed by `(device, timeline_or_fence_token)` so teardown can drain
    /// per device and values from different devices never collide.
    pub compute_texture_staging_pool:
        HashMap<(DeviceHandle, u64), Vec<(vk::Buffer, vk::DeviceMemory)>>,
    /// Per-device staging belts for batched WriteBuffer uploads.
    pub(super) staging_belts: HashMap<DeviceHandle, crate::backend::vulkan::staging::StagingBelt>,
    /// Command buffers to free once the device timeline reaches the given value
    /// (one submit may register multiple buffers at the same timeline point).
    pub timeline_cmd_buffers: HashMap<u64, Vec<(DeviceHandle, vk::CommandBuffer)>>,
    pub transient_heaps: HashMap<TransientHeapHandle, TransientHeapEntry>,
    pub next_transient_heap_handle: TransientHeapHandle,
}
