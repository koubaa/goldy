//! Vulkan backend internal types.
//!
//! This module contains all the state structs used by the Vulkan backend.
//!
//! ## Bindless Architecture
//!
//! The Vulkan backend uses descriptor indexing (Vulkan 1.2+) for bindless resource access:
//! - A global descriptor set contains arrays of all resource types
//! - Resources are registered at creation time and assigned indices
//! - Shaders access resources by index using nonuniformEXT qualifier
//! - Update-after-bind allows descriptor updates without pipeline barriers

use super::super::{BufferHandle, DeviceHandle, SamplerHandle, TextureHandle};
use crate::types::{DepthFormat, TextureFormat};
use ash::vk;
use std::collections::HashMap;

/// Maximum number of descriptors per resource type in the global bindless set
pub const MAX_BINDLESS_RESOURCES: u32 = 16384;

/// Descriptor set index for the global bindless set
#[allow(dead_code)]
pub const BINDLESS_SET_INDEX: u32 = 0;

/// Binding indices within the global bindless descriptor set
pub mod bindless_bindings {
    pub const STORAGE_BUFFERS: u32 = 0;
    pub const UNIFORM_BUFFERS: u32 = 1;
    pub const SAMPLED_IMAGES: u32 = 2;
    pub const SAMPLERS: u32 = 3;
}

/// Maximum number of resource indices in push constants
pub const MAX_PUSH_CONSTANT_INDICES: usize = 16;

/// Push constants for passing bindless resource indices to shaders.
/// This is used to tell shaders which indices in the global descriptor arrays to access.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct BindlessIndices {
    /// Resource indices (buffers, textures, samplers packed sequentially)
    pub indices: [u32; MAX_PUSH_CONSTANT_INDICES],
}

// Safety: BindlessIndices is a POD type with known layout
unsafe impl bytemuck::Pod for BindlessIndices {}
unsafe impl bytemuck::Zeroable for BindlessIndices {}

/// Registry for tracking bindless resource indices
#[derive(Default)]
pub(crate) struct ResourceRegistry {
    next_storage_buffer_index: u32,
    next_uniform_buffer_index: u32,
    next_texture_index: u32,
    next_sampler_index: u32,
    pub buffer_indices: HashMap<BufferHandle, u32>,
    pub texture_indices: HashMap<TextureHandle, u32>,
    pub sampler_indices: HashMap<SamplerHandle, u32>,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            next_storage_buffer_index: 0,
            next_uniform_buffer_index: 0,
            next_texture_index: 0,
            next_sampler_index: 0,
            buffer_indices: HashMap::new(),
            texture_indices: HashMap::new(),
            sampler_indices: HashMap::new(),
        }
    }

    pub fn register_buffer(&mut self, handle: BufferHandle, is_storage: bool) -> u32 {
        let index = if is_storage {
            let idx = self.next_storage_buffer_index;
            self.next_storage_buffer_index += 1;
            idx
        } else {
            let idx = self.next_uniform_buffer_index;
            self.next_uniform_buffer_index += 1;
            idx
        };
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

    // Bindless infrastructure
    /// Whether bindless descriptor indexing is enabled
    pub bindless_enabled: bool,
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
}

/// GPU buffer state.
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    /// Index in the global bindless descriptor set (if bindless enabled)
    #[allow(dead_code)]
    pub bindless_index: Option<u32>,
    /// Whether this is a storage buffer (vs uniform buffer)
    #[allow(dead_code)]
    pub is_storage: bool,
}

/// Shader module state with cached compiled stages.
pub(crate) struct ShaderState {
    pub device_handle: DeviceHandle,
    pub slang_source: String,
    /// Search paths for Slang module resolution
    pub search_paths: Vec<String>,
    /// Cached compiled vertex shader module
    pub vertex_module: Option<vk::ShaderModule>,
    /// Cached compiled fragment shader module
    pub fragment_module: Option<vk::ShaderModule>,
    /// Cached compiled compute shader module
    pub compute_module: Option<vk::ShaderModule>,
    /// Reflection data for bindless rendering (ParameterBlock layouts)
    pub reflection: Option<crate::slang::ShaderReflection>,
}

/// Graphics pipeline state.
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    /// Whether this pipeline owns its layout (false when using bindless_pipeline_layout)
    pub owns_layout: bool,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    #[allow(dead_code)]
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
}

/// Compute pipeline state.
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    /// Whether this pipeline owns its layout (false when using bindless_pipeline_layout)
    pub owns_layout: bool,
    /// ParameterBlock layouts from shader reflection (for bindless rendering)
    #[allow(dead_code)]
    pub parameter_block_layouts: Vec<crate::slang::ParameterBlockLayout>,
}

/// Bind group layout (descriptor set layout) state.
pub(crate) struct BindGroupLayoutState {
    #[allow(dead_code)]
    pub device_handle: DeviceHandle,
    pub layout: vk::DescriptorSetLayout,
    /// Maps binding index to descriptor type for correct bind group creation.
    pub binding_types: std::collections::HashMap<u32, ash::vk::DescriptorType>,
}

/// Tracks a single binding entry for bindless index lookup.
#[derive(Clone)]
pub(crate) enum BindGroupResourceRef {
    Buffer(BufferHandle),
    Texture(TextureHandle),
    Sampler(SamplerHandle),
}

/// Bind group (descriptor set) state.
pub(crate) struct BindGroupState {
    pub device_handle: DeviceHandle,
    pub descriptor_set: vk::DescriptorSet,
    pub pool: vk::DescriptorPool,
    /// Resource references for bindless index lookup (binding -> resource).
    /// Used to retrieve bindless indices when SetBindGroup is called.
    pub entries: Vec<(u32, BindGroupResourceRef)>,
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
    #[allow(dead_code)]
    pub bindless_index: Option<u32>,
}

/// GPU sampler state.
pub(crate) struct SamplerState {
    pub device_handle: DeviceHandle,
    pub sampler: vk::Sampler,
    /// Index in the global bindless descriptor set (if bindless enabled)
    #[allow(dead_code)]
    pub bindless_index: Option<u32>,
}

/// Maximum number of frames that can be in-flight at once.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Per-frame synchronization resources for proper swapchain pipelining.
pub(crate) struct FrameSync {
    pub command_buffer: vk::CommandBuffer,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub in_flight_fence: vk::Fence,
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
    /// Current frame index (0..MAX_FRAMES_IN_FLIGHT)
    pub current_frame: usize,
    /// Currently acquired swapchain image index
    pub current_image_index: Option<u32>,
    /// Per-frame synchronization resources
    pub frame_sync: Vec<FrameSync>,
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
/// Resources are kept alive until the frame they were last used in completes.
pub(crate) enum PendingDeletion {
    Buffer {
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
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
    /// Resources pending deletion, tagged with the frame they were queued on
    pub pending: Vec<(u64, PendingDeletion)>,
    /// Current frame counter (incremented each present)
    pub current_frame: u64,
}

impl DeletionQueue {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            current_frame: 0,
        }
    }

    /// Queue a resource for deferred deletion
    pub fn queue(&mut self, resource: PendingDeletion) {
        self.pending.push((self.current_frame, resource));
    }

    /// Advance the frame counter (called after present)
    pub fn advance_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Process deletions for frames that have completed.
    /// `completed_frame` is the frame number that has finished executing on the GPU.
    pub fn process_deletions(&mut self, device: &ash::Device, completed_frame: u64) {
        // Keep resources from frames that haven't completed yet
        let (to_delete, to_keep): (Vec<_>, Vec<_>) = self.pending
            .drain(..)
            .partition(|(frame, _)| *frame <= completed_frame);

        self.pending = to_keep;

        // Delete resources from completed frames
        for (_, resource) in to_delete {
            unsafe {
                match resource {
                    PendingDeletion::Buffer { buffer, memory } => {
                        device.destroy_buffer(buffer, None);
                        device.free_memory(memory, None);
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
                    PendingDeletion::Buffer { buffer, memory } => {
                        device.destroy_buffer(buffer, None);
                        device.free_memory(memory, None);
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
