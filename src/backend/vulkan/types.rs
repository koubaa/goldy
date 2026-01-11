//! Vulkan backend internal types.
//!
//! This module contains all the state structs used by the Vulkan backend.

use crate::types::{DepthFormat, TextureFormat};
use ash::vk;
use super::super::{DeviceHandle, BufferHandle};

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
}

/// GPU buffer state.
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
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
}

/// Graphics pipeline state.
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

/// Compute pipeline state.
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

/// Bind group layout (descriptor set layout) state.
pub(crate) struct BindGroupLayoutState {
    #[allow(dead_code)]
    pub device_handle: DeviceHandle,
    pub layout: vk::DescriptorSetLayout,
    /// Maps binding index to descriptor type for correct bind group creation.
    pub binding_types: std::collections::HashMap<u32, ash::vk::DescriptorType>,
}

/// Bind group (descriptor set) state.
pub(crate) struct BindGroupState {
    pub device_handle: DeviceHandle,
    pub descriptor_set: vk::DescriptorSet,
    pub pool: vk::DescriptorPool,
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
}

/// GPU sampler state.
pub(crate) struct SamplerState {
    pub device_handle: DeviceHandle,
    pub sampler: vk::Sampler,
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

