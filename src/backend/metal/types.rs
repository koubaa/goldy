//! Metal backend internal types.
//!
//! This module contains all the state structs used by the Metal backend.

#![allow(dead_code)] // Some fields are for future use

use crate::types::{DepthFormat, TextureFormat};
use super::super::{DeviceHandle, BufferHandle, BindGroupLayoutEntry};
// Use explicit crate path to avoid collision with our module name
use ::metal as mtl;
use mtl::{Buffer as MTLBuffer, CommandQueue, Device as MTLDevice, Library, Texture as MTLTexture, SamplerState, RenderPipelineState, ComputePipelineState as MTLComputePipelineState, DepthStencilState as MTLDepthStencilState};

/// A logical Metal device with associated resources.
pub(crate) struct LogicalDevice {
    pub device: MTLDevice,
    pub command_queue: CommandQueue,
    pub adapter_id: u32,
}

/// GPU buffer state.
pub(crate) struct BufferState {
    pub device_handle: DeviceHandle,
    pub buffer: MTLBuffer,
    pub size: u64,
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
}

/// Graphics pipeline state.
pub(crate) struct PipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: RenderPipelineState,
    pub depth_stencil: Option<MTLDepthStencilState>,
}

/// Compute pipeline state.
pub(crate) struct ComputePipelineState {
    pub device_handle: DeviceHandle,
    pub pipeline: MTLComputePipelineState,
}

/// Bind group layout state.
/// Metal uses argument buffers, but for simplicity we track binding metadata.
pub(crate) struct BindGroupLayoutState {
    pub device_handle: DeviceHandle,
    pub entries: Vec<BindGroupLayoutEntry>,
}

/// Bind group state.
/// Stores the actual buffer/texture/sampler bindings.
pub(crate) struct BindGroupState {
    pub device_handle: DeviceHandle,
    pub layout_handle: super::super::BindGroupLayoutHandle,
    pub bindings: Vec<BindingState>,
}

/// Individual binding within a bind group.
#[derive(Clone)]
pub(crate) enum BindingState {
    Buffer { buffer: BufferHandle, offset: u64, size: u64 },
    Texture(super::super::TextureHandle),
    Sampler(super::super::SamplerHandle),
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
}

/// GPU sampler state.
pub(crate) struct SamplerState_ {
    pub device_handle: DeviceHandle,
    pub sampler: SamplerState,
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

