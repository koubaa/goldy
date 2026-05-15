//! # Goldy - Modern GPU Library
//!
//! A modern GPU library targeting Vulkan 1.4+, DX12, and Metal.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use goldy::{Instance, DeviceType};
//!
//! let instance = Instance::new().unwrap();
//! let device = instance.create_device(DeviceType::DiscreteGpu).unwrap();
//! ```

pub mod backend;
pub mod buffer;
pub mod common_types;
pub mod compute;
pub mod device;
pub mod encoder;
pub mod examples;
pub mod pipeline;
pub mod render_target;
pub mod sampler;
pub mod shader;
pub mod shader_library;
pub mod shaders;
pub mod surface;
pub mod task_graph;
pub mod texture;
pub mod texture_pool;
pub mod types;

pub mod shader_cache;
pub mod slang;
pub mod validation_env;

// Structured instrumentation for debugging and profiling
pub mod instrumentation;
pub mod timeline;
pub mod placement_heap;
pub mod transient_allocator;
pub mod vram_allocator;

// Re-export main types
pub use buffer::{Buffer, BufferPool, BufferSource, BufferView, StructuredBufferElement};
pub use common_types::{FrameUniforms, Instance2D, Particle2D, Particle3D, Transform2D};
pub use compute::{ComputeEncoder, ComputePass, ComputePipeline};
pub use timeline::TimelineValue;

pub use backend::GraphCommand;
pub use device::{Adapter, Device, DeviceCapabilities, Instance};
pub use encoder::{CommandEncoder, RenderPass};
pub use goldy_derive::LayoutCheckable;
pub use goldy_derive::StructuredBufferElement;
pub use pipeline::{RenderPipeline, RenderPipelineDesc};
pub use render_target::RenderTarget;
pub use sampler::Sampler;
pub use shader::{builtins, ShaderModule};
pub use shader_library::ShaderLibrary;
pub use slang::{layout_validation_enabled, LayoutCheck, StructFieldLayout, StructLayout};
pub use surface::{Frame, Surface};
pub use task_graph::{
    GraphIR, NodeAccess, NodeBuilder, RenderPassBuilder, TaskGraph, TransientBufferSpec,
    TransientId, TransientTextureId,
};

pub use texture::Texture;
pub use texture_pool::{TexturePool, TexturePoolConfig, TexturePoolStats};
pub use transient_allocator::{
    BumpResetAllocator, EpochRegionsAllocator, TransientAllocator, TransientAllocatorConfig,
    TransientAllocatorStrategy,
};
pub use types::*;
pub use types::{PresentMode, SurfaceConfig};

#[cfg(all(feature = "dx12", target_os = "windows"))]
pub use backend::dx12::WARP_ADAPTER_ID;
