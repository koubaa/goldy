//! # Goldy - Modern GPU Library
//!
//! A modern GPU library targeting Vulkan 1.3+, DX12, and Metal.
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
pub mod bind_group;
pub mod buffer;
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
pub mod texture;
pub mod types;

// Slang compiler is only available on native targets (not WASM)
// For web builds, use slang-wasm in JavaScript
#[cfg(not(target_arch = "wasm32"))]
pub mod slang;

// Re-export main types
pub use bind_group::{
    BindGroup, BindGroupLayout, BindGroupLayoutBinding, BindingType, BufferBinding, SamplerBinding,
    ShaderStages, TextureBinding,
};
pub use buffer::Buffer;
pub use compute::{ComputeEncoder, ComputePass, ComputePipeline, ComputePipelineDesc};
pub use device::{Adapter, Device, DeviceCapabilities, Instance};
pub use encoder::{CommandEncoder, RenderPass};
pub use pipeline::{RenderPipeline, RenderPipelineDesc};
pub use render_target::RenderTarget;
pub use sampler::Sampler;
pub use shader::{builtins, ShaderModule};
pub use shader_library::ShaderLibrary;
pub use surface::{Surface, SurfaceFrame};
pub use texture::Texture;
pub use types::*;
