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

pub mod types;
pub mod backend;
pub mod device;
pub mod buffer;
pub mod bind_group;
pub mod shader;
pub mod shader_library;
pub mod shaders;
pub mod pipeline;
pub mod compute;
pub mod encoder;
pub mod render_target;
pub mod surface;
pub mod texture;
pub mod sampler;
pub mod examples;

// Slang compiler is only available on native targets (not WASM)
// For web builds, use slang-wasm in JavaScript
#[cfg(not(target_arch = "wasm32"))]
pub mod slang;

// Re-export main types
pub use types::*;
pub use device::{Instance, Device, Adapter, DeviceCapabilities};
pub use buffer::Buffer;
pub use bind_group::{BindGroup, BindGroupLayout, BindGroupLayoutBinding, BufferBinding, TextureBinding, SamplerBinding, BindingType, ShaderStages};
pub use shader::{ShaderModule, builtins};
pub use shader_library::ShaderLibrary;
pub use pipeline::{RenderPipeline, RenderPipelineDesc};
pub use compute::{ComputePipeline, ComputePipelineDesc, ComputeEncoder, ComputePass};
pub use encoder::{CommandEncoder, RenderPass};
pub use render_target::RenderTarget;
pub use surface::{Surface, SurfaceFrame};
pub use texture::Texture;
pub use sampler::Sampler;

