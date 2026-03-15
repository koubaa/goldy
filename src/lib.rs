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
pub mod texture;
pub mod types;

pub mod slang;

// Structured instrumentation for debugging and profiling
pub mod instrumentation;

// Re-export main types
pub use buffer::{Buffer, BufferPool, BufferSource, BufferView};
pub use common_types::{FrameUniforms, Instance2D, Particle2D, Particle3D, Transform2D};
pub use compute::{ComputeEncoder, ComputePass, ComputePipeline};
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
