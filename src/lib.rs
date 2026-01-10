//! # RAG - Rust Abstract GPU
//!
//! A modern GPU abstraction library targeting Vulkan 1.4+, Metal 2+, and DX12.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rag::{Instance, DeviceType};
//!
//! let instance = Instance::new().unwrap();
//! let device = instance.create_device(DeviceType::DiscreteGpu).unwrap();
//! ```

pub mod types;
pub mod backend;
pub mod device;
pub mod buffer;
pub mod shader;
pub mod shaders;
pub mod pipeline;
pub mod encoder;
pub mod frame;
pub mod render_target;
pub mod surface;
pub mod examples;

// Slang compiler is only available on native targets (not WASM)
// For web builds, use slang-wasm in JavaScript
#[cfg(not(target_arch = "wasm32"))]
pub mod slang;

// Re-export main types
pub use types::*;
pub use device::{Instance, Device, Adapter};
pub use buffer::Buffer;
pub use shader::ShaderModule;
pub use pipeline::{RenderPipeline, RenderPipelineDesc};
pub use encoder::{CommandEncoder, RenderPass};
pub use render_target::RenderTarget;
pub use surface::{Surface, SurfaceFrame};
#[allow(deprecated)]
pub use frame::FrameOutput;

