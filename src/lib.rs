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
pub mod pipeline;
pub mod encoder;
pub mod frame;

// Re-export main types
pub use types::*;
pub use device::{Instance, Device, Adapter};
pub use buffer::Buffer;
pub use shader::ShaderModule;
pub use pipeline::{RenderPipeline, RenderPipelineDesc};
pub use encoder::{CommandEncoder, RenderPass};
pub use frame::FrameOutput;

