//! Rust RAII client for the Goldy C ABI (`libgoldy_ffi`).
//!
//! This crate links **`libgoldy_ffi` dynamically** and calls the stable C API
//! via runtime-loaded C bindings (libloading) — the same path as C++/Python/.NET clients.
//! Wraps raw FFI with `Result`, resource ownership, and builder-style
//! task-graph recording.

mod adapter;
mod buffer;
mod compute;
mod device;
mod error;
mod instance;
mod pipeline;
mod render_target;
mod shader_module;
mod surface;
mod sys;
mod task_graph;
mod types;

pub use adapter::Adapter;
pub use buffer::Buffer;
pub use compute::{ComputeEncoder, ComputePipeline};
pub use device::Device;
pub use error::{GoldyError, Result};
pub use instance::{AdapterInfo, Instance};
pub use pipeline::RenderPipeline;
pub use render_target::RenderTarget;
pub use shader_module::ShaderModule;
pub use surface::{Frame, Surface};
pub use task_graph::{ComputeNodeBuilder, RenderPassBuilder, SwapchainOutputHandle, TaskGraph};
pub use types::{
    BufferKind, Color, CompareFunction, DepthFormat, DepthStencilState, DeviceDescriptor, DeviceType, NodeAccess,
    PowerPreference, PrimitiveTopology, RenderPipelineDesc, RequestAdapterOptions, ResourceAccess, TextureFormat,
    Vertex2D, VertexAttribute, VertexBufferLayout, VertexFormat,
};

/// Built-in shader sources (`shader::builtins`, matching native Goldy).
pub mod shader {
    pub use crate::shader_module::builtins;
}
