//! Rust RAII client for the Goldy C ABI (`libgoldy_ffi`).
//!
//! This crate links **`libgoldy_ffi` dynamically** and calls the stable C API
//! via runtime-loaded C bindings (libloading) — the same path as C++/Python/.NET clients.
//! Wraps raw FFI with `Result`, resource ownership, and builder-style
//! task-graph recording.

mod adapter;
mod buffer;
mod compute;
mod context;
mod device;
mod error;
mod exchange;
mod instance;
mod parcel;
mod pipeline;
mod render_target;
mod retained_pool;
mod scheme;
mod shader_module;
mod surface;
mod surface_exchange;
mod sys;
mod texture;
mod types;

pub use adapter::Adapter;
pub use buffer::Buffer;
pub use compute::ComputePipeline;
pub use context::Context;
pub use device::Device;
pub use error::{GoldyError, Result};
pub use exchange::{Claim, Transaction};
pub use instance::{AdapterInfo, Instance};
pub use parcel::Parcel;
pub use pipeline::RenderPipeline;
pub use render_target::RenderTarget;
pub use retained_pool::{RecordBuilder, RecordField, RetainedPool};
pub use scheme::{
    ComputeNodeBuilder as SchemeComputeNodeBuilder, PresentLease, ReadGrant, ReplayStats, Scheme,
    SchemeRenderPassBuilder, SchemeRenderTargetLease, SchemeSubmission,
};
pub use shader_module::ShaderModule;
pub use surface::{Frame, Surface};
pub use surface_exchange::SurfaceExchange;
pub use texture::Texture;
pub use types::{
    BufferKind, Color, CompareFunction, DepthFormat, DepthStencilState, DeviceDescriptor, DeviceType, IndexFormat,
    NodeAccess, PowerPreference, PrimitiveTopology, RenderPipelineDesc, RequestAdapterOptions, ResourceAccess,
    ResourceCategory, ResourceHandle, TextureFlags, TextureFormat, TextureKind, Vertex2D, VertexAttribute,
    VertexBufferLayout, VertexFormat,
};

/// Built-in shader sources (`shader::builtins`, matching native Goldy).
pub mod shader {
    pub use crate::shader_module::builtins;
}
