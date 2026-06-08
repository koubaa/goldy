//! Goldy FFI - C bindings for the Goldy GPU library.
//!
//! This crate provides a stable C ABI for interoperating with Goldy from
//! other languages (C#, Python via ctypes, etc.).

mod buffer;
mod compute;
mod device;
mod error;
mod instance;
mod pipeline;
mod render_target;
mod sampler;
mod shader;
mod surface;
mod task_graph;
mod texture;
mod types;

pub use buffer::*;
pub use compute::*;
pub use device::*;
pub use error::*;
pub use instance::*;
pub use pipeline::*;
pub use render_target::*;
pub use sampler::*;
pub use shader::*;
pub use surface::*;
pub use task_graph::*;
pub use texture::*;
pub use types::*;
