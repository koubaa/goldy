//! Goldy FFI - C bindings for the Goldy GPU library.
//!
//! This crate provides a stable C ABI for interoperating with Goldy from
//! other languages (C#, Python via ctypes, etc.).

mod compute;
mod device;
mod error;
mod instance;
mod pipeline;
mod render_target;
mod retained_pool;
mod sampler;
mod shader;
mod surface;
mod task_graph;
mod types;

pub use compute::*;
pub use device::*;
pub use error::*;
pub use instance::*;
pub use pipeline::*;
pub use render_target::*;
pub use retained_pool::*;
pub use sampler::*;
pub use shader::*;
pub use surface::*;
pub use task_graph::*;
pub use types::*;
