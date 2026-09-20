//! Goldy FFI - C bindings for the Goldy GPU library.
//!
//! This crate provides a stable C ABI for interoperating with Goldy from
//! other languages (C#, Python via ctypes, etc.).

mod compute;
mod context;
mod error;
mod instance;
mod memory_exchange;
mod pipeline;
mod retained_pool;
mod runtime;
mod sampler;
mod scheme;
mod shader;
mod surface_exchange;
#[cfg(feature = "tensor")]
mod tensor;
mod types;

pub use compute::*;
pub use context::*;
pub use error::*;
pub use instance::*;
pub use memory_exchange::*;
pub use pipeline::*;
pub use retained_pool::*;
pub use runtime::*;
pub use sampler::*;
pub use scheme::*;
pub use shader::*;
pub use surface_exchange::*;
#[cfg(feature = "tensor")]
pub use tensor::*;
pub use types::*;
