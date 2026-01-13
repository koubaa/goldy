//! Goldy FFI - C bindings for the Goldy GPU library.
//!
//! This crate provides a stable C ABI for interoperating with Goldy from
//! other languages (C#, Python via ctypes, etc.).

mod error;
mod types;
mod instance;
mod device;
mod buffer;
mod render_target;
mod shader;
mod pipeline;
mod encoder;
mod bind_group;
mod compute;
mod texture;
mod sampler;
mod surface;

pub use error::*;
pub use types::*;
pub use instance::*;
pub use device::*;
pub use buffer::*;
pub use render_target::*;
pub use shader::*;
pub use pipeline::*;
pub use encoder::*;
pub use bind_group::*;
pub use compute::*;
pub use texture::*;
pub use sampler::*;
pub use surface::*;

