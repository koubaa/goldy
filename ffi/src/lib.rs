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

/// Cross-platform surface creation for winit windows (Rust examples only).
///
/// C/C++ clients use platform entry points from the generated header (e.g. `goldy_surface_create_win32` on Windows).
/// This helper is not part of the stable C header.
#[cfg(feature = "examples")]
pub mod winit_surface;

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
