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

// Slang compiler is only available on native targets (not WASM)
// For web builds, use slang-wasm in JavaScript
#[cfg(not(target_arch = "wasm32"))]
pub mod slang;

// Structured instrumentation for debugging and profiling
pub mod instrumentation;

// Re-export main types
pub use buffer::Buffer;
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

/// Test utilities for setting up the test environment.
#[cfg(test)]
pub(crate) mod test_utils {
    use std::sync::Once;

    static INIT: Once = Once::new();

    /// Initialize the test environment.
    ///
    /// This sets up `GOLDY_SLANG_PATH` to point to the vendored Slang libraries
    /// in the repository, making tests runnable without external setup.
    pub fn init() {
        INIT.call_once(|| {
            // Only set if not already set (allows override)
            if std::env::var("GOLDY_SLANG_PATH").is_err() {
                // Find the slang libraries relative to the cargo manifest directory
                let manifest_dir = env!("CARGO_MANIFEST_DIR");
                let platform = if cfg!(target_os = "windows") {
                    if cfg!(target_arch = "x86_64") {
                        "windows-x86_64"
                    } else {
                        "windows-aarch64"
                    }
                } else if cfg!(target_os = "macos") {
                    if cfg!(target_arch = "aarch64") {
                        "macos-aarch64"
                    } else {
                        "macos-x86_64"
                    }
                } else if cfg!(target_os = "linux") {
                    if cfg!(target_arch = "aarch64") {
                        "linux-aarch64"
                    } else {
                        "linux-x86_64"
                    }
                } else {
                    panic!("Unsupported platform for tests");
                };

                let lib_name = if cfg!(target_os = "windows") {
                    "slang-compiler.dll"
                } else if cfg!(target_os = "macos") {
                    "libslang-compiler.dylib"
                } else {
                    "libslang-compiler.so"
                };

                let slang_path = format!("{}/slang/bin/{}/{}", manifest_dir, platform, lib_name);

                if std::path::Path::new(&slang_path).exists() {
                    std::env::set_var("GOLDY_SLANG_PATH", &slang_path);
                } else {
                    eprintln!(
                        "Warning: Slang library not found at {}. Run slang/download.sh first.",
                        slang_path
                    );
                }
            }
        });
    }
}
