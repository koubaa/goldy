//! Slang shader compiler integration.
//!
//! This module provides Rust bindings to the [Slang](https://shader-slang.org/) shader compiler.
//! Slang is Goldy's sole shader language, supporting compilation to:
//! - SPIR-V (Vulkan)
//! - WGSL (WebGPU)
//! - HLSL (DirectX)
//! - MSL (Metal)
//! - GLSL
//!
//! **Note**: The native Slang compiler is not available on WASM targets.
//! For web builds, use slang-wasm in JavaScript to compile Slang to WGSL.
//!
//! # Example
//!
//! ```rust,no_run
//! use goldy::slang::{SlangCompiler, ShaderTarget};
//!
//! let compiler = SlangCompiler::new().unwrap();
//!
//! let source = r#"
//!     [shader("vertex")]
//!     float4 vs_main(float2 pos : POSITION) : SV_Position {
//!         return float4(pos, 0, 1);
//!     }
//!
//!     [shader("fragment")]
//!     float4 fs_main() : SV_Target {
//!         return float4(1, 0, 0, 1);
//!     }
//! "#;
//!
//! let spirv = compiler.compile(source, ShaderTarget::Spirv).unwrap();
//! println!("Compiled {} bytes of SPIR-V", spirv.data.len());
//! ```

#[cfg(not(target_arch = "wasm32"))]
pub mod ffi;
#[cfg(not(target_arch = "wasm32"))]
pub mod loader;
#[cfg(not(target_arch = "wasm32"))]
pub mod compiler;

#[cfg(not(target_arch = "wasm32"))]
pub use compiler::{CompiledShader, ShaderTarget, SlangCompiler, global_compiler};
#[cfg(not(target_arch = "wasm32"))]
pub use ffi::SlangStage;

