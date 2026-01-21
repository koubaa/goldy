//! Slang shader compiler integration.
//!
//! This module provides Rust bindings to the [Slang](https://shader-slang.org/) shader compiler.
//! Slang is Goldy's sole shader language, supporting compilation to:
//! - SPIR-V (Vulkan)
//! - DXIL (DirectX 12)
//! - MSL (Metal)
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

pub mod compiler;
pub mod ffi;
pub mod loader;

pub use compiler::{
    CompiledShader, CompiledShaderWithReflection, FieldLayout, ParameterBlockLayout, ResourceKind,
    ShaderReflection, ShaderTarget, SlangCompiler,
};
pub use ffi::SlangStage;
