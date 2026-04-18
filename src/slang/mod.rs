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
//! use goldy::slang::{SlangCompiler, ShaderTarget, SlangStage};
//! use goldy::types::OptimizationLevel;
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
//! let out = compiler
//!     .compile_bindless_with_reflection_and_defines(
//!         source,
//!         ShaderTarget::Spirv,
//!         &[("vs_main", SlangStage::Vertex)],
//!         &[],
//!         &[],
//!         &[],
//!         OptimizationLevel::Default,
//!     )
//!     .unwrap();
//! println!("Compiled {} bytes of SPIR-V", out.shader.data.len());
//! ```

pub mod compiler;
pub mod ffi;
pub mod loader;

pub use compiler::{
    analyze_push_constant_categories_from_source, analyze_push_constant_stride_hints_from_source,
    layout_validation_enabled, CompiledShader, CompiledShaderWithReflection, FieldLayout,
    LayoutCheck, OwnedLayoutCheck, ParameterBlockLayout, PushConstantStrideHint, ResourceKind,
    ShaderReflection, ShaderTarget, SlangCompiler, StructFieldLayout, StructLayout,
};
pub use ffi::SlangStage;
