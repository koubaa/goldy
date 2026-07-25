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
pub mod virtual_main;

pub use compiler::{
    layout_validation_enabled, CompiledShader, CompiledShaderWithReflection, FieldLayout, LayoutCheck,
    OwnedLayoutCheck, ParameterBlockLayout, ResourceKind, ShaderReflection, ShaderTarget, SlangCompiler,
    StructFieldLayout, StructLayout,
};
pub use ffi::SlangStage;
pub use virtual_main::{emit_wrapper_from_kernel_def, entry_def_from_kernel_def, try_kernel_def_from_source};

/// Parse `[numthreads(x, y, z)]` from Slang shader source.
///
/// The input may be the full source string or just the inner content of the
/// `[numthreads(...)]` attribute (e.g. `"numthreads(64, 1, 1)"` or `"64, 1, 1"`).
///
/// Returns `None` if the attribute is absent or cannot be parsed; callers
/// should fall back to a suitable default (e.g. `[64, 1, 1]`).
pub fn parse_numthreads(source: &str) -> Option<[u32; 3]> {
    // Locate the `numthreads` keyword anywhere in the input.
    let kw_pos = source.find("numthreads")?;
    let after_kw = source[kw_pos + "numthreads".len()..].trim_start();
    // Accept both `numthreads(x,y,z)` and bare `x,y,z` (already inside parens).
    let inner = if let Some(stripped) = after_kw.strip_prefix('(') {
        let close = stripped.find(')')?;
        &stripped[..close]
    } else {
        // Input was already the content between parens.
        after_kw
    };
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() < 3 {
        return None;
    }
    let x: u32 = parts[0].parse().ok()?;
    let y: u32 = parts[1].parse().ok()?;
    let z: u32 = parts[2].parse().ok()?;
    Some([x, y, z])
}
