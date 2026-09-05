//! Slang shader compiler integration.
//!
//! This module provides Rust bindings to the [Slang](https://shader-slang.org/) shader compiler.
//! Slang is Goldy's sole shader language, supporting compilation to:
//! - SPIR-V (Vulkan)
//! - DXIL (DirectX 12)
//! - MSL (Metal)
//! - CUDA PTX / CUDA C++ (`GOLDY_DUMP_SHADERS`)
//! - Host-callable CPU JIT (`ShaderTarget::HostCallable`, debug only)
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

pub mod bounds_analysis;
pub mod compiler;
pub mod ffi;
mod gpu_type;
pub mod graphics_link;
pub mod loader;
pub mod virtual_main;

pub use bounds_analysis::{BoundsAnalysisError, BoundsDiagnostic, BoundsReport, SourceLocation};
pub use compiler::{
    layout_validation_enabled, CompiledShader, CompiledShaderWithReflection, FieldLayout, LayoutCheck,
    OwnedLayoutCheck, ParameterBlockLayout, ResourceKind, ShaderReflection, ShaderTarget, SlangCompiler,
    StructFieldLayout, StructLayout,
};
pub use ffi::SlangStage;
pub use gpu_type::{GpuField, GpuFieldType, GpuType, PackedGpuField, PackedGpuLayout};
pub use graphics_link::{
    GraphicsPipelineInterface, InterpolationMode, PipelineResource, PipelineResourceContract, StageInterface,
    StageIoField,
};
pub use virtual_main::{
    emit_wrapper_from_kernel_def, entry_def_from_kernel_def, transform_virtual_main_cpu, try_kernel_def_from_source,
};

pub fn canonical_entry_point(stage: SlangStage) -> Option<&'static str> {
    match stage {
        SlangStage::Vertex => Some("vs_main"),
        SlangStage::Fragment => Some("fs_main"),
        SlangStage::Compute => Some("cs_main"),
        SlangStage::RayGeneration => Some("rgen_main"),
        SlangStage::Intersection => Some("rint_main"),
        SlangStage::AnyHit => Some("rahit_main"),
        SlangStage::ClosestHit => Some("rchit_main"),
        SlangStage::Miss => Some("rmiss_main"),
        SlangStage::Callable => Some("rcall_main"),
        SlangStage::Mesh => Some("mesh_main"),
        SlangStage::Amplification => Some("amp_main"),
        SlangStage::None | SlangStage::Hull | SlangStage::Domain | SlangStage::Geometry => None,
    }
}

pub(crate) fn slang_stage_to_virtual_main(stage: SlangStage) -> Option<virtual_main::Stage> {
    match stage {
        SlangStage::Vertex => Some(virtual_main::Stage::Vertex),
        SlangStage::Fragment => Some(virtual_main::Stage::Fragment),
        SlangStage::Compute => Some(virtual_main::Stage::Compute),
        SlangStage::RayGeneration => Some(virtual_main::Stage::RayGeneration),
        SlangStage::Intersection => Some(virtual_main::Stage::Intersection),
        SlangStage::AnyHit => Some(virtual_main::Stage::AnyHit),
        SlangStage::ClosestHit => Some(virtual_main::Stage::ClosestHit),
        SlangStage::Miss => Some(virtual_main::Stage::Miss),
        SlangStage::Mesh => Some(virtual_main::Stage::Mesh),
        SlangStage::Amplification => Some(virtual_main::Stage::Amplification),
        _ => None,
    }
}

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

#[cfg(test)]
mod canonical_entry_point_tests {
    use super::{canonical_entry_point, SlangStage};

    #[test]
    fn maps_graphics_compute_rt_and_mesh() {
        assert_eq!(canonical_entry_point(SlangStage::Vertex), Some("vs_main"));
        assert_eq!(canonical_entry_point(SlangStage::Fragment), Some("fs_main"));
        assert_eq!(canonical_entry_point(SlangStage::Compute), Some("cs_main"));
        assert_eq!(canonical_entry_point(SlangStage::RayGeneration), Some("rgen_main"));
        assert_eq!(canonical_entry_point(SlangStage::Intersection), Some("rint_main"));
        assert_eq!(canonical_entry_point(SlangStage::AnyHit), Some("rahit_main"));
        assert_eq!(canonical_entry_point(SlangStage::ClosestHit), Some("rchit_main"));
        assert_eq!(canonical_entry_point(SlangStage::Miss), Some("rmiss_main"));
        assert_eq!(canonical_entry_point(SlangStage::Callable), Some("rcall_main"));
        assert_eq!(canonical_entry_point(SlangStage::Mesh), Some("mesh_main"));
        assert_eq!(canonical_entry_point(SlangStage::Amplification), Some("amp_main"));
        assert_eq!(canonical_entry_point(SlangStage::Hull), None);
        assert_eq!(canonical_entry_point(SlangStage::Domain), None);
        assert_eq!(canonical_entry_point(SlangStage::Geometry), None);
        assert_eq!(canonical_entry_point(SlangStage::None), None);
    }
}
