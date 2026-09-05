//! Shader module management.
//!
//! Goldy provides two ways to work with shaders:
//!
//! 1. **Built-in shaders** (`shader::builtins`) - Complete, self-contained shaders
//!    for common use cases. No imports, no file system access needed.
//!
//! 2. **Shader libraries** - Reusable Slang modules that your shaders can import.
//!    The `goldy` library is registered by default on every device.
//!
//! # Using Shader Libraries
//!
//! Shaders can import registered libraries:
//!
//! ```slang
//! import goldy;  // Uses the built-in goldy library
//!
//! [shader("vertex")]
//! FullscreenVarying vs_main(FullscreenVertex input) {
//!     return vs_fullscreen(input);
//! }
//!
//! [shader("fragment")]
//! float4 fs_main(FullscreenVarying input) : SV_Target {
//!     return float4(rainbow(input.uv.x), 1.0);
//! }
//! ```
//!
//! # Custom Libraries
//!
//! Register your own libraries with [`Device::register_library`](crate::Device::register_library):
//!
//! ```rust,ignore
//! use goldy::ShaderLibrary;
//!
//! device.register_library(ShaderLibrary::from_source("myutils", r#"
//!     module myutils;
//!     public float3 my_effect() { return float3(1, 0, 0); }
//! "#))?;
//!
//! // Now your shaders can use: import myutils;
//! ```

use crate::backend::{GpuBackend, ShaderHandle};
use crate::device::Device;
use crate::slang::{layout_validation_enabled, GpuType, LayoutCheck, OwnedLayoutCheck};
use anyhow::{bail, Context, Result};
use std::borrow::Cow;
use std::sync::{Arc, Mutex, OnceLock};

/// The compile inputs a [`ShaderModule`] was built from.
///
/// Retained by every module (and by every [`crate::ComputePipeline`] built from one) so
/// goldy can compile *variants* of a shader after the fact — the retained-scheme
/// specialization predictor recompiles a dispatch's shader with scalar params baked in
/// (see `docs/src/design/shader-specialization.md`). Cheap to share: every field is an
/// `Arc`, and the `id` is unique per module so variant caches can key on it.
pub(crate) struct ShaderProvenance {
    id: u64,
    /// Author-facing Slang after optional GpuType preamble (before virtual-main rewrite).
    pub(crate) source: Arc<str>,
    /// Library + extra search paths recorded at construction (not re-merged on variants).
    pub(crate) search_paths: Arc<[String]>,
    pub(crate) defines: Arc<[(String, String)]>,
    pub(crate) optimization_level: crate::types::OptimizationLevel,
    pub(crate) layout_checks: Arc<[OwnedLayoutCheck]>,
    /// User function name of the single `[goldy_compute]` entry, if the source has exactly one.
    compute_entry: OnceLock<Option<String>>,
}

impl ShaderProvenance {
    fn new(
        source: Arc<str>,
        search_paths: Arc<[String]>,
        defines: Arc<[(String, String)]>,
        optimization_level: crate::types::OptimizationLevel,
        layout_checks: Arc<[OwnedLayoutCheck]>,
    ) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            source,
            search_paths,
            defines,
            optimization_level,
            layout_checks,
            compute_entry: OnceLock::new(),
        }
    }

    /// Process-unique identity of the module these inputs produced.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// User function name of the `[goldy_compute]` entry, when the source has exactly one.
    ///
    /// Scalar-param bake macros are scoped by this name
    /// ([`crate::slang::virtual_main::scalar_specialization_macro`]); with zero or several
    /// compute entries there is no unambiguous macro to define, and specialization skips
    /// the shader.
    pub(crate) fn compute_entry(&self) -> Option<&str> {
        self.compute_entry
            .get_or_init(|| crate::slang::virtual_main::single_compute_entry_name(&self.source))
            .as_deref()
    }

    /// This module's defines with `extra_defines` merged in (matching keys are overridden).
    pub(crate) fn merged_defines(&self, extra_defines: &[(&str, &str)]) -> Vec<(String, String)> {
        let mut merged: Vec<(String, String)> = self.defines.iter().cloned().collect();
        for &(key, value) in extra_defines {
            if let Some(existing) = merged.iter_mut().find(|(k, _)| k == key) {
                existing.1 = value.to_string();
            } else {
                merged.push((key.to_string(), value.to_string()));
            }
        }
        merged
    }
}

/// A compiled shader module.
pub struct ShaderModule {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: ShaderHandle,
    /// Everything needed to compile this module again (or a variant of it).
    provenance: Arc<ShaderProvenance>,
    /// Post-virtual-main source, filled on first compile / [`Self::effective_source`].
    effective_source: OnceLock<Arc<str>>,
}

impl ShaderModule {
    /// Create a shader module from Slang source.
    ///
    /// The source is compiled using Slang and can import any registered
    /// shader libraries (including the built-in `goldy` library).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use goldy::ShaderModule;
    ///
    /// let shader = ShaderModule::from_slang(&device, r#"
    ///     import goldy;
    ///
    ///     [shader("vertex")]
    ///     FullscreenVarying vs_main(FullscreenVertex input) {
    ///         return vs_fullscreen(input);
    ///     }
    ///
    ///     [shader("fragment")]
    ///     float4 fs_main(FullscreenVarying input) : SV_Target {
    ///         return float4(rainbow(input.uv.x), 1.0);
    ///     }
    /// "#)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn from_slang(device: &Device, source: &str) -> Result<Self> {
        Self::from_slang_with_options(device, source, &[], &[], Default::default(), &[])
    }

    /// Create a shader module after injecting Slang declarations generated from Rust GPU types.
    pub fn from_slang_with_gpu_types(device: &Device, source: &str, gpu_types: &[GpuType<'_>]) -> Result<Self> {
        Self::from_slang_with_gpu_types_and_options(device, source, &[], &[], Default::default(), &[], gpu_types)
    }

    /// Create a shader module with additional search paths.
    ///
    /// This is useful when your shaders also need to access modules from
    /// additional filesystem directories, beyond the registered libraries.
    ///
    /// Registered libraries are always included automatically.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use goldy::ShaderModule;
    ///
    /// // Shader can import from both registered libraries AND the "my_project" directory
    /// let shader = ShaderModule::from_slang_with_paths(
    ///     &device,
    ///     source,
    ///     &["my_project/shaders"],
    /// )?;
    /// ```
    pub fn from_slang_with_paths(device: &Device, source: &str, extra_paths: &[&str]) -> Result<Self> {
        Self::from_slang_with_options(device, source, extra_paths, &[], Default::default(), &[])
    }

    /// Create a shader module with search paths and preprocessor defines.
    ///
    /// Use for shader variants like MSAA (`msaa`, `msaa8`, `msaa16`).
    pub fn from_slang_with_paths_and_defines(
        device: &Device,
        source: &str,
        extra_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<Self> {
        Self::from_slang_with_options(device, source, extra_paths, defines, Default::default(), &[])
    }

    /// Create a shader module with full control over compilation options.
    ///
    /// `layout_checks` declares Rust struct layouts to validate against Slang reflection.
    /// Validation only runs when layout validation is enabled (`GOLDY_VALIDATE_LAYOUTS`,
    /// `GOLDY_VALIDATION=layout`, etc. — see `validation_env`); otherwise the checks
    /// are ignored (zero cost). Pass `&[]` when no validation is needed.
    ///
    /// Use `OptimizationLevel::None` to disable compiler optimizations for
    /// shaders that hit driver bugs on software renderers (e.g. lavapipe).
    pub fn from_slang_with_options(
        device: &Device,
        source: &str,
        extra_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
        layout_checks: &[LayoutCheck<'_>],
    ) -> Result<Self> {
        Self::from_slang_with_gpu_types_and_options(
            device,
            source,
            extra_paths,
            defines,
            optimization_level,
            layout_checks,
            &[],
        )
    }

    /// Create a shader module with full options and Rust-generated Slang struct declarations.
    ///
    /// Generated types are always reflection-validated. Authored `layout_checks` retain their
    /// existing opt-in validation behavior.
    #[allow(clippy::too_many_arguments)]
    /// Compile `source` and always validate `gpu_types` without injecting their declarations.
    ///
    /// Use this when the types already exist in `source` or an imported shader library
    /// (for example after [`crate::ShaderLibrary::from_source_with_gpu_types`]).
    pub fn validate_existing_gpu_types(device: &Device, source: &str, gpu_types: &[GpuType<'_>]) -> Result<()> {
        let mut generated_checks = Vec::with_capacity(gpu_types.len());
        let mut names = std::collections::HashSet::with_capacity(gpu_types.len());
        for gpu_type in gpu_types {
            if !names.insert(gpu_type.type_name) {
                bail!("duplicate generated GpuType `{}`", gpu_type.type_name);
            }
            generated_checks.push(gpu_type.generate()?.check);
        }
        if generated_checks.is_empty() {
            return Ok(());
        }

        let library_paths = device
            .get_shader_search_paths()
            .context("Failed to prepare shader library paths")?;
        let all_paths: Vec<String> = library_paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        let path_refs: Vec<&str> = all_paths.iter().map(|s| s.as_str()).collect();

        let mut backend = device.inner.backend.lock().unwrap();
        let handle = backend.create_shader_with_checks(
            device.inner.handle,
            source,
            &path_refs,
            &[],
            Default::default(),
            generated_checks,
        )?;
        backend.destroy_shader(handle);
        Ok(())
    }

    pub fn from_slang_with_gpu_types_and_options(
        device: &Device,
        source: &str,
        extra_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
        layout_checks: &[LayoutCheck<'_>],
        gpu_types: &[GpuType<'_>],
    ) -> Result<Self> {
        let mut generated_source = String::new();
        let mut generated_checks = Vec::with_capacity(gpu_types.len());
        let mut names = std::collections::HashSet::with_capacity(gpu_types.len());
        for gpu_type in gpu_types {
            if !names.insert(gpu_type.type_name) {
                bail!("duplicate generated GpuType `{}`", gpu_type.type_name);
            }
            let generated = gpu_type.generate()?;
            generated_source.push_str(&generated.source);
            generated_source.push('\n');
            generated_checks.push(generated.check);
        }
        let effective_source;
        let source = if generated_source.is_empty() {
            source
        } else {
            generated_source.push_str("#line 1 \"shader.slang\"\n");
            generated_source.push_str(source);
            effective_source = generated_source;
            effective_source.as_str()
        };

        let validate_authored = layout_validation_enabled() && !layout_checks.is_empty();
        let validate = validate_authored || !generated_checks.is_empty();

        tracing::debug!(
            source_len = source.len(),
            extra_paths = extra_paths.len(),
            defines = defines.len(),
            layout_checks = layout_checks.len(),
            generated_types = gpu_types.len(),
            validate,
            ?optimization_level,
            "Compiling shader module"
        );

        let library_paths = device
            .get_shader_search_paths()
            .context("Failed to prepare shader library paths")?;

        let all_paths: Vec<String> = library_paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .chain(extra_paths.iter().map(|s| s.to_string()))
            .collect();

        let owned_defines: Vec<(String, String)> = defines
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        let mut owned_checks = generated_checks;
        if validate_authored {
            owned_checks.extend(layout_checks.iter().map(OwnedLayoutCheck::from_layout_check));
        }

        Self::create_retained(
            device,
            Arc::from(source),
            all_paths.into(),
            owned_defines.into(),
            optimization_level,
            owned_checks.into(),
        )
    }

    /// New module with extra preprocessor defines merged into this module's defines.
    ///
    /// Keys in `extra_defines` override matching keys. Source, search paths (including
    /// registered libraries), optimization level, and layout checks are reused. The
    /// original module is unchanged.
    pub fn variant(&self, extra_defines: &[(&str, &str)]) -> Result<Self> {
        Self::from_provenance(&self._device, &self.provenance, extra_defines)
    }

    /// Compile a module from another module's retained inputs plus `extra_defines`.
    ///
    /// This is [`Self::variant`] without needing the original module to still exist —
    /// a [`crate::ComputePipeline`] keeps its shader's provenance alive so the
    /// specialization predictor can compile variants of a shader whose module the
    /// caller already dropped.
    pub(crate) fn from_provenance(
        device: &Device,
        provenance: &ShaderProvenance,
        extra_defines: &[(&str, &str)],
    ) -> Result<Self> {
        Self::create_retained(
            device,
            Arc::clone(&provenance.source),
            Arc::clone(&provenance.search_paths),
            provenance.merged_defines(extra_defines).into(),
            provenance.optimization_level,
            Arc::clone(&provenance.layout_checks),
        )
    }

    fn create_retained(
        device: &Device,
        source: Arc<str>,
        search_paths: Arc<[String]>,
        defines: Arc<[(String, String)]>,
        optimization_level: crate::types::OptimizationLevel,
        layout_checks: Arc<[OwnedLayoutCheck]>,
    ) -> Result<Self> {
        let path_refs: Vec<&str> = search_paths.iter().map(|s| s.as_str()).collect();
        let define_refs: Vec<(&str, &str)> = defines.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();

        let mut backend = device.inner.backend.lock().unwrap();
        let handle = if layout_checks.is_empty() {
            backend.create_shader_with_paths(
                device.inner.handle,
                source.as_ref(),
                &path_refs,
                &define_refs,
                optimization_level,
            )?
        } else {
            backend.create_shader_with_checks(
                device.inner.handle,
                source.as_ref(),
                &path_refs,
                &define_refs,
                optimization_level,
                layout_checks.to_vec(),
            )?
        };
        drop(backend);

        tracing::debug!("Shader module created");

        Ok(Self {
            _device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            provenance: Arc::new(ShaderProvenance::new(
                source,
                search_paths,
                defines,
                optimization_level,
                layout_checks,
            )),
            effective_source: OnceLock::new(),
        })
    }

    /// Retained compile inputs, shared with every pipeline built from this module.
    pub(crate) fn provenance(&self) -> &Arc<ShaderProvenance> {
        &self.provenance
    }

    pub(crate) fn source(&self) -> &str {
        &self.provenance.source
    }

    pub(crate) fn search_paths(&self) -> &[String] {
        &self.provenance.search_paths
    }

    pub(crate) fn defines(&self) -> &[(String, String)] {
        &self.provenance.defines
    }

    pub(crate) fn optimization_level(&self) -> crate::types::OptimizationLevel {
        self.provenance.optimization_level
    }

    pub(crate) fn layout_checks(&self) -> &[OwnedLayoutCheck] {
        &self.provenance.layout_checks
    }

    /// Post-virtual-main translation unit, computed once per module.
    pub(crate) fn effective_source(&self) -> &str {
        self.effective_source
            .get_or_init(|| {
                match crate::slang::virtual_main::effective_slang_source_for_compile(&self.provenance.source) {
                    Cow::Borrowed(_) => Arc::clone(&self.provenance.source),
                    Cow::Owned(transformed) => Arc::from(transformed),
                }
            })
            .as_ref()
    }

    #[cfg(test)]
    pub(crate) fn effective_source_arc(&self) -> Arc<str> {
        self.effective_source();
        Arc::clone(self.effective_source.get().expect("initialized above"))
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        tracing::trace!("Destroying shader module");
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_shader(self.handle);
    }
}

/// Built-in shaders for common use cases.
///
/// All shaders are written in Slang (HLSL-like syntax).
pub mod builtins {
    /// Simple 2D vertex + fragment shader for colored vertices.
    pub const VERTEX_COLOR_2D: &str = r#"
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
"#;

    /// Simple solid color fragment shader.
    pub const SOLID_COLOR: &str = r#"
struct VertexInput {
    float2 position : POSITION;
};

struct VertexOutput {
    float4 position : SV_Position;
};

cbuffer Uniforms {
    float4 color;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return color;
}
"#;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::compute::ComputePipeline;

    fn mock_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).expect("mock device")
    }

    #[test]
    fn variant_merges_defines_and_keeps_original() {
        let device = mock_device();
        let base =
            ShaderModule::from_slang_with_paths_and_defines(&device, "void main() {}", &[], &[("A", "1"), ("B", "2")])
                .expect("shader");
        let variant = base.variant(&[("B", "9"), ("C", "3")]).expect("variant");

        assert_ne!(base.handle, variant.handle);
        assert_eq!(
            base.defines(),
            &[("A".to_string(), "1".to_string()), ("B".to_string(), "2".to_string())]
        );
        assert_eq!(
            variant.defines(),
            &[
                ("A".to_string(), "1".to_string()),
                ("B".to_string(), "9".to_string()),
                ("C".to_string(), "3".to_string()),
            ]
        );
        assert_eq!(variant.source(), base.source());
        assert_eq!(variant.search_paths(), base.search_paths());
    }

    #[test]
    fn effective_source_is_cached_per_module() {
        let device = mock_device();
        let shader = ShaderModule::from_slang(
            &device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1,1,1)]
void cs_main(Scattered<uint> buf, ThreadId id) { buf[0] = 1; }
"#,
        )
        .expect("shader");
        let first = shader.effective_source_arc();
        let second = shader.effective_source_arc();
        assert!(Arc::ptr_eq(&first, &second));
        assert_ne!(first.as_ref(), shader.source());
    }

    #[test]
    fn compute_pipeline_new_on_mock_does_not_need_slang_target() {
        let device = mock_device();
        let shader = ShaderModule::from_slang(&device, "void main() {}").expect("shader");
        {
            let backend = device.inner.backend.lock().unwrap();
            assert!(backend.compute_shader_target().is_none());
        }
        ComputePipeline::new(&device, &shader).expect("mock compute pipeline");
    }
}
