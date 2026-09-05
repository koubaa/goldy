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
use std::sync::{Arc, Mutex};

/// A compiled shader module.
pub struct ShaderModule {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: ShaderHandle,
    /// Authored Slang source used for graphics linking (virtual-main parse).
    #[cfg_attr(not(feature = "graphics"), allow(dead_code))]
    pub(crate) slang_source: String,
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

        let path_refs: Vec<&str> = all_paths.iter().map(|s| s.as_str()).collect();

        let mut backend = device.inner.backend.lock().unwrap();
        let handle = if validate {
            let mut owned_checks = generated_checks;
            if validate_authored {
                owned_checks.extend(layout_checks.iter().map(OwnedLayoutCheck::from_layout_check));
            }
            backend.create_shader_with_checks(
                device.inner.handle,
                source,
                &path_refs,
                defines,
                optimization_level,
                owned_checks,
            )?
        } else {
            backend.create_shader_with_paths(device.inner.handle, source, &path_refs, defines, optimization_level)?
        };

        tracing::debug!("Shader module created");

        Ok(Self {
            _device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            slang_source: source.to_string(),
        })
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
