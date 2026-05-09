//! High-level Slang compiler API.
//!
//! Provides a safe, ergonomic interface for compiling Slang shaders.

use anyhow::{Context, Result};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

use super::ffi::*;
use super::loader::SlangLibrary;
use super::virtual_main::transform_virtual_main;
use crate::types::OptimizationLevel;
use crate::{goldy_event, goldy_span};

/// Returns `true` when layout validation is enabled.
///
/// This is on when:
/// - `GOLDY_VALIDATE_LAYOUTS` is `1`, `true`, or `yes` (unchanged), or
/// - `GOLDY_VALIDATION` lists `layout` / `layouts` / `all` (see `validation_env`).
///
/// Note: `GOLDY_VALIDATION=1|true|yes` enables **GPU API** validation only, not layout checks.
///
/// Controls both struct layout checks (at compile time) and buffer element-stride
/// checks (at dispatch time). Reads the environment on every call so that tests
/// can toggle the flag without restarting the process.
pub fn layout_validation_enabled() -> bool {
    crate::validation_env::layout_validation_enabled()
}

// ============================================================================
// Reflection data structures
// ============================================================================

/// Kind of resource in a parameter block field
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    /// A buffer (StructuredBuffer, RWStructuredBuffer, etc.)
    Buffer,
    /// A mutable buffer (RWStructuredBuffer, RWByteAddressBuffer)
    MutableBuffer,
    /// A texture (Texture2D, etc.)
    Texture,
    /// A mutable texture (RWTexture2D)
    MutableTexture,
    /// A sampler state
    Sampler,
    /// A constant buffer / uniform block
    ConstantBuffer,
    /// A nested parameter block
    ParameterBlock,
    /// Other/unknown
    Other,
}

/// Layout information for a single field within a ParameterBlock
#[derive(Debug, Clone)]
pub struct FieldLayout {
    /// Name of the field
    pub name: String,
    /// Offset in bytes from the start of the containing struct
    pub offset: usize,
    /// Size in bytes
    pub size: usize,
    /// What kind of resource this field represents
    pub resource_kind: ResourceKind,
    /// Type name (e.g., `StructuredBuffer<Particle>`)
    pub type_name: String,
}

/// Layout information for a ParameterBlock
#[derive(Debug, Clone)]
pub struct ParameterBlockLayout {
    /// Name of the parameter (from shader)
    pub name: String,
    /// Binding slot (for Metal: buffer index)
    pub binding_slot: u32,
    /// Binding space/set
    pub binding_space: u32,
    /// Total size of the argument buffer in bytes
    pub size: usize,
    /// Alignment requirement
    pub alignment: usize,
    /// Fields within the parameter block
    pub fields: Vec<FieldLayout>,
}

/// Complete reflection information for a compiled shader
#[derive(Debug, Clone, Default)]
pub struct ShaderReflection {
    /// All parameter blocks found in the shader
    pub parameter_blocks: Vec<ParameterBlockLayout>,
    /// Per push-constant slot, the [`BindlessCategory`](crate::types::BindlessCategory)
    /// the shader expects. Populated from `[goldy_*]` entry-point analysis at compile time.
    /// Used by backend validation when `BindResourcesTyped` is used to catch category
    /// mismatches against the shader's reflected expectations.
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
}

/// Byte layout of a Slang `struct` under uniform / constant-buffer rules (`SlangLayoutRules::Default`).
///
/// Used with [`StructLayout::validate`] to compare against a Rust `#[repr(C)]` type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructLayout {
    /// Reflected type name (as requested, e.g. `SceneUniforms`).
    pub name: String,
    /// Total size in bytes.
    pub size: usize,
    /// Alignment in bytes (Slang-reported for the uniform category).
    pub alignment: usize,
    pub fields: Vec<StructFieldLayout>,
}

/// One field in a [`StructLayout`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructFieldLayout {
    pub name: String,
    /// Byte offset from the start of the struct.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
    /// Slang type name (e.g. `float4x4`, `float2`).
    pub type_name: String,
}

impl StructLayout {
    /// Compare this Slang layout against Rust `size_of` / `offset_of!` / per-field `size_of`.
    ///
    /// Validation rules:
    /// - Every field declared in the shader must exist in the Rust struct with a matching name,
    ///   byte offset, and size. A missing or mismatched shader field is a hard error.
    /// - The Rust struct must be large enough to cover all shader-declared data (i.e. ≥ the last
    ///   shader field's end byte). Tail padding added by constant-buffer alignment rules is *not*
    ///   required to be present in the Rust struct.
    /// - Extra Rust fields that have no counterpart in the shader are allowed (they are padding or
    ///   bookkeeping). A warning is emitted for non-`_`-prefixed extras so genuine "forgot to add
    ///   this field to the shader" mistakes are visible. Prefix with `_` to silence the warning.
    pub fn validate(&self, rust_size: usize, rust_fields: &[(&str, usize, usize)]) -> Result<()> {
        let mut errors: Vec<String> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();

        // Data extent: the last byte actually declared by the shader (excludes CB tail padding).
        let slang_data_extent = self
            .fields
            .iter()
            .map(|f| f.offset + f.size)
            .max()
            .unwrap_or(0);

        if rust_size < slang_data_extent {
            errors.push(format!(
                "Rust struct ({rust_size} bytes) is smaller than the shader's data extent \
                 ({slang_data_extent} bytes); all shader fields must fit inside the Rust struct"
            ));
        }

        // Direction 1: every Slang field must exist in Rust with matching offset and size.
        for sf in &self.fields {
            match rust_fields.iter().find(|&&(name, _, _)| name == sf.name) {
                Some(&(_, rust_offset, rust_size_field)) => {
                    if sf.offset != rust_offset {
                        errors.push(format!(
                            "field `{}`: offset Slang {} vs Rust {}",
                            sf.name, sf.offset, rust_offset
                        ));
                    }
                    if sf.size != rust_size_field {
                        errors.push(format!(
                            "field `{}`: size Slang {} vs Rust {}",
                            sf.name, sf.size, rust_size_field
                        ));
                    }
                }
                None => {
                    errors.push(format!(
                        "field `{}` is declared in the shader but missing from the Rust struct",
                        sf.name
                    ));
                }
            }
        }

        // Direction 2: Rust fields absent from the shader — warn for non-`_`-prefixed ones.
        for &(name, _, _) in rust_fields {
            if !self.fields.iter().any(|sf| sf.name == name) && !name.starts_with('_') {
                warnings.push(format!(
                    "field `{name}` is in the Rust struct but not in the shader \
                     (prefix with `_` to suppress this warning)"
                ));
            }
        }

        if !warnings.is_empty() {
            tracing::warn!("Layout check for `{}`: {}", self.name, warnings.join("; "));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            anyhow::bail!(
                "Struct layout mismatch for `{}`:\n{}",
                self.name,
                errors.join("\n")
            );
        }
    }
}

/// Opt-in Rust vs Slang struct layout checks run during the same compile as GPU codegen.
///
/// Pass a slice to [`ShaderModule::from_slang_with_options`](crate::ShaderModule::from_slang_with_options).
/// Checks only run when `GOLDY_VALIDATE_LAYOUTS=1` is set.
#[derive(Debug, Clone, Copy)]
pub struct LayoutCheck<'a> {
    pub type_name: &'a str,
    pub rust_size: usize,
    pub rust_fields: &'a [(&'a str, usize, usize)],
}

/// Stored on backend [`ShaderState`](crate::backend) for deferred per-stage compilation.
#[derive(Debug, Clone)]
pub struct OwnedLayoutCheck {
    pub type_name: String,
    pub rust_size: usize,
    pub rust_fields: Vec<(String, usize, usize)>,
}

impl OwnedLayoutCheck {
    pub fn from_layout_check(c: &LayoutCheck<'_>) -> Self {
        Self {
            type_name: c.type_name.to_string(),
            rust_size: c.rust_size,
            rust_fields: c
                .rust_fields
                .iter()
                .map(|(n, o, s)| ((*n).to_string(), *o, *s))
                .collect(),
        }
    }
}

/// Compiled shader output with optional reflection data.
#[derive(Debug, Clone)]
pub struct CompiledShaderWithReflection {
    /// The compiled shader
    pub shader: CompiledShader,
    /// Reflection data (if requested)
    pub reflection: ShaderReflection,
}

/// Shader compilation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderTarget {
    /// SPIR-V bytecode for Vulkan
    Spirv,
    /// DXIL bytecode for DirectX 12 (binary, SM 6.6 for bindless)
    Dxil,
    /// Metal Shading Language
    Metal,
}

impl ShaderTarget {
    fn to_slang_target(self) -> SlangCompileTarget {
        match self {
            ShaderTarget::Spirv => SlangCompileTarget::Spirv,
            ShaderTarget::Dxil => SlangCompileTarget::Dxil,
            ShaderTarget::Metal => SlangCompileTarget::Metal,
        }
    }

    /// Returns true if this target produces binary bytecode (not text).
    pub fn is_binary(self) -> bool {
        matches!(self, ShaderTarget::Spirv | ShaderTarget::Dxil)
    }
}

/// Compiled shader output.
#[derive(Debug, Clone)]
pub struct CompiledShader {
    /// The compiled bytecode or source code
    pub data: Vec<u8>,
    /// The target format
    pub target: ShaderTarget,
}

impl CompiledShader {
    /// Get the data as a string (for text-based targets like Metal).
    pub fn as_str(&self) -> Option<&str> {
        match self.target {
            ShaderTarget::Metal => std::str::from_utf8(&self.data).ok(),
            ShaderTarget::Spirv | ShaderTarget::Dxil => None,
        }
    }

    /// Get the data as SPIR-V words (for Vulkan).
    pub fn as_spirv(&self) -> Option<&[u32]> {
        if self.target == ShaderTarget::Spirv && self.data.len().is_multiple_of(4) {
            Some(bytemuck::cast_slice(&self.data))
        } else {
            None
        }
    }

    /// Get the data as DXIL bytecode (for DirectX 12).
    pub fn as_dxil(&self) -> Option<&[u8]> {
        if self.target == ShaderTarget::Dxil {
            Some(&self.data)
        } else {
            None
        }
    }
}

/// Slang shader compiler.
///
/// Thread-safe wrapper around the Slang compilation API.
pub struct SlangCompiler {
    library: Arc<SlangLibrary>,
    global_session: *mut IGlobalSession,
}

// SlangCompiler is Send + Sync because each compilation creates its own request
unsafe impl Send for SlangCompiler {}
unsafe impl Sync for SlangCompiler {}

impl SlangCompiler {
    /// Create a new Slang compiler instance.
    pub fn new() -> Result<Self> {
        let _span = goldy_span!("slang.compiler.init").entered();

        let library = Arc::new(SlangLibrary::load()?);

        // Create global session using the new COM API
        let mut global_session: *mut IGlobalSession = ptr::null_mut();
        let global_desc = SlangGlobalSessionDesc::default();
        tracing::debug!(
            "Creating global session with desc size: {}",
            std::mem::size_of::<SlangGlobalSessionDesc>()
        );
        let result = unsafe { (library.create_global_session)(&global_desc, &mut global_session) };

        if !slang_succeeded(result) || global_session.is_null() {
            anyhow::bail!(
                "Failed to create Slang global session (result={}, ptr={:?})",
                result,
                global_session
            );
        }
        tracing::debug!("Global session created: {:?}", global_session);

        goldy_event!("slang.session.create", success = true);
        tracing::info!("Slang compiler initialized");

        Ok(Self {
            library,
            global_session,
        })
    }

    /// Compile with reflection data.
    ///
    /// Returns both the compiled shader and reflection information about
    /// ParameterBlocks, which is needed to properly set up argument buffers.
    ///
    /// Target-specific preprocessor defines (`__SPIRV__`, `__DX12__`, `__METAL__`) are applied
    /// automatically (same as [`Self::compile_bindless_with_reflection_and_defines`] with no
    /// extra defines).
    pub fn compile_bindless_with_reflection(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
    ) -> Result<CompiledShaderWithReflection> {
        self.compile_bindless_with_reflection_and_defines(
            source,
            target,
            entry_points,
            search_paths,
            &[],
            &[],
            OptimizationLevel::Default,
        )
    }

    /// Like [`Self::compile_bindless_with_reflection`] but with extra preprocessor defines.
    ///
    /// Extra defines are merged with target-specific defines (e.g. `__SPIRV__`, `__DX12__`).
    /// Use for shader variants like `msaa`, `msaa8`, `msaa16`.
    #[allow(clippy::too_many_arguments)] // layout_checks + defines + paths are all required at call sites
    pub fn compile_bindless_with_reflection_and_defines(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        extra_defines: &[(&str, &str)],
        layout_checks: &[OwnedLayoutCheck],
        optimization_level: OptimizationLevel,
    ) -> Result<CompiledShaderWithReflection> {
        let mut defines = Self::bindless_defines_for_target(target);
        defines.extend_from_slice(extra_defines);
        self.compile_with_reflection(
            source,
            target,
            entry_points,
            search_paths,
            &defines,
            layout_checks,
            optimization_level,
        )
    }

    /// Get preprocessor defines for the given target.
    fn bindless_defines_for_target(target: ShaderTarget) -> Vec<(&'static str, &'static str)> {
        match target {
            ShaderTarget::Spirv => vec![("__SPIRV__", "1")],
            ShaderTarget::Dxil => vec![("__DX12__", "1")],
            ShaderTarget::Metal => vec![("__METAL__", "1")],
        }
    }

    /// Shared session + compile path; invokes `f` with the live compile request after `spCompile` succeeds.
    #[allow(clippy::too_many_arguments)] // mirrors public compile entry points
    fn with_compiled_request<R>(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: OptimizationLevel,
        f: impl FnOnce(&Self, *mut SlangCompileRequest, i32) -> Result<R>,
    ) -> Result<R> {
        // Create session with session-level preprocessor macros.
        let define_names: Vec<CString> = defines
            .iter()
            .map(|(k, _)| CString::new(*k).unwrap())
            .collect();
        let define_values: Vec<CString> = defines
            .iter()
            .map(|(_, v)| CString::new(*v).unwrap())
            .collect();
        let macro_descs: Vec<PreprocessorMacroDesc> = define_names
            .iter()
            .zip(define_values.iter())
            .map(|(name, value)| PreprocessorMacroDesc {
                name: name.as_ptr(),
                value: value.as_ptr(),
            })
            .collect();

        let search_path_cstrings: Vec<CString> = search_paths
            .iter()
            .map(|p| CString::new(*p).unwrap())
            .collect();
        let search_path_ptrs: Vec<*const c_char> =
            search_path_cstrings.iter().map(|s| s.as_ptr()).collect();

        let mut session_desc = SessionDesc::default();
        if !search_path_ptrs.is_empty() {
            session_desc.search_paths = search_path_ptrs.as_ptr();
            session_desc.search_path_count = search_path_ptrs.len() as i64;
        }
        if !macro_descs.is_empty() {
            session_desc.preprocessor_macros = macro_descs.as_ptr();
            session_desc.preprocessor_macro_count = macro_descs.len() as i64;
        }

        tracing::debug!(
            "Creating session with {} macros, SessionDesc size: {}",
            macro_descs.len(),
            std::mem::size_of::<SessionDesc>()
        );
        let mut session: *mut ISession = ptr::null_mut();
        let result = unsafe {
            global_session_create_session(self.global_session, &session_desc, &mut session)
        };
        if !slang_succeeded(result) || session.is_null() {
            anyhow::bail!(
                "Failed to create Slang session with preprocessor defines (result={}, ptr={:?})",
                result,
                session
            );
        }
        tracing::debug!("Session with defines created: {:?}", session);

        let _session_guard = scopeguard::guard(session, |s| {
            unsafe { session_release(s) };
        });

        let mut request: *mut SlangCompileRequest = ptr::null_mut();
        let result = unsafe { session_create_compile_request(session, &mut request) };
        if !slang_succeeded(result) || request.is_null() {
            anyhow::bail!(
                "Failed to create Slang compile request (result={}, ptr={:?})",
                result,
                request
            );
        }
        tracing::debug!("Compile request created: {:?}", request);

        let library = self.library.clone();
        let _guard = scopeguard::guard(request, |req| {
            unsafe { (library.destroy_compile_request)(req) };
        });

        let target_index =
            unsafe { (self.library.add_code_gen_target)(request, target.to_slang_target() as i32) };
        if target_index < 0 {
            anyhow::bail!("Failed to add code generation target");
        }

        if target == ShaderTarget::Dxil {
            let profile_name = CString::new("sm_6_6").unwrap();
            let profile_id =
                unsafe { global_session_find_profile(self.global_session, profile_name.as_ptr()) };
            if profile_id > 0 {
                unsafe {
                    (self.library.set_target_profile)(request, target_index, profile_id);
                }
                tracing::debug!("Set DXIL target profile to sm_6_6 (id={})", profile_id);
            } else {
                tracing::warn!("Could not find sm_6_6 profile, using default");
            }
            unsafe {
                (self.library.set_target_floating_point_mode)(
                    request,
                    target_index,
                    SLANG_FLOATING_POINT_MODE_PRECISE,
                );
            }
        }

        let unit_name = CString::new("shader").unwrap();
        let translation_unit = unsafe {
            (self.library.add_translation_unit)(
                request,
                SlangSourceLanguage::Slang as i32,
                unit_name.as_ptr(),
            )
        };
        if translation_unit < 0 {
            anyhow::bail!("Failed to add translation unit");
        }

        // Apply virtual-main transform: [goldy_*] entry points → [shader(...)] wrappers.
        let transformed_source;
        let source = if source.contains("[goldy_compute]")
            || source.contains("[goldy_vertex]")
            || source.contains("[goldy_fragment]")
        {
            transformed_source = transform_virtual_main(source);
            &transformed_source as &str
        } else {
            source
        };

        let source_path = CString::new("shader.slang").unwrap();
        let source_cstr = CString::new(source).context("Source contains null bytes")?;
        unsafe {
            (self.library.add_translation_unit_source_string)(
                request,
                translation_unit,
                source_path.as_ptr(),
                source_cstr.as_ptr(),
            );
        }

        for (name, stage) in entry_points {
            let name_cstr = CString::new(*name).context("Entry point name contains null bytes")?;
            let entry_index = unsafe {
                (self.library.add_entry_point)(
                    request,
                    translation_unit,
                    name_cstr.as_ptr(),
                    *stage as i32,
                )
            };
            if entry_index < 0 {
                anyhow::bail!("Failed to add entry point: {}", name);
            }
        }

        if optimization_level != OptimizationLevel::Default {
            let ffi_level = match optimization_level {
                OptimizationLevel::None => SLANG_OPTIMIZATION_LEVEL_NONE,
                OptimizationLevel::Default => unreachable!(),
                OptimizationLevel::High => SLANG_OPTIMIZATION_LEVEL_HIGH,
                OptimizationLevel::Maximal => SLANG_OPTIMIZATION_LEVEL_MAXIMAL,
            };
            unsafe { (self.library.set_optimization_level)(request, ffi_level) };
            tracing::info!("Slang optimization level set to {:?}", optimization_level);
        }

        let result = unsafe { (self.library.compile)(request) };
        if !slang_succeeded(result) {
            let diag_ptr = unsafe { (self.library.get_diagnostic_output)(request) };
            let diagnostic = if !diag_ptr.is_null() {
                unsafe { CStr::from_ptr(diag_ptr) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown compilation error".to_string()
            };
            anyhow::bail!("Slang compilation failed:\n{}", diagnostic);
        }

        f(self, request, target_index)
    }

    /// Compile with reflection data.
    ///
    /// This performs compilation and extracts reflection information about
    /// all parameters, especially ParameterBlocks for bindless rendering.
    ///
    /// When `layout_checks` is non-empty, each struct is reflected from the same compile request
    /// and validated before returning (see [`OwnedLayoutCheck`]).
    #[allow(clippy::too_many_arguments)] // Slang compile inputs are naturally wide
    pub fn compile_with_reflection(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        defines: &[(&str, &str)],
        layout_checks: &[OwnedLayoutCheck],
        optimization_level: OptimizationLevel,
    ) -> Result<CompiledShaderWithReflection> {
        self.with_compiled_request(
            source,
            target,
            entry_points,
            search_paths,
            defines,
            optimization_level,
            |slf, request, target_index| {
                let mut blob: *mut ISlangBlob = ptr::null_mut();
                let result = unsafe {
                    (slf.library.get_entry_point_code_blob)(request, 0, target_index, &mut blob)
                };

                if !slang_succeeded(result) || blob.is_null() {
                    anyhow::bail!("Failed to get compiled shader code");
                }

                let (data_ptr, data_size) = unsafe { blob_get_data(blob) };
                let data = unsafe { std::slice::from_raw_parts(data_ptr, data_size) }.to_vec();
                unsafe { blob_release(blob) };

                let reflection = slf.extract_reflection(request)?;

                if !layout_checks.is_empty() {
                    slf.validate_owned_layout_checks(request, layout_checks)?;
                }

                Ok(CompiledShaderWithReflection {
                    shader: CompiledShader { data, target },
                    reflection,
                })
            },
        )
    }

    fn validate_owned_layout_checks(
        &self,
        request: *mut SlangCompileRequest,
        checks: &[OwnedLayoutCheck],
    ) -> Result<()> {
        for owned in checks {
            let layout = self.reflect_named_struct_from_request(request, &owned.type_name)?;
            let field_refs: Vec<(&str, usize, usize)> = owned
                .rust_fields
                .iter()
                .map(|(n, o, s)| (n.as_str(), *o, *s))
                .collect();
            layout.validate(owned.rust_size, &field_refs)?;
        }
        Ok(())
    }

    /// Compile `shader_source` with bindless target defines and return the uniform layout of struct `type_name`.
    ///
    /// `shader_source` must declare a vertex entry point named **`vs_main`** (same convention as typical goldy shaders).
    pub fn reflect_struct_layout(
        &self,
        shader_source: &str,
        target: ShaderTarget,
        search_paths: &[&str],
        type_name: &str,
    ) -> Result<StructLayout> {
        let defines = Self::bindless_defines_for_target(target);
        let entry_points = &[("vs_main", SlangStage::Vertex)];
        self.with_compiled_request(
            shader_source,
            target,
            entry_points,
            search_paths,
            &defines,
            OptimizationLevel::Default,
            |slf, request, _target_index| slf.reflect_named_struct_from_request(request, type_name),
        )
    }

    fn reflect_named_struct_from_request(
        &self,
        request: *mut SlangCompileRequest,
        type_name: &str,
    ) -> Result<StructLayout> {
        let reflection_ptr = unsafe { (self.library.get_reflection)(request) };
        if reflection_ptr.is_null() {
            anyhow::bail!("No Slang reflection available after compile");
        }

        let name_cstr = CString::new(type_name).context("type_name contains null bytes")?;
        let ty = unsafe {
            (self.library.reflection_find_type_by_name)(reflection_ptr, name_cstr.as_ptr())
        };
        if ty.is_null() {
            anyhow::bail!("Slang reflection: type `{type_name}` not found");
        }

        let layout_ptr = unsafe {
            (self.library.reflection_get_type_layout)(reflection_ptr, ty, SlangLayoutRules::Default)
        };
        if layout_ptr.is_null() {
            anyhow::bail!("Slang reflection: failed to get layout for `{type_name}`");
        }

        self.extract_struct_layout_uniform(layout_ptr, type_name)
    }

    fn extract_struct_layout_uniform(
        &self,
        type_layout: *mut SlangReflectionTypeLayout,
        struct_name: &str,
    ) -> Result<StructLayout> {
        let cat = SlangParameterCategory::Uniform as i32;
        let size = unsafe { (self.library.reflection_type_layout_get_size)(type_layout, cat) };
        let alignment =
            unsafe { (self.library.reflection_type_layout_get_alignment)(type_layout, cat) };

        let field_count =
            unsafe { (self.library.reflection_type_layout_get_field_count)(type_layout) };

        let mut fields = Vec::new();
        for i in 0..field_count {
            let field_var =
                unsafe { (self.library.reflection_type_layout_get_field_by_index)(type_layout, i) };
            if field_var.is_null() {
                continue;
            }

            let variable =
                unsafe { (self.library.reflection_variable_layout_get_variable)(field_var) };
            let name = if !variable.is_null() {
                let name_ptr = unsafe { (self.library.reflection_variable_get_name)(variable) };
                if !name_ptr.is_null() {
                    unsafe { CStr::from_ptr(name_ptr) }
                        .to_string_lossy()
                        .into_owned()
                } else {
                    format!("field_{i}")
                }
            } else {
                format!("field_{i}")
            };

            let field_type_layout =
                unsafe { (self.library.reflection_variable_layout_get_type_layout)(field_var) };
            if field_type_layout.is_null() {
                continue;
            }

            let offset =
                unsafe { (self.library.reflection_variable_layout_get_offset)(field_var, cat) };
            let fsize =
                unsafe { (self.library.reflection_type_layout_get_size)(field_type_layout, cat) };

            let field_type =
                unsafe { (self.library.reflection_type_layout_get_type)(field_type_layout) };
            let type_name = if !field_type.is_null() {
                let type_name_ptr = unsafe { (self.library.reflection_type_get_name)(field_type) };
                if !type_name_ptr.is_null() {
                    unsafe { CStr::from_ptr(type_name_ptr) }
                        .to_string_lossy()
                        .into_owned()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            fields.push(StructFieldLayout {
                name,
                offset,
                size: fsize,
                type_name,
            });
        }

        Ok(StructLayout {
            name: struct_name.to_string(),
            size,
            alignment,
            fields,
        })
    }

    /// Extract reflection data from a compiled request.
    fn extract_reflection(&self, request: *mut SlangCompileRequest) -> Result<ShaderReflection> {
        let _span = goldy_span!("slang.reflection.extract").entered();

        let reflection_ptr = unsafe { (self.library.get_reflection)(request) };
        if reflection_ptr.is_null() {
            return Ok(ShaderReflection::default());
        }

        let mut parameter_blocks = Vec::new();

        // Get parameter count
        let param_count = unsafe { (self.library.reflection_get_parameter_count)(reflection_ptr) };

        for i in 0..param_count {
            let param =
                unsafe { (self.library.reflection_get_parameter_by_index)(reflection_ptr, i) };
            if param.is_null() {
                continue;
            }

            // Get parameter name (parameter -> variable -> name)
            let variable = unsafe { (self.library.reflection_variable_layout_get_variable)(param) };
            let name = if !variable.is_null() {
                let name_ptr = unsafe { (self.library.reflection_variable_get_name)(variable) };
                if !name_ptr.is_null() {
                    unsafe { CStr::from_ptr(name_ptr) }
                        .to_string_lossy()
                        .into_owned()
                } else {
                    format!("param_{}", i)
                }
            } else {
                format!("param_{}", i)
            };

            // Get type layout
            let type_layout = unsafe { (self.library.reflection_parameter_get_type_layout)(param) };
            if type_layout.is_null() {
                continue;
            }

            // Get the type to check if it's a ParameterBlock
            let type_ptr = unsafe { (self.library.reflection_type_layout_get_type)(type_layout) };
            if type_ptr.is_null() {
                continue;
            }

            let type_kind = unsafe { (self.library.reflection_type_get_kind)(type_ptr) };

            // Check if this is a ParameterBlock
            if type_kind == SlangTypeKind::ParameterBlock as i32 {
                let block_layout =
                    self.extract_parameter_block_layout(param, type_layout, &name)?;
                parameter_blocks.push(block_layout);
            }
        }

        goldy_event!(
            "slang.reflection.extract",
            parameter_blocks = parameter_blocks.len(),
            total_fields = parameter_blocks
                .iter()
                .map(|pb| pb.fields.len())
                .sum::<usize>()
        );

        Ok(ShaderReflection {
            parameter_blocks,
            push_constant_categories: Vec::new(),
        })
    }

    /// Extract layout information for a ParameterBlock.
    fn extract_parameter_block_layout(
        &self,
        param: *mut SlangReflectionParameter,
        type_layout: *mut SlangReflectionTypeLayout,
        name: &str,
    ) -> Result<ParameterBlockLayout> {
        // Get binding information
        let binding_slot =
            unsafe { (self.library.reflection_parameter_get_binding_index)(param) } as u32;
        let binding_space =
            unsafe { (self.library.reflection_parameter_get_binding_space)(param) } as u32;

        // Get the element type layout (the T in ParameterBlock<T>)
        let element_type_layout =
            unsafe { (self.library.reflection_type_layout_get_element_type_layout)(type_layout) };

        // Get size, alignment, and fields from the element type
        // Note: Slang returns slot counts, not byte sizes. Each slot = 8 bytes.
        const SLOT_SIZE_BYTES: usize = 8;

        let (mut size, alignment, fields) = if !element_type_layout.is_null() {
            // Try MetalArgumentBufferElement first (for argument buffers with resources)
            let size_slots = unsafe {
                (self.library.reflection_type_layout_get_size)(
                    element_type_layout,
                    SlangParameterCategory::MetalArgumentBufferElement as i32,
                )
            };
            let alignment = unsafe {
                (self.library.reflection_type_layout_get_alignment)(
                    element_type_layout,
                    SlangParameterCategory::MetalArgumentBufferElement as i32,
                )
            };
            let fields = self.extract_struct_fields(element_type_layout)?;
            // Convert slots to bytes
            let size = size_slots * SLOT_SIZE_BYTES;
            (size, alignment, fields)
        } else {
            // Fallback: use the type_layout directly
            let size = unsafe {
                (self.library.reflection_type_layout_get_size)(
                    type_layout,
                    SlangParameterCategory::Uniform as i32,
                )
            };
            let alignment = unsafe {
                (self.library.reflection_type_layout_get_alignment)(
                    type_layout,
                    SlangParameterCategory::Uniform as i32,
                )
            };
            (size, alignment, Vec::new())
        };

        // If size is still 0, calculate from fields (each resource pointer is 8 bytes)
        if size == 0 && !fields.is_empty() {
            size = fields.iter().map(|f| f.offset + f.size).max().unwrap_or(0);
        }

        // Alignment from Slang reflection is also in slots, convert to bytes
        // For Metal argument buffers, minimum alignment is 8 bytes (pointer size)
        let alignment_bytes = if alignment > 0 {
            alignment * SLOT_SIZE_BYTES
        } else {
            SLOT_SIZE_BYTES // Default to 8-byte alignment
        };

        Ok(ParameterBlockLayout {
            name: name.to_string(),
            binding_slot,
            binding_space,
            size,
            alignment: alignment_bytes,
            fields,
        })
    }

    /// Extract field layouts from a struct type (used for ParameterBlock element types).
    fn extract_struct_fields(
        &self,
        type_layout: *mut SlangReflectionTypeLayout,
    ) -> Result<Vec<FieldLayout>> {
        let mut fields = Vec::new();

        let field_count =
            unsafe { (self.library.reflection_type_layout_get_field_count)(type_layout) };

        for i in 0..field_count {
            let field_var =
                unsafe { (self.library.reflection_type_layout_get_field_by_index)(type_layout, i) };
            if field_var.is_null() {
                continue;
            }

            // Get field name (variable layout -> variable -> name)
            let variable =
                unsafe { (self.library.reflection_variable_layout_get_variable)(field_var) };
            let name = if !variable.is_null() {
                let name_ptr = unsafe { (self.library.reflection_variable_get_name)(variable) };
                if !name_ptr.is_null() {
                    unsafe { CStr::from_ptr(name_ptr) }
                        .to_string_lossy()
                        .into_owned()
                } else {
                    format!("field_{}", i)
                }
            } else {
                format!("field_{}", i)
            };

            // Get field type layout
            let field_type_layout =
                unsafe { (self.library.reflection_variable_layout_get_type_layout)(field_var) };
            if field_type_layout.is_null() {
                continue;
            }

            // Determine resource kind
            let resource_kind = self.determine_resource_kind(field_type_layout);

            // For Metal argument buffers (ParameterBlock context), try MetalArgumentBufferElement
            // category first. This handles buffers, textures, and other resources correctly.
            // Slang returns SLOT indices, not byte offsets. Each slot is 8 bytes (GPU pointer size).
            let offset_slots = unsafe {
                (self.library.reflection_variable_layout_get_offset)(
                    field_var,
                    SlangParameterCategory::MetalArgumentBufferElement as i32,
                )
            };
            let size_slots = unsafe {
                (self.library.reflection_type_layout_get_size)(
                    field_type_layout,
                    SlangParameterCategory::MetalArgumentBufferElement as i32,
                )
            };

            // Convert slot counts to byte offsets/sizes (each slot = 8 bytes = GPU pointer)
            const SLOT_SIZE_BYTES: usize = 8;
            let offset = offset_slots * SLOT_SIZE_BYTES;
            let size = if size_slots > 0 {
                size_slots * SLOT_SIZE_BYTES
            } else {
                SLOT_SIZE_BYTES
            };

            tracing::trace!(
                "Field {} (index {}): offset_slots={}, size_slots={} -> offset={}, size={}, resource_kind={:?}",
                name, i, offset_slots, size_slots, offset, size, resource_kind
            );

            // Get type name
            let field_type =
                unsafe { (self.library.reflection_type_layout_get_type)(field_type_layout) };
            let type_name = if !field_type.is_null() {
                let type_name_ptr = unsafe { (self.library.reflection_type_get_name)(field_type) };
                if !type_name_ptr.is_null() {
                    unsafe { CStr::from_ptr(type_name_ptr) }
                        .to_string_lossy()
                        .into_owned()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            fields.push(FieldLayout {
                name,
                offset,
                size,
                resource_kind,
                type_name,
            });
        }

        Ok(fields)
    }

    /// Determine the resource kind from a type layout.
    fn determine_resource_kind(&self, type_layout: *mut SlangReflectionTypeLayout) -> ResourceKind {
        let type_ptr = unsafe { (self.library.reflection_type_layout_get_type)(type_layout) };
        if type_ptr.is_null() {
            return ResourceKind::Other;
        }

        let type_kind = unsafe { (self.library.reflection_type_get_kind)(type_ptr) };
        let binding_type =
            unsafe { (self.library.reflection_type_layout_get_binding_type)(type_layout) };

        // Debug logging for type detection
        tracing::trace!(
            "determine_resource_kind: type_kind={}, binding_type={}",
            type_kind,
            binding_type
        );

        match type_kind {
            k if k == SlangTypeKind::SamplerState as i32 => ResourceKind::Sampler,
            k if k == SlangTypeKind::ConstantBuffer as i32 => ResourceKind::ConstantBuffer,
            k if k == SlangTypeKind::ParameterBlock as i32 => ResourceKind::ParameterBlock,
            k if k == SlangTypeKind::Resource as i32 => {
                // Check binding type to distinguish buffer vs texture, mutable vs immutable
                match binding_type {
                    b if b == SlangBindingType::Texture as i32 => ResourceKind::Texture,
                    b if b == SlangBindingType::MutableTexture as i32 => {
                        ResourceKind::MutableTexture
                    }
                    b if b == SlangBindingType::TypedBuffer as i32 => ResourceKind::Buffer,
                    b if b == SlangBindingType::MutableTypedBuffer as i32 => {
                        ResourceKind::MutableBuffer
                    }
                    b if b == SlangBindingType::RawBuffer as i32 => ResourceKind::Buffer,
                    b if b == SlangBindingType::MutableRawBuffer as i32 => {
                        ResourceKind::MutableBuffer
                    }
                    _ => ResourceKind::Other,
                }
            }
            k if k == SlangTypeKind::ShaderStorageBuffer as i32 => ResourceKind::MutableBuffer,
            _ => {
                // Try to infer from binding type if type_kind doesn't match expected values
                // This helps with StructuredBuffer which may have different type_kind
                match binding_type {
                    b if b == SlangBindingType::TypedBuffer as i32 => ResourceKind::Buffer,
                    b if b == SlangBindingType::MutableTypedBuffer as i32 => {
                        ResourceKind::MutableBuffer
                    }
                    b if b == SlangBindingType::RawBuffer as i32 => ResourceKind::Buffer,
                    b if b == SlangBindingType::MutableRawBuffer as i32 => {
                        ResourceKind::MutableBuffer
                    }
                    b if b == SlangBindingType::Texture as i32 => ResourceKind::Texture,
                    b if b == SlangBindingType::MutableTexture as i32 => {
                        ResourceKind::MutableTexture
                    }
                    b if b == SlangBindingType::Sampler as i32 => ResourceKind::Sampler,
                    b if b == SlangBindingType::ConstantBuffer as i32 => {
                        ResourceKind::ConstantBuffer
                    }
                    _ => ResourceKind::Other,
                }
            }
        }
    }
}

#[cfg(test)]
mod struct_layout_validate_tests {
    use super::{StructFieldLayout, StructLayout};
    use crate as goldy;

    fn two_float_layout() -> StructLayout {
        StructLayout {
            name: "S".into(),
            size: 8,
            alignment: 4,
            fields: vec![
                StructFieldLayout {
                    name: "a".into(),
                    offset: 0,
                    size: 4,
                    type_name: "float".into(),
                },
                StructFieldLayout {
                    name: "b".into(),
                    offset: 4,
                    size: 4,
                    type_name: "float".into(),
                },
            ],
        }
    }

    /// Slang CB layout: total size padded to 16 with one `float` field (GPU tail padding).
    fn layout_time_only_cb_padded() -> StructLayout {
        StructLayout {
            name: "TimeUniforms".into(),
            size: 16,
            alignment: 16,
            fields: vec![StructFieldLayout {
                name: "time".into(),
                offset: 0,
                size: 4,
                type_name: "float".into(),
            }],
        }
    }

    #[test]
    fn validate_cb_padded_single_field_passes() {
        let slang = layout_time_only_cb_padded();
        let rust_fields = [("time", 0usize, 4usize)];
        slang
            .validate(4, &rust_fields)
            .expect("Rust 4-byte struct should cover Slang data extent");
    }

    #[test]
    fn validate_shader_field_missing_from_rust_errors() {
        let slang = StructLayout {
            name: "U".into(),
            size: 8,
            alignment: 4,
            fields: vec![
                StructFieldLayout {
                    name: "time".into(),
                    offset: 0,
                    size: 4,
                    type_name: "float".into(),
                },
                StructFieldLayout {
                    name: "brightness".into(),
                    offset: 4,
                    size: 4,
                    type_name: "float".into(),
                },
            ],
        };
        let rust_fields = [("time", 0usize, 4usize)];
        let err = slang.validate(4, &rust_fields).unwrap_err();
        let s = err.to_string();
        assert!(
            s.contains("brightness") && s.contains("missing"),
            "expected missing-field error, got: {s}"
        );
    }

    #[test]
    fn validate_rust_too_small_for_data_extent_errors() {
        let slang = layout_time_only_cb_padded();
        let rust_fields = [("time", 0usize, 4usize)];
        let err = slang.validate(2, &rust_fields).unwrap_err();
        assert!(
            err.to_string()
                .contains("smaller than the shader's data extent"),
            "{err}"
        );
    }

    #[test]
    fn validate_extra_rust_field_without_underscore_passes() {
        let slang = layout_time_only_cb_padded();
        let rust_fields = [("time", 0usize, 4usize), ("brightness", 4usize, 4usize)];
        slang
            .validate(8, &rust_fields)
            .expect("extra Rust field is not an error");
    }

    #[test]
    fn validate_extra_rust_field_with_underscore_passes() {
        let slang = layout_time_only_cb_padded();
        let rust_fields = [("time", 0usize, 4usize), ("_pad0", 4usize, 4usize)];
        slang
            .validate(8, &rust_fields)
            .expect("_prefixed extra field is silent");
    }

    #[test]
    fn validate_ok_when_matching() {
        two_float_layout()
            .validate(8, &[("a", 0, 4), ("b", 4, 4)])
            .unwrap();
    }

    #[test]
    fn validate_does_not_require_rust_struct_to_match_slang_cb_padding() {
        // Slang total `size` can be 16 due to cbuffer rules; Rust only needs to cover data bytes.
        let mut layout = two_float_layout();
        layout.size = 16;
        layout
            .validate(8, &[("a", 0, 4), ("b", 4, 4)])
            .expect("Slang padded size must not force Rust to pad");
    }

    #[test]
    fn validate_err_on_field_count_mismatch() {
        let err = two_float_layout()
            .validate(8, &[("a", 0, 4)])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`b`") && err.contains("missing"),
            "expected shader field b missing in Rust: {err}"
        );
    }

    #[test]
    fn validate_allows_extra_rust_fields_not_in_shader() {
        two_float_layout()
            .validate(12, &[("a", 0, 4), ("b", 4, 4), ("c", 8, 4)])
            .expect("extra Rust-only field should not fail validation");
    }

    #[test]
    fn validate_err_on_field_offset_mismatch() {
        let err = two_float_layout()
            .validate(8, &[("a", 0, 4), ("b", 0, 4)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("offset"), "expected offset mismatch: {err}");
        assert!(err.contains("`b`"), "expected field name b: {err}");
    }

    #[test]
    fn validate_err_on_field_size_mismatch() {
        let err = two_float_layout()
            .validate(8, &[("a", 0, 8), ("b", 4, 4)])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("size") && err.contains("`a`"),
            "expected field size mismatch for a: {err}"
        );
    }

    #[test]
    fn validate_err_on_field_name_mismatch() {
        let err = two_float_layout()
            .validate(8, &[("x", 0, 4), ("b", 4, 4)])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`a`") && err.contains("missing"),
            "expected shader field `a` missing from Rust (got `x` instead): {err}"
        );
    }

    #[test]
    fn validate_reports_multiple_errors() {
        // Only `a` in Rust, wrong offset — missing `b` and offset error for `a`.
        let err = two_float_layout()
            .validate(8, &[("a", 4, 4)])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("offset") && err.contains("`a`"),
            "expected offset error for a: {err}"
        );
        assert!(
            err.contains("`b`") && err.contains("missing"),
            "expected missing b: {err}"
        );
    }

    #[test]
    fn layout_checkable_derive_generates_correct_const() {
        #[derive(goldy_derive::LayoutCheckable)]
        #[repr(C)]
        struct TestStruct {
            pos: [f32; 2],
            color: [f32; 4],
        }

        let check = TestStruct::LAYOUT_CHECK;
        assert_eq!(check.type_name, "TestStruct");
        assert_eq!(check.rust_size, std::mem::size_of::<TestStruct>());
        assert_eq!(check.rust_fields.len(), 2);

        let (name, offset, size) = check.rust_fields[0];
        assert_eq!(name, "pos");
        assert_eq!(offset, 0);
        assert_eq!(size, std::mem::size_of::<[f32; 2]>());

        let (name, offset, size) = check.rust_fields[1];
        assert_eq!(name, "color");
        assert_eq!(offset, std::mem::size_of::<[f32; 2]>());
        assert_eq!(size, std::mem::size_of::<[f32; 4]>());
    }

    #[test]
    fn layout_check_validates_against_matching_slang_layout() {
        #[derive(goldy_derive::LayoutCheckable)]
        #[repr(C)]
        struct Uniforms {
            x: f32,
            y: f32,
        }

        let slang = StructLayout {
            name: "Uniforms".into(),
            size: 8,
            alignment: 4,
            fields: vec![
                StructFieldLayout {
                    name: "x".into(),
                    offset: 0,
                    size: 4,
                    type_name: "float".into(),
                },
                StructFieldLayout {
                    name: "y".into(),
                    offset: 4,
                    size: 4,
                    type_name: "float".into(),
                },
            ],
        };

        let check = Uniforms::LAYOUT_CHECK;
        slang.validate(check.rust_size, check.rust_fields).unwrap();
    }

    #[test]
    fn layout_check_detects_mismatch_against_slang_layout() {
        #[derive(goldy_derive::LayoutCheckable)]
        #[repr(C)]
        struct Uniforms {
            x: f32,
            y: f32,
        }

        let slang_with_wrong_offset = StructLayout {
            name: "Uniforms".into(),
            size: 8,
            alignment: 4,
            fields: vec![
                StructFieldLayout {
                    name: "x".into(),
                    offset: 0,
                    size: 4,
                    type_name: "float".into(),
                },
                StructFieldLayout {
                    name: "y".into(),
                    offset: 8,
                    size: 4,
                    type_name: "float".into(),
                },
            ],
        };

        let check = Uniforms::LAYOUT_CHECK;
        let err = slang_with_wrong_offset
            .validate(check.rust_size, check.rust_fields)
            .unwrap_err()
            .to_string();
        assert!(err.contains("offset"), "expected offset mismatch: {err}");
        assert!(err.contains("`y`"), "expected field y: {err}");
    }

    /// Integration test: feeds a deliberate layout mismatch through the full
    /// `compile_with_reflection` path and checks that the error message is
    /// actionable (contains the struct name and "offset").
    ///
    /// This is the path an agent hits when `GOLDY_VALIDATE_LAYOUTS=1` is set
    /// and a `#[derive(LayoutCheckable)]` struct drifts from its Slang counterpart.
    #[test]
    fn layout_validation_end_to_end_catches_mismatch() {
        use super::{OwnedLayoutCheck, ShaderTarget, SlangCompiler, SlangStage};
        use crate::types::OptimizationLevel;

        let compiler = SlangCompiler::new().expect("Slang compiler unavailable; skipping");

        // A minimal compute shader that declares MyUniforms.
        let source = r#"
            struct MyUniforms { float x; float y; };
            [shader("compute")]
            [numthreads(1, 1, 1)]
            void cs_main() {}
        "#;

        // Correct Rust layout for { float x; float y; } — size=8, y at offset 4.
        // We intentionally claim y is at offset 8, which Slang will disagree with.
        let bad_check = OwnedLayoutCheck {
            type_name: "MyUniforms".into(),
            rust_size: 8,
            rust_fields: vec![
                ("x".into(), 0, 4),
                ("y".into(), 8, 4), // wrong: Slang reflects offset 4
            ],
        };

        let err = compiler
            .compile_with_reflection(
                source,
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[],
                &[],
                &[bad_check],
                OptimizationLevel::None,
            )
            .unwrap_err()
            .to_string();

        assert!(
            err.contains("MyUniforms"),
            "error should name the mismatched struct: {err}"
        );
        assert!(
            err.contains("offset"),
            "error should describe the offset mismatch: {err}"
        );
        assert!(
            err.contains("`y`"),
            "error should name the offending field: {err}"
        );
    }
}

impl Drop for SlangCompiler {
    fn drop(&mut self) {
        if !self.global_session.is_null() {
            unsafe { global_session_release(self.global_session) };
        }
    }
}

/// Verify that `uniform` entry-point parameters (the replacement for
/// `gGoldyDynamic`) compile correctly for all three backends, and that
/// the resulting code accesses the expected resource-slot / argument-buffer
/// locations.
///
/// SPIR-V: `uniform` params → implemented via push constants at offset 0 (std430).
/// DXIL:   `uniform` params → implemented via root constants at b0/space0.
/// Metal:  Slang wraps them in an `EntryPointParams` struct at buffer index 1.
#[cfg(test)]
mod uniform_entry_point_param_binding_tests {
    use super::*;

    /// A minimal compute shader with typed params and no gGoldyDynamic.
    const TEST_SHADER: &str = r#"
        import goldy_exp;

        [goldy_compute]
        [numthreads(64, 1, 1)]
        void cs_main(BufRO<uint> src, Scattered<uint> dst, uint base, ThreadId id) {
            uint ix = id.x + base;
            dst[ix] = src[ix];
        }
    "#;

    fn shader_path() -> String {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join("shaders").to_string_lossy().into_owned()
    }

    #[test]
    fn uniform_params_compile_spirv() {
        let compiler = SlangCompiler::new().expect("Slang unavailable");
        let path = shader_path();

        let output = compiler
            .compile_bindless_with_reflection_and_defines(
                TEST_SHADER,
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("SPIR-V compilation failed for uniform entry-point params");

        assert!(!output.shader.data.is_empty(), "SPIR-V output is empty");

        // Verify SPIR-V magic word (0x07230203).
        let words = output.shader.as_spirv().expect("should be valid SPIR-V");
        assert_eq!(words[0], 0x07230203, "SPIR-V magic number mismatch");

        // StorageClass::PushConstant == 9. This value should appear as a word in
        // the SPIR-V binary when uniform entry-point params are mapped to resource slots.
        assert!(
            words.contains(&9),
            "Expected PushConstant storage class (9) in SPIR-V for uniform params"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn uniform_params_compile_dxil() {
        let compiler = SlangCompiler::new().expect("Slang unavailable");
        let path = shader_path();

        let output = compiler
            .compile_bindless_with_reflection_and_defines(
                TEST_SHADER,
                ShaderTarget::Dxil,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("DXIL compilation failed for uniform entry-point params");

        assert!(!output.shader.data.is_empty(), "DXIL output is empty");
        // DXIL container magic: "DXBC" = 0x43425844 at byte offset 0.
        let magic = u32::from_le_bytes(output.shader.data[..4].try_into().unwrap());
        assert_eq!(magic, 0x43425844, "DXIL magic 'DXBC' mismatch");
    }

    #[test]
    fn uniform_params_compile_metal() {
        let compiler = SlangCompiler::new().expect("Slang unavailable");
        let path = shader_path();

        let output = compiler
            .compile_bindless_with_reflection_and_defines(
                TEST_SHADER,
                ShaderTarget::Metal,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("Metal MSL compilation failed for uniform entry-point params");

        let msl = String::from_utf8_lossy(&output.shader.data);
        assert!(!msl.is_empty(), "Metal MSL output is empty");

        // Slang emits uniform entry-point params as a generated struct (EntryPointParams
        // or similar) passed at a specific buffer slot. The struct name may vary by Slang
        // version, but the kernel's argument list should contain a [[buffer(...)]] binding.
        assert!(
            msl.contains("[[buffer(") || msl.contains("buffer("),
            "Expected Metal buffer binding for uniform params in MSL:\n{msl}"
        );
    }

    /// Regression: compiled Metal output must not contain gGoldyDynamic or
    /// GoldyDynamicSlots — both were removed in the gGoldyDynamic migration.
    #[test]
    fn no_ggoldydynamic_in_compiled_output() {
        let compiler = SlangCompiler::new().expect("Slang unavailable");
        let path = shader_path();

        let output = compiler
            .compile_bindless_with_reflection_and_defines(
                TEST_SHADER,
                ShaderTarget::Metal,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("Metal MSL compilation failed");

        let msl = String::from_utf8_lossy(&output.shader.data);
        assert!(
            !msl.contains("gGoldyDynamic"),
            "gGoldyDynamic must not appear in output MSL"
        );
        assert!(
            !msl.contains("GoldyDynamicSlots"),
            "GoldyDynamicSlots must not appear in output MSL"
        );
    }
}
