//! High-level Slang compiler API.
//!
//! Provides a safe, ergonomic interface for compiling Slang shaders.

use anyhow::{Context, Result};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::{Arc, Mutex};

/// Serializes Slang global-session create/destroy and compilation.
///
/// Parallel `Instance::new` / `release_idle_shader_compiler` paths can otherwise
/// call `create_global_session` and `global_session_release` concurrently and SIGSEGV.
static SLANG_PROCESS_LOCK: Mutex<()> = Mutex::new(());

use super::ffi::*;
use super::loader::SlangLibrary;
use super::virtual_main::effective_slang_source_for_compile;
use crate::types::{OptimizationLevel, ResourceCategory};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    /// A ray-tracing acceleration structure (`RaytracingAccelerationStructure`)
    AccelerationStructure,
}

/// Layout information for a single field within a ParameterBlock
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ShaderReflection {
    /// All parameter blocks found in the shader
    pub parameter_blocks: Vec<ParameterBlockLayout>,
    /// Per push-constant slot, the [`ResourceCategory`]
    /// the shader expects. Populated from `[goldy_*]` entry-point analysis at compile time.
    /// Used by backend validation when `BindResourcesTyped` is used to catch category
    /// mismatches against the shader's reflected expectations.
    pub push_constant_categories: Vec<Option<crate::types::ResourceCategory>>,
    /// Per push-constant slot, the DX12 bindless view kind the shader expects
    /// (`Scattered<T>` → UAV, `BufRO<T>` → SRV). Empty when source has no
    /// `[goldy_*]` annotations. Re-derived from source at compile time (not serialized).
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    #[serde(skip)]
    pub(crate) push_constant_slot_kinds: Vec<Option<crate::types::BindlessSlotKind>>,
    /// Per push-constant slot, the expected element stride (bytes) of the bound
    /// buffer. Populated from `[goldy_*]` source analysis + Slang reflection at
    /// compile time.  At dispatch time, backends compare each bound buffer's
    /// `element_stride` against this value when layout validation is enabled
    /// (`GOLDY_VALIDATE_LAYOUTS`, `GOLDY_VALIDATION=layout`).
    #[serde(default)]
    pub binding_element_strides: Vec<Option<u32>>,
    /// Per-entry graphics stage I/O (vertex attributes, interpolants, fragment outputs).
    #[serde(default)]
    pub stage_interfaces: Vec<crate::slang::graphics_link::StageInterface>,
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
        let slang_data_extent = self.fields.iter().map(|f| f.offset + f.size).max().unwrap_or(0);

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
            anyhow::bail!("Struct layout mismatch for `{}`:\n{}", self.name, errors.join("\n"));
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

/// Stored on backend shader state for deferred per-stage compilation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompiledShaderWithReflection {
    /// Code for `entry_points[0]` of the compile request.
    pub shader: CompiledShader,
    /// Reflection data (if requested)
    pub reflection: ShaderReflection,
    /// Code for `entry_points[1..]` in request order. Empty for a single-entry compile.
    #[serde(default)]
    pub extra_entry_points: Vec<CompiledShader>,
}

impl CompiledShaderWithReflection {
    /// Compiled blob for `entry_points[index]` from the compile request.
    pub fn entry_point(&self, index: usize) -> Option<&CompiledShader> {
        match index {
            0 => Some(&self.shader),
            i => self.extra_entry_points.get(i - 1),
        }
    }
}

/// Shader compilation target.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ShaderTarget {
    /// SPIR-V bytecode for Vulkan
    Spirv,
    /// DXIL bytecode for DirectX 12 (binary, SM 6.6 for bindless)
    Dxil,
    /// Metal Shading Language
    Metal,
    /// WebGPU Shading Language
    Wgsl,
    /// CUDA PTX (Slang → CUDA C++ → NVRTC)
    Ptx,
    /// CUDA C++ generated by Slang (`GOLDY_DUMP_SHADERS`). Not a runtime backend.
    CudaSource,
    /// Host-callable CPU JIT (`SLANG_SHADER_HOST_CALLABLE`). Debug only — not a production backend.
    HostCallable,
}

impl ShaderTarget {
    fn to_slang_target(self) -> SlangCompileTarget {
        match self {
            ShaderTarget::Spirv => SlangCompileTarget::Spirv,
            ShaderTarget::Dxil => SlangCompileTarget::Dxil,
            ShaderTarget::Metal => SlangCompileTarget::Metal,
            ShaderTarget::Wgsl => SlangCompileTarget::Wgsl,
            ShaderTarget::Ptx => SlangCompileTarget::Ptx,
            ShaderTarget::CudaSource => SlangCompileTarget::CudaSource,
            ShaderTarget::HostCallable => SlangCompileTarget::ShaderHostCallable,
        }
    }

    /// Returns true if this target produces binary bytecode (not text).
    pub fn is_binary(self) -> bool {
        matches!(self, ShaderTarget::Spirv | ShaderTarget::Dxil)
    }
}

/// Compiled shader output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
            ShaderTarget::Metal | ShaderTarget::Wgsl | ShaderTarget::Ptx | ShaderTarget::CudaSource => {
                std::str::from_utf8(&self.data).ok()
            }
            ShaderTarget::Spirv | ShaderTarget::Dxil | ShaderTarget::HostCallable => None,
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

fn entry_point_code_blob(
    library: &SlangLibrary,
    request: *mut SlangCompileRequest,
    entry_index: i32,
    target_index: i32,
) -> Result<Vec<u8>> {
    let mut blob: *mut ISlangBlob = ptr::null_mut();
    let result = unsafe { (library.get_entry_point_code_blob)(request, entry_index, target_index, &mut blob) };
    if !slang_succeeded(result) || blob.is_null() {
        anyhow::bail!("Failed to get compiled shader code for entry point {entry_index}");
    }
    let (data_ptr, data_size) = unsafe { blob_get_data(blob) };
    let data = unsafe { std::slice::from_raw_parts(data_ptr, data_size) }.to_vec();
    unsafe { blob_release(blob) };
    Ok(data)
}

/// Return the byte stride of a Slang built-in scalar/vector/matrix type, or
/// `None` for user-defined structs that require Slang reflection.
pub fn builtin_type_stride(name: &str) -> Option<u32> {
    match name {
        "uint" | "int" | "float" | "bool" | "dword" => Some(4),
        "half" | "float16_t" => Some(2),
        "double" | "uint64_t" | "int64_t" => Some(8),
        "uint2" | "int2" | "float2" => Some(8),
        "half2" => Some(4),
        "uint3" | "int3" | "float3" => Some(12),
        "half3" => Some(6),
        "uint4" | "int4" | "float4" => Some(16),
        "half4" => Some(8),
        "float2x2" => Some(16),
        "float3x3" => Some(36),
        "float4x4" => Some(64),
        "DispatchShape" => Some(12),
        _ => None,
    }
}

/// Slang shader compiler.
///
/// Thread-safe wrapper around the Slang compilation API.
pub struct SlangCompiler {
    library: Arc<SlangLibrary>,
    global_session: *mut IGlobalSession,
    shader_disk_cache: std::sync::Mutex<crate::shader_cache::ShaderBytecodeDiskCache>,
    /// Compile keys already run through the opt-in bounds analysis (one report per shader).
    bounds_analyzed: std::sync::Mutex<std::collections::HashSet<u64>>,
    /// IR containers of imported library modules, keyed by module name, source and defines.
    ir_library_cache: std::sync::Mutex<std::collections::HashMap<u64, Arc<Vec<u8>>>>,
}

/// What a Slang compile request is asked to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestOutput {
    /// Target code for the requested [`ShaderTarget`].
    Code,
    /// The front-end IR container (`.slang-module`) with standard debug info; no target code.
    IrContainer,
}

// SlangCompiler is Send + Sync because each compilation creates its own request
unsafe impl Send for SlangCompiler {}
unsafe impl Sync for SlangCompiler {}

impl SlangCompiler {
    /// Create a new Slang compiler instance.
    pub fn new() -> Result<Self> {
        let _span = goldy_span!("slang.compiler.init").entered();

        let _guard = SLANG_PROCESS_LOCK.lock().unwrap();

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
            shader_disk_cache: std::sync::Mutex::new(crate::shader_cache::ShaderBytecodeDiskCache::new_load_or_empty()),
            bounds_analyzed: std::sync::Mutex::new(std::collections::HashSet::new()),
            ir_library_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
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
            ShaderTarget::Wgsl => vec![("__WGSL__", "1")],
            ShaderTarget::Ptx | ShaderTarget::CudaSource => vec![("__CUDA__", "1")],
            ShaderTarget::HostCallable => vec![("__CPU__", "1")],
        }
    }

    /// Shared session + compile path; invokes `f` with the live compile request after `spCompile` succeeds.
    ///
    /// `source` must be the effective Slang translation-unit text (after
    /// [`super::virtual_main::effective_slang_source_for_compile`] when applicable). Virtual-main
    /// rewriting is not applied here so it runs once per logical compile.
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
        self.with_compiled_request_opts(
            source,
            target,
            entry_points,
            search_paths,
            defines,
            optimization_level,
            RequestOutput::Code,
            f,
        )
    }

    /// [`Self::with_compiled_request`] with an explicit choice of request output.
    ///
    /// [`RequestOutput::IrContainer`] requests `SLANG_DEBUG_INFO_LEVEL_STANDARD` so the
    /// serialized IR carries source locations, and adds no code-generation target; `target`
    /// then only selects the preprocessor defines.
    #[allow(clippy::too_many_arguments)]
    fn with_compiled_request_opts<R>(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: OptimizationLevel,
        output: RequestOutput,
        f: impl FnOnce(&Self, *mut SlangCompileRequest, i32) -> Result<R>,
    ) -> Result<R> {
        let debug_info = output == RequestOutput::IrContainer;
        let _guard = SLANG_PROCESS_LOCK.lock().unwrap();

        // Create session with session-level preprocessor macros.
        let define_names: Vec<CString> = defines.iter().map(|(k, _)| CString::new(*k).unwrap()).collect();
        let define_values: Vec<CString> = defines.iter().map(|(_, v)| CString::new(*v).unwrap()).collect();
        let macro_descs: Vec<PreprocessorMacroDesc> = define_names
            .iter()
            .zip(define_values.iter())
            .map(|(name, value)| PreprocessorMacroDesc {
                name: name.as_ptr(),
                value: value.as_ptr(),
            })
            .collect();

        let search_path_cstrings: Vec<CString> = search_paths.iter().map(|p| CString::new(*p).unwrap()).collect();
        let search_path_ptrs: Vec<*const c_char> = search_path_cstrings.iter().map(|s| s.as_ptr()).collect();

        let mut session_desc = SessionDesc::default();
        if !search_path_ptrs.is_empty() {
            session_desc.search_paths = search_path_ptrs.as_ptr();
            session_desc.search_path_count = search_path_ptrs.len() as i64;
        }
        if !macro_descs.is_empty() {
            session_desc.preprocessor_macros = macro_descs.as_ptr();
            session_desc.preprocessor_macro_count = macro_descs.len() as i64;
        }
        // Debug info has to be a *session* option: request-level `spSetDebugInfoLevel` is not
        // honored for requests created from `ISession::createCompileRequest`.
        let option_entries: Vec<CompilerOptionEntry> = if debug_info {
            vec![CompilerOptionEntry::int(
                COMPILER_OPTION_DEBUG_INFORMATION,
                SLANG_DEBUG_INFO_LEVEL_STANDARD,
            )]
        } else {
            Vec::new()
        };
        if !option_entries.is_empty() {
            session_desc.compiler_option_entries = option_entries.as_ptr().cast();
            session_desc.compiler_option_entry_count = option_entries.len() as u32;
        }

        tracing::debug!(
            "Creating session with {} macros, SessionDesc size: {}",
            macro_descs.len(),
            std::mem::size_of::<SessionDesc>()
        );
        let mut session: *mut ISession = ptr::null_mut();
        let result = unsafe { global_session_create_session(self.global_session, &session_desc, &mut session) };
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

        let target_index = if output == RequestOutput::IrContainer {
            unsafe { (self.library.set_output_container_format)(request, SLANG_CONTAINER_FORMAT_SLANG_MODULE) };
            -1
        } else {
            let target_index = unsafe { (self.library.add_code_gen_target)(request, target.to_slang_target() as i32) };
            if target_index < 0 {
                anyhow::bail!("Failed to add code generation target");
            }
            target_index
        };

        if output == RequestOutput::Code && target == ShaderTarget::Dxil {
            let profile_name = CString::new("sm_6_6").unwrap();
            let profile_id = unsafe { global_session_find_profile(self.global_session, profile_name.as_ptr()) };
            if profile_id > 0 {
                unsafe {
                    (self.library.set_target_profile)(request, target_index, profile_id);
                }
                tracing::debug!("Set DXIL target profile to sm_6_6 (id={})", profile_id);
            } else {
                tracing::warn!("Could not find sm_6_6 profile, using default");
            }
            unsafe {
                (self.library.set_target_floating_point_mode)(request, target_index, SLANG_FLOATING_POINT_MODE_PRECISE);
            }
        }

        let unit_name = CString::new("shader").unwrap();
        let translation_unit = unsafe {
            (self.library.add_translation_unit)(request, SlangSourceLanguage::Slang as i32, unit_name.as_ptr())
        };
        if translation_unit < 0 {
            anyhow::bail!("Failed to add translation unit");
        }

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
            let entry_index =
                unsafe { (self.library.add_entry_point)(request, translation_unit, name_cstr.as_ptr(), *stage as i32) };
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

        if output == RequestOutput::IrContainer {
            // Slang only emits the IR container when its option parser ran and saw neither a
            // code-gen target nor an output path (`slangc -emit-ir` behavior); an empty
            // argument list is enough to trigger that.
            let result = unsafe { (self.library.process_command_line_arguments)(request, ptr::null(), 0) };
            if !slang_succeeded(result) {
                anyhow::bail!("Failed to configure Slang request for IR output (result={result})");
            }
        }

        let result = unsafe { (self.library.compile)(request) };
        if !slang_succeeded(result) {
            let diag_ptr = unsafe { (self.library.get_diagnostic_output)(request) };
            let diagnostic = if !diag_ptr.is_null() {
                unsafe { CStr::from_ptr(diag_ptr) }.to_string_lossy().into_owned()
            } else {
                "Unknown compilation error".to_string()
            };
            anyhow::bail!("Slang compilation failed:\n{}", diagnostic);
        }

        f(self, request, target_index)
    }

    /// Compile a compute kernel to Slang host-callable JIT (`SLANG_SHADER_HOST_CALLABLE`).
    ///
    /// `source` is compiled as-is (apply [`super::virtual_main::transform_virtual_main_cpu`] first
    /// for `[goldy_compute]`). The returned library stays valid while this compiler's Slang
    /// shared library remains loaded.
    pub(crate) fn compile_host_callable_library(
        &self,
        source: &str,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        extra_defines: &[(&str, &str)],
        optimization_level: OptimizationLevel,
    ) -> Result<(*mut ISlangSharedLibrary, Arc<SlangLibrary>)> {
        let mut defines = Self::bindless_defines_for_target(ShaderTarget::HostCallable);
        defines.extend_from_slice(extra_defines);
        self.with_compiled_request(
            source,
            ShaderTarget::HostCallable,
            entry_points,
            search_paths,
            &defines,
            optimization_level,
            |slf, request, target_index| {
                let mut lib: *mut ISlangSharedLibrary = ptr::null_mut();
                let result = unsafe { (slf.library.get_entry_point_host_callable)(request, 0, target_index, &mut lib) };
                if !slang_succeeded(result) || lib.is_null() {
                    let diag_ptr = unsafe { (slf.library.get_diagnostic_output)(request) };
                    let diagnostic = if !diag_ptr.is_null() {
                        unsafe { CStr::from_ptr(diag_ptr) }.to_string_lossy().into_owned()
                    } else {
                        format!("getEntryPointHostCallable failed (result={result})")
                    };
                    anyhow::bail!("Slang host-callable JIT failed:\n{diagnostic}");
                }
                Ok((lib, Arc::clone(&slf.library)))
            },
        )
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
        if target == ShaderTarget::HostCallable {
            anyhow::bail!("ShaderTarget::HostCallable does not produce bytecode; use goldy::cpu_shaders::compile");
        }
        let t0 = std::time::Instant::now();
        let effective = effective_slang_source_for_compile(source);
        crate::shader_timing::record("slang.transform", "", t0.elapsed());
        self.compile_with_reflection_effective(
            source,
            effective.as_ref(),
            target,
            entry_points,
            search_paths,
            defines,
            layout_checks,
            optimization_level,
        )
    }

    /// Like [`Self::compile_bindless_with_reflection_and_defines`], but uses a pre-transformed
    /// translation unit so virtual-main rewrite can be cached on the frontend module.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_bindless_with_reflection_effective(
        &self,
        original_source: &str,
        effective_source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        extra_defines: &[(&str, &str)],
        layout_checks: &[OwnedLayoutCheck],
        optimization_level: OptimizationLevel,
    ) -> Result<CompiledShaderWithReflection> {
        let mut defines = Self::bindless_defines_for_target(target);
        defines.extend_from_slice(extra_defines);
        self.compile_with_reflection_effective(
            original_source,
            effective_source,
            target,
            entry_points,
            search_paths,
            &defines,
            layout_checks,
            optimization_level,
        )
    }

    /// Compile already-rewritten Slang. `original_source` is used for `[goldy_*]` binding
    /// analysis; `effective_source` is hashed and passed to Slang.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_with_reflection_effective(
        &self,
        original_source: &str,
        effective_source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        defines: &[(&str, &str)],
        layout_checks: &[OwnedLayoutCheck],
        optimization_level: OptimizationLevel,
    ) -> Result<CompiledShaderWithReflection> {
        if target == ShaderTarget::HostCallable {
            anyhow::bail!("ShaderTarget::HostCallable does not produce bytecode; use goldy::cpu_shaders::compile");
        }
        let t1 = std::time::Instant::now();
        let cache_key = crate::shader_cache::compile_cache_key(
            effective_source,
            target,
            entry_points,
            search_paths,
            defines,
            layout_checks,
            optimization_level,
        );
        crate::shader_timing::record("slang.compile_cache_key", "", t1.elapsed());

        {
            let t2 = std::time::Instant::now();
            let mut disk = self.shader_disk_cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(hit) = disk.get(cache_key) {
                crate::shader_timing::record("slang.disk_hit", "", t2.elapsed());
                drop(disk);
                // Diagnostics are opt-in and independent of the cached bytecode, so a cache
                // hit must still surface them.
                if crate::validation_env::bounds_validation_enabled() {
                    self.report_bounds_analysis(effective_source, target, entry_points, search_paths, defines);
                }
                return hit.with_context(|| "decode shader disk cache");
            }
            crate::shader_timing::record("slang.disk_miss_lookup", "", t2.elapsed());
        }

        let t_compile = std::time::Instant::now();
        let binding_type_names = super::virtual_main::extract_binding_element_type_names(original_source);
        let binding_categories = super::virtual_main::extract_push_constant_categories(original_source);

        let out = self.with_compiled_request(
            effective_source,
            target,
            entry_points,
            search_paths,
            defines,
            optimization_level,
            |slf, request, target_index| {
                let ep_count = entry_points.len().max(1);
                let mut blobs = Vec::with_capacity(ep_count);
                for i in 0..ep_count {
                    blobs.push(entry_point_code_blob(&slf.library, request, i as i32, target_index)?);
                }
                let data = blobs.remove(0);
                let extra_entry_points = blobs.into_iter().map(|data| CompiledShader { data, target }).collect();

                let mut reflection = slf.extract_reflection(request)?;

                if !layout_checks.is_empty() {
                    slf.validate_owned_layout_checks(request, layout_checks)?;
                }

                let strides: Vec<Option<u32>> = binding_type_names
                    .iter()
                    .enumerate()
                    .map(|(i, opt_name)| {
                        let cat = binding_categories.get(i).copied().unwrap_or(None);
                        opt_name.as_deref().and_then(|name| {
                            builtin_type_stride(name).or_else(|| slf.reflect_binding_element_stride(request, name, cat))
                        })
                    })
                    .collect();
                reflection.binding_element_strides = strides;

                Ok(CompiledShaderWithReflection {
                    shader: CompiledShader { data, target },
                    reflection,
                    extra_entry_points,
                })
            },
        )?;
        crate::shader_timing::record("slang.compile_miss", "", t_compile.elapsed());

        {
            let mut disk = self.shader_disk_cache.lock().unwrap_or_else(|p| p.into_inner());
            if let Err(e) = disk.insert(cache_key, &out) {
                tracing::warn!(?e, "failed to serialize shader disk cache entry");
            }
        }

        if crate::validation_env::bounds_validation_enabled() {
            self.report_bounds_analysis(effective_source, target, entry_points, search_paths, defines);
        }

        Ok(out)
    }

    /// Compile `source` to Slang's front-end IR container (`.slang-module` bytes) with standard
    /// debug info. No target code is generated; `target` only selects the preprocessor defines.
    fn compile_ir_container(
        &self,
        source: &str,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<Vec<u8>> {
        self.with_compiled_request_opts(
            source,
            ShaderTarget::Spirv,
            entry_points,
            search_paths,
            defines,
            OptimizationLevel::Default,
            RequestOutput::IrContainer,
            |slf, request, _| {
                let mut blob: *mut ISlangBlob = ptr::null_mut();
                let result = unsafe { (slf.library.get_container_code)(request, &mut blob) };
                if !slang_succeeded(result) || blob.is_null() {
                    anyhow::bail!("Slang produced no IR container (result={result})");
                }
                let (data_ptr, data_size) = unsafe { blob_get_data(blob) };
                let data = unsafe { std::slice::from_raw_parts(data_ptr, data_size) }.to_vec();
                unsafe { blob_release(blob) };
                Ok(data)
            },
        )
    }

    /// IR containers of the modules `container` imports (transitively), compiled from
    /// `<name>.slang` on `search_paths` with the same defines and cached by source content.
    /// Modules that cannot be found or compiled are skipped with a debug log; the analysis
    /// then treats calls into them as unknown.
    fn imported_library_containers(
        &self,
        container: &[u8],
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Vec<Arc<Vec<u8>>> {
        let mut out: Vec<Arc<Vec<u8>>> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut pending = match super::bounds_analysis::imported_modules(container) {
            Ok(names) => names,
            Err(e) => {
                tracing::debug!(?e, "could not list imported modules");
                return out;
            }
        };
        while let Some(name) = pending.pop() {
            if !seen.insert(name.clone()) {
                continue;
            }
            let Some((path, source)) = search_paths.iter().find_map(|sp| {
                let path = std::path::Path::new(sp).join(format!("{name}.slang"));
                std::fs::read_to_string(&path).ok().map(|src| (path, src))
            }) else {
                tracing::debug!(module = %name, "imported module not found on search paths");
                continue;
            };
            let mut key = crate::shader_cache::compile_cache_key(
                &source,
                ShaderTarget::Spirv,
                &[],
                search_paths,
                defines,
                &[],
                OptimizationLevel::Default,
            );
            for b in name.bytes() {
                key = key.wrapping_mul(0x0100_0000_01b3) ^ u64::from(b);
            }
            let cached = {
                let cache = self.ir_library_cache.lock().unwrap_or_else(|p| p.into_inner());
                cache.get(&key).cloned()
            };
            let lib = match cached {
                Some(lib) => lib,
                None => match self.compile_ir_container(&source, &[], search_paths, defines) {
                    Ok(bytes) => {
                        let lib = Arc::new(bytes);
                        let mut cache = self.ir_library_cache.lock().unwrap_or_else(|p| p.into_inner());
                        cache.insert(key, Arc::clone(&lib));
                        lib
                    }
                    Err(e) => {
                        tracing::debug!(module = %name, path = %path.display(), ?e, "imported module failed to compile for analysis");
                        continue;
                    }
                },
            };
            if let Ok(more) = super::bounds_analysis::imported_modules(&lib) {
                pending.extend(more);
            }
            out.push(lib);
        }
        out
    }

    /// Compile `source` to Slang IR (with standard debug info) and run the static bounds
    /// analysis over it and the modules it imports.
    ///
    /// This is a separate Slang request from the production compile so the bytecode handed
    /// to the driver is unchanged; the IR build only feeds the analysis. `target` selects the
    /// preprocessor defines the shader is analyzed under. Returns the report, or the
    /// compile/analysis error.
    pub fn analyze_bounds(
        &self,
        source: &str,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        defines: &[(&str, &str)],
        target: ShaderTarget,
    ) -> Result<super::bounds_analysis::BoundsReport> {
        let mut all_defines = Self::bindless_defines_for_target(target);
        for d in defines {
            if !all_defines.contains(d) {
                all_defines.push(*d);
            }
        }
        let effective = effective_slang_source_for_compile(source);
        let container = self.compile_ir_container(effective.as_ref(), entry_points, search_paths, &all_defines)?;
        if let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") {
            let dir = std::path::Path::new(&dump_dir);
            let _ = std::fs::create_dir_all(dir);
            let name = entry_points.first().map(|(n, _)| *n).unwrap_or("shader");
            let path = dir.join(format!("{name}_bounds.slang-module"));
            if std::fs::write(&path, &container).is_ok() {
                tracing::info!("Dumped Slang IR container for bounds analysis to {}", path.display());
            }
        }
        let libraries = self.imported_library_containers(&container, search_paths, &all_defines);
        let library_refs: Vec<&[u8]> = libraries.iter().map(|l| l.as_slice()).collect();
        Ok(super::bounds_analysis::analyze_container(&container, &library_refs)?)
    }

    /// Opt-in (`GOLDY_VALIDATION=bounds`) hook: analyze once per distinct compile and log
    /// every unproven dynamic index as a warning. Never fails the compile.
    fn report_bounds_analysis(
        &self,
        effective_source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) {
        let key = crate::shader_cache::compile_cache_key(
            effective_source,
            target,
            entry_points,
            search_paths,
            defines,
            &[],
            OptimizationLevel::Default,
        );
        {
            let mut seen = self.bounds_analyzed.lock().unwrap_or_else(|p| p.into_inner());
            if !seen.insert(key) {
                return;
            }
        }
        let t0 = std::time::Instant::now();
        // `analyze_bounds` re-applies the virtual-main rewrite; on already-effective source
        // that is a no-op, so pass it straight through.
        match self.analyze_bounds(effective_source, entry_points, search_paths, defines, target) {
            Ok(report) => {
                crate::shader_timing::record("slang.bounds_analysis", "", t0.elapsed());
                tracing::debug!(
                    checked = report.checked_accesses,
                    proven_safe = report.proven_safe,
                    unproven = report.diagnostics.len(),
                    "shader bounds analysis"
                );
                for d in &report.diagnostics {
                    tracing::warn!("shader bounds: {d}");
                }
            }
            Err(e) => tracing::debug!(?e, "shader bounds analysis skipped"),
        }
    }

    fn validate_owned_layout_checks(
        &self,
        request: *mut SlangCompileRequest,
        checks: &[OwnedLayoutCheck],
    ) -> Result<()> {
        for owned in checks {
            let layout = self.reflect_named_struct_from_request(request, &owned.type_name)?;
            let field_refs: Vec<(&str, usize, usize)> =
                owned.rust_fields.iter().map(|(n, o, s)| (n.as_str(), *o, *s)).collect();
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
        let effective = effective_slang_source_for_compile(shader_source);
        self.with_compiled_request(
            effective.as_ref(),
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
        let ty = unsafe { (self.library.reflection_find_type_by_name)(reflection_ptr, name_cstr.as_ptr()) };
        if ty.is_null() {
            anyhow::bail!("Slang reflection: type `{type_name}` not found");
        }

        let layout_ptr =
            unsafe { (self.library.reflection_get_type_layout)(reflection_ptr, ty, SlangLayoutRules::Default) };
        if layout_ptr.is_null() {
            anyhow::bail!("Slang reflection: failed to get layout for `{type_name}`");
        }

        self.extract_struct_layout_uniform(layout_ptr, type_name)
    }

    /// Query the per-element byte stride for a bindless slot's element type.
    ///
    /// Broadcast uniforms use std140-style `Uniform` layout. Storage-buffer element
    /// types (`Scattered<T>`, `BufRO<T>`, etc.) use `ShaderResource` layout so
    /// simple structs like `{ uint a; uint b; }` report 8 bytes, not the 16-byte
    /// uniform round-up that would mismatch GPU structured-buffer indexing.
    fn reflect_binding_element_stride(
        &self,
        request: *mut SlangCompileRequest,
        type_name: &str,
        category: Option<ResourceCategory>,
    ) -> Option<u32> {
        // Broadcast (constant-buffer) params: use struct_storage_stride which sums
        // field extents without std140 tail-padding.  reflect_type_size_with_category
        // with Uniform returns the cbuffer-padded whole-struct size (e.g. 16 for a
        // single-float struct), which does not match the buffer's element_stride set
        // at allocation time.
        if matches!(category, Some(ResourceCategory::Broadcast)) {
            return self.reflect_struct_storage_stride(request, type_name, SlangParameterCategory::Uniform);
        }

        let layout_cat = match category {
            Some(ResourceCategory::StorageImage) => SlangParameterCategory::UnorderedAccess,
            Some(ResourceCategory::Scattered)
            | Some(ResourceCategory::Texture)
            | Some(ResourceCategory::Sampler)
            | Some(ResourceCategory::Accel)
            | None => SlangParameterCategory::ShaderResource,
            Some(ResourceCategory::Broadcast) => unreachable!("handled above"),
        };
        self.reflect_type_size_with_category(request, type_name, layout_cat)
            .or_else(|| {
                if matches!(
                    category,
                    Some(ResourceCategory::Scattered)
                        | Some(ResourceCategory::StorageImage)
                        | Some(ResourceCategory::Texture)
                        | None
                ) {
                    self.reflect_struct_storage_stride(request, type_name, layout_cat)
                } else {
                    None
                }
            })
    }

    fn reflect_type_size_with_category(
        &self,
        request: *mut SlangCompileRequest,
        type_name: &str,
        layout_cat: SlangParameterCategory,
    ) -> Option<u32> {
        let layout_ptr = self.reflect_type_layout_ptr(request, type_name)?;
        let size = unsafe { (self.library.reflection_type_layout_get_size)(layout_ptr, layout_cat as i32) } as u32;
        if size > 0 {
            Some(size)
        } else {
            None
        }
    }

    /// Structured-buffer element size from per-field offsets (natural struct extent).
    ///
    /// Whole-type `Uniform` layout includes std140 tail padding (e.g. 8-byte struct → 16),
    /// which does not match GPU structured-buffer indexing. Field extents under `Uniform`
    /// omit that padding and match storage-buffer element sizes.
    fn reflect_struct_storage_stride(
        &self,
        request: *mut SlangCompileRequest,
        type_name: &str,
        _layout_cat: SlangParameterCategory,
    ) -> Option<u32> {
        let layout_ptr = self.reflect_type_layout_ptr(request, type_name)?;
        let field_count = unsafe { (self.library.reflection_type_layout_get_field_count)(layout_ptr) };
        if field_count == 0 {
            return None;
        }

        let field_cat = SlangParameterCategory::Uniform as i32;
        let mut extent = 0u32;
        for i in 0..field_count {
            let field_var = unsafe { (self.library.reflection_type_layout_get_field_by_index)(layout_ptr, i) };
            if field_var.is_null() {
                continue;
            }
            let field_type_layout = unsafe { (self.library.reflection_variable_layout_get_type_layout)(field_var) };
            if field_type_layout.is_null() {
                continue;
            }
            let offset = unsafe { (self.library.reflection_variable_layout_get_offset)(field_var, field_cat) } as u32;
            let field_size =
                unsafe { (self.library.reflection_type_layout_get_size)(field_type_layout, field_cat) } as u32;
            let field_extent = offset.saturating_add(field_size.max(1));
            extent = extent.max(field_extent);
        }

        if extent > 0 {
            Some(extent)
        } else {
            None
        }
    }

    fn reflect_type_layout_ptr(
        &self,
        request: *mut SlangCompileRequest,
        type_name: &str,
    ) -> Option<*mut SlangReflectionTypeLayout> {
        let reflection_ptr = unsafe { (self.library.get_reflection)(request) };
        if reflection_ptr.is_null() {
            return None;
        }

        let mut candidates = vec![type_name.to_string()];
        if !type_name.contains('.') {
            candidates.push(format!("shader.{type_name}"));
        }

        for candidate in &candidates {
            let name_cstr = CString::new(candidate.as_str()).ok()?;
            let ty = unsafe { (self.library.reflection_find_type_by_name)(reflection_ptr, name_cstr.as_ptr()) };
            if ty.is_null() {
                continue;
            }
            let layout_ptr =
                unsafe { (self.library.reflection_get_type_layout)(reflection_ptr, ty, SlangLayoutRules::Default) };
            if !layout_ptr.is_null() {
                return Some(layout_ptr);
            }
        }
        None
    }

    fn extract_struct_layout_uniform(
        &self,
        type_layout: *mut SlangReflectionTypeLayout,
        struct_name: &str,
    ) -> Result<StructLayout> {
        let cat = SlangParameterCategory::Uniform as i32;
        let size = unsafe { (self.library.reflection_type_layout_get_size)(type_layout, cat) };
        let alignment = unsafe { (self.library.reflection_type_layout_get_alignment)(type_layout, cat) };

        let field_count = unsafe { (self.library.reflection_type_layout_get_field_count)(type_layout) };

        let mut fields = Vec::new();
        for i in 0..field_count {
            let field_var = unsafe { (self.library.reflection_type_layout_get_field_by_index)(type_layout, i) };
            if field_var.is_null() {
                continue;
            }

            let variable = unsafe { (self.library.reflection_variable_layout_get_variable)(field_var) };
            let name = if !variable.is_null() {
                let name_ptr = unsafe { (self.library.reflection_variable_get_name)(variable) };
                if !name_ptr.is_null() {
                    unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy().into_owned()
                } else {
                    format!("field_{i}")
                }
            } else {
                format!("field_{i}")
            };

            let field_type_layout = unsafe { (self.library.reflection_variable_layout_get_type_layout)(field_var) };
            if field_type_layout.is_null() {
                continue;
            }

            let offset = unsafe { (self.library.reflection_variable_layout_get_offset)(field_var, cat) };
            let fsize = unsafe { (self.library.reflection_type_layout_get_size)(field_type_layout, cat) };

            let field_type = unsafe { (self.library.reflection_type_layout_get_type)(field_type_layout) };
            let type_name = if !field_type.is_null() {
                let type_name_ptr = unsafe { (self.library.reflection_type_get_name)(field_type) };
                if !type_name_ptr.is_null() {
                    unsafe { CStr::from_ptr(type_name_ptr) }.to_string_lossy().into_owned()
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
            let param = unsafe { (self.library.reflection_get_parameter_by_index)(reflection_ptr, i) };
            if param.is_null() {
                continue;
            }

            // Get parameter name (parameter -> variable -> name)
            let variable = unsafe { (self.library.reflection_variable_layout_get_variable)(param) };
            let name = if !variable.is_null() {
                let name_ptr = unsafe { (self.library.reflection_variable_get_name)(variable) };
                if !name_ptr.is_null() {
                    unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy().into_owned()
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
                let block_layout = self.extract_parameter_block_layout(param, type_layout, &name)?;
                parameter_blocks.push(block_layout);
            }
        }

        goldy_event!(
            "slang.reflection.extract",
            parameter_blocks = parameter_blocks.len(),
            total_fields = parameter_blocks.iter().map(|pb| pb.fields.len()).sum::<usize>()
        );

        Ok(ShaderReflection {
            parameter_blocks,
            push_constant_categories: Vec::new(),
            #[cfg(all(feature = "dx12", target_os = "windows"))]
            push_constant_slot_kinds: Vec::new(),
            binding_element_strides: Vec::new(),
            stage_interfaces: self.extract_stage_interfaces(reflection_ptr),
        })
    }

    fn extract_stage_interfaces(
        &self,
        reflection_ptr: *mut SlangReflection,
    ) -> Vec<crate::slang::graphics_link::StageInterface> {
        use crate::slang::ffi::{SlangParameterCategory, SlangTypeKind};
        use crate::slang::graphics_link::StageInterface;

        let mut out = Vec::new();
        let ep_count = unsafe { (self.library.reflection_get_entry_point_count)(reflection_ptr) };
        for i in 0..ep_count {
            let ep = unsafe { (self.library.reflection_get_entry_point_by_index)(reflection_ptr, i) };
            if ep.is_null() {
                continue;
            }
            let name = unsafe {
                let ptr = (self.library.reflection_entry_point_get_name)(ep);
                if ptr.is_null() {
                    format!("entry_{i}")
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };
            let stage_id = unsafe { (self.library.reflection_entry_point_get_stage)(ep) };
            let stage_name = slang_stage_name(stage_id);

            let mut iface = StageInterface {
                stage: stage_name.to_string(),
                entry_name: name,
                vertex_inputs: Vec::new(),
                payload_inputs: Vec::new(),
                payload_outputs: Vec::new(),
                fragment_outputs: Vec::new(),
            };

            let result = unsafe { (self.library.reflection_entry_point_get_result_var_layout)(ep) };
            if !result.is_null() {
                let mut fields = Vec::new();
                self.collect_io_fields(result, "", &mut fields);
                fields.retain(is_graphics_payload_field);
                if stage_name == "fragment" {
                    iface.fragment_outputs = fields;
                } else {
                    iface.payload_outputs = fields;
                }
            }

            let param_count = unsafe { (self.library.reflection_entry_point_get_parameter_count)(ep) };
            for p in 0..param_count {
                let param = unsafe { (self.library.reflection_entry_point_get_parameter_by_index)(ep, p) };
                if param.is_null() {
                    continue;
                }
                let type_layout = unsafe { (self.library.reflection_variable_layout_get_type_layout)(param) };
                if type_layout.is_null() {
                    continue;
                }
                let category = unsafe { (self.library.reflection_type_layout_get_category)(type_layout) };
                if category == SlangParameterCategory::Uniform as i32
                    || category == SlangParameterCategory::PushConstantBuffer as i32
                    || category == SlangParameterCategory::ConstantBuffer as i32
                {
                    continue;
                }
                let is_varying = category == SlangParameterCategory::VaryingInput as i32
                    || category == SlangParameterCategory::VaryingOutput as i32
                    || category == SlangParameterCategory::Mixed as i32
                    || category == SlangParameterCategory::MetalPayload as i32;
                let is_meshish = stage_name == "mesh" || stage_name == "amplification";
                if !is_varying && !is_meshish {
                    continue;
                }
                let mut fields = Vec::new();
                self.collect_io_fields(param, "", &mut fields);
                fields.retain(is_graphics_payload_field);
                if fields.is_empty() {
                    continue;
                }
                if category == SlangParameterCategory::VaryingOutput as i32
                    || category == SlangParameterCategory::MetalPayload as i32
                    || (is_meshish && stage_name == "mesh")
                {
                    iface.payload_outputs.extend(fields);
                } else if stage_name == "vertex" {
                    iface.vertex_inputs.extend(fields);
                } else {
                    iface.payload_inputs.extend(fields);
                }
            }

            let _ = SlangTypeKind::None;
            out.push(iface);
        }
        out
    }

    fn collect_io_fields(
        &self,
        var_layout: *mut super::ffi::SlangReflectionVariableLayout,
        parent_struct: &str,
        out: &mut Vec<crate::slang::graphics_link::StageIoField>,
    ) {
        use crate::slang::ffi::{SlangScalarType, SlangTypeKind};
        use crate::slang::graphics_link::{parse_semantic, InterpolationMode, StageIoField};

        let variable = unsafe { (self.library.reflection_variable_layout_get_variable)(var_layout) };
        let field_name = if !variable.is_null() {
            let ptr = unsafe { (self.library.reflection_variable_get_name)(variable) };
            if ptr.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
            }
        } else {
            String::new()
        };

        let type_layout = unsafe { (self.library.reflection_variable_layout_get_type_layout)(var_layout) };
        if type_layout.is_null() {
            return;
        }
        let type_ptr = unsafe { (self.library.reflection_type_layout_get_type)(type_layout) };
        if type_ptr.is_null() {
            return;
        }
        let kind = unsafe { (self.library.reflection_type_get_kind)(type_ptr) };
        let type_name = unsafe {
            let ptr = (self.library.reflection_type_get_name)(type_ptr);
            if ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };

        if kind == SlangTypeKind::Struct as i32 {
            let field_count = unsafe { (self.library.reflection_type_layout_get_field_count)(type_layout) };
            if field_count > 0 {
                let struct_name = if type_name.is_empty() {
                    field_name.clone()
                } else {
                    type_name
                };
                for i in 0..field_count {
                    let field = unsafe { (self.library.reflection_type_layout_get_field_by_index)(type_layout, i) };
                    if !field.is_null() {
                        self.collect_io_fields(field, &struct_name, out);
                    }
                }
                return;
            }
        }

        if kind == SlangTypeKind::Array as i32 {
            let elem_layout = unsafe { (self.library.reflection_type_layout_get_element_type_layout)(type_layout) };
            if !elem_layout.is_null() {
                let elem_ty = unsafe { (self.library.reflection_type_layout_get_type)(elem_layout) };
                if !elem_ty.is_null() {
                    let elem_kind = unsafe { (self.library.reflection_type_get_kind)(elem_ty) };
                    if elem_kind == SlangTypeKind::Struct as i32 {
                        let elem_name = unsafe {
                            let ptr = (self.library.reflection_type_get_name)(elem_ty);
                            if ptr.is_null() {
                                field_name.clone()
                            } else {
                                CStr::from_ptr(ptr).to_string_lossy().into_owned()
                            }
                        };
                        let field_count = unsafe { (self.library.reflection_type_layout_get_field_count)(elem_layout) };
                        for i in 0..field_count {
                            let field =
                                unsafe { (self.library.reflection_type_layout_get_field_by_index)(elem_layout, i) };
                            if !field.is_null() {
                                self.collect_io_fields(field, &elem_name, out);
                            }
                        }
                        return;
                    }
                }
            }
        }

        let semantic_ptr = unsafe { (self.library.reflection_variable_layout_get_semantic_name)(var_layout) };
        let semantic_raw = if semantic_ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(semantic_ptr) }.to_string_lossy().into_owned()
        };
        if semantic_raw.is_empty() && kind == SlangTypeKind::Struct as i32 {
            return;
        }
        let semantic_index = unsafe { (self.library.reflection_variable_layout_get_semantic_index)(var_layout) } as u32;
        let (semantic, parsed_index) = if semantic_raw.is_empty() {
            (field_name.to_ascii_uppercase(), 0)
        } else {
            let (n, i) = parse_semantic(&semantic_raw);
            (n, i)
        };
        let semantic_index = if parsed_index != 0 {
            parsed_index
        } else {
            semantic_index
        };

        let (scalar_type, vector_size) = io_type_shape(&self.library, type_ptr, kind, &type_name);
        let _ = SlangScalarType::None;
        out.push(StageIoField {
            field_name,
            struct_name: if parent_struct.is_empty() {
                type_name
            } else {
                parent_struct.to_string()
            },
            semantic,
            semantic_index,
            scalar_type,
            vector_size,
            interpolation: InterpolationMode::Perspective,
        });
    }
}

fn is_graphics_payload_field(field: &crate::slang::graphics_link::StageIoField) -> bool {
    if field.field_name.starts_with('_') {
        return false;
    }
    let sem = field.semantic.to_ascii_uppercase();
    if sem.is_empty() {
        return false;
    }
    if sem.starts_with("SV_GROUP")
        || sem.starts_with("SV_DISPATCH")
        || sem == "SV_VERTEXID"
        || sem == "SV_INSTANCEID"
        || sem == "SV_ISFRONTFACE"
        || sem == "SV_PRIMITIVEID"
    {
        return false;
    }
    sem.starts_with("SV_")
        || sem.starts_with("TEXCOORD")
        || sem.starts_with("COLOR")
        || sem.starts_with("NORMAL")
        || sem.starts_with("TANGENT")
        || sem.starts_with("BINORMAL")
        || sem.starts_with("BLEND")
        || sem == "POSITION"
        || sem == "PSIZE"
}

fn slang_stage_name(stage: i32) -> &'static str {
    // Matches SlangStage in ffi.rs / slang.h
    match stage {
        1 => "vertex",
        2 => "hull",
        3 => "domain",
        4 => "geometry",
        5 => "fragment",
        6 => "compute",
        7 => "raygen",
        8 => "intersection",
        9 => "anyhit",
        10 => "closesthit",
        11 => "miss",
        12 => "callable",
        13 => "mesh",
        14 => "amplification",
        _ => "unknown",
    }
}

fn io_type_shape(
    library: &super::loader::SlangLibrary,
    type_ptr: *mut super::ffi::SlangReflectionType,
    kind: i32,
    type_name: &str,
) -> (String, u32) {
    use crate::slang::ffi::{SlangScalarType, SlangTypeKind};
    use crate::slang::graphics_link::parse_value_shape;

    if kind == SlangTypeKind::Vector as i32 {
        let cols = unsafe { (library.reflection_type_get_column_count)(type_ptr) }.max(1);
        let elem = unsafe { (library.reflection_type_get_element_type)(type_ptr) };
        let scalar = if elem.is_null() {
            scalar_type_name(unsafe { (library.reflection_type_get_scalar_type)(type_ptr) })
        } else {
            scalar_type_name(unsafe { (library.reflection_type_get_scalar_type)(elem) })
        };
        return (scalar, cols);
    }
    if kind == SlangTypeKind::Scalar as i32 {
        return (
            scalar_type_name(unsafe { (library.reflection_type_get_scalar_type)(type_ptr) }),
            1,
        );
    }
    let _ = SlangScalarType::None;
    parse_value_shape(type_name)
}

fn scalar_type_name(scalar: i32) -> String {
    use crate::slang::ffi::SlangScalarType;
    match scalar {
        x if x == SlangScalarType::Bool as i32 => "bool",
        x if x == SlangScalarType::Int32 as i32 => "int",
        x if x == SlangScalarType::Uint32 as i32 => "uint",
        x if x == SlangScalarType::Float16 as i32 => "half",
        x if x == SlangScalarType::Float32 as i32 => "float",
        x if x == SlangScalarType::Float64 as i32 => "double",
        x if x == SlangScalarType::Int8 as i32 => "int",
        x if x == SlangScalarType::Uint8 as i32 => "uint",
        x if x == SlangScalarType::Int16 as i32 => "int",
        x if x == SlangScalarType::Uint16 as i32 => "uint",
        _ => "float",
    }
    .to_string()
}

impl SlangCompiler {
    /// Extract layout information for a ParameterBlock.
    fn extract_parameter_block_layout(
        &self,
        param: *mut SlangReflectionParameter,
        type_layout: *mut SlangReflectionTypeLayout,
        name: &str,
    ) -> Result<ParameterBlockLayout> {
        // Get binding information
        let binding_slot = unsafe { (self.library.reflection_parameter_get_binding_index)(param) } as u32;
        let binding_space = unsafe { (self.library.reflection_parameter_get_binding_space)(param) } as u32;

        // Get the element type layout (the T in ParameterBlock<T>)
        let element_type_layout = unsafe { (self.library.reflection_type_layout_get_element_type_layout)(type_layout) };

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
                (self.library.reflection_type_layout_get_size)(type_layout, SlangParameterCategory::Uniform as i32)
            };
            let alignment = unsafe {
                (self.library.reflection_type_layout_get_alignment)(type_layout, SlangParameterCategory::Uniform as i32)
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
    fn extract_struct_fields(&self, type_layout: *mut SlangReflectionTypeLayout) -> Result<Vec<FieldLayout>> {
        let mut fields = Vec::new();

        let field_count = unsafe { (self.library.reflection_type_layout_get_field_count)(type_layout) };

        for i in 0..field_count {
            let field_var = unsafe { (self.library.reflection_type_layout_get_field_by_index)(type_layout, i) };
            if field_var.is_null() {
                continue;
            }

            // Get field name (variable layout -> variable -> name)
            let variable = unsafe { (self.library.reflection_variable_layout_get_variable)(field_var) };
            let name = if !variable.is_null() {
                let name_ptr = unsafe { (self.library.reflection_variable_get_name)(variable) };
                if !name_ptr.is_null() {
                    unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy().into_owned()
                } else {
                    format!("field_{}", i)
                }
            } else {
                format!("field_{}", i)
            };

            // Get field type layout
            let field_type_layout = unsafe { (self.library.reflection_variable_layout_get_type_layout)(field_var) };
            if field_type_layout.is_null() {
                continue;
            }

            // Get type name
            let field_type = unsafe { (self.library.reflection_type_layout_get_type)(field_type_layout) };
            let type_name = if !field_type.is_null() {
                let type_name_ptr = unsafe { (self.library.reflection_type_get_name)(field_type) };
                if !type_name_ptr.is_null() {
                    unsafe { CStr::from_ptr(type_name_ptr) }.to_string_lossy().into_owned()
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let resource_kind = self.determine_resource_kind(field_type_layout, &type_name);

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
                name,
                i,
                offset_slots,
                size_slots,
                offset,
                size,
                resource_kind
            );

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
    fn determine_resource_kind(&self, type_layout: *mut SlangReflectionTypeLayout, type_name: &str) -> ResourceKind {
        let type_ptr = unsafe { (self.library.reflection_type_layout_get_type)(type_layout) };
        if type_ptr.is_null() {
            return ResourceKind::Other;
        }

        let type_kind = unsafe { (self.library.reflection_type_get_kind)(type_ptr) };
        let binding_type = unsafe { (self.library.reflection_type_layout_get_binding_type)(type_layout) };

        tracing::trace!(
            "determine_resource_kind: type_kind={}, binding_type={}, type_name={}",
            type_kind,
            binding_type,
            type_name
        );

        let kind = match type_kind {
            k if k == SlangTypeKind::SamplerState as i32 => ResourceKind::Sampler,
            k if k == SlangTypeKind::ConstantBuffer as i32 => ResourceKind::ConstantBuffer,
            k if k == SlangTypeKind::ParameterBlock as i32 => ResourceKind::ParameterBlock,
            k if k == SlangTypeKind::Resource as i32 => match binding_type {
                b if b == SlangBindingType::Texture as i32 => ResourceKind::Texture,
                b if b == SlangBindingType::MutableTexture as i32 => ResourceKind::MutableTexture,
                b if b == SlangBindingType::TypedBuffer as i32 => ResourceKind::Buffer,
                b if b == SlangBindingType::MutableTypedBuffer as i32 => ResourceKind::MutableBuffer,
                b if b == SlangBindingType::RawBuffer as i32 => ResourceKind::Buffer,
                b if b == SlangBindingType::MutableRawBuffer as i32 => ResourceKind::MutableBuffer,
                b if b == SlangBindingType::RayTracingAccelerationStructure as i32 => {
                    ResourceKind::AccelerationStructure
                }
                _ => ResourceKind::Other,
            },
            k if k == SlangTypeKind::ShaderStorageBuffer as i32 => ResourceKind::MutableBuffer,
            _ => match binding_type {
                b if b == SlangBindingType::TypedBuffer as i32 => ResourceKind::Buffer,
                b if b == SlangBindingType::MutableTypedBuffer as i32 => ResourceKind::MutableBuffer,
                b if b == SlangBindingType::RawBuffer as i32 => ResourceKind::Buffer,
                b if b == SlangBindingType::MutableRawBuffer as i32 => ResourceKind::MutableBuffer,
                b if b == SlangBindingType::Texture as i32 => ResourceKind::Texture,
                b if b == SlangBindingType::MutableTexture as i32 => ResourceKind::MutableTexture,
                b if b == SlangBindingType::Sampler as i32 => ResourceKind::Sampler,
                b if b == SlangBindingType::ConstantBuffer as i32 => ResourceKind::ConstantBuffer,
                b if b == SlangBindingType::RayTracingAccelerationStructure as i32 => {
                    ResourceKind::AccelerationStructure
                }
                _ => ResourceKind::Other,
            },
        };
        if kind == ResourceKind::Other && type_name.contains("AccelerationStructure") {
            ResourceKind::AccelerationStructure
        } else {
            kind
        }
    }
}

#[cfg(test)]
mod builtin_stride_tests {
    use super::builtin_type_stride;

    #[test]
    fn scalar_types() {
        assert_eq!(builtin_type_stride("uint"), Some(4));
        assert_eq!(builtin_type_stride("int"), Some(4));
        assert_eq!(builtin_type_stride("float"), Some(4));
        assert_eq!(builtin_type_stride("half"), Some(2));
        assert_eq!(builtin_type_stride("double"), Some(8));
    }

    #[test]
    fn vector_types() {
        assert_eq!(builtin_type_stride("float2"), Some(8));
        assert_eq!(builtin_type_stride("float3"), Some(12));
        assert_eq!(builtin_type_stride("float4"), Some(16));
        assert_eq!(builtin_type_stride("uint4"), Some(16));
    }

    #[test]
    fn matrix_types() {
        assert_eq!(builtin_type_stride("float4x4"), Some(64));
    }

    #[test]
    fn user_struct_returns_none() {
        assert_eq!(builtin_type_stride("MyStruct"), None);
        assert_eq!(builtin_type_stride("Particle"), None);
    }

    #[test]
    fn dispatch_shape_stride() {
        assert_eq!(builtin_type_stride("DispatchShape"), Some(12));
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
            err.to_string().contains("smaller than the shader's data extent"),
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
        two_float_layout().validate(8, &[("a", 0, 4), ("b", 4, 4)]).unwrap();
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
        let err = two_float_layout().validate(8, &[("a", 0, 4)]).unwrap_err().to_string();
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
        let err = two_float_layout().validate(8, &[("a", 4, 4)]).unwrap_err().to_string();
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
    fn gpu_type_derive_generates_portable_field_metadata() {
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, goldy_derive::GpuType)]
        #[repr(C)]
        struct GeneratedVertex {
            pos: [f32; 3],
            uv: [f32; 2],
            light: u32,
        }

        let ty = GeneratedVertex::GPU_TYPE;
        assert_eq!(ty.type_name, "GeneratedVertex");
        assert_eq!(ty.rust_size, 24);
        assert_eq!(ty.fields.len(), 3);
        assert_eq!(ty.fields[0].offset, 0);
        assert_eq!(ty.fields[0].ty, crate::GpuFieldType::F32x3);
        assert_eq!(ty.fields[1].offset, 12);
        assert_eq!(ty.fields[1].ty, crate::GpuFieldType::F32x2);
        assert_eq!(ty.fields[2].offset, 20);
        assert_eq!(ty.fields[2].ty, crate::GpuFieldType::U32);
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
        assert!(err.contains("`y`"), "error should name the offending field: {err}");
    }

    /// Integration test: compiles a `[goldy_compute]` shader that uses
    /// `Scattered<uint>` and a broadcast struct, then verifies that the
    /// reflected `binding_element_strides` contain the expected values.
    #[test]
    fn stride_extraction_end_to_end() {
        use super::{ShaderTarget, SlangCompiler, SlangStage};
        use crate::types::OptimizationLevel;

        let compiler = SlangCompiler::new().expect("Slang compiler unavailable; skipping");

        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_dir.join("shaders").to_string_lossy().into_owned();

        let source = r#"
            import goldy_exp;

            struct Params { float x; float y; };

            [goldy_compute]
            [numthreads(64, 1, 1)]
            void cs_main(Params cfg, Scattered<uint> data, ThreadId id) {
                data[id.x] = uint(cfg.x);
            }
        "#;

        let result = compiler
            .compile_with_reflection(
                source,
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[("__SPIRV__", "1")],
                &[],
                OptimizationLevel::None,
            )
            .expect("compilation failed");

        let strides = &result.reflection.binding_element_strides;
        assert_eq!(strides.len(), 2, "expected 2 binding slots: {strides:?}");

        // Broadcast params use reflect_struct_storage_stride: field extent without
        // std140 tail-padding.  Params { float x; float y } = 2 × 4 = 8 bytes.
        assert_eq!(
            strides[0],
            Some(8),
            "Broadcast Params {{float x; float y}} natural stride should be 8 (not cbuffer 16): {strides:?}"
        );
        assert_eq!(
            strides[1],
            Some(4),
            "Scattered<uint> element stride should be 4: {strides:?}"
        );
    }

    #[test]
    fn stride_extraction_structured_buffer_element_uses_storage_layout() {
        use super::{ShaderTarget, SlangCompiler, SlangStage};
        use crate::types::OptimizationLevel;

        let compiler = SlangCompiler::new().expect("Slang compiler unavailable; skipping");

        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_dir.join("shaders").to_string_lossy().into_owned();

        let source = r#"
            import goldy_exp;

            struct Pair { uint a; uint b; };

            [goldy_compute]
            [numthreads(64, 1, 1)]
            void cs_main(BufRO<Pair> input, Scattered<Pair> output, ThreadId id) {
                output[id.x] = input[id.x];
            }
        "#;

        let result = compiler
            .compile_with_reflection(
                source,
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[("__SPIRV__", "1")],
                &[],
                OptimizationLevel::None,
            )
            .expect("compilation failed");

        let strides = &result.reflection.binding_element_strides;
        assert_eq!(strides.len(), 2, "expected 2 binding slots: {strides:?}");
        assert_eq!(
            strides[0],
            Some(8),
            "BufRO<Pair> element stride should be 8 (not uniform 16): {strides:?}"
        );
        assert_eq!(
            strides[1],
            Some(8),
            "Scattered<Pair> element stride should be 8: {strides:?}"
        );
    }

    /// Regression: Broadcast params (plain struct without Scattered<>) must use
    /// struct_storage_stride (natural field extent) — NOT the std140-padded
    /// cbuffer size.  Before the fix, `reflect_type_size_with_category(Uniform)`
    /// returned 16 for a single-float struct (cbuffer alignment), causing
    /// validate_binding_strides to reject a correctly-created buffer with
    /// element_stride = 4.
    #[test]
    fn broadcast_param_stride_matches_natural_struct_size_not_cbuffer_padded() {
        use super::{ShaderTarget, SlangCompiler, SlangStage};
        use crate::types::OptimizationLevel;

        let compiler = SlangCompiler::new().expect("Slang compiler unavailable; skipping");

        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_dir.join("shaders").to_string_lossy().into_owned();

        // SimParams { float deltaTime; } — natural size 4, cbuffer-padded size 16.
        let source = r#"
            import goldy_exp;

            struct SimParams { float deltaTime; };

            [goldy_compute]
            [numthreads(64, 1, 1)]
            void cs_main(Scattered<uint> data, SimParams params, ThreadId id) {
                data[id.x] = uint(params.deltaTime);
            }
        "#;

        let result = compiler
            .compile_with_reflection(
                source,
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[("__SPIRV__", "1")],
                &[],
                OptimizationLevel::None,
            )
            .expect("compilation failed");

        let strides = &result.reflection.binding_element_strides;
        assert_eq!(strides.len(), 2, "expected 2 binding slots: {strides:?}");
        assert_eq!(
            strides[0],
            Some(4),
            "Scattered<uint> element stride should be 4: {strides:?}"
        );
        // This was the bug: cbuffer-padded layout returned 16 here, causing
        // validate_binding_strides to fail for a correctly-created buffer.
        assert_eq!(
            strides[1],
            Some(4),
            "Broadcast SimParams{{float deltaTime}} natural stride should be 4, not cbuffer 16: {strides:?}"
        );
    }

    /// Multi-field Broadcast struct: stride must be the sum of fields, not
    /// the std140 whole-struct size.  `Params { float x; float y; }` = 8 bytes
    /// naturally; std140 would pad to 16.
    #[test]
    fn broadcast_two_float_struct_stride_is_eight_not_sixteen() {
        use super::{ShaderTarget, SlangCompiler, SlangStage};
        use crate::types::OptimizationLevel;

        let compiler = SlangCompiler::new().expect("Slang compiler unavailable; skipping");

        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_dir.join("shaders").to_string_lossy().into_owned();

        let source = r#"
            import goldy_exp;

            struct Params { float x; float y; };

            [goldy_compute]
            [numthreads(64, 1, 1)]
            void cs_main(Params cfg, Scattered<uint> data, ThreadId id) {
                data[id.x] = uint(cfg.x + cfg.y);
            }
        "#;

        let result = compiler
            .compile_with_reflection(
                source,
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[("__SPIRV__", "1")],
                &[],
                OptimizationLevel::None,
            )
            .expect("compilation failed");

        let strides = &result.reflection.binding_element_strides;
        assert_eq!(strides.len(), 2, "expected 2 binding slots: {strides:?}");
        assert_eq!(
            strides[0],
            Some(8),
            "Broadcast Params{{float x; float y}} natural stride = 8: {strides:?}"
        );
        assert_eq!(strides[1], Some(4), "Scattered<uint> = 4: {strides:?}");
    }

    /// Validate that `validate_binding_strides` correctly catches a stride
    /// mismatch (expected vs actual) and passes when they agree.
    #[test]
    fn validate_binding_strides_passes_and_fails_correctly() {
        use crate::backend::validate_binding_strides;

        // Matching strides — must pass.
        let actual = vec![Some(16u32), Some(4u32)];
        let expected = vec![Some(16u32), Some(4u32)];
        assert!(validate_binding_strides(&actual, &expected, "test").is_ok());

        // Slot 1 mismatch: 16 expected, 4 actual — must fail with the slot number.
        let actual_bad = vec![Some(16u32), Some(4u32)];
        let expected_bad = vec![Some(16u32), Some(16u32)];
        let err =
            validate_binding_strides(&actual_bad, &expected_bad, "myshader").expect_err("should fail on mismatch");
        let msg = err.to_string();
        assert!(msg.contains("slot 1"), "error should name the slot: {msg}");
        assert!(msg.contains("myshader"), "error should name the shader: {msg}");
    }
}

impl Drop for SlangCompiler {
    fn drop(&mut self) {
        let _guard = SLANG_PROCESS_LOCK.lock().unwrap();
        if !self.global_session.is_null() {
            unsafe { global_session_release(self.global_session) };
            self.global_session = std::ptr::null_mut();
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
    fn pretransformed_source_matches_default_spirv_compile() {
        let compiler = SlangCompiler::new().expect("Slang unavailable");
        let path = shader_path();
        let effective = crate::slang::virtual_main::effective_slang_source_for_compile(TEST_SHADER);
        let via_default = compiler
            .compile_bindless_with_reflection_and_defines(
                TEST_SHADER,
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("default SPIR-V compile");
        let via_effective = compiler
            .compile_bindless_with_reflection_effective(
                TEST_SHADER,
                effective.as_ref(),
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("effective SPIR-V compile");
        assert_eq!(via_default.shader.data, via_effective.shader.data);
        assert_eq!(
            via_default.reflection.push_constant_categories,
            via_effective.reflection.push_constant_categories
        );
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

#[cfg(test)]
mod multi_entry_wgsl_tests {
    use super::*;

    #[test]
    fn compile_vertex_and_fragment_wgsl_in_one_request() {
        let compiler = SlangCompiler::new().expect("Slang unavailable");
        let src = r#"
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
        let compiled = compiler
            .compile_bindless_with_reflection_and_defines(
                src,
                ShaderTarget::Wgsl,
                &[("vs_main", SlangStage::Vertex), ("fs_main", SlangStage::Fragment)],
                &[],
                &[],
                &[],
                OptimizationLevel::Default,
            )
            .expect("combined vs/fs WGSL compile");
        let vs = compiled.shader.as_str().expect("vs WGSL");
        let fs = compiled.entry_point(1).and_then(|s| s.as_str()).expect("fs WGSL");
        assert!(vs.contains("@vertex"), "missing vertex entry:\n{vs}");
        assert!(fs.contains("@fragment"), "missing fragment entry:\n{fs}");
        assert_eq!(compiled.extra_entry_points.len(), 1);
    }
}

/// Spike: CUDA `DirectSpatial<float4, uint8_t4>` `__subscript` must compile through NVRTC.
#[cfg(all(test, feature = "cuda"))]
mod cuda_direct_spatial_rgba8_view_tests {
    use super::*;

    fn shader_path() -> String {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join("shaders").to_string_lossy().into_owned()
    }

    const RGBA8_VIEW_SHADER: &str = r#"
        import goldy_exp;

        [shader("compute")]
        [numthreads(8, 8, 1)]
        void cs_main(uniform RWTexture2D<uint8_t4> raw, uint3 id : SV_DispatchThreadID) {
            DirectSpatial<float4, uint8_t4> view = DirectSpatial<float4, uint8_t4>(raw);
            uint2 dims;
            view.GetDimensions(dims.x, dims.y);
            if (id.x < dims.x && id.y < dims.y) {
                float4 v = view[int2(id.xy)];
                view[uint2(id.xy)] = v + float4(0.1, 0.0, 0.0, 0.0);
                view[int2(id.xy)] = float4(1.0, 0.0, 0.0, 1.0);
            }
        }
    "#;

    #[test]
    fn rgba8_view_subscript_compiles_ptx() {
        let compiler = SlangCompiler::new().expect("Slang unavailable");
        let path = shader_path();
        let transformed = crate::slang::virtual_main::transform_virtual_main_cuda_compute(RGBA8_VIEW_SHADER, &[])
            .unwrap_or_else(|_| RGBA8_VIEW_SHADER.to_string());
        // Plain shader (no [goldy_compute]) passes through unchanged.
        let output = compiler
            .compile_bindless_with_reflection_and_defines(
                &transformed,
                ShaderTarget::Ptx,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("CUDA PTX compilation failed for DirectSpatial<float4, uint8_t4> subscript");
        let ptx = output.shader.as_str().expect("PTX text");
        assert!(!ptx.is_empty(), "PTX output is empty");
    }
}

#[cfg(test)]
mod rt_mesh_compile_tests {
    use super::*;

    #[test]
    fn compile_raygen_spirv() {
        let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
        let source = r#"
            [shader("raygeneration")]
            void rgen_main() {}
        "#;
        let out = compiler
            .compile_with_reflection(
                source,
                ShaderTarget::Spirv,
                &[("rgen_main", SlangStage::RayGeneration)],
                &[],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("raygeneration SPIR-V compile failed");
        assert!(!out.shader.data.is_empty());
    }

    #[test]
    fn compile_mesh_spirv() {
        let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
        let source = r#"
            struct MeshOutput {
                float4 pos : SV_Position;
            };
            [shader("mesh")]
            [numthreads(1, 1, 1)]
            [outputtopology("triangle")]
            void mesh_main(out vertices MeshOutput verts[3], out indices uint3 tris[1]) {
                SetMeshOutputCounts(3, 1);
                verts[0].pos = float4(-1, -1, 0, 1);
                verts[1].pos = float4(3, -1, 0, 1);
                verts[2].pos = float4(-1, 3, 0, 1);
                tris[0] = uint3(0, 1, 2);
            }
        "#;
        let out = compiler
            .compile_with_reflection(
                source,
                ShaderTarget::Spirv,
                &[("mesh_main", SlangStage::Mesh)],
                &[],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("mesh SPIR-V compile failed");
        assert!(!out.shader.data.is_empty());
    }

    #[test]
    fn compile_mesh_metal_assigns_whole_vertex_struct() {
        let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("shaders")
            .to_string_lossy()
            .into_owned();
        let source = r#"
            import goldy_exp;
            struct MeshOutput {
                float4 pos : SV_Position;
                float4 color : COLOR;
            };
            [goldy_mesh]
            [numthreads(1, 1, 1)]
            [outputtopology("triangle")]
            void mesh_main(out vertices MeshOutput verts[3], out indices uint3 tris[1]) {
                SetMeshOutputCounts(3, 1);
                verts[0] = { float4(0.0, -0.5, 0.0, 1.0), float4(1.0, 0.0, 0.0, 1.0) };
                verts[1] = { float4(-0.5, 0.5, 0.0, 1.0), float4(0.0, 1.0, 0.0, 1.0) };
                verts[2] = { float4(0.5, 0.5, 0.0, 1.0), float4(0.0, 0.0, 1.0, 1.0) };
                tris[0] = uint3(0, 1, 2);
            }
        "#;
        let out = compiler
            .compile_with_reflection(
                source,
                ShaderTarget::Metal,
                &[("mesh_main", SlangStage::Mesh)],
                &[&path],
                &[("__METAL__", "1")],
                &[],
                OptimizationLevel::None,
            )
            .expect("mesh Metal compile failed");
        assert!(!out.shader.data.is_empty());
    }

    #[test]
    fn reflect_acceleration_structure_parameter_block() {
        let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
        let source = r#"
            struct Scene {
                RaytracingAccelerationStructure tlas;
            };
            ParameterBlock<Scene> gScene;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void cs_main(uint3 tid : SV_DispatchThreadID) {
                RayDesc ray;
                ray.Origin = float3(0, 0, 0);
                ray.Direction = float3(0, 0, 1);
                ray.TMin = 0;
                ray.TMax = 1;
                RayQuery<RAY_FLAG_NONE> q;
                q.TraceRayInline(gScene.tlas, RAY_FLAG_NONE, 0xff, ray);
            }
        "#;
        let out = compiler
            .compile_with_reflection(
                source,
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("acceleration-structure compute compile failed");
        let kinds: Vec<_> = out
            .reflection
            .parameter_blocks
            .iter()
            .flat_map(|pb| pb.fields.iter().map(|f| f.resource_kind))
            .collect();
        assert!(
            kinds.contains(&ResourceKind::AccelerationStructure),
            "expected AccelerationStructure in reflection, got {kinds:?} blocks={:?}",
            out.reflection.parameter_blocks
        );
    }

    #[test]
    fn compile_goldy_compute_accel_spirv() {
        let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
        let path = {
            let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            manifest_dir.join("shaders").to_string_lossy().into_owned()
        };
        let source = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Accel scene, Scattered<uint> hits, ThreadId id)
{
    RayDesc ray;
    ray.Origin = float3(0.0, 0.0, -2.0);
    ray.TMin = 0.001;
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMax = 100.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    q.Proceed();
    hits[id.x] = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 1 : 0;
}
"#;
        let out = compiler
            .compile_bindless_with_reflection_and_defines(
                source,
                ShaderTarget::Spirv,
                &[("cs_main", SlangStage::Compute)],
                &[&path],
                &[("GOLDY_RAY_QUERY", "1")],
                &[],
                OptimizationLevel::None,
            )
            .expect("goldy_compute Accel SPIR-V compile failed");
        assert!(!out.shader.data.is_empty());
        let words = out.shader.as_spirv().expect("SPIR-V");
        assert_eq!(words[0], 0x07230203);
    }
}

#[cfg(test)]
mod stage_io_reflection_tests {
    use super::*;

    fn shader_path() -> String {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir.join("shaders").to_string_lossy().into_owned()
    }

    const VS_FS: &str = r#"
import goldy_exp;

struct VertIn {
    float3 pos : POSITION;
    float2 uv  : TEXCOORD0;
};

struct Varying {
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

[goldy_vertex]
Varying vs_main(VertIn input) {
    Varying o;
    o.position = float4(input.pos, 1);
    o.uv = input.uv;
    return o;
}

[goldy_fragment]
float4 fs_main(Varying input) : SV_Target {
    return float4(input.uv, 0, 1);
}
"#;

    #[test]
    fn spirv_reflects_vertex_and_fragment_io() {
        let compiler = SlangCompiler::new().expect("Slang unavailable");
        let path = shader_path();
        let vs = compiler
            .compile_bindless_with_reflection_and_defines(
                VS_FS,
                ShaderTarget::Spirv,
                &[("vs_main", SlangStage::Vertex)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("VS SPIR-V");
        let fs = compiler
            .compile_bindless_with_reflection_and_defines(
                VS_FS,
                ShaderTarget::Spirv,
                &[("fs_main", SlangStage::Fragment)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("FS SPIR-V");
        let vs_io = vs
            .reflection
            .stage_interfaces
            .iter()
            .find(|s| s.stage == "vertex")
            .expect("vertex interface");
        let fs_io = fs
            .reflection
            .stage_interfaces
            .iter()
            .find(|s| s.stage == "fragment")
            .expect("fragment interface");
        assert!(
            !vs_io.payload_outputs.is_empty() || !vs_io.vertex_inputs.is_empty(),
            "expected vertex I/O fields, got {vs_io:?}"
        );
        crate::slang::graphics_link::refine_payload_link("vertex", "fragment", vs_io, fs_io)
            .expect("reflected VS/FS payload should link");
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn dxil_reflects_vertex_and_fragment_io() {
        let compiler = SlangCompiler::new().expect("Slang unavailable");
        let path = shader_path();
        let vs = compiler
            .compile_bindless_with_reflection_and_defines(
                VS_FS,
                ShaderTarget::Dxil,
                &[("vs_main", SlangStage::Vertex)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("VS DXIL");
        assert!(!vs.reflection.stage_interfaces.is_empty());
    }

    const MESH_FS: &str = r#"
import goldy_exp;

struct MeshOut {
    float4 pos : SV_Position;
    float4 color : COLOR;
};
struct FsIn {
    float4 pos : SV_Position;
    float4 color : COLOR;
};

[goldy_mesh]
[numthreads(1,1,1)]
[outputtopology("triangle")]
void mesh_main(out vertices MeshOut verts[3], out indices uint3 tris[1]) {
    SetMeshOutputCounts(3, 1);
    verts[0] = { float4(0,0,0,1), float4(1,0,0,1) };
    tris[0] = uint3(0, 1, 2);
}

[goldy_fragment]
float4 fs_main(FsIn input) : SV_Target { return input.color; }
"#;

    #[test]
    fn spirv_reflects_mesh_vertex_payload() {
        let compiler = SlangCompiler::new().expect("Slang unavailable");
        let path = shader_path();
        let mesh = compiler
            .compile_bindless_with_reflection_and_defines(
                MESH_FS,
                ShaderTarget::Spirv,
                &[("mesh_main", SlangStage::Mesh)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("mesh SPIR-V");
        let fs = compiler
            .compile_bindless_with_reflection_and_defines(
                MESH_FS,
                ShaderTarget::Spirv,
                &[("fs_main", SlangStage::Fragment)],
                &[&path],
                &[],
                &[],
                OptimizationLevel::None,
            )
            .expect("FS SPIR-V");
        let fs_io = fs
            .reflection
            .stage_interfaces
            .iter()
            .find(|s| s.stage == "fragment")
            .expect("fragment interface");
        assert!(
            fs_io
                .payload_inputs
                .iter()
                .any(|f| f.semantic.eq_ignore_ascii_case("SV_POSITION")),
            "expected fragment payload SV_Position, got {fs_io:?}"
        );
        let mesh_io = mesh
            .reflection
            .stage_interfaces
            .iter()
            .find(|s| s.stage == "mesh")
            .expect("mesh interface");
        let producer_outs = if mesh_io.payload_outputs.is_empty() {
            &mesh_io.payload_inputs
        } else {
            &mesh_io.payload_outputs
        };
        if producer_outs
            .iter()
            .any(|f| f.semantic.eq_ignore_ascii_case("SV_POSITION"))
        {
            let mut producer = mesh_io.clone();
            producer.payload_outputs = producer_outs.clone();
            crate::slang::graphics_link::refine_payload_link("mesh", "fragment", &producer, fs_io)
                .expect("reflected mesh/FS payload should link");
        }
    }
}
