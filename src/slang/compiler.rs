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
use crate::types::OptimizationLevel;
use crate::{goldy_event, goldy_span};

/// Returns `true` when `GOLDY_VALIDATE_LAYOUTS=1` (or any truthy value).
///
/// Controls both struct layout checks (at compile time) and buffer element-stride
/// checks (at dispatch time). Reads the environment on every call so that tests
/// can toggle the flag without restarting the process.
pub fn layout_validation_enabled() -> bool {
    std::env::var("GOLDY_VALIDATE_LAYOUTS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
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

/// Scan Slang source for `goldy_dyn_*(N)` calls with literal integer slot
/// arguments and build a per-slot vector of the
/// [`crate::types::BindlessCategory`] the shader expects at
/// each push-constant slot. Slots accessed via dynamic/computed indices (e.g.
/// `goldy_dyn_scattered(i)`) are left as `None`.
///
/// Multiple categories for the same slot are treated as a conflict and collapse
/// to `None` so the error surface is the `SetPushConstantsTyped` validator,
/// which can produce a context-rich diagnostic rather than a shader-load
/// failure.
///
/// Comments and preprocessor-disabled blocks are not stripped; this is a
/// best-effort conservative heuristic. False positives (a category inferred
/// for a slot the user no longer uses) only constrain what handle type the
/// user can bind — the user's recourse is to update the shader or bind the
/// right handle type. False negatives (slot left as `None`) silently fall back
/// to no validation for that slot, matching pre-reflection behavior.
pub fn analyze_push_constant_categories_from_source(
    source: &str,
) -> Vec<Option<crate::types::BindlessCategory>> {
    use crate::types::BindlessCategory;

    // Function name -> category it reads at the nth push-constant slot.
    // Keep in sync with `shaders/goldy_exp/access.slang` — this list MUST
    // match the public `goldy_dyn_*` surface exactly.
    const DYN_FUNCS: &[(&str, BindlessCategory)] = &[
        ("goldy_dyn_scattered", BindlessCategory::Scattered),
        ("goldy_dyn_buf_ro", BindlessCategory::Scattered),
        ("goldy_dyn_byte_address", BindlessCategory::Scattered),
        ("goldy_dyn_broadcast", BindlessCategory::Broadcast),
        ("goldy_dyn_direct_spatial", BindlessCategory::StorageImage),
        ("goldy_dyn_interpolated", BindlessCategory::Texture),
        ("goldy_dyn_filter", BindlessCategory::Sampler),
    ];

    // Observed category per slot; `Conflict` means ≥ 2 incompatible categories.
    #[derive(Copy, Clone, PartialEq)]
    enum Slot {
        Unknown,
        One(BindlessCategory),
        Conflict,
    }
    let mut slots = [Slot::Unknown; 16];

    for (fn_name, category) in DYN_FUNCS {
        let mut search_from = 0usize;
        while let Some(rel) = source[search_from..].find(fn_name) {
            let abs = search_from + rel;
            search_from = abs + fn_name.len();

            // Require a word boundary before the match so `goldy_dyn_scattered`
            // doesn't trigger on `__goldy_dyn_scattered_impl`.
            if abs > 0 {
                let prev = source.as_bytes()[abs - 1];
                if prev.is_ascii_alphanumeric() || prev == b'_' {
                    continue;
                }
            }

            // Skip an optional `<...>` generic argument list.
            let mut cursor = search_from;
            let bytes = source.as_bytes();
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            if cursor < bytes.len() && bytes[cursor] == b'<' {
                let mut depth = 1i32;
                cursor += 1;
                while cursor < bytes.len() && depth > 0 {
                    match bytes[cursor] {
                        b'<' => depth += 1,
                        b'>' => depth -= 1,
                        _ => {}
                    }
                    cursor += 1;
                }
            }
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }

            // Expect `(` then a decimal literal then `)` (or `,`/whitespace
            // before `)`). Anything else -> unresolvable, leave slot unknown.
            if cursor >= bytes.len() || bytes[cursor] != b'(' {
                continue;
            }
            cursor += 1;
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            let num_start = cursor;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor == num_start {
                continue;
            }
            // Only accept a clean `N)` or `N )` — anything else (identifiers,
            // arithmetic, casts) leaves the slot unknown on purpose.
            let mut end = cursor;
            while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                end += 1;
            }
            if end >= bytes.len() || (bytes[end] != b')' && bytes[end] != b'u') {
                continue;
            }
            // Accept `Nu` / `N u` (Slang unsigned suffix) -> must then close.
            if bytes[end] == b'u' {
                end += 1;
                while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                    end += 1;
                }
                if end >= bytes.len() || bytes[end] != b')' {
                    continue;
                }
            }

            let Ok(n) = source[num_start..cursor].parse::<usize>() else {
                continue;
            };
            if n >= slots.len() {
                continue;
            }
            slots[n] = match slots[n] {
                Slot::Unknown => Slot::One(*category),
                Slot::One(prev) if prev.is_compatible_with(*category) => Slot::One(prev),
                _ => Slot::Conflict,
            };
        }
    }

    // Trim trailing Unknown slots so short shaders don't carry 16-long vectors.
    let mut last_known = 0usize;
    for (i, s) in slots.iter().enumerate() {
        if !matches!(s, Slot::Unknown) {
            last_known = i + 1;
        }
    }
    slots[..last_known]
        .iter()
        .map(|s| match s {
            Slot::One(c) => Some(*c),
            Slot::Unknown | Slot::Conflict => None,
        })
        .collect()
}

/// Parsed stride hint for a push-constant slot (before Slang resolution).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PushConstantStrideHint {
    /// Reflect `T` from the same compile request to get the element byte stride.
    Structured { type_name: String },
    /// `goldy_dyn_byte_address` — host buffer should use byte stride 1.
    ByteAddress,
}

/// Scan Slang source for `goldy_dyn_buf_ro<T>(N)`, `goldy_dyn_scattered<T>(N)`,
/// `goldy_dyn_broadcast<T>(N)`, and `goldy_dyn_byte_address(N)` with literal
/// slot indices. Returns a sparse vector aligned to slot indices (same trimming
/// rules as [`analyze_push_constant_categories_from_source`]).
///
/// Conflicting hints for the same slot (e.g. different `T` or typed buffer vs
/// byte-address) collapse to `None` so validation is skipped for that slot.
pub fn analyze_push_constant_stride_hints_from_source(
    source: &str,
) -> Vec<Option<PushConstantStrideHint>> {
    #[derive(Clone, PartialEq, Eq)]
    enum Slot {
        Unknown,
        One(PushConstantStrideHint),
        Conflict,
    }
    let mut slots = vec![Slot::Unknown; 16];

    #[derive(Clone, Copy)]
    struct DynStrideFunc {
        name: &'static str,
        /// `true` = requires a `<T>` generic before `(`.
        needs_type_arg: bool,
    }

    const STRIDE_FUNCS: &[DynStrideFunc] = &[
        DynStrideFunc {
            name: "goldy_dyn_scattered",
            needs_type_arg: true,
        },
        DynStrideFunc {
            name: "goldy_dyn_buf_ro",
            needs_type_arg: true,
        },
        DynStrideFunc {
            name: "goldy_dyn_broadcast",
            needs_type_arg: true,
        },
        DynStrideFunc {
            name: "goldy_dyn_byte_address",
            needs_type_arg: false,
        },
    ];

    for spec in STRIDE_FUNCS {
        let fn_name = spec.name;
        let mut search_from = 0usize;
        while let Some(rel) = source[search_from..].find(fn_name) {
            let abs = search_from + rel;
            search_from = abs + fn_name.len();

            if abs > 0 {
                let prev = source.as_bytes()[abs - 1];
                if prev.is_ascii_alphanumeric() || prev == b'_' {
                    continue;
                }
            }

            let mut cursor = search_from;
            let bytes = source.as_bytes();

            let hint = if spec.needs_type_arg {
                while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                    cursor += 1;
                }
                if cursor >= bytes.len() || bytes[cursor] != b'<' {
                    continue;
                }
                let mut depth = 1i32;
                let type_start = cursor + 1;
                cursor += 1;
                while cursor < bytes.len() && depth > 0 {
                    match bytes[cursor] {
                        b'<' => depth += 1,
                        b'>' => depth -= 1,
                        _ => {}
                    }
                    cursor += 1;
                }
                if depth != 0 {
                    continue;
                }
                let type_end = cursor - 1;
                let type_name = source[type_start..type_end].trim();
                if type_name.is_empty() {
                    continue;
                }
                PushConstantStrideHint::Structured {
                    type_name: type_name.to_string(),
                }
            } else {
                while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                    cursor += 1;
                }
                PushConstantStrideHint::ByteAddress
            };

            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }

            if cursor >= bytes.len() || bytes[cursor] != b'(' {
                continue;
            }
            cursor += 1;
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            let num_start = cursor;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor == num_start {
                continue;
            }
            let mut end = cursor;
            while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                end += 1;
            }
            if end >= bytes.len() || (bytes[end] != b')' && bytes[end] != b'u') {
                continue;
            }
            if bytes[end] == b'u' {
                end += 1;
                while end < bytes.len() && bytes[end].is_ascii_whitespace() {
                    end += 1;
                }
                if end >= bytes.len() || bytes[end] != b')' {
                    continue;
                }
            }

            let Ok(n) = source[num_start..cursor].parse::<usize>() else {
                continue;
            };
            if n >= slots.len() {
                continue;
            }

            slots[n] = match (&slots[n], &hint) {
                (Slot::Unknown, _) => Slot::One(hint.clone()),
                (Slot::One(prev), next) if prev == next => Slot::One(prev.clone()),
                _ => Slot::Conflict,
            };
        }
    }

    let mut last_known = 0usize;
    for (i, s) in slots.iter().enumerate() {
        if !matches!(s, Slot::Unknown) {
            last_known = i + 1;
        }
    }
    slots[..last_known]
        .iter()
        .map(|s| match s {
            Slot::One(h) => Some(h.clone()),
            Slot::Unknown | Slot::Conflict => None,
        })
        .collect()
}

/// Complete reflection information for a compiled shader
#[derive(Debug, Clone, Default)]
pub struct ShaderReflection {
    /// All parameter blocks found in the shader
    pub parameter_blocks: Vec<ParameterBlockLayout>,
    /// Per push-constant slot (index = slot number), the
    /// [`crate::types::BindlessCategory`] the shader reads
    /// that slot as, or `None` if reflection couldn't infer it (e.g. the slot
    /// was only accessed via a dynamic index). Populated by source-level
    /// analysis of `goldy_dyn_*(N)` calls in the Slang source; may be sparse
    /// up to `MAX_PUSH_CONSTANT_INDICES`. Used by
    /// [`crate::types::BindlessHandle`]-typed push-constant setters to catch
    /// category mismatches at dispatch time.
    pub push_constant_categories: Vec<Option<crate::types::BindlessCategory>>,
    /// Per push-constant slot, the structured-buffer / uniform element size in
    /// bytes the shader expects for `goldy_dyn_*<T>(slot)` (or `1` for
    /// `goldy_dyn_byte_address`), or `None` when stride checking doesn't apply
    /// or couldn't be resolved. Populated at shader compile time from source
    /// analysis + Slang reflection. Used when `GOLDY_VALIDATE_BUFFER_STRIDES`
    /// is enabled.
    pub push_constant_buffer_strides: Vec<Option<u32>>,
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
    /// `rust_fields` must be in declaration order and use the same field names as Slang.
    pub fn validate(&self, rust_size: usize, rust_fields: &[(&str, usize, usize)]) -> Result<()> {
        let mut errors: Vec<String> = Vec::new();

        if self.size != rust_size {
            errors.push(format!(
                "struct size: Slang {} bytes vs Rust {} bytes",
                self.size, rust_size
            ));
        }
        if self.fields.len() != rust_fields.len() {
            errors.push(format!(
                "field count: Slang {} vs Rust {}",
                self.fields.len(),
                rust_fields.len()
            ));
        }

        for (i, &(name, rust_offset, rust_size_field)) in rust_fields.iter().enumerate() {
            match self.fields.get(i) {
                Some(sf) => {
                    if sf.name != name {
                        errors.push(format!(
                            "field[{i}]: expected name `{name}`, Slang has `{}`",
                            sf.name
                        ));
                    }
                    if sf.offset != rust_offset {
                        errors.push(format!(
                            "field `{name}`: offset Slang {} vs Rust {}",
                            sf.offset, rust_offset
                        ));
                    }
                    if sf.size != rust_size_field {
                        errors.push(format!(
                            "field `{name}`: size Slang {} vs Rust {}",
                            sf.size, rust_size_field
                        ));
                    }
                }
                None => {
                    errors.push(format!(
                        "field[{i}] (`{name}`): missing in Slang reflection"
                    ));
                }
            }
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

                let mut reflection = slf.extract_reflection(request)?;
                reflection.push_constant_categories =
                    analyze_push_constant_categories_from_source(source);
                let stride_hints = analyze_push_constant_stride_hints_from_source(source);
                reflection.push_constant_buffer_strides =
                    slf.resolve_push_constant_buffer_strides(request, &stride_hints);

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

    /// Byte size of `type_name` for buffer / structured-buffer layout, using the
    /// first non-zero size among Slang categories (shader resource, uniform,
    /// constant buffer).
    fn reflect_type_byte_stride_for_buffer(
        &self,
        request: *mut SlangCompileRequest,
        type_name: &str,
    ) -> Result<u32> {
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

        for cat in [
            SlangParameterCategory::ShaderResource,
            SlangParameterCategory::Uniform,
            SlangParameterCategory::ConstantBuffer,
        ] {
            let size =
                unsafe { (self.library.reflection_type_layout_get_size)(layout_ptr, cat as i32) };
            if size > 0 {
                return Ok(size as u32);
            }
        }

        anyhow::bail!("Slang reflection: zero byte size for `{type_name}` in buffer layouts")
    }

    fn resolve_push_constant_buffer_strides(
        &self,
        request: *mut SlangCompileRequest,
        hints: &[Option<PushConstantStrideHint>],
    ) -> Vec<Option<u32>> {
        hints
            .iter()
            .map(|h| match h {
                None => None,
                Some(PushConstantStrideHint::ByteAddress) => Some(1),
                Some(PushConstantStrideHint::Structured { type_name }) => {
                    match self.reflect_type_byte_stride_for_buffer(request, type_name) {
                        Ok(s) => Some(s),
                        Err(e) => {
                            tracing::warn!(
                                target: "goldy::slang",
                                "could not resolve push-constant stride for Slang type `{type_name}`: {e}"
                            );
                            None
                        }
                    }
                }
            })
            .collect()
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
            push_constant_buffer_strides: Vec::new(),
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

    #[test]
    fn validate_ok_when_matching() {
        two_float_layout()
            .validate(8, &[("a", 0, 4), ("b", 4, 4)])
            .unwrap();
    }

    #[test]
    fn validate_err_on_struct_size_mismatch() {
        let mut layout = two_float_layout();
        layout.size = 16;
        let err = layout
            .validate(8, &[("a", 0, 4), ("b", 4, 4)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("size"), "expected size mismatch: {err}");
        assert!(err.contains("16"), "expected Slang size 16: {err}");
        assert!(err.contains("8"), "expected Rust size 8: {err}");
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
        assert!(err.contains("`x`"), "expected name x in message: {err}");
        assert!(err.contains("`a`"), "expected name a in message: {err}");
    }

    #[test]
    fn validate_err_on_field_count_mismatch() {
        let err = two_float_layout()
            .validate(8, &[("a", 0, 4)])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("field count"),
            "expected field count mismatch: {err}"
        );
    }

    #[test]
    fn validate_err_on_extra_rust_fields() {
        let err = two_float_layout()
            .validate(8, &[("a", 0, 4), ("b", 4, 4), ("c", 8, 4)])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("field count") || err.contains("missing"),
            "expected field count or missing field error: {err}"
        );
    }

    #[test]
    fn validate_reports_multiple_errors() {
        let mut layout = two_float_layout();
        layout.size = 16;
        let err = layout
            .validate(8, &[("x", 0, 4), ("b", 0, 4)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("size"), "expected size error: {err}");
        assert!(err.contains("`x`"), "expected name error: {err}");
        assert!(err.contains("offset"), "expected offset error: {err}");
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

#[cfg(test)]
mod push_constant_source_analysis_tests {
    use super::analyze_push_constant_categories_from_source;
    use crate::types::BindlessCategory;

    #[test]
    fn single_scattered_slot() {
        let src = "StorageBuffer<uint> b = goldy_dyn_scattered<uint>(0);";
        let cats = analyze_push_constant_categories_from_source(src);
        assert_eq!(cats, vec![Some(BindlessCategory::Scattered)]);
    }

    #[test]
    fn mixed_broadcast_and_scattered() {
        let src = r#"
            MyUniforms u = goldy_dyn_broadcast<MyUniforms>(0);
            StorageBuffer<int> b = goldy_dyn_scattered<int>(2);
        "#;
        let cats = analyze_push_constant_categories_from_source(src);
        assert_eq!(
            cats,
            vec![
                Some(BindlessCategory::Broadcast),
                None,
                Some(BindlessCategory::Scattered),
            ]
        );
    }

    #[test]
    fn buf_ro_maps_to_scattered() {
        let src = "ReadOnlyBuffer<Particle> p = goldy_dyn_buf_ro<Particle>(1);";
        let cats = analyze_push_constant_categories_from_source(src);
        assert_eq!(cats, vec![None, Some(BindlessCategory::Scattered)]);
    }

    #[test]
    fn storage_image_and_texture_and_sampler() {
        let src = r#"
            RWTexture2D<float4> img = goldy_dyn_direct_spatial<float4>(0);
            Texture2D<float4> tex = goldy_dyn_interpolated<float4>(1);
            SamplerState s = goldy_dyn_filter(2);
        "#;
        let cats = analyze_push_constant_categories_from_source(src);
        assert_eq!(
            cats,
            vec![
                Some(BindlessCategory::StorageImage),
                Some(BindlessCategory::Texture),
                Some(BindlessCategory::Sampler),
            ]
        );
    }

    #[test]
    fn dynamic_index_leaves_slot_unknown() {
        // `i` is not a literal — we must not report a category.
        let src = "StorageBuffer<uint> b = goldy_dyn_scattered<uint>(i);";
        let cats = analyze_push_constant_categories_from_source(src);
        assert!(cats.is_empty());
    }

    #[test]
    fn conflicting_uses_collapse_to_none() {
        let src = r#"
            MyUniforms u = goldy_dyn_broadcast<MyUniforms>(0);
            StorageBuffer<int> b = goldy_dyn_scattered<int>(0);
        "#;
        let cats = analyze_push_constant_categories_from_source(src);
        // Slot 0 saw conflicting categories; forced to None so the dispatch-time
        // validator reports a clean error instead of a stale expectation.
        assert_eq!(cats, vec![None]);
    }

    #[test]
    fn slot_suffixed_with_u_is_accepted() {
        let src = "let x = goldy_dyn_broadcast<MyUniforms>(3u).time;";
        let cats = analyze_push_constant_categories_from_source(src);
        assert_eq!(
            cats,
            vec![None, None, None, Some(BindlessCategory::Broadcast)]
        );
    }

    #[test]
    fn word_boundary_prevents_prefix_match() {
        // A hypothetical wrapper shouldn't be counted as `goldy_dyn_scattered`.
        let src = "StorageBuffer<uint> b = __my_goldy_dyn_scattered<uint>(0);";
        let cats = analyze_push_constant_categories_from_source(src);
        assert!(cats.is_empty());
    }

    #[test]
    fn slot_above_16_is_ignored() {
        let src = "let b = goldy_dyn_scattered<uint>(99);";
        let cats = analyze_push_constant_categories_from_source(src);
        assert!(cats.is_empty());
    }

    #[test]
    fn trims_trailing_unknown_slots() {
        let src = "let u = goldy_dyn_broadcast<MyUniforms>(2);";
        let cats = analyze_push_constant_categories_from_source(src);
        // 3 slots total: None, None, Some(Broadcast) — nothing after 2.
        assert_eq!(cats.len(), 3);
        assert_eq!(cats[2], Some(BindlessCategory::Broadcast));
    }

    #[test]
    fn stride_hints_buf_ro() {
        use super::{analyze_push_constant_stride_hints_from_source, PushConstantStrideHint};
        let src = "ReadOnlyBuffer<Particle> p = goldy_dyn_buf_ro<Particle>(1);";
        let h = analyze_push_constant_stride_hints_from_source(src);
        assert_eq!(h.len(), 2);
        assert_eq!(
            h[1],
            Some(PushConstantStrideHint::Structured {
                type_name: "Particle".into()
            })
        );
    }

    #[test]
    fn stride_hints_byte_address() {
        use super::{analyze_push_constant_stride_hints_from_source, PushConstantStrideHint};
        let src = "ByteAddressView v = goldy_dyn_byte_address(0);";
        let h = analyze_push_constant_stride_hints_from_source(src);
        assert_eq!(h, vec![Some(PushConstantStrideHint::ByteAddress)]);
    }

    #[test]
    fn stride_hints_conflict_on_same_slot() {
        use super::analyze_push_constant_stride_hints_from_source;
        let src = r#"
            ReadOnlyBuffer<A> a = goldy_dyn_buf_ro<A>(0);
            ReadOnlyBuffer<B> b = goldy_dyn_buf_ro<B>(0);
        "#;
        let h = analyze_push_constant_stride_hints_from_source(src);
        assert_eq!(h, vec![None]);
    }
}
