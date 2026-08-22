//! Dynamic library loading for Slang compiler.
//!
//! Handles finding and loading the Slang shared library across platforms.
//! Slang binaries are embedded at compile time and extracted to a cache directory.

use anyhow::{Context, Result};
use libloading::Library;
use std::fs;
use std::path::PathBuf;
use std::sync::Once;

use super::ffi::*;
use crate::{goldy_event, goldy_span};

// Include the generated embedded module from build.rs
include!(concat!(env!("OUT_DIR"), "/slang_embedded.rs"));

#[cfg(windows)]
#[link(name = "kernel32", kind = "dylib")]
extern "system" {
    fn SetDllDirectoryW(lp_path_name: *const u16) -> i32;
}

/// On Windows, dependent DLLs (`slang-glslang.dll`, `spirv-opt`, etc.) live next to the primary
/// Slang DLL. `LoadLibraryW` does not search that directory for dependencies unless we add it via
/// [`SetDllDirectoryW`] (see DLL search order).
#[cfg(windows)]
fn set_dll_directory_for_slang_dependencies(lib_path: &std::path::Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    let Some(dir) = lib_path.parent() else {
        return Ok(());
    };
    let mut wide: Vec<u16> = dir.as_os_str().encode_wide().collect();
    wide.push(0);
    let ok = unsafe { SetDllDirectoryW(wide.as_ptr()) };
    if ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Loaded Slang library with function pointers.
pub struct SlangLibrary {
    _library: Library,
    // Core session management (deprecated C API)
    pub create_session: FnSpCreateSession,
    pub destroy_session: FnSpDestroySession,
    // New COM-style API for session-level preprocessor macros
    pub create_global_session: FnSlangCreateGlobalSession2,
    // Compile request management
    pub create_compile_request: FnSpCreateCompileRequest,
    pub destroy_compile_request: FnSpDestroyCompileRequest,
    // Target configuration
    pub add_code_gen_target: FnSpAddCodeGenTarget,
    pub set_target_profile: FnSpSetTargetProfile,
    pub set_target_floating_point_mode: FnSpSetTargetFloatingPointMode,
    pub find_profile: FnSpFindProfile,
    // Source input
    pub add_translation_unit: FnSpAddTranslationUnit,
    pub add_translation_unit_source_string: FnSpAddTranslationUnitSourceString,
    pub add_entry_point: FnSpAddEntryPoint,
    pub add_search_path: FnSpAddSearchPath,
    pub add_preprocessor_define: FnSpAddPreprocessorDefine,
    // Compilation
    pub set_optimization_level: FnSpSetOptimizationLevel,
    pub compile: FnSpCompile,
    pub get_diagnostic_output: FnSpGetDiagnosticOutput,
    // Output
    pub get_entry_point_code_blob: FnSpGetEntryPointCodeBlob,
    pub get_target_code_blob: FnSpGetTargetCodeBlob,
    // Reflection API
    pub get_reflection: FnSpGetReflection,
    pub reflection_find_type_by_name: FnSpReflectionFindTypeByName,
    pub reflection_get_type_layout: FnSpReflectionGetTypeLayout,
    pub reflection_get_parameter_count: FnSpReflectionGetParameterCount,
    pub reflection_get_parameter_by_index: FnSpReflectionGetParameterByIndex,
    pub reflection_parameter_get_type_layout: FnSpReflectionParameterGetTypeLayout,
    pub reflection_variable_layout_get_variable: FnSpReflectionVariableLayoutGetVariable,
    pub reflection_variable_get_name: FnSpReflectionVariableGetName,
    pub reflection_parameter_get_binding_index: FnSpReflectionParameterGetBindingIndex,
    pub reflection_parameter_get_binding_space: FnSpReflectionParameterGetBindingSpace,
    pub reflection_type_layout_get_size: FnSpReflectionTypeLayoutGetSize,
    pub reflection_type_layout_get_stride: FnSpReflectionTypeLayoutGetStride,
    pub reflection_type_layout_get_alignment: FnSpReflectionTypeLayoutGetAlignment,
    pub reflection_type_layout_get_field_count: FnSpReflectionTypeLayoutGetFieldCount,
    pub reflection_type_layout_get_field_by_index: FnSpReflectionTypeLayoutGetFieldByIndex,
    pub reflection_type_layout_get_type: FnSpReflectionTypeLayoutGetType,
    pub reflection_type_get_kind: FnSpReflectionTypeGetKind,
    pub reflection_type_get_name: FnSpReflectionTypeGetName,
    pub reflection_type_layout_get_element_type_layout: FnSpReflectionTypeLayoutGetElementTypeLayout,
    pub reflection_variable_layout_get_type_layout: FnSpReflectionVariableLayoutGetTypeLayout,
    pub reflection_variable_layout_get_offset: FnSpReflectionVariableLayoutGetOffset,
    pub reflection_type_layout_get_binding_type: FnSpReflectionTypeLayoutGetBindingType,
    pub reflection_type_layout_get_category: FnSpReflectionTypeLayoutGetCategory,
}

impl SlangLibrary {
    /// Load the Slang library.
    ///
    /// Search order:
    /// 1. GOLDY_SLANG_PATH environment variable (user override)
    /// 2. Next to executable (bundled distribution)
    /// 3. Cache directory (extracted from embedded)
    pub fn load() -> Result<Self> {
        let _span = goldy_span!("slang.library.load").entered();

        let lib_path = Self::find_library()?;
        tracing::info!("Loading Slang library from: {}", lib_path.display());

        #[cfg(windows)]
        set_dll_directory_for_slang_dependencies(&lib_path).with_context(|| {
            format!(
                "SetDllDirectoryW failed for Slang dependency search ({})",
                lib_path.display()
            )
        })?;

        // Safety: We're loading a known library with a stable C ABI
        let library = unsafe { Library::new(&lib_path) }
            .with_context(|| format!("Failed to load Slang library from {}", lib_path.display()))?;

        goldy_event!(
            "slang.library.load",
            path = %lib_path.display(),
            success = true
        );

        // Load function pointers
        // Safety: These are all C functions with stable ABI from the Slang library
        unsafe {
            let create_session: FnSpCreateSession = *library
                .get(b"spCreateSession\0")
                .context("Failed to load spCreateSession")?;
            let destroy_session: FnSpDestroySession = *library
                .get(b"spDestroySession\0")
                .context("Failed to load spDestroySession")?;
            let create_global_session: FnSlangCreateGlobalSession2 = *library
                .get(b"slang_createGlobalSession2\0")
                .context("Failed to load slang_createGlobalSession2")?;
            let create_compile_request: FnSpCreateCompileRequest = *library
                .get(b"spCreateCompileRequest\0")
                .context("Failed to load spCreateCompileRequest")?;
            let destroy_compile_request: FnSpDestroyCompileRequest = *library
                .get(b"spDestroyCompileRequest\0")
                .context("Failed to load spDestroyCompileRequest")?;
            let add_code_gen_target: FnSpAddCodeGenTarget = *library
                .get(b"spAddCodeGenTarget\0")
                .context("Failed to load spAddCodeGenTarget")?;
            let set_target_profile: FnSpSetTargetProfile = *library
                .get(b"spSetTargetProfile\0")
                .context("Failed to load spSetTargetProfile")?;
            let set_target_floating_point_mode: FnSpSetTargetFloatingPointMode = *library
                .get(b"spSetTargetFloatingPointMode\0")
                .context("Failed to load spSetTargetFloatingPointMode")?;
            let find_profile: FnSpFindProfile = *library
                .get(b"spFindProfile\0")
                .context("Failed to load spFindProfile")?;
            let add_translation_unit: FnSpAddTranslationUnit = *library
                .get(b"spAddTranslationUnit\0")
                .context("Failed to load spAddTranslationUnit")?;
            let add_translation_unit_source_string: FnSpAddTranslationUnitSourceString = *library
                .get(b"spAddTranslationUnitSourceString\0")
                .context("Failed to load spAddTranslationUnitSourceString")?;
            let add_entry_point: FnSpAddEntryPoint = *library
                .get(b"spAddEntryPoint\0")
                .context("Failed to load spAddEntryPoint")?;
            let add_search_path: FnSpAddSearchPath = *library
                .get(b"spAddSearchPath\0")
                .context("Failed to load spAddSearchPath")?;
            let add_preprocessor_define: FnSpAddPreprocessorDefine = *library
                .get(b"spAddPreprocessorDefine\0")
                .context("Failed to load spAddPreprocessorDefine")?;
            let set_optimization_level: FnSpSetOptimizationLevel = *library
                .get(b"spSetOptimizationLevel\0")
                .context("Failed to load spSetOptimizationLevel")?;
            let compile: FnSpCompile = *library.get(b"spCompile\0").context("Failed to load spCompile")?;
            let get_diagnostic_output: FnSpGetDiagnosticOutput = *library
                .get(b"spGetDiagnosticOutput\0")
                .context("Failed to load spGetDiagnosticOutput")?;
            let get_entry_point_code_blob: FnSpGetEntryPointCodeBlob = *library
                .get(b"spGetEntryPointCodeBlob\0")
                .context("Failed to load spGetEntryPointCodeBlob")?;
            let get_target_code_blob: FnSpGetTargetCodeBlob = *library
                .get(b"spGetTargetCodeBlob\0")
                .context("Failed to load spGetTargetCodeBlob")?;

            goldy_event!("slang.ffi.core_symbols", loaded = true);

            // Reflection API
            let get_reflection: FnSpGetReflection = *library
                .get(b"spGetReflection\0")
                .context("Failed to load spGetReflection")?;
            let reflection_find_type_by_name: FnSpReflectionFindTypeByName = *library
                .get(b"spReflection_FindTypeByName\0")
                .context("Failed to load spReflection_FindTypeByName")?;
            let reflection_get_type_layout: FnSpReflectionGetTypeLayout = *library
                .get(b"spReflection_GetTypeLayout\0")
                .context("Failed to load spReflection_GetTypeLayout")?;
            let reflection_get_parameter_count: FnSpReflectionGetParameterCount = *library
                .get(b"spReflection_GetParameterCount\0")
                .context("Failed to load spReflection_GetParameterCount")?;
            let reflection_get_parameter_by_index: FnSpReflectionGetParameterByIndex = *library
                .get(b"spReflection_GetParameterByIndex\0")
                .context("Failed to load spReflection_GetParameterByIndex")?;
            let reflection_parameter_get_type_layout: FnSpReflectionParameterGetTypeLayout = *library
                .get(b"spReflectionVariableLayout_GetTypeLayout\0")
                .context("Failed to load spReflectionVariableLayout_GetTypeLayout")?;
            let reflection_variable_layout_get_variable: FnSpReflectionVariableLayoutGetVariable = *library
                .get(b"spReflectionVariableLayout_GetVariable\0")
                .context("Failed to load spReflectionVariableLayout_GetVariable")?;
            let reflection_variable_get_name: FnSpReflectionVariableGetName = *library
                .get(b"spReflectionVariable_GetName\0")
                .context("Failed to load spReflectionVariable_GetName")?;

            let reflection_parameter_get_binding_index: FnSpReflectionParameterGetBindingIndex = *library
                .get(b"spReflectionParameter_GetBindingIndex\0")
                .context("Failed to load spReflectionParameter_GetBindingIndex")?;
            let reflection_parameter_get_binding_space: FnSpReflectionParameterGetBindingSpace = *library
                .get(b"spReflectionParameter_GetBindingSpace\0")
                .context("Failed to load spReflectionParameter_GetBindingSpace")?;
            let reflection_type_layout_get_size: FnSpReflectionTypeLayoutGetSize = *library
                .get(b"spReflectionTypeLayout_GetSize\0")
                .context("Failed to load spReflectionTypeLayout_GetSize")?;
            let reflection_type_layout_get_stride: FnSpReflectionTypeLayoutGetStride = *library
                .get(b"spReflectionTypeLayout_GetStride\0")
                .context("Failed to load spReflectionTypeLayout_GetStride")?;
            let reflection_type_layout_get_alignment: FnSpReflectionTypeLayoutGetAlignment = *library
                .get(b"spReflectionTypeLayout_getAlignment\0")
                .context("Failed to load spReflectionTypeLayout_getAlignment")?;

            let reflection_type_layout_get_field_count: FnSpReflectionTypeLayoutGetFieldCount = *library
                .get(b"spReflectionTypeLayout_GetFieldCount\0")
                .context("Failed to load spReflectionTypeLayout_GetFieldCount")?;
            let reflection_type_layout_get_field_by_index: FnSpReflectionTypeLayoutGetFieldByIndex = *library
                .get(b"spReflectionTypeLayout_GetFieldByIndex\0")
                .context("Failed to load spReflectionTypeLayout_GetFieldByIndex")?;
            let reflection_type_layout_get_type: FnSpReflectionTypeLayoutGetType = *library
                .get(b"spReflectionTypeLayout_GetType\0")
                .context("Failed to load spReflectionTypeLayout_GetType")?;
            let reflection_type_get_kind: FnSpReflectionTypeGetKind = *library
                .get(b"spReflectionType_GetKind\0")
                .context("Failed to load spReflectionType_GetKind")?;
            let reflection_type_get_name: FnSpReflectionTypeGetName = *library
                .get(b"spReflectionType_GetName\0")
                .context("Failed to load spReflectionType_GetName")?;
            let reflection_type_layout_get_element_type_layout: FnSpReflectionTypeLayoutGetElementTypeLayout = *library
                .get(b"spReflectionTypeLayout_GetElementTypeLayout\0")
                .context("Failed to load spReflectionTypeLayout_GetElementTypeLayout")?;
            let reflection_variable_layout_get_type_layout: FnSpReflectionVariableLayoutGetTypeLayout = *library
                .get(b"spReflectionVariableLayout_GetTypeLayout\0")
                .context("Failed to load spReflectionVariableLayout_GetTypeLayout")?;
            let reflection_variable_layout_get_offset: FnSpReflectionVariableLayoutGetOffset = *library
                .get(b"spReflectionVariableLayout_GetOffset\0")
                .context("Failed to load spReflectionVariableLayout_GetOffset")?;

            let reflection_type_layout_get_binding_type: FnSpReflectionTypeLayoutGetBindingType = *library
                .get(b"spReflectionTypeLayout_getDescriptorSetDescriptorRangeType\0")
                .context("Failed to load spReflectionTypeLayout_getDescriptorSetDescriptorRangeType")?;

            let reflection_type_layout_get_category: FnSpReflectionTypeLayoutGetCategory = *library
                .get(b"spReflectionTypeLayout_GetParameterCategory\0")
                .context("Failed to load spReflectionTypeLayout_GetParameterCategory")?;

            goldy_event!("slang.ffi.reflection_symbols", loaded = true);

            Ok(Self {
                _library: library,
                create_session,
                destroy_session,
                create_global_session,
                create_compile_request,
                destroy_compile_request,
                add_code_gen_target,
                set_target_profile,
                set_target_floating_point_mode,
                find_profile,
                add_translation_unit,
                add_translation_unit_source_string,
                add_entry_point,
                add_search_path,
                add_preprocessor_define,
                set_optimization_level,
                compile,
                get_diagnostic_output,
                get_entry_point_code_blob,
                get_target_code_blob,
                // Reflection API
                get_reflection,
                reflection_find_type_by_name,
                reflection_get_type_layout,
                reflection_get_parameter_count,
                reflection_get_parameter_by_index,
                reflection_parameter_get_type_layout,
                reflection_variable_layout_get_variable,
                reflection_variable_get_name,
                reflection_parameter_get_binding_index,
                reflection_parameter_get_binding_space,
                reflection_type_layout_get_size,
                reflection_type_layout_get_stride,
                reflection_type_layout_get_alignment,
                reflection_type_layout_get_field_count,
                reflection_type_layout_get_field_by_index,
                reflection_type_layout_get_type,
                reflection_type_get_kind,
                reflection_type_get_name,
                reflection_type_layout_get_element_type_layout,
                reflection_variable_layout_get_type_layout,
                reflection_variable_layout_get_offset,
                reflection_type_layout_get_binding_type,
                reflection_type_layout_get_category,
            })
        }
    }

    /// Find the Slang library path.
    ///
    /// Search order:
    /// 1. GOLDY_SLANG_PATH environment variable (user override)
    /// 2. Next to executable (bundled distribution)
    /// 3. Cache directory (extracted from embedded binaries)
    fn find_library() -> Result<PathBuf> {
        // 1. Check GOLDY_SLANG_PATH environment variable (user override)
        if let Ok(path) = std::env::var("GOLDY_SLANG_PATH") {
            let path = PathBuf::from(path);
            if path.exists() {
                tracing::debug!("Using Slang from GOLDY_SLANG_PATH: {}", path.display());
                return Ok(path);
            }
            tracing::warn!("GOLDY_SLANG_PATH set but file not found: {}", path.display());
        }

        // 2. Check vendored binaries next to executable
        if let Some(path) = Self::find_vendored_library() {
            tracing::debug!("Using Slang from executable directory: {}", path.display());
            return Ok(path);
        }

        // 3. Check/extract from cache
        if let Some(path) = Self::ensure_cached_library() {
            tracing::debug!("Using Slang from cache: {}", path.display());
            return Ok(path);
        }

        anyhow::bail!(
            "Could not find or extract Slang library. Options:\n\
             1. Ensure slang is bundled next to the executable\n\
             2. Set GOLDY_SLANG_PATH environment variable\n\
             3. Slang version {} may not be embedded",
            SLANG_VERSION
        )
    }

    /// Find vendored library next to the executable.
    ///
    /// This is the happy path for distribution - Slang libraries should be
    /// copied alongside the executable.
    fn find_vendored_library() -> Option<PathBuf> {
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let path = exe_dir.join(SLANG_PRIMARY);
                if path.exists() {
                    return Some(path);
                }
            }
        }
        None
    }

    /// Get the cache directory for Slang libraries.
    ///
    /// Returns platform-idiomatic cache directory:
    /// - Windows: %LOCALAPPDATA%\goldy\slang\{version}\
    /// - Linux: ~/.cache/goldy/slang/{version}/
    /// - macOS: ~/Library/Caches/goldy/slang/{version}/
    fn cache_dir() -> Option<PathBuf> {
        if SLANG_VERSION.is_empty() {
            return None;
        }
        let base = std::env::var_os("GOLDY_CACHE_DIR")
            .map(PathBuf::from)
            .or_else(dirs::cache_dir)
            .or_else(|| Some(std::env::temp_dir().join("goldy-cache")));
        base.map(|d| {
            let mut p = d.join("goldy").join("slang").join(SLANG_VERSION);
            if !SLANG_EMBED_PLATFORM.is_empty() {
                p.push(SLANG_EMBED_PLATFORM);
            }
            p
        })
    }

    /// Ensure Slang libraries are extracted to the cache directory.
    ///
    /// Returns the path to the primary library if successful.
    fn ensure_cached_library() -> Option<PathBuf> {
        if SLANG_FILES.is_empty() {
            tracing::debug!("No embedded Slang binaries available");
            return None;
        }

        let cache_dir = Self::cache_dir()?;
        let primary_path = cache_dir.join(SLANG_PRIMARY);
        let sentinel_path = cache_dir.join("version.txt");

        // Cache is valid only if the sentinel matches the embedded version.
        let cache_valid = fs::read_to_string(&sentinel_path)
            .map(|v| v.trim() == SLANG_VERSION)
            .unwrap_or(false);

        if cache_valid && primary_path.exists() {
            return Some(primary_path);
        }

        // Cache is missing or stale — extract. Multiple processes (e.g. nextest
        // workers) may race here; extract_to_cache uses PID-unique temp files so
        // concurrent extractions don't interfere with each other.
        static EXTRACT_ONCE: Once = Once::new();
        let mut extract_result = Ok(());

        EXTRACT_ONCE.call_once(|| {
            extract_result = Self::extract_to_cache(&cache_dir);
        });

        if let Err(ref e) = extract_result {
            tracing::warn!("Slang extraction failed: {e:#}");
        }

        // Even if *our* extraction failed, another process may have succeeded.
        if primary_path.exists() {
            Some(primary_path)
        } else {
            None
        }
    }

    /// Extract all embedded Slang files to the cache directory.
    fn extract_to_cache(cache_dir: &PathBuf) -> Result<()> {
        tracing::info!("Extracting Slang {} to cache: {}", SLANG_VERSION, cache_dir.display());

        // Create cache directory
        fs::create_dir_all(cache_dir)
            .with_context(|| format!("Failed to create cache directory: {}", cache_dir.display()))?;

        // Extract each file. PID-scoped temp names avoid conflicts when multiple
        // processes extract concurrently (e.g. cargo-nextest parallel workers).
        let pid = std::process::id();
        for (filename, bytes) in SLANG_FILES {
            let dest_path = cache_dir.join(filename);
            let temp_path = cache_dir.join(format!("{}.{}.tmp", filename, pid));

            fs::write(&temp_path, bytes)
                .with_context(|| format!("Failed to write Slang library: {}", temp_path.display()))?;

            // rename is atomic on POSIX; on Windows it replaces the target.
            // If another process already placed the file, that's fine.
            if let Err(e) = fs::rename(&temp_path, &dest_path) {
                if dest_path.exists() {
                    let _ = fs::remove_file(&temp_path);
                    tracing::debug!("Extracted by another process: {filename}");
                } else {
                    return Err(e).with_context(|| {
                        format!("Failed to rename {} to {}", temp_path.display(), dest_path.display())
                    });
                }
            } else {
                tracing::debug!("Extracted: {filename} ({} bytes)", bytes.len());
            }
        }

        // Write the sentinel last so a partial extraction never leaves a valid marker.
        let sentinel_tmp = cache_dir.join(format!("version.{}.txt", pid));
        fs::write(&sentinel_tmp, SLANG_VERSION).context("Failed to write Slang version sentinel")?;
        let _ = fs::rename(&sentinel_tmp, cache_dir.join("version.txt"));

        goldy_event!(
            "slang.cache.extracted",
            version = SLANG_VERSION,
            path = %cache_dir.display(),
            file_count = SLANG_FILES.len()
        );

        Ok(())
    }
}
