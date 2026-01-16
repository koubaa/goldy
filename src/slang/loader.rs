//! Dynamic library loading for Slang compiler.
//!
//! Handles finding and loading the Slang shared library across platforms.

use anyhow::{Context, Result};
use libloading::Library;
use std::path::PathBuf;

use super::ffi::*;

/// Loaded Slang library with function pointers.
pub struct SlangLibrary {
    _library: Library,
    // Core session management
    pub create_session: FnSpCreateSession,
    pub destroy_session: FnSpDestroySession,
    // Compile request management
    pub create_compile_request: FnSpCreateCompileRequest,
    pub destroy_compile_request: FnSpDestroyCompileRequest,
    // Target configuration
    pub add_code_gen_target: FnSpAddCodeGenTarget,
    // Source input
    pub add_translation_unit: FnSpAddTranslationUnit,
    pub add_translation_unit_source_string: FnSpAddTranslationUnitSourceString,
    pub add_entry_point: FnSpAddEntryPoint,
    pub add_search_path: FnSpAddSearchPath,
    pub add_preprocessor_define: FnSpAddPreprocessorDefine,
    // Compilation
    pub compile: FnSpCompile,
    pub get_diagnostic_output: FnSpGetDiagnosticOutput,
    // Output
    pub get_entry_point_code_blob: FnSpGetEntryPointCodeBlob,
    pub get_target_code_blob: FnSpGetTargetCodeBlob,
    // Reflection API
    pub get_reflection: FnSpGetReflection,
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
    pub reflection_type_layout_get_element_type_layout:
        FnSpReflectionTypeLayoutGetElementTypeLayout,
    pub reflection_variable_layout_get_type_layout: FnSpReflectionVariableLayoutGetTypeLayout,
    pub reflection_variable_layout_get_offset: FnSpReflectionVariableLayoutGetOffset,
    pub reflection_type_layout_get_binding_type: FnSpReflectionTypeLayoutGetBindingType,
    pub reflection_type_layout_get_category: FnSpReflectionTypeLayoutGetCategory,
}

impl SlangLibrary {
    /// Load the Slang library.
    ///
    /// Search order:
    /// 1. RAG_SLANG_PATH environment variable
    /// 2. Vendored binaries in goldy/slang/bin/{platform}/
    /// 3. Vulkan SDK (Windows only, for development)
    pub fn load() -> Result<Self> {
        let lib_path = Self::find_library()?;
        tracing::info!("Loading Slang library from: {}", lib_path.display());

        // Safety: We're loading a known library with a stable C ABI
        let library = unsafe { Library::new(&lib_path) }
            .with_context(|| format!("Failed to load Slang library from {}", lib_path.display()))?;

        // Load function pointers
        // Safety: These are all C functions with stable ABI from the Slang library
        unsafe {
            let create_session: FnSpCreateSession = *library
                .get(b"spCreateSession\0")
                .context("Failed to load spCreateSession")?;
            let destroy_session: FnSpDestroySession = *library
                .get(b"spDestroySession\0")
                .context("Failed to load spDestroySession")?;
            let create_compile_request: FnSpCreateCompileRequest = *library
                .get(b"spCreateCompileRequest\0")
                .context("Failed to load spCreateCompileRequest")?;
            let destroy_compile_request: FnSpDestroyCompileRequest = *library
                .get(b"spDestroyCompileRequest\0")
                .context("Failed to load spDestroyCompileRequest")?;
            let add_code_gen_target: FnSpAddCodeGenTarget =
                *library
                    .get(b"spAddCodeGenTarget\0")
                    .context("Failed to load spAddCodeGenTarget")?;
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
            let compile: FnSpCompile = *library
                .get(b"spCompile\0")
                .context("Failed to load spCompile")?;
            let get_diagnostic_output: FnSpGetDiagnosticOutput = *library
                .get(b"spGetDiagnosticOutput\0")
                .context("Failed to load spGetDiagnosticOutput")?;
            let get_entry_point_code_blob: FnSpGetEntryPointCodeBlob = *library
                .get(b"spGetEntryPointCodeBlob\0")
                .context("Failed to load spGetEntryPointCodeBlob")?;
            let get_target_code_blob: FnSpGetTargetCodeBlob = *library
                .get(b"spGetTargetCodeBlob\0")
                .context("Failed to load spGetTargetCodeBlob")?;

            // Reflection API
            let get_reflection: FnSpGetReflection = *library
                .get(b"spGetReflection\0")
                .context("Failed to load spGetReflection")?;
            let reflection_get_parameter_count: FnSpReflectionGetParameterCount = *library
                .get(b"spReflection_GetParameterCount\0")
                .context("Failed to load spReflection_GetParameterCount")?;
            let reflection_get_parameter_by_index: FnSpReflectionGetParameterByIndex = *library
                .get(b"spReflection_GetParameterByIndex\0")
                .context("Failed to load spReflection_GetParameterByIndex")?;
            let reflection_parameter_get_type_layout: FnSpReflectionParameterGetTypeLayout =
                *library
                    .get(b"spReflectionVariableLayout_GetTypeLayout\0")
                    .context("Failed to load spReflectionVariableLayout_GetTypeLayout")?;
            let reflection_variable_layout_get_variable: FnSpReflectionVariableLayoutGetVariable =
                *library
                    .get(b"spReflectionVariableLayout_GetVariable\0")
                    .context("Failed to load spReflectionVariableLayout_GetVariable")?;
            let reflection_variable_get_name: FnSpReflectionVariableGetName = *library
                .get(b"spReflectionVariable_GetName\0")
                .context("Failed to load spReflectionVariable_GetName")?;

            let reflection_parameter_get_binding_index: FnSpReflectionParameterGetBindingIndex =
                *library
                    .get(b"spReflectionParameter_GetBindingIndex\0")
                    .context("Failed to load spReflectionParameter_GetBindingIndex")?;
            let reflection_parameter_get_binding_space: FnSpReflectionParameterGetBindingSpace =
                *library
                    .get(b"spReflectionParameter_GetBindingSpace\0")
                    .context("Failed to load spReflectionParameter_GetBindingSpace")?;
            let reflection_type_layout_get_size: FnSpReflectionTypeLayoutGetSize = *library
                .get(b"spReflectionTypeLayout_GetSize\0")
                .context("Failed to load spReflectionTypeLayout_GetSize")?;
            let reflection_type_layout_get_stride: FnSpReflectionTypeLayoutGetStride = *library
                .get(b"spReflectionTypeLayout_GetStride\0")
                .context("Failed to load spReflectionTypeLayout_GetStride")?;
            let reflection_type_layout_get_alignment: FnSpReflectionTypeLayoutGetAlignment =
                *library
                    .get(b"spReflectionTypeLayout_getAlignment\0")
                    .context("Failed to load spReflectionTypeLayout_getAlignment")?;

            let reflection_type_layout_get_field_count: FnSpReflectionTypeLayoutGetFieldCount =
                *library
                    .get(b"spReflectionTypeLayout_GetFieldCount\0")
                    .context("Failed to load spReflectionTypeLayout_GetFieldCount")?;
            let reflection_type_layout_get_field_by_index: FnSpReflectionTypeLayoutGetFieldByIndex =
                *library
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
            let reflection_variable_layout_get_offset: FnSpReflectionVariableLayoutGetOffset =
                *library
                    .get(b"spReflectionVariableLayout_GetOffset\0")
                    .context("Failed to load spReflectionVariableLayout_GetOffset")?;

            let reflection_type_layout_get_binding_type: FnSpReflectionTypeLayoutGetBindingType =
                *library
                    .get(b"spReflectionTypeLayout_getDescriptorSetDescriptorRangeType\0")
                    .context(
                        "Failed to load spReflectionTypeLayout_getDescriptorSetDescriptorRangeType",
                    )?;

            let reflection_type_layout_get_category: FnSpReflectionTypeLayoutGetCategory = *library
                .get(b"spReflectionTypeLayout_GetParameterCategory\0")
                .context("Failed to load spReflectionTypeLayout_GetParameterCategory")?;

            Ok(Self {
                _library: library,
                create_session,
                destroy_session,
                create_compile_request,
                destroy_compile_request,
                add_code_gen_target,
                add_translation_unit,
                add_translation_unit_source_string,
                add_entry_point,
                add_search_path,
                add_preprocessor_define,
                compile,
                get_diagnostic_output,
                get_entry_point_code_blob,
                get_target_code_blob,
                // Reflection API
                get_reflection,
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
    fn find_library() -> Result<PathBuf> {
        // 1. Check GOLDY_SLANG_PATH or RAG_SLANG_PATH environment variable
        for env_var in ["GOLDY_SLANG_PATH", "RAG_SLANG_PATH"] {
            if let Ok(path) = std::env::var(env_var) {
                let path = PathBuf::from(path);
                if path.exists() {
                    return Ok(path);
                }
                tracing::warn!("{} set but file not found: {}", env_var, path.display());
            }
        }

        // 2. Check build.rs downloaded binaries (via GOLDY_SLANG_DIR compile-time env)
        if let Some(path) = Self::find_build_script_library() {
            return Ok(path);
        }

        // 3. Check vendored binaries (for development)
        if let Some(path) = Self::find_vendored_library() {
            return Ok(path);
        }

        // 4. Check Vulkan SDK (Windows development fallback)
        #[cfg(target_os = "windows")]
        if let Some(path) = Self::find_vulkan_sdk_library() {
            return Ok(path);
        }

        anyhow::bail!(
            "Could not find Slang library. Options:\n\
             1. Set GOLDY_SLANG_PATH environment variable\n\
             2. Install Vulkan SDK 1.3.296+ (Windows)\n\
             3. For development: run slang/download.sh"
        )
    }

    /// Find library downloaded by build.rs.
    fn find_build_script_library() -> Option<PathBuf> {
        // GOLDY_SLANG_DIR is set at compile time by build.rs
        let slang_dir = option_env!("GOLDY_SLANG_DIR")?;
        let lib_name = Self::library_name();
        let path = PathBuf::from(slang_dir).join(lib_name);
        if path.exists() {
            return Some(path);
        }
        None
    }

    /// Find vendored library based on platform.
    fn find_vendored_library() -> Option<PathBuf> {
        let lib_name = Self::library_name();
        let platform_dir = Self::platform_dir();

        // Try relative to executable
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                // Check in slang/bin/{platform}/ relative to exe
                let path = exe_dir
                    .join("slang")
                    .join("bin")
                    .join(platform_dir)
                    .join(lib_name);
                if path.exists() {
                    return Some(path);
                }

                // Check in ../slang/bin/{platform}/ (for running from target/debug)
                let path = exe_dir
                    .join("..")
                    .join("..")
                    .join("slang")
                    .join("bin")
                    .join(platform_dir)
                    .join(lib_name);
                if path.exists() {
                    return Some(path);
                }
            }
        }

        // Try relative to current directory
        let path = PathBuf::from("slang")
            .join("bin")
            .join(platform_dir)
            .join(lib_name);
        if path.exists() {
            return Some(path);
        }

        None
    }

    /// Find Slang in Vulkan SDK (Windows only).
    #[cfg(target_os = "windows")]
    fn find_vulkan_sdk_library() -> Option<PathBuf> {
        // Check VULKAN_SDK environment variable
        if let Ok(sdk_path) = std::env::var("VULKAN_SDK") {
            // The older slang.dll is in Bin/, newer is slang-compiler.dll
            let path = PathBuf::from(&sdk_path).join("Bin").join("slang.dll");
            if path.exists() {
                return Some(path);
            }
            let path = PathBuf::from(&sdk_path)
                .join("Bin")
                .join("slang-compiler.dll");
            if path.exists() {
                return Some(path);
            }
        }

        // Try common Vulkan SDK locations
        for version in ["1.3.296.0", "1.3.290.0", "1.3.283.0"] {
            let path = PathBuf::from(format!("C:\\VulkanSDK\\{}\\Bin\\slang.dll", version));
            if path.exists() {
                return Some(path);
            }
        }

        None
    }

    /// Get the library filename for the current platform.
    fn library_name() -> &'static str {
        #[cfg(target_os = "windows")]
        {
            "slang-compiler.dll"
        }
        #[cfg(target_os = "linux")]
        {
            "libslang-compiler.so"
        }
        #[cfg(target_os = "macos")]
        {
            "libslang-compiler.dylib"
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        {
            compile_error!("Unsupported platform for Slang library")
        }
    }

    /// Get the platform directory name.
    fn platform_dir() -> &'static str {
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        {
            "windows-x86_64"
        }
        #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
        {
            "windows-aarch64"
        }
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            "linux-x86_64"
        }
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            "linux-aarch64"
        }
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        {
            "macos-x86_64"
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            "macos-aarch64"
        }
        #[cfg(not(any(
            all(target_os = "windows", target_arch = "x86_64"),
            all(target_os = "windows", target_arch = "aarch64"),
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "aarch64"),
            all(target_os = "macos", target_arch = "x86_64"),
            all(target_os = "macos", target_arch = "aarch64"),
        )))]
        {
            compile_error!("Unsupported platform/architecture combination")
        }
    }
}
