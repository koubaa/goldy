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
    // Compilation
    pub compile: FnSpCompile,
    pub get_diagnostic_output: FnSpGetDiagnosticOutput,
    // Output
    pub get_entry_point_code_blob: FnSpGetEntryPointCodeBlob,
    pub get_target_code_blob: FnSpGetTargetCodeBlob,
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
            let add_code_gen_target: FnSpAddCodeGenTarget = *library
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
                compile,
                get_diagnostic_output,
                get_entry_point_code_blob,
                get_target_code_blob,
            })
        }
    }
    
    /// Find the Slang library path.
    fn find_library() -> Result<PathBuf> {
        // 1. Check RAG_SLANG_PATH environment variable
        if let Ok(path) = std::env::var("RAG_SLANG_PATH") {
            let path = PathBuf::from(path);
            if path.exists() {
                return Ok(path);
            }
            tracing::warn!("RAG_SLANG_PATH set but file not found: {}", path.display());
        }
        
        // 2. Check vendored binaries
        if let Some(path) = Self::find_vendored_library() {
            return Ok(path);
        }
        
        // 3. Check Vulkan SDK (Windows development fallback)
        #[cfg(target_os = "windows")]
        if let Some(path) = Self::find_vulkan_sdk_library() {
            return Ok(path);
        }
        
        anyhow::bail!(
            "Could not find Slang library. Options:\n\
             1. Set RAG_SLANG_PATH environment variable\n\
             2. Run goldy/slang/download.sh to fetch vendored binaries\n\
             3. Install Vulkan SDK 1.3.296+ (Windows)"
        )
    }
    
    /// Find vendored library based on platform.
    fn find_vendored_library() -> Option<PathBuf> {
        let lib_name = Self::library_name();
        let platform_dir = Self::platform_dir();
        
        // Try relative to executable
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                // Check in slang/bin/{platform}/ relative to exe
                let path = exe_dir.join("slang").join("bin").join(&platform_dir).join(&lib_name);
                if path.exists() {
                    return Some(path);
                }
                
                // Check in ../slang/bin/{platform}/ (for running from target/debug)
                let path = exe_dir.join("..").join("..").join("slang").join("bin").join(&platform_dir).join(&lib_name);
                if path.exists() {
                    return Some(path);
                }
            }
        }
        
        // Try relative to current directory
        let path = PathBuf::from("slang").join("bin").join(&platform_dir).join(&lib_name);
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
            let path = PathBuf::from(&sdk_path).join("Bin").join("slang-compiler.dll");
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

