//! High-level Slang compiler API.
//!
//! Provides a safe, ergonomic interface for compiling Slang shaders.

use anyhow::{Context, Result};
use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

use super::ffi::*;
use super::loader::SlangLibrary;

/// Shader compilation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderTarget {
    /// SPIR-V bytecode for Vulkan
    Spirv,
    /// WGSL source for WebGPU
    Wgsl,
    /// HLSL source for DirectX
    Hlsl,
    /// Metal Shading Language
    Metal,
    /// GLSL source
    Glsl,
}

impl ShaderTarget {
    fn to_slang_target(self) -> SlangCompileTarget {
        match self {
            ShaderTarget::Spirv => SlangCompileTarget::Spirv,
            ShaderTarget::Wgsl => SlangCompileTarget::Wgsl,
            ShaderTarget::Hlsl => SlangCompileTarget::Hlsl,
            ShaderTarget::Metal => SlangCompileTarget::Metal,
            ShaderTarget::Glsl => SlangCompileTarget::Glsl,
        }
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
    /// Get the data as a string (for text-based targets like WGSL, HLSL, GLSL).
    pub fn as_str(&self) -> Option<&str> {
        match self.target {
            ShaderTarget::Wgsl | ShaderTarget::Hlsl | ShaderTarget::Metal | ShaderTarget::Glsl => {
                std::str::from_utf8(&self.data).ok()
            }
            ShaderTarget::Spirv => None,
        }
    }
    
    /// Get the data as SPIR-V words (for Vulkan).
    pub fn as_spirv(&self) -> Option<&[u32]> {
        if self.target == ShaderTarget::Spirv && self.data.len() % 4 == 0 {
            Some(bytemuck::cast_slice(&self.data))
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
    session: *mut SlangSession,
}

// SlangCompiler is Send + Sync because each compilation creates its own request
unsafe impl Send for SlangCompiler {}
unsafe impl Sync for SlangCompiler {}

impl SlangCompiler {
    /// Create a new Slang compiler instance.
    pub fn new() -> Result<Self> {
        let library = Arc::new(SlangLibrary::load()?);
        
        // Create global session
        let session = unsafe { (library.create_session)(ptr::null()) };
        if session.is_null() {
            anyhow::bail!("Failed to create Slang session");
        }
        
        tracing::info!("Slang compiler initialized");
        
        Ok(Self { library, session })
    }
    
    /// Compile Slang source code to the specified target.
    ///
    /// Compiles a single entry point. For shaders with both vertex and fragment,
    /// call this twice with different entry point names.
    pub fn compile(&self, source: &str, target: ShaderTarget) -> Result<CompiledShader> {
        // When no entry point is specified, compile all detected entry points
        self.compile_entry_point(source, target, None)
    }
    
    /// Compile a specific entry point to the specified target.
    pub fn compile_entry_point(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_point: Option<(&str, SlangStage)>,
    ) -> Result<CompiledShader> {
        let entry_points: Vec<(&str, SlangStage)> = match entry_point {
            Some(ep) => vec![ep],
            None => vec![],
        };
        self.compile_with_entry_points(source, target, &entry_points)
    }
    
    /// Compile with explicit entry points.
    ///
    /// If `entry_points` is empty, entry points are detected from shader attributes.
    pub fn compile_with_entry_points(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
    ) -> Result<CompiledShader> {
        // Create compile request
        let request = unsafe { (self.library.create_compile_request)(self.session) };
        if request.is_null() {
            anyhow::bail!("Failed to create Slang compile request");
        }
        
        // Ensure cleanup on all paths
        let _guard = scopeguard::guard(request, |req| {
            unsafe { (self.library.destroy_compile_request)(req) };
        });
        
        // Add target
        let target_index = unsafe {
            (self.library.add_code_gen_target)(request, target.to_slang_target() as i32)
        };
        if target_index < 0 {
            anyhow::bail!("Failed to add code generation target");
        }
        
        // Add translation unit (the source file)
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
        
        // Add source code
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
        
        // Add explicit entry points if provided
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
        
        // Compile
        let result = unsafe { (self.library.compile)(request) };
        if !slang_succeeded(result) {
            // Get diagnostic output
            let diag_ptr = unsafe { (self.library.get_diagnostic_output)(request) };
            let diagnostic = if !diag_ptr.is_null() {
                unsafe { std::ffi::CStr::from_ptr(diag_ptr) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown compilation error".to_string()
            };
            anyhow::bail!("Slang compilation failed:\n{}", diagnostic);
        }
        
        // Get output code for each entry point and combine
        // For now, get the first entry point's code
        let mut blob: *mut ISlangBlob = ptr::null_mut();
        let result = unsafe {
            (self.library.get_entry_point_code_blob)(request, 0, target_index, &mut blob)
        };
        
        if !slang_succeeded(result) || blob.is_null() {
            anyhow::bail!("Failed to get compiled shader code");
        }
        
        // Copy data from blob
        let (data_ptr, data_size) = unsafe { blob_get_data(blob) };
        let data = unsafe { std::slice::from_raw_parts(data_ptr, data_size) }.to_vec();
        
        // Release blob
        unsafe { blob_release(blob) };
        
        Ok(CompiledShader { data, target })
    }
}

impl Drop for SlangCompiler {
    fn drop(&mut self) {
        if !self.session.is_null() {
            unsafe { (self.library.destroy_session)(self.session) };
        }
    }
}

/// Global Slang compiler instance.
///
/// Lazily initialized on first use.
///
/// **Deprecated**: Each VulkanBackend now owns its own SlangCompiler to avoid
/// test isolation issues. This global remains for backward compatibility.
static GLOBAL_COMPILER: std::sync::OnceLock<Result<SlangCompiler, String>> = std::sync::OnceLock::new();

/// Get or create the global Slang compiler.
///
/// **Deprecated**: Prefer creating a `SlangCompiler::new()` per context to avoid
/// session state pollution when compiling the same shader source multiple times.
/// The Vulkan backend now uses per-backend compiler instances.
#[deprecated(since = "0.2.0", note = "Use SlangCompiler::new() per context instead")]
pub fn global_compiler() -> Result<&'static SlangCompiler> {
    GLOBAL_COMPILER
        .get_or_init(|| SlangCompiler::new().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{}", e))
}

