//! High-level Slang compiler API.
//!
//! Provides a safe, ergonomic interface for compiling Slang shaders.

use anyhow::{Context, Result};
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Arc;

use super::ffi::*;
use super::loader::SlangLibrary;
use crate::{goldy_event, goldy_span};

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
    /// WGSL source for WebGPU
    Wgsl,
    /// HLSL source for DirectX (text, requires FXC/DXC to compile)
    Hlsl,
    /// DXIL bytecode for DirectX 12 (binary, SM 6.6 for bindless)
    Dxil,
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
            ShaderTarget::Dxil => SlangCompileTarget::Dxil,
            ShaderTarget::Metal => SlangCompileTarget::Metal,
            ShaderTarget::Glsl => SlangCompileTarget::Glsl,
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
    /// Get the data as a string (for text-based targets like WGSL, HLSL, GLSL).
    pub fn as_str(&self) -> Option<&str> {
        match self.target {
            ShaderTarget::Wgsl | ShaderTarget::Hlsl | ShaderTarget::Metal | ShaderTarget::Glsl => {
                std::str::from_utf8(&self.data).ok()
            }
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
    session: *mut SlangSession,
}

// SlangCompiler is Send + Sync because each compilation creates its own request
unsafe impl Send for SlangCompiler {}
unsafe impl Sync for SlangCompiler {}

impl SlangCompiler {
    /// Create a new Slang compiler instance.
    pub fn new() -> Result<Self> {
        let _span = goldy_span!("slang.compiler.init").entered();

        let library = Arc::new(SlangLibrary::load()?);

        // Create global session
        let session = unsafe { (library.create_session)(ptr::null()) };

        if session.is_null() {
            anyhow::bail!("Failed to create Slang session");
        }

        goldy_event!("slang.session.create", success = true);
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

    /// Compile for bindless rendering (adds __BINDLESS__ preprocessor define).
    ///
    /// This is used by backends that support bindless resource access.
    /// Shaders can check for `#ifdef __BINDLESS__` to use bindless patterns.
    pub fn compile_bindless(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
    ) -> Result<CompiledShader> {
        self.compile_with_defines(
            source,
            target,
            entry_points,
            search_paths,
            &[("__BINDLESS__", "1")],
        )
    }

    /// Compile for bindless rendering with reflection data.
    ///
    /// Returns both the compiled shader and reflection information about
    /// ParameterBlocks, which is needed to properly set up argument buffers.
    pub fn compile_bindless_with_reflection(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
    ) -> Result<CompiledShaderWithReflection> {
        self.compile_with_reflection(
            source,
            target,
            entry_points,
            search_paths,
            &[("__BINDLESS__", "1")],
        )
    }

    /// Compile with reflection data.
    ///
    /// This performs compilation and extracts reflection information about
    /// all parameters, especially ParameterBlocks for bindless rendering.
    pub fn compile_with_reflection(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<CompiledShaderWithReflection> {
        // Create compile request
        let request = unsafe { (self.library.create_compile_request)(self.session) };
        if request.is_null() {
            anyhow::bail!("Failed to create Slang compile request");
        }

        // Ensure cleanup on all paths
        let library = self.library.clone();
        let _guard = scopeguard::guard(request, |req| {
            unsafe { (library.destroy_compile_request)(req) };
        });

        // Add search paths for module resolution
        for path in search_paths {
            let path_cstr = CString::new(*path).context("Search path contains null bytes")?;
            unsafe {
                (self.library.add_search_path)(request, path_cstr.as_ptr());
            }
        }

        // Add preprocessor defines
        for (key, value) in defines {
            let key_cstr = CString::new(*key).context("Define key contains null bytes")?;
            let value_cstr = CString::new(*value).context("Define value contains null bytes")?;
            unsafe {
                (self.library.add_preprocessor_define)(
                    request,
                    key_cstr.as_ptr(),
                    value_cstr.as_ptr(),
                );
            }
        }

        // Add target
        let target_index =
            unsafe { (self.library.add_code_gen_target)(request, target.to_slang_target() as i32) };
        if target_index < 0 {
            anyhow::bail!("Failed to add code generation target");
        }

        // Set profile for DXIL target (SM 6.6 for bindless support)
        if target == ShaderTarget::Dxil {
            let profile_name = CString::new("sm_6_6").unwrap();
            let profile_id =
                unsafe { (self.library.find_profile)(self.session, profile_name.as_ptr()) };
            if profile_id > 0 {
                unsafe {
                    (self.library.set_target_profile)(request, target_index, profile_id);
                }
                tracing::debug!("Set DXIL target profile to sm_6_6 (id={})", profile_id);
            } else {
                tracing::warn!("Could not find sm_6_6 profile, using default");
            }
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
                unsafe { CStr::from_ptr(diag_ptr) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown compilation error".to_string()
            };
            anyhow::bail!("Slang compilation failed:\n{}", diagnostic);
        }

        // Get compiled code
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

        // Extract reflection data
        let reflection = self.extract_reflection(request)?;

        Ok(CompiledShaderWithReflection {
            shader: CompiledShader { data, target },
            reflection,
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

        Ok(ShaderReflection { parameter_blocks })
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

    /// Compile with explicit entry points.
    ///
    /// If `entry_points` is empty, entry points are detected from shader attributes.
    pub fn compile_with_entry_points(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
    ) -> Result<CompiledShader> {
        self.compile_with_options(source, target, entry_points, &[])
    }

    /// Compile with explicit entry points and search paths.
    ///
    /// Search paths are used to resolve `import` statements in Slang modules.
    /// If `entry_points` is empty, entry points are detected from shader attributes.
    pub fn compile_with_options(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
    ) -> Result<CompiledShader> {
        self.compile_with_defines(source, target, entry_points, search_paths, &[])
    }

    /// Compile with explicit entry points, search paths, and preprocessor defines.
    ///
    /// Search paths are used to resolve `import` statements in Slang modules.
    /// If `entry_points` is empty, entry points are detected from shader attributes.
    /// Defines are passed to the preprocessor as key=value pairs (value can be empty).
    pub fn compile_with_defines(
        &self,
        source: &str,
        target: ShaderTarget,
        entry_points: &[(&str, SlangStage)],
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<CompiledShader> {
        let _span = goldy_span!(
            "slang.compile",
            target = ?target,
            entry_points = entry_points.len(),
            bindless = defines.iter().any(|(k, _)| *k == "__BINDLESS__")
        )
        .entered();

        goldy_event!(
            "slang.compile.start",
            target = ?target,
            entry_points = entry_points.len(),
            source_len = source.len()
        );

        // Create compile request
        let request = unsafe { (self.library.create_compile_request)(self.session) };
        if request.is_null() {
            anyhow::bail!("Failed to create Slang compile request");
        }

        // Ensure cleanup on all paths
        let _guard = scopeguard::guard(request, |req| {
            unsafe { (self.library.destroy_compile_request)(req) };
        });

        // Add search paths for module resolution
        for path in search_paths {
            let path_cstr = CString::new(*path).context("Search path contains null bytes")?;
            unsafe {
                (self.library.add_search_path)(request, path_cstr.as_ptr());
            }
        }

        // Add preprocessor defines
        for (key, value) in defines {
            let key_cstr = CString::new(*key).context("Define key contains null bytes")?;
            let value_cstr = CString::new(*value).context("Define value contains null bytes")?;
            unsafe {
                (self.library.add_preprocessor_define)(
                    request,
                    key_cstr.as_ptr(),
                    value_cstr.as_ptr(),
                );
            }
        }

        // Add target
        let target_index =
            unsafe { (self.library.add_code_gen_target)(request, target.to_slang_target() as i32) };
        if target_index < 0 {
            anyhow::bail!("Failed to add code generation target");
        }

        // Set profile for DXIL target (SM 6.6 for bindless support)
        if target == ShaderTarget::Dxil {
            let profile_name = CString::new("sm_6_6").unwrap();
            let profile_id =
                unsafe { (self.library.find_profile)(self.session, profile_name.as_ptr()) };
            if profile_id > 0 {
                unsafe {
                    (self.library.set_target_profile)(request, target_index, profile_id);
                }
                tracing::debug!("Set DXIL target profile to sm_6_6 (id={})", profile_id);
            } else {
                tracing::warn!("Could not find sm_6_6 profile, using default");
            }
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

        goldy_event!(
            "slang.compile.end",
            output_size = data.len(),
            success = true
        );

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
static GLOBAL_COMPILER: std::sync::OnceLock<Result<SlangCompiler, String>> =
    std::sync::OnceLock::new();

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
