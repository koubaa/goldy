//! Raw FFI bindings to the Slang shader compiler.
//!
//! These are manual bindings to the Slang C API (sp* functions) and COM interfaces.
//! We use dynamic loading via libloading to avoid build-time dependencies.

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};

/// Opaque session handle (deprecated C API)
pub type SlangSession = c_void;

/// Opaque global session handle (COM interface)
pub type IGlobalSession = c_void;

/// Opaque session handle (COM interface)
pub type ISession = c_void;

/// Slang API version
pub const SLANG_API_VERSION: i64 = 0;

// ============================================================================
// Session Description Structures (for COM API)
// ============================================================================

/// Preprocessor macro for session-level defines
#[repr(C)]
#[derive(Debug, Clone)]
pub struct PreprocessorMacroDesc {
    pub name: *const c_char,
    pub value: *const c_char,
}

/// Target description for code generation
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TargetDesc {
    pub structure_size: usize,
    pub format: SlangCompileTarget,
    pub profile: SlangProfileID,
    pub flags: u32,
    pub floating_point_mode: i32,
    pub line_directive_mode: i32,
    pub force_glsl_scalar_buffer_layout: bool,
    pub compiler_option_entries: *const c_void,
    pub compiler_option_entry_count: u32,
}

impl Default for TargetDesc {
    fn default() -> Self {
        Self {
            structure_size: std::mem::size_of::<TargetDesc>(),
            format: SlangCompileTarget::Unknown,
            profile: 0,
            flags: 0,
            floating_point_mode: 0,
            line_directive_mode: 0,
            force_glsl_scalar_buffer_layout: false,
            compiler_option_entries: std::ptr::null(),
            compiler_option_entry_count: 0,
        }
    }
}

/// Slang matrix layout modes (matches SlangMatrixLayoutMode enum in slang.h).
/// Controls how `float4x4` in constant buffers is interpreted in memory.
pub const SLANG_MATRIX_LAYOUT_MODE_UNKNOWN: i32 = 0;
pub const SLANG_MATRIX_LAYOUT_ROW_MAJOR: i32 = 1;
pub const SLANG_MATRIX_LAYOUT_COLUMN_MAJOR: i32 = 2;

/// Session description for creating sessions with options.
/// This must match the C++ SessionDesc struct exactly (including padding).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SessionDesc {
    pub structure_size: usize,                             // size_t (8 bytes)
    pub targets: *const TargetDesc,                        // pointer (8 bytes)
    pub target_count: i64,                                 // SlangInt = int64_t (8 bytes)
    pub flags: u32,                                        // SessionFlags = uint32_t (4 bytes)
    pub default_matrix_layout_mode: i32,                   // SlangMatrixLayoutMode = int (4 bytes)
    pub search_paths: *const *const c_char,                // char const* const* (8 bytes)
    pub search_path_count: i64,                            // SlangInt = int64_t (8 bytes)
    pub preprocessor_macros: *const PreprocessorMacroDesc, // pointer (8 bytes)
    pub preprocessor_macro_count: i64,                     // SlangInt = int64_t (8 bytes)
    pub file_system: *const c_void,                        // ISlangFileSystem* (8 bytes)
    pub enable_effect_annotations: bool,                   // bool (1 byte)
    pub allow_glsl_syntax: bool,                           // bool (1 byte)
    _padding1: [u8; 6],                                    // padding to align next pointer
    pub compiler_option_entries: *const c_void,            // CompilerOptionEntry* (8 bytes)
    pub compiler_option_entry_count: u32,                  // uint32_t (4 bytes)
    pub skip_spirv_validation: bool,                       // bool (1 byte)
    _padding2: [u8; 3],                                    // padding to reach struct alignment
}

impl Default for SessionDesc {
    fn default() -> Self {
        Self {
            structure_size: std::mem::size_of::<SessionDesc>(),
            targets: std::ptr::null(),
            target_count: 0,
            flags: 0,
            default_matrix_layout_mode: SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
            search_paths: std::ptr::null(),
            search_path_count: 0,
            preprocessor_macros: std::ptr::null(),
            preprocessor_macro_count: 0,
            file_system: std::ptr::null(),
            enable_effect_annotations: false,
            allow_glsl_syntax: false,
            _padding1: [0; 6],
            compiler_option_entries: std::ptr::null(),
            compiler_option_entry_count: 0,
            skip_spirv_validation: false,
            _padding2: [0; 3],
        }
    }
}

/// Global session description
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SlangGlobalSessionDesc {
    pub structure_size: u32,
    pub api_version: u32,
    pub min_language_version: u32,
    pub enable_glsl: bool,
    _padding: [u8; 3], // Padding after bool to align reserved array
    pub reserved: [u32; 16],
}

impl Default for SlangGlobalSessionDesc {
    fn default() -> Self {
        Self {
            structure_size: std::mem::size_of::<SlangGlobalSessionDesc>() as u32,
            api_version: SLANG_API_VERSION as u32,
            min_language_version: 2025, // SLANG_LANGUAGE_VERSION_2025
            enable_glsl: false,
            _padding: [0; 3],
            reserved: [0; 16],
        }
    }
}

// ============================================================================
// COM Interface VTables
// ============================================================================

/// IGlobalSession vtable (COM-style)
/// The vtable layout must exactly match the C++ vtable order.
#[repr(C)]
pub struct IGlobalSessionVtable {
    // ISlangUnknown methods (3 methods)
    pub query_interface: unsafe extern "C" fn(
        this: *mut IGlobalSession,
        uuid: *const c_void,
        out_object: *mut *mut c_void,
    ) -> SlangResult,
    pub add_ref: unsafe extern "C" fn(this: *mut IGlobalSession) -> u32,
    pub release: unsafe extern "C" fn(this: *mut IGlobalSession) -> u32,
    // IGlobalSession methods
    pub create_session: unsafe extern "C" fn(
        this: *mut IGlobalSession,
        desc: *const SessionDesc,
        out_session: *mut *mut ISession,
    ) -> SlangResult,
    pub find_profile: unsafe extern "C" fn(this: *mut IGlobalSession, name: *const c_char) -> SlangProfileID,
    // ... more methods we don't need, but we need to know about them for vtable offset
}

/// ISession vtable (COM-style)
/// We need to include all methods up to createCompileRequest (vtable index 14)
#[repr(C)]
pub struct ISessionVtable {
    // ISlangUnknown methods (3 methods, indices 0-2)
    pub query_interface:
        unsafe extern "C" fn(this: *mut ISession, uuid: *const c_void, out_object: *mut *mut c_void) -> SlangResult,
    pub add_ref: unsafe extern "C" fn(this: *mut ISession) -> u32,
    pub release: unsafe extern "C" fn(this: *mut ISession) -> u32,
    // ISession methods (indices 3-13)
    pub get_global_session: *const c_void,                         // 3
    pub load_module: *const c_void,                                // 4
    pub load_module_from_source: *const c_void,                    // 5
    pub create_composite_component_type: *const c_void,            // 6
    pub specialize_type: *const c_void,                            // 7
    pub get_type_layout: *const c_void,                            // 8
    pub get_container_type: *const c_void,                         // 9
    pub get_dynamic_type: *const c_void,                           // 10
    pub get_type_rtti_mangled_name: *const c_void,                 // 11
    pub get_type_conformance_witness_mangled_name: *const c_void,  // 12
    pub get_type_conformance_witness_sequential_id: *const c_void, // 13
    // createCompileRequest (index 14)
    pub create_compile_request:
        unsafe extern "C" fn(this: *mut ISession, out_compile_request: *mut *mut SlangCompileRequest) -> SlangResult,
}

/// Opaque compile request handle  
pub type SlangCompileRequest = c_void;

/// Opaque blob handle for output data
pub type ISlangBlob = c_void;

// ============================================================================
// Reflection API types
// ============================================================================

/// Opaque reflection handle (returned after compilation)
pub type SlangReflection = c_void;

/// Opaque reflection parameter handle
pub type SlangReflectionParameter = c_void;

/// Opaque reflection type layout handle
pub type SlangReflectionTypeLayout = c_void;

/// Opaque reflection type handle
pub type SlangReflectionType = c_void;

/// Opaque reflection variable layout handle
pub type SlangReflectionVariableLayout = c_void;

/// Slang type kind enum
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlangTypeKind {
    None = 0,
    Struct = 1,
    Array = 2,
    Matrix = 3,
    Vector = 4,
    Scalar = 5,
    ConstantBuffer = 6,
    Resource = 7,
    SamplerState = 8,
    TextureBuffer = 9,
    ShaderStorageBuffer = 10,
    ParameterBlock = 11,
    GenericTypeParameter = 12,
    Interface = 13,
    OutputStream = 14,
    Specialized = 15,
    Feedback = 16,
    Pointer = 17,
    DynamicResource = 18,
}

/// Slang binding type enum (for reflection)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlangBindingType {
    Unknown = 0,
    Sampler = 1,
    Texture = 2,
    ConstantBuffer = 3,
    ParameterBlock = 4,
    TypedBuffer = 5,
    RawBuffer = 6,
    CombinedTextureSampler = 7,
    InputRenderTarget = 8,
    InlineUniformData = 9,
    RayTracingAccelerationStructure = 10,
    VaryingInput = 11,
    VaryingOutput = 12,
    ExistentialValue = 13,
    PushConstant = 14,
    MutableFlag = 0x100,
    MutableTexture = 0x102,     // Texture | MutableFlag
    MutableTypedBuffer = 0x105, // TypedBuffer | MutableFlag
    MutableRawBuffer = 0x106,   // RawBuffer | MutableFlag
}

/// Slang parameter category (for layout)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlangParameterCategory {
    None = 0,
    Mixed = 1,
    ConstantBuffer = 2,
    ShaderResource = 3,
    UnorderedAccess = 4,
    VaryingInput = 5,
    VaryingOutput = 6,
    SamplerState = 7,
    Uniform = 8,
    DescriptorTableSlot = 9,
    SpecializationConstant = 10,
    PushConstantBuffer = 11,
    RegisterSpace = 12,
    GenericResource = 13,
    RayPayload = 14,
    HitAttributes = 15,
    CallablePayload = 16,
    ShaderRecord = 17,
    ExistentialTypeParam = 18,
    ExistentialObjectParam = 19,
    SubElementRegisterSpace = 20,
    InputAttachmentIndex = 21,
    MetalArgumentBufferElement = 22,
    MetalAttribute = 23,
    MetalPayload = 24,
}

/// Result type (HRESULT-style)
pub type SlangResult = i32;

/// Success result
pub const SLANG_OK: SlangResult = 0;

/// Floating point modes
pub const SLANG_FLOATING_POINT_MODE_DEFAULT: c_int = 0;
pub const SLANG_FLOATING_POINT_MODE_FAST: c_int = 1;
pub const SLANG_FLOATING_POINT_MODE_PRECISE: c_int = 2;

/// Check if result is successful
#[inline]
pub fn slang_succeeded(result: SlangResult) -> bool {
    result >= 0
}

/// Compile target enum
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlangCompileTarget {
    Unknown = 0,
    None = 1,
    Glsl = 2,
    GlslVulkanDeprecated = 3,
    GlslVulkanOneDescDeprecated = 4,
    Hlsl = 5,
    Spirv = 6,
    SpirvAsm = 7,
    Dxbc = 8,
    DxbcAsm = 9,
    Dxil = 10,
    DxilAsm = 11,
    CSource = 12,
    CppSource = 13,
    HostExecutable = 14,
    ShaderSharedLibrary = 15,
    ShaderHostCallable = 16,
    CudaSource = 17,
    Ptx = 18,
    CudaObjectCode = 19,
    ObjectCode = 20,
    HostCppSource = 21,
    HostHostCallable = 22,
    CppPytorchBinding = 23,
    Metal = 24,
    MetalLib = 25,
    MetalLibAsm = 26,
    HostSharedLibrary = 27,
    Wgsl = 28,
}

/// Shader stage enum
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SlangStage {
    None = 0,
    Vertex = 1,
    Hull = 2,
    Domain = 3,
    Geometry = 4,
    Fragment = 5,
    Compute = 6,
    RayGeneration = 7,
    Intersection = 8,
    AnyHit = 9,
    ClosestHit = 10,
    Miss = 11,
    Callable = 12,
    Mesh = 13,
    Amplification = 14,
}

/// Source language enum
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlangSourceLanguage {
    Unknown = 0,
    Slang = 1,
    Hlsl = 2,
    Glsl = 3,
    C = 4,
    Cpp = 5,
    Cuda = 6,
    Spirv = 7,
}

/// Profile ID type (opaque integral type for shader profiles like sm_6_6)
pub type SlangProfileID = c_int;

/// Slang optimization levels (matches SlangOptimizationLevel in slang.h).
pub const SLANG_OPTIMIZATION_LEVEL_NONE: c_int = 0;
pub const SLANG_OPTIMIZATION_LEVEL_DEFAULT: c_int = 1;
pub const SLANG_OPTIMIZATION_LEVEL_HIGH: c_int = 2;
pub const SLANG_OPTIMIZATION_LEVEL_MAXIMAL: c_int = 3;

/// Function pointer types for dynamic loading
pub type FnSpCreateSession = unsafe extern "C" fn(deprecated: *const c_char) -> *mut SlangSession;
pub type FnSpDestroySession = unsafe extern "C" fn(session: *mut SlangSession);
pub type FnSpCreateCompileRequest = unsafe extern "C" fn(session: *mut SlangSession) -> *mut SlangCompileRequest;
pub type FnSpDestroyCompileRequest = unsafe extern "C" fn(request: *mut SlangCompileRequest);
pub type FnSpSetCodeGenTarget = unsafe extern "C" fn(request: *mut SlangCompileRequest, target: c_int);
pub type FnSpAddCodeGenTarget = unsafe extern "C" fn(request: *mut SlangCompileRequest, target: c_int) -> c_int;
pub type FnSpSetTargetProfile =
    unsafe extern "C" fn(request: *mut SlangCompileRequest, target_index: c_int, profile: c_int);
pub type FnSpSetTargetFloatingPointMode =
    unsafe extern "C" fn(request: *mut SlangCompileRequest, target_index: c_int, mode: c_int);
/// Find a profile by name (e.g., "sm_6_6") - uses the global session
pub type FnSpFindProfile = unsafe extern "C" fn(session: *mut SlangSession, name: *const c_char) -> SlangProfileID;
pub type FnSpAddTranslationUnit =
    unsafe extern "C" fn(request: *mut SlangCompileRequest, language: c_int, name: *const c_char) -> c_int;
pub type FnSpAddTranslationUnitSourceString = unsafe extern "C" fn(
    request: *mut SlangCompileRequest,
    translation_unit_index: c_int,
    path: *const c_char,
    source: *const c_char,
);
pub type FnSpAddSearchPath = unsafe extern "C" fn(request: *mut SlangCompileRequest, path: *const c_char);
pub type FnSpAddPreprocessorDefine =
    unsafe extern "C" fn(request: *mut SlangCompileRequest, key: *const c_char, value: *const c_char);
pub type FnSpAddEntryPoint = unsafe extern "C" fn(
    request: *mut SlangCompileRequest,
    translation_unit_index: c_int,
    name: *const c_char,
    stage: c_int,
) -> c_int;
pub type FnSpSetOptimizationLevel = unsafe extern "C" fn(request: *mut SlangCompileRequest, level: c_int);
pub type FnSpCompile = unsafe extern "C" fn(request: *mut SlangCompileRequest) -> SlangResult;
pub type FnSpGetDiagnosticOutput = unsafe extern "C" fn(request: *mut SlangCompileRequest) -> *const c_char;
pub type FnSpGetEntryPointCodeBlob = unsafe extern "C" fn(
    request: *mut SlangCompileRequest,
    entry_point_index: c_int,
    target_index: c_int,
    out_blob: *mut *mut ISlangBlob,
) -> SlangResult;
pub type FnSpGetTargetCodeBlob = unsafe extern "C" fn(
    request: *mut SlangCompileRequest,
    target_index: c_int,
    out_blob: *mut *mut ISlangBlob,
) -> SlangResult;

// New COM-style API function pointers
pub type FnSlangCreateGlobalSession2 = unsafe extern "C" fn(
    desc: *const SlangGlobalSessionDesc,
    out_global_session: *mut *mut IGlobalSession,
) -> SlangResult;

// ============================================================================
// Reflection API function pointer types
// ============================================================================

/// Get reflection data from a compile request (call after spCompile succeeds)
pub type FnSpGetReflection = unsafe extern "C" fn(request: *mut SlangCompileRequest) -> *mut SlangReflection;

/// Layout rules for [`FnSpReflectionGetTypeLayout`] (matches `SlangLayoutRules` in slang.h).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlangLayoutRules {
    Default = 0,
    MetalArgumentBufferTier2 = 1,
}

/// Find a named type in the compiled program's reflection.
pub type FnSpReflectionFindTypeByName =
    unsafe extern "C" fn(reflection: *mut SlangReflection, name: *const c_char) -> *mut SlangReflectionType;

/// Get the memory layout of a type under the given rules.
pub type FnSpReflectionGetTypeLayout = unsafe extern "C" fn(
    reflection: *mut SlangReflection,
    reflection_type: *mut SlangReflectionType,
    rules: SlangLayoutRules,
) -> *mut SlangReflectionTypeLayout;

/// Get the number of parameters in the program
pub type FnSpReflectionGetParameterCount = unsafe extern "C" fn(reflection: *mut SlangReflection) -> c_int;

/// Get a parameter by index
pub type FnSpReflectionGetParameterByIndex =
    unsafe extern "C" fn(reflection: *mut SlangReflection, index: c_int) -> *mut SlangReflectionParameter;

/// Get the type layout of a variable layout (parameter)
/// Note: In Slang, parameters are VariableLayouts
pub type FnSpReflectionParameterGetTypeLayout =
    unsafe extern "C" fn(var_layout: *mut SlangReflectionVariableLayout) -> *mut SlangReflectionTypeLayout;

/// Get the underlying variable from a variable layout
pub type FnSpReflectionVariableLayoutGetVariable =
    unsafe extern "C" fn(var_layout: *mut SlangReflectionVariableLayout) -> *mut SlangReflectionVariable;

/// Opaque reflection variable handle
pub type SlangReflectionVariable = c_void;

/// Get the name of a variable
pub type FnSpReflectionVariableGetName = unsafe extern "C" fn(variable: *mut SlangReflectionVariable) -> *const c_char;

/// Get the binding index for a parameter
pub type FnSpReflectionParameterGetBindingIndex =
    unsafe extern "C" fn(parameter: *mut SlangReflectionParameter) -> c_int;

/// Get the binding space/set for a parameter  
pub type FnSpReflectionParameterGetBindingSpace =
    unsafe extern "C" fn(parameter: *mut SlangReflectionParameter) -> c_int;

/// Get the size of a type layout in bytes
pub type FnSpReflectionTypeLayoutGetSize =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout, category: c_int) -> usize;

/// Get the stride of a type layout (for arrays/buffers)
pub type FnSpReflectionTypeLayoutGetStride =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout, category: c_int) -> usize;

/// Get the alignment of a type layout
pub type FnSpReflectionTypeLayoutGetAlignment =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout, category: c_int) -> usize;

/// Get the number of fields in a struct type layout
pub type FnSpReflectionTypeLayoutGetFieldCount =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout) -> c_int;

/// Get a field by index from a type layout
pub type FnSpReflectionTypeLayoutGetFieldByIndex = unsafe extern "C" fn(
    type_layout: *mut SlangReflectionTypeLayout,
    index: c_int,
) -> *mut SlangReflectionVariableLayout;

/// Get the type from a type layout
pub type FnSpReflectionTypeLayoutGetType =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout) -> *mut SlangReflectionType;

/// Get the kind of a type
pub type FnSpReflectionTypeGetKind = unsafe extern "C" fn(type_: *mut SlangReflectionType) -> c_int;

/// Get the name of a type
pub type FnSpReflectionTypeGetName = unsafe extern "C" fn(type_: *mut SlangReflectionType) -> *const c_char;

/// Get the element type layout (for arrays, buffers, parameter blocks)
pub type FnSpReflectionTypeLayoutGetElementTypeLayout =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout) -> *mut SlangReflectionTypeLayout;

/// Get the type layout of a variable layout (field)
pub type FnSpReflectionVariableLayoutGetTypeLayout =
    unsafe extern "C" fn(var_layout: *mut SlangReflectionVariableLayout) -> *mut SlangReflectionTypeLayout;

/// Get the offset of a variable layout within its container
pub type FnSpReflectionVariableLayoutGetOffset =
    unsafe extern "C" fn(var_layout: *mut SlangReflectionVariableLayout, category: c_int) -> usize;

/// Get the binding type of a type layout (for resources)
pub type FnSpReflectionTypeLayoutGetBindingType =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout) -> c_int;

/// Get the category of a type layout
pub type FnSpReflectionTypeLayoutGetCategory =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout) -> c_int;

// ============================================================================
// User-defined attribute reflection
// ============================================================================

// ISlangBlob interface methods (COM-style vtable)
pub type FnBlobGetBufferPointer = unsafe extern "C" fn(blob: *mut ISlangBlob) -> *const c_void;
pub type FnBlobGetBufferSize = unsafe extern "C" fn(blob: *mut ISlangBlob) -> usize;
pub type FnBlobRelease = unsafe extern "C" fn(blob: *mut ISlangBlob) -> u32;

/// ISlangBlob vtable layout (COM-style)
#[repr(C)]
pub struct ISlangBlobVtable {
    // ISlangUnknown methods
    pub query_interface: *const c_void,
    pub add_ref: *const c_void,
    pub release: unsafe extern "C" fn(this: *mut ISlangBlob) -> u32,
    // ISlangBlob methods
    pub get_buffer_pointer: unsafe extern "C" fn(this: *mut ISlangBlob) -> *const c_void,
    pub get_buffer_size: unsafe extern "C" fn(this: *mut ISlangBlob) -> usize,
}

/// Helper to get blob data
///
/// # Safety
/// The blob pointer must be valid and point to a valid ISlangBlob COM object.
pub unsafe fn blob_get_data(blob: *mut ISlangBlob) -> (*const u8, usize) {
    // ISlangBlob is a COM object, first pointer is vtable
    let vtable_ptr = *(blob as *const *const ISlangBlobVtable);
    let vtable = &*vtable_ptr;

    let ptr = (vtable.get_buffer_pointer)(blob) as *const u8;
    let size = (vtable.get_buffer_size)(blob);

    (ptr, size)
}

/// Helper to release blob
///
/// # Safety
/// The blob pointer must be valid and point to a valid ISlangBlob COM object.
pub unsafe fn blob_release(blob: *mut ISlangBlob) {
    let vtable_ptr = *(blob as *const *const ISlangBlobVtable);
    let vtable = &*vtable_ptr;
    (vtable.release)(blob);
}

// ============================================================================
// COM Interface Helpers
// ============================================================================

/// Create a session from a global session with preprocessor macros
///
/// # Safety
/// The global_session pointer must be valid.
pub unsafe fn global_session_create_session(
    global_session: *mut IGlobalSession,
    desc: *const SessionDesc,
    out_session: *mut *mut ISession,
) -> SlangResult {
    let vtable_ptr = *(global_session as *const *const IGlobalSessionVtable);
    let vtable = &*vtable_ptr;
    (vtable.create_session)(global_session, desc, out_session)
}

/// Find a profile by name using the global session
///
/// # Safety
/// The global_session pointer must be valid.
pub unsafe fn global_session_find_profile(global_session: *mut IGlobalSession, name: *const c_char) -> SlangProfileID {
    let vtable_ptr = *(global_session as *const *const IGlobalSessionVtable);
    let vtable = &*vtable_ptr;
    (vtable.find_profile)(global_session, name)
}

/// Release a global session
///
/// # Safety
/// The global_session pointer must be valid.
pub unsafe fn global_session_release(global_session: *mut IGlobalSession) -> u32 {
    let vtable_ptr = *(global_session as *const *const IGlobalSessionVtable);
    let vtable = &*vtable_ptr;
    (vtable.release)(global_session)
}

/// Release a session
///
/// # Safety
/// The session pointer must be valid.
pub unsafe fn session_release(session: *mut ISession) -> u32 {
    let vtable_ptr = *(session as *const *const ISessionVtable);
    let vtable = &*vtable_ptr;
    (vtable.release)(session)
}

/// Create a compile request from an ISession
///
/// # Safety
/// The session pointer must be valid.
pub unsafe fn session_create_compile_request(
    session: *mut ISession,
    out_compile_request: *mut *mut SlangCompileRequest,
) -> SlangResult {
    let vtable_ptr = *(session as *const *const ISessionVtable);
    let vtable = &*vtable_ptr;
    (vtable.create_compile_request)(session, out_compile_request)
}

// ============================================================================
// Size assertions to catch FFI layout mismatches
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        // SlangGlobalSessionDesc: 4 + 4 + 4 + 1 + 3(pad) + 64 = 80 bytes
        assert_eq!(
            std::mem::size_of::<SlangGlobalSessionDesc>(),
            80,
            "SlangGlobalSessionDesc size mismatch"
        );

        // SessionDesc: Expected ~96 bytes on 64-bit
        // 8 + 8 + 8 + 4 + 4 + 8 + 8 + 8 + 8 + 8 + 1 + 1 + 6 + 8 + 4 + 1 + 3 = 96
        assert_eq!(std::mem::size_of::<SessionDesc>(), 96, "SessionDesc size mismatch");

        // PreprocessorMacroDesc: 2 pointers = 16 bytes
        assert_eq!(
            std::mem::size_of::<PreprocessorMacroDesc>(),
            16,
            "PreprocessorMacroDesc size mismatch"
        );
    }
}
