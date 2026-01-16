//! Raw FFI bindings to the Slang shader compiler.
//!
//! These are manual bindings to the Slang C API (sp* functions).
//! We use dynamic loading via libloading to avoid build-time dependencies.

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};

/// Opaque session handle
pub type SlangSession = c_void;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Function pointer types for dynamic loading
pub type FnSpCreateSession = unsafe extern "C" fn(deprecated: *const c_char) -> *mut SlangSession;
pub type FnSpDestroySession = unsafe extern "C" fn(session: *mut SlangSession);
pub type FnSpCreateCompileRequest =
    unsafe extern "C" fn(session: *mut SlangSession) -> *mut SlangCompileRequest;
pub type FnSpDestroyCompileRequest = unsafe extern "C" fn(request: *mut SlangCompileRequest);
pub type FnSpSetCodeGenTarget =
    unsafe extern "C" fn(request: *mut SlangCompileRequest, target: c_int);
pub type FnSpAddCodeGenTarget =
    unsafe extern "C" fn(request: *mut SlangCompileRequest, target: c_int) -> c_int;
pub type FnSpSetTargetProfile =
    unsafe extern "C" fn(request: *mut SlangCompileRequest, target_index: c_int, profile: c_int);
pub type FnSpAddTranslationUnit = unsafe extern "C" fn(
    request: *mut SlangCompileRequest,
    language: c_int,
    name: *const c_char,
) -> c_int;
pub type FnSpAddTranslationUnitSourceString = unsafe extern "C" fn(
    request: *mut SlangCompileRequest,
    translation_unit_index: c_int,
    path: *const c_char,
    source: *const c_char,
);
pub type FnSpAddSearchPath =
    unsafe extern "C" fn(request: *mut SlangCompileRequest, path: *const c_char);
pub type FnSpAddPreprocessorDefine = unsafe extern "C" fn(
    request: *mut SlangCompileRequest,
    key: *const c_char,
    value: *const c_char,
);
pub type FnSpAddEntryPoint = unsafe extern "C" fn(
    request: *mut SlangCompileRequest,
    translation_unit_index: c_int,
    name: *const c_char,
    stage: c_int,
) -> c_int;
pub type FnSpCompile = unsafe extern "C" fn(request: *mut SlangCompileRequest) -> SlangResult;
pub type FnSpGetDiagnosticOutput =
    unsafe extern "C" fn(request: *mut SlangCompileRequest) -> *const c_char;
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

// ============================================================================
// Reflection API function pointer types
// ============================================================================

/// Get reflection data from a compile request (call after spCompile succeeds)
pub type FnSpGetReflection =
    unsafe extern "C" fn(request: *mut SlangCompileRequest) -> *mut SlangReflection;

/// Get the number of parameters in the program
pub type FnSpReflectionGetParameterCount =
    unsafe extern "C" fn(reflection: *mut SlangReflection) -> c_int;

/// Get a parameter by index
pub type FnSpReflectionGetParameterByIndex = unsafe extern "C" fn(
    reflection: *mut SlangReflection,
    index: c_int,
) -> *mut SlangReflectionParameter;

/// Get the type layout of a variable layout (parameter)
/// Note: In Slang, parameters are VariableLayouts
pub type FnSpReflectionParameterGetTypeLayout =
    unsafe extern "C" fn(
        var_layout: *mut SlangReflectionVariableLayout,
    ) -> *mut SlangReflectionTypeLayout;

/// Get the underlying variable from a variable layout
pub type FnSpReflectionVariableLayoutGetVariable =
    unsafe extern "C" fn(
        var_layout: *mut SlangReflectionVariableLayout,
    ) -> *mut SlangReflectionVariable;

/// Opaque reflection variable handle
pub type SlangReflectionVariable = c_void;

/// Get the name of a variable
pub type FnSpReflectionVariableGetName =
    unsafe extern "C" fn(variable: *mut SlangReflectionVariable) -> *const c_char;

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
pub type FnSpReflectionTypeLayoutGetFieldByIndex =
    unsafe extern "C" fn(
        type_layout: *mut SlangReflectionTypeLayout,
        index: c_int,
    ) -> *mut SlangReflectionVariableLayout;

/// Get the type from a type layout
pub type FnSpReflectionTypeLayoutGetType =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout) -> *mut SlangReflectionType;

/// Get the kind of a type
pub type FnSpReflectionTypeGetKind = unsafe extern "C" fn(type_: *mut SlangReflectionType) -> c_int;

/// Get the name of a type
pub type FnSpReflectionTypeGetName =
    unsafe extern "C" fn(type_: *mut SlangReflectionType) -> *const c_char;

/// Get the element type layout (for arrays, buffers, parameter blocks)
pub type FnSpReflectionTypeLayoutGetElementTypeLayout =
    unsafe extern "C" fn(
        type_layout: *mut SlangReflectionTypeLayout,
    ) -> *mut SlangReflectionTypeLayout;

/// Get the type layout of a variable layout (field)
pub type FnSpReflectionVariableLayoutGetTypeLayout =
    unsafe extern "C" fn(
        var_layout: *mut SlangReflectionVariableLayout,
    ) -> *mut SlangReflectionTypeLayout;

/// Get the offset of a variable layout within its container
pub type FnSpReflectionVariableLayoutGetOffset =
    unsafe extern "C" fn(var_layout: *mut SlangReflectionVariableLayout, category: c_int) -> usize;

/// Get the binding type of a type layout (for resources)
pub type FnSpReflectionTypeLayoutGetBindingType =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout) -> c_int;

/// Get the category of a type layout
pub type FnSpReflectionTypeLayoutGetCategory =
    unsafe extern "C" fn(type_layout: *mut SlangReflectionTypeLayout) -> c_int;

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
