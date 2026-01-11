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
pub type FnSpCreateCompileRequest = unsafe extern "C" fn(session: *mut SlangSession) -> *mut SlangCompileRequest;
pub type FnSpDestroyCompileRequest = unsafe extern "C" fn(request: *mut SlangCompileRequest);
pub type FnSpSetCodeGenTarget = unsafe extern "C" fn(request: *mut SlangCompileRequest, target: c_int);
pub type FnSpAddCodeGenTarget = unsafe extern "C" fn(request: *mut SlangCompileRequest, target: c_int) -> c_int;
pub type FnSpSetTargetProfile = unsafe extern "C" fn(request: *mut SlangCompileRequest, target_index: c_int, profile: c_int);
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
pub type FnSpAddSearchPath = unsafe extern "C" fn(
    request: *mut SlangCompileRequest,
    path: *const c_char,
);
pub type FnSpAddEntryPoint = unsafe extern "C" fn(
    request: *mut SlangCompileRequest,
    translation_unit_index: c_int,
    name: *const c_char,
    stage: c_int,
) -> c_int;
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

