//! Function pointer types for the Goldy C ABI.

use super::types::*;
use std::ffi::{c_char, c_void};
use std::os::raw::c_int;

pub type FnGoldyBufferAccess = unsafe extern "C" fn(*const GoldyBuffer) -> GoldyBufferKind;
pub type FnGoldyBufferCreate = unsafe extern "C" fn(*const GoldyDevice, u64, GoldyBufferKind) -> *mut GoldyBuffer;
pub type FnGoldyBufferCreateWithData =
    unsafe extern "C" fn(*const GoldyDevice, *const u8, usize, GoldyBufferKind) -> *mut GoldyBuffer;
pub type FnGoldyBufferCreateWithDataStride =
    unsafe extern "C" fn(*const GoldyDevice, *const u8, usize, GoldyBufferKind, u32) -> *mut GoldyBuffer;
pub type FnGoldyBufferDestroy = unsafe extern "C" fn(*mut GoldyBuffer);
pub type FnGoldyBufferSize = unsafe extern "C" fn(*const GoldyBuffer) -> u64;
pub type FnGoldyBufferReadToCpu =
    unsafe extern "C" fn(*const GoldyBuffer, *const GoldyDevice, *mut u8, usize) -> GoldyResult;
pub type FnGoldyBufferResourceIndex = unsafe extern "C" fn(*const GoldyBuffer, GoldyResourceAccess) -> u32;
pub type FnGoldyBufferWrite = unsafe extern "C" fn(*const GoldyBuffer, u64, *const u8, usize) -> GoldyResult;

pub type FnGoldyClearError = unsafe extern "C" fn();

pub type FnGoldyComputePipelineCreate =
    unsafe extern "C" fn(*const GoldyDevice, *const GoldyShaderModule) -> *mut GoldyComputePipeline;
pub type FnGoldyComputePipelineDestroy = unsafe extern "C" fn(*mut GoldyComputePipeline);

pub type FnGoldyDeviceAdapterId = unsafe extern "C" fn(*const GoldyDevice) -> u32;
pub type FnGoldyDeviceDestroy = unsafe extern "C" fn(*mut GoldyDevice);
pub type FnGoldyDeviceHasLibrary = unsafe extern "C" fn(*const GoldyDevice, *const c_char) -> bool;
pub type FnGoldyDeviceIsValid = unsafe extern "C" fn(*const GoldyDevice) -> bool;

pub type FnGoldyGetLastError = unsafe extern "C" fn() -> *const c_char;

pub type FnGoldyInstanceAdapterCount = unsafe extern "C" fn(*const GoldyInstance) -> u32;
pub type FnGoldyInstanceBackendType = unsafe extern "C" fn(*const GoldyInstance) -> GoldyBackendType;
pub type FnGoldyInstanceCreate = unsafe extern "C" fn() -> *mut GoldyInstance;
pub type FnGoldyInstanceCreateDeviceForAdapter = unsafe extern "C" fn(*const GoldyInstance, u32) -> *mut GoldyDevice;
pub type FnGoldyInstanceDestroy = unsafe extern "C" fn(*mut GoldyInstance);
pub type FnGoldyInstanceGetAdapter =
    unsafe extern "C" fn(*const GoldyInstance, u32, *mut GoldyAdapterInfo) -> GoldyResult;

pub type FnGoldyRenderPipelineCreate = unsafe extern "C" fn(
    *const GoldyDevice,
    *const GoldyShaderModule,
    *const GoldyShaderModule,
    *const GoldyRenderPipelineDesc,
) -> *mut GoldyRenderPipeline;
pub type FnGoldyRenderPipelineDestroy = unsafe extern "C" fn(*mut GoldyRenderPipeline);

pub type FnGoldyRenderTargetBufferSize = unsafe extern "C" fn(*const GoldyRenderTarget) -> usize;
pub type FnGoldyRenderTargetCreate =
    unsafe extern "C" fn(*const GoldyDevice, u32, u32, GoldyTextureFormat) -> *mut GoldyRenderTarget;
pub type FnGoldyRenderTargetCreateWithDepth =
    unsafe extern "C" fn(*const GoldyDevice, u32, u32, GoldyTextureFormat, GoldyDepthFormat) -> *mut GoldyRenderTarget;
pub type FnGoldyRenderTargetDestroy = unsafe extern "C" fn(*mut GoldyRenderTarget);
pub type FnGoldyRenderTargetFormat = unsafe extern "C" fn(*const GoldyRenderTarget) -> GoldyTextureFormat;
pub type FnGoldyRenderTargetHasDepth = unsafe extern "C" fn(*const GoldyRenderTarget) -> bool;
pub type FnGoldyRenderTargetHeight = unsafe extern "C" fn(*const GoldyRenderTarget) -> u32;
pub type FnGoldyRenderTargetReadToBuffer =
    unsafe extern "C" fn(*const GoldyRenderTarget, *mut u8, usize) -> GoldyResult;
pub type FnGoldyRenderTargetWidth = unsafe extern "C" fn(*const GoldyRenderTarget) -> u32;

pub type FnGoldySamplerCreate = unsafe extern "C" fn(*const GoldyDevice, *const GoldySamplerDesc) -> *mut GoldySampler;
pub type FnGoldySamplerCreateDefault = unsafe extern "C" fn(*const GoldyDevice) -> *mut GoldySampler;
pub type FnGoldySamplerDestroy = unsafe extern "C" fn(*mut GoldySampler);

pub type FnGoldyShaderBuiltinVertexColor2d = unsafe extern "C" fn() -> *const c_char;
pub type FnGoldyShaderCreate = unsafe extern "C" fn(*const GoldyDevice, *const c_char) -> *mut GoldyShaderModule;
pub type FnGoldyShaderDestroy = unsafe extern "C" fn(*mut GoldyShaderModule);

pub type FnGoldySurfaceAcquire = unsafe extern "C" fn(*const GoldySurface) -> *mut GoldySurfaceFrame;
pub type FnGoldySurfaceCreateAppkit = unsafe extern "C" fn(*const GoldyDevice, *mut c_void) -> *mut GoldySurface;
pub type FnGoldySurfaceCreateWin32 = unsafe extern "C" fn(*const GoldyDevice, *mut c_void) -> *mut GoldySurface;
pub type FnGoldySurfaceDestroy = unsafe extern "C" fn(*mut GoldySurface);
pub type FnGoldySurfaceFormat = unsafe extern "C" fn(*const GoldySurface) -> GoldyTextureFormat;
pub type FnGoldySurfaceFrameHeight = unsafe extern "C" fn(*const GoldySurfaceFrame) -> u32;
pub type FnGoldySurfaceFrameWidth = unsafe extern "C" fn(*const GoldySurfaceFrame) -> u32;
pub type FnGoldySurfaceHeight = unsafe extern "C" fn(*const GoldySurface) -> u32;
pub type FnGoldySurfacePresent = unsafe extern "C" fn(*const GoldySurface, *mut GoldySurfaceFrame) -> GoldyResult;
pub type FnGoldySurfaceResize = unsafe extern "C" fn(*mut GoldySurface, u32, u32) -> GoldyResult;
pub type FnGoldySurfaceSubmitGraphToFrame =
    unsafe extern "C" fn(*const GoldySurface, *mut GoldyTaskGraph, *mut GoldySurfaceFrame) -> GoldyResult;
pub type FnGoldySurfaceWidth = unsafe extern "C" fn(*const GoldySurface) -> u32;

pub type FnGoldyTaskGraphClear = unsafe extern "C" fn(*mut GoldyTaskGraph) -> GoldyResult;
pub type FnGoldyTaskGraphCopyRenderTargetToSwapchain =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyRenderTarget, *const GoldySwapchainOutput) -> GoldyResult;
pub type FnGoldyTaskGraphCreate = unsafe extern "C" fn() -> *mut GoldyTaskGraph;
pub type FnGoldyTaskGraphDeclareSwapchainOutput =
    unsafe extern "C" fn(*mut GoldyTaskGraph) -> *mut GoldySwapchainOutput;
pub type FnGoldyTaskGraphDestroy = unsafe extern "C" fn(*mut GoldyTaskGraph);
pub type FnGoldyTaskGraphComputeNodeBegin =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const c_char, *const GoldyComputePipeline) -> GoldyResult;
pub type FnGoldyTaskGraphComputeNodeBindBuffer =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyBuffer, GoldyNodeAccess) -> GoldyResult;
pub type FnGoldyTaskGraphComputeNodeBindParcel =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyParcel, GoldyNodeAccess) -> GoldyResult;
pub type FnGoldyTaskGraphComputeNodeBindResourcesRaw =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const u32, u32) -> GoldyResult;
pub type FnGoldyTaskGraphComputeNodeDispatch = unsafe extern "C" fn(*mut GoldyTaskGraph, u32, u32, u32) -> GoldyResult;
pub type FnGoldyTaskGraphDispatch = unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyDevice) -> GoldyResult;
pub type FnGoldyTaskGraphWriteBuffer =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyBuffer, u64, *const u8, usize) -> GoldyResult;
pub type FnGoldyTaskGraphWriteParcel =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyParcel, u64, *const u8, usize) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassBegin =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const c_char, *const GoldyRenderTarget) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassBindBuffer =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyBuffer, GoldyNodeAccess) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassBindParcel =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyParcel, GoldyNodeAccess) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassBindResources =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const *const GoldyBuffer, u32) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassBindResourcesTyped =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const u32, u32) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassClear = unsafe extern "C" fn(*mut GoldyTaskGraph, GoldyColor) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassClearDepth = unsafe extern "C" fn(*mut GoldyTaskGraph, f32) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassDraw = unsafe extern "C" fn(*mut GoldyTaskGraph, u32, u32, u32, u32) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassDrawFullscreen = unsafe extern "C" fn(*mut GoldyTaskGraph) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassDrawIndexed =
    unsafe extern "C" fn(*mut GoldyTaskGraph, u32, u32, c_int, u32, u32) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassFinish = unsafe extern "C" fn(*mut GoldyTaskGraph) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassSetIndexBuffer =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyBuffer, GoldyIndexFormat) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassSetPipeline =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyRenderPipeline) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassSetVertexBuffer =
    unsafe extern "C" fn(*mut GoldyTaskGraph, u32, *const GoldyBuffer) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassSetVertexBufferParcel =
    unsafe extern "C" fn(*mut GoldyTaskGraph, u32, *const GoldyParcel) -> GoldyResult;

pub type FnGoldyRetainedPoolAcquireBuffer =
    unsafe extern "C" fn(*mut GoldyRetainedPool, u64, GoldyBufferKind, u32, *const u8, usize) -> *mut GoldyParcel;
pub type FnGoldyRetainedPoolCreate = unsafe extern "C" fn(*const GoldyDevice) -> *mut GoldyRetainedPool;
pub type FnGoldyRetainedPoolDestroy = unsafe extern "C" fn(*mut GoldyRetainedPool);
pub type FnGoldyParcelByteSize = unsafe extern "C" fn(*const GoldyParcel) -> u64;
pub type FnGoldyParcelDestroy = unsafe extern "C" fn(*mut GoldyParcel);
pub type FnGoldyParcelResourceIndex = unsafe extern "C" fn(*const GoldyParcel, GoldyResourceAccess) -> u32;
pub type FnGoldyParcelReadToCpu =
    unsafe extern "C" fn(*const GoldyParcel, *const GoldyDevice, *mut u8, usize) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassSetVertexBufferOffset =
    unsafe extern "C" fn(*mut GoldyTaskGraph, u32, *const GoldyBuffer, u64) -> GoldyResult;

pub type FnGoldyTextureCreate = unsafe extern "C" fn(
    *const GoldyDevice,
    u32,
    u32,
    GoldyTextureFormat,
    GoldyTextureKind,
    GoldyTextureFlags,
) -> *mut GoldyTexture;
pub type FnGoldyTextureDestroy = unsafe extern "C" fn(*mut GoldyTexture);
pub type FnGoldyTextureFormat = unsafe extern "C" fn(*const GoldyTexture) -> GoldyTextureFormat;
pub type FnGoldyTextureHeight = unsafe extern "C" fn(*const GoldyTexture) -> u32;
pub type FnGoldyTextureWidth = unsafe extern "C" fn(*const GoldyTexture) -> u32;
