//! Function pointer types for the Goldy C ABI.

use super::types::*;
use std::ffi::{c_char, c_void};
use std::os::raw::c_int;

pub type FnGoldyClearError = unsafe extern "C" fn();

pub type FnGoldyComputePipelineCreate =
    unsafe extern "C" fn(*const GoldyDevice, *const GoldyShaderModule) -> *mut GoldyComputePipeline;
pub type FnGoldyComputePipelineDestroy = unsafe extern "C" fn(*mut GoldyComputePipeline);

pub type FnGoldyContextCreate = unsafe extern "C" fn(*const GoldyDevice) -> *mut GoldyContext;
pub type FnGoldyContextDestroy = unsafe extern "C" fn(*mut GoldyContext);

pub type FnGoldySchemeCreate = unsafe extern "C" fn(*const GoldyContext) -> *mut GoldyScheme;
pub type FnGoldySchemeDestroy = unsafe extern "C" fn(*mut GoldyScheme);
pub type FnGoldySchemeLen = unsafe extern "C" fn(*const GoldyScheme) -> u32;
pub type FnGoldySchemeIsDirty = unsafe extern "C" fn(*const GoldyScheme) -> bool;
pub type FnGoldySchemeReplayStats = unsafe extern "C" fn(*const GoldyScheme, *mut GoldyReplayStats) -> GoldyResult;
pub type FnGoldySchemeComputeNodeBegin =
    unsafe extern "C" fn(*mut GoldyScheme, *const c_char, *const GoldyComputePipeline) -> GoldyResult;
pub type FnGoldySchemeComputeNodeDeclareParcel =
    unsafe extern "C" fn(*mut GoldyScheme, *const GoldyParcel, GoldyNodeAccess, GoldyResourceAccess) -> GoldyResult;
pub type FnGoldySchemeComputeNodeDeclareParcelView = unsafe extern "C" fn(
    *mut GoldyScheme,
    *const GoldyParcel,
    u32,
    GoldyNodeAccess,
    GoldyResourceAccess,
) -> GoldyResult;
pub type FnGoldySchemeComputeNodeDispatch = unsafe extern "C" fn(*mut GoldyScheme, u32, u32, u32) -> GoldyResult;
pub type FnGoldySchemeSubmit = unsafe extern "C" fn(*mut GoldyScheme, *mut *mut GoldySchemeSubmission) -> GoldyResult;
pub type FnGoldySchemeSubmissionDestroy = unsafe extern "C" fn(*mut GoldySchemeSubmission);
pub type FnGoldySchemeSubmissionTimelineValue = unsafe extern "C" fn(*const GoldySchemeSubmission) -> u64;
pub type FnGoldySchemeSubmissionWait =
    unsafe extern "C" fn(*const GoldyContext, *const GoldySchemeSubmission) -> GoldyResult;
pub type FnGoldySchemeGrantRead = unsafe extern "C" fn(*mut GoldyScheme, *const GoldyParcel) -> *mut GoldyReadGrant;
pub type FnGoldyReadGrantDestroy = unsafe extern "C" fn(*mut GoldyReadGrant);
pub type FnGoldyReadGrantByteSize = unsafe extern "C" fn(*const GoldyReadGrant) -> u64;
pub type FnGoldyReadGrantConsume =
    unsafe extern "C" fn(*const GoldyReadGrant, *const GoldySchemeSubmission, *mut u8, usize) -> GoldyResult;

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
pub type FnGoldyTaskGraphComputeNodeBindParcel =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyParcel, GoldyNodeAccess) -> GoldyResult;
pub type FnGoldyTaskGraphComputeNodeBindResourcesRaw =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const u32, u32) -> GoldyResult;
pub type FnGoldyTaskGraphComputeNodeDispatch = unsafe extern "C" fn(*mut GoldyTaskGraph, u32, u32, u32) -> GoldyResult;
pub type FnGoldyTaskGraphDispatch = unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyDevice) -> GoldyResult;
pub type FnGoldyTaskGraphWriteParcel =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyParcel, u64, *const u8, usize) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassBegin =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const c_char, *const GoldyRenderTarget) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassBindParcel =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyParcel, GoldyNodeAccess) -> GoldyResult;
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
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyParcel, GoldyIndexFormat) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassSetPipeline =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyRenderPipeline) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassSetVertexBufferParcel =
    unsafe extern "C" fn(*mut GoldyTaskGraph, u32, *const GoldyParcel) -> GoldyResult;

pub type FnGoldyRetainedPoolAcquireBuffer =
    unsafe extern "C" fn(*mut GoldyRetainedPool, u64, GoldyBufferKind, u32, *const u8, usize) -> *mut GoldyParcel;
pub type FnGoldyRetainedPoolCreate = unsafe extern "C" fn(*const GoldyDevice) -> *mut GoldyRetainedPool;
pub type FnGoldyRetainedPoolDestroy = unsafe extern "C" fn(*mut GoldyRetainedPool);
pub type FnGoldyMosaicBuilderCreate = unsafe extern "C" fn() -> *mut GoldyMosaicBuilder;
pub type FnGoldyMosaicBuilderDestroy = unsafe extern "C" fn(*mut GoldyMosaicBuilder);
pub type FnGoldyMosaicBuilderEmplace = unsafe extern "C" fn(*mut GoldyMosaicBuilder, *const u8, usize, u64, u32) -> u32;
pub type FnGoldyMosaicBuilderBuild =
    unsafe extern "C" fn(*mut GoldyMosaicBuilder, *mut GoldyRetainedPool) -> *mut GoldyParcel;
pub type FnGoldyParcelByteSize = unsafe extern "C" fn(*const GoldyParcel) -> u64;
pub type FnGoldyParcelDestroy = unsafe extern "C" fn(*mut GoldyParcel);
pub type FnGoldyParcelMosaicViewResourceIndex =
    unsafe extern "C" fn(*const GoldyParcel, u32, GoldyResourceAccess) -> u32;
pub type FnGoldyParcelMosaicViewReadToCpu =
    unsafe extern "C" fn(*const GoldyParcel, u32, *const GoldyDevice, *mut u8, usize) -> GoldyResult;
pub type FnGoldyParcelMosaicViewSize = unsafe extern "C" fn(*const GoldyParcel, u32) -> u64;
pub type FnGoldyTaskGraphComputeNodeBindParcelView =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyParcel, u32, GoldyNodeAccess) -> GoldyResult;
pub type FnGoldyTaskGraphRenderPassBindParcelView =
    unsafe extern "C" fn(*mut GoldyTaskGraph, *const GoldyParcel, u32, GoldyNodeAccess) -> GoldyResult;
