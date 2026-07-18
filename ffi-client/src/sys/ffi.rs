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
pub type FnGoldySchemeComputeNodeWithParcel =
    unsafe extern "C" fn(*mut GoldyScheme, *const GoldyParcel, GoldyNodeAccess) -> GoldyResult;
pub type FnGoldySchemeComputeNodeWithBufferUnit =
    unsafe extern "C" fn(*mut GoldyScheme, *const GoldyBuffer, u32, GoldyNodeAccess) -> GoldyResult;
pub type FnGoldySchemeRenderPassWithBufferUnit =
    unsafe extern "C" fn(*mut GoldyScheme, *const GoldyBuffer, u32, GoldyNodeAccess) -> GoldyResult;
pub type FnGoldySchemeGrantRead = unsafe extern "C" fn(*mut GoldyScheme, *const GoldyBuffer) -> *mut GoldyReadGrant;
pub type FnGoldySchemeComputeNodeWithParam = unsafe extern "C" fn(*mut GoldyScheme, u32) -> GoldyResult;
pub type FnGoldySchemeComputeNodeDispatch = unsafe extern "C" fn(*mut GoldyScheme, u32, u32, u32) -> GoldyResult;
pub type FnGoldySchemeSubmit = unsafe extern "C" fn(*mut GoldyScheme, *mut *mut GoldySchemeSubmission) -> GoldyResult;
pub type FnGoldySchemeSubmissionDestroy = unsafe extern "C" fn(*mut GoldySchemeSubmission);
pub type FnGoldySchemeSubmissionTimelineValue = unsafe extern "C" fn(*const GoldySchemeSubmission) -> u64;
pub type FnGoldySchemeSubmissionWait =
    unsafe extern "C" fn(*const GoldyContext, *const GoldySchemeSubmission) -> GoldyResult;
pub type FnGoldyReadGrantDestroy = unsafe extern "C" fn(*mut GoldyReadGrant);
pub type FnGoldyReadGrantByteSize = unsafe extern "C" fn(*const GoldyReadGrant) -> u64;
pub type FnGoldyReadGrantConsume =
    unsafe extern "C" fn(*const GoldyReadGrant, *const GoldySchemeSubmission, *mut u8, usize) -> GoldyResult;

pub type FnGoldySchemeLeaseRenderTarget = unsafe extern "C" fn(
    *mut GoldyScheme,
    u32,
    u32,
    GoldyTextureFormat,
    bool,
    GoldyDepthFormat,
) -> *mut GoldySchemeRenderTargetLease;
pub type FnGoldySchemeRenderTargetLeaseDestroy = unsafe extern "C" fn(*mut GoldySchemeRenderTargetLease);
pub type FnGoldySchemeRenderPassBegin =
    unsafe extern "C" fn(*mut GoldyScheme, *const c_char, *const GoldySchemeRenderTargetLease) -> GoldyResult;
pub type FnGoldySchemeRenderPassWithParcel =
    unsafe extern "C" fn(*mut GoldyScheme, *const GoldyParcel, GoldyNodeAccess) -> GoldyResult;
pub type FnGoldySchemeRenderPassClear = unsafe extern "C" fn(*mut GoldyScheme, GoldyColor) -> GoldyResult;
pub type FnGoldySchemeRenderPassClearDepth = unsafe extern "C" fn(*mut GoldyScheme, f32) -> GoldyResult;
pub type FnGoldySchemeRenderPassSetPipeline =
    unsafe extern "C" fn(*mut GoldyScheme, *const GoldyRenderPipeline) -> GoldyResult;
pub type FnGoldySchemeRenderPassSetVertexBufferParcel =
    unsafe extern "C" fn(*mut GoldyScheme, u32, *const GoldyParcel) -> GoldyResult;
pub type FnGoldySchemeRenderPassSetIndexBuffer =
    unsafe extern "C" fn(*mut GoldyScheme, *const GoldyParcel, GoldyIndexFormat) -> GoldyResult;
pub type FnGoldySchemeRenderPassDraw = unsafe extern "C" fn(*mut GoldyScheme, u32, u32, u32, u32) -> GoldyResult;
pub type FnGoldySchemeRenderPassDrawIndexed =
    unsafe extern "C" fn(*mut GoldyScheme, u32, u32, c_int, u32, u32) -> GoldyResult;
pub type FnGoldySchemeRenderPassDrawFullscreen = unsafe extern "C" fn(*mut GoldyScheme) -> GoldyResult;
pub type FnGoldySchemeRenderPassFinish = unsafe extern "C" fn(*mut GoldyScheme) -> GoldyResult;
pub type FnGoldySchemeCopyToTexture =
    unsafe extern "C" fn(*mut GoldyScheme, *const GoldySchemeRenderTargetLease, *const GoldyTexture) -> GoldyResult;
pub type FnGoldySchemeGrantReadTexture =
    unsafe extern "C" fn(*mut GoldyScheme, *const GoldyTexture) -> *mut GoldyReadGrant;
pub type FnGoldyRetainedPoolAcquireTexture = unsafe extern "C" fn(
    *mut GoldyRetainedPool,
    u32,
    u32,
    GoldyTextureFormat,
    GoldyTextureKind,
    GoldyTextureFlags,
    *const u8,
    usize,
) -> *mut GoldyTexture;
pub type FnGoldyPresentLeaseDestroy = unsafe extern "C" fn(*mut GoldyPresentLease);

pub type FnGoldySurfaceExchangeDestroy = unsafe extern "C" fn(*mut GoldySurfaceExchange);
pub type FnGoldySurfaceExchangeWidth = unsafe extern "C" fn(*const GoldySurfaceExchange) -> u32;
pub type FnGoldySurfaceExchangeHeight = unsafe extern "C" fn(*const GoldySurfaceExchange) -> u32;
pub type FnGoldySurfaceExchangeFormat = unsafe extern "C" fn(*const GoldySurfaceExchange) -> GoldyTextureFormat;
pub type FnGoldySurfaceExchangeGeneration = unsafe extern "C" fn(*const GoldySurfaceExchange) -> u64;
pub type FnGoldySurfaceExchangeResize = unsafe extern "C" fn(*mut GoldySurfaceExchange, u32, u32) -> GoldyResult;
pub type FnGoldySurfaceExchangeLease = unsafe extern "C" fn(*const GoldySurfaceExchange) -> *mut GoldyPresentLease;
pub type FnGoldySurfaceExchangeBindRenderTarget = unsafe extern "C" fn(
    *const GoldySurfaceExchange,
    *mut GoldyScheme,
    *const GoldySchemeRenderTargetLease,
) -> *mut GoldyTransaction;
pub type FnGoldySurfaceExchangeBind =
    unsafe extern "C" fn(*const GoldySurfaceExchange, *mut GoldyScheme, *const GoldyTexture) -> *mut GoldyTransaction;
pub type FnGoldySurfaceExchangeBindDestination = unsafe extern "C" fn(
    *const GoldySurfaceExchange,
    *mut GoldyScheme,
    *mut GoldySurfaceExchangeBindDestinationOut,
) -> GoldyResult;
pub type FnGoldyTransactionDestroy = unsafe extern "C" fn(*mut GoldyTransaction);
pub type FnGoldyTransactionBindingId = unsafe extern "C" fn(*const GoldyTransaction) -> u32;
pub type FnGoldyTransactionGeneration = unsafe extern "C" fn(*const GoldyTransaction) -> u64;
pub type FnGoldyTransactionClaim =
    unsafe extern "C" fn(*const GoldyTransaction, *mut GoldySchemeSubmission) -> *mut GoldyClaim;
pub type FnGoldyClaimDestroy = unsafe extern "C" fn(*mut GoldyClaim);
pub type FnGoldyClaimConsume = unsafe extern "C" fn(*mut GoldyClaim) -> GoldyResult;
pub type FnGoldyClaimDiscard = unsafe extern "C" fn(*mut GoldyClaim) -> GoldyResult;
#[cfg(windows)]
pub type FnGoldySurfaceExchangeCreateWin32 =
    unsafe extern "C" fn(*const GoldyContext, *mut c_void, u32) -> *mut GoldySurfaceExchange;
#[cfg(target_os = "macos")]
pub type FnGoldySurfaceExchangeCreateAppkit =
    unsafe extern "C" fn(*const GoldyContext, *mut c_void, u32) -> *mut GoldySurfaceExchange;
#[cfg(target_os = "linux")]
pub type FnGoldySurfaceExchangeCreateWayland =
    unsafe extern "C" fn(*const GoldyContext, *mut c_void, *mut c_void, u32) -> *mut GoldySurfaceExchange;

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
pub type FnGoldySurfaceWidth = unsafe extern "C" fn(*const GoldySurface) -> u32;

pub type FnGoldyRetainedPoolAcquireBuffer =
    unsafe extern "C" fn(*mut GoldyRetainedPool, u64, GoldyBufferKind, u32, *const u8, usize) -> *mut GoldyBuffer;
pub type FnGoldyRetainedPoolCreate = unsafe extern "C" fn(*const GoldyDevice) -> *mut GoldyRetainedPool;
pub type FnGoldyRetainedPoolDestroy = unsafe extern "C" fn(*mut GoldyRetainedPool);
pub type FnGoldyRecordBuilderCreate = unsafe extern "C" fn() -> *mut GoldyRecordBuilder;
pub type FnGoldyRecordBuilderDestroy = unsafe extern "C" fn(*mut GoldyRecordBuilder);
pub type FnGoldyRecordBuilderEmplace =
    unsafe extern "C" fn(*mut GoldyRecordBuilder, *const c_char, *const u8, usize, u64, u32) -> u32;
pub type FnGoldyRecordBuilderBuild =
    unsafe extern "C" fn(*mut GoldyRecordBuilder, *mut GoldyRetainedPool) -> *mut GoldyBuffer;
pub type FnGoldyBufferDestroy = unsafe extern "C" fn(*mut GoldyBuffer);
pub type FnGoldyBufferByteSize = unsafe extern "C" fn(*const GoldyBuffer) -> u64;
pub type FnGoldyBufferUnitCount = unsafe extern "C" fn(*const GoldyBuffer) -> u32;
pub type FnGoldyBufferUnitByteSize = unsafe extern "C" fn(*const GoldyBuffer, u32) -> u64;
pub type FnGoldyBufferUnitReadToCpu =
    unsafe extern "C" fn(*const GoldyBuffer, u32, *const GoldyDevice, *mut u8, usize) -> GoldyResult;
pub type FnGoldyBufferField = unsafe extern "C" fn(*const GoldyBuffer, u32) -> *mut GoldyParcel;
pub type FnGoldyTextureByteSize = unsafe extern "C" fn(*const GoldyTexture) -> u64;
pub type FnGoldyTextureDestroy = unsafe extern "C" fn(*mut GoldyTexture);
pub type FnGoldyParcelByteSize = unsafe extern "C" fn(*const GoldyParcel) -> u64;
pub type FnGoldyParcelDestroy = unsafe extern "C" fn(*mut GoldyParcel);
