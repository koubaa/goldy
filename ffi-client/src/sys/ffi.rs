//! Function pointer types for the Goldy C ABI.

use super::types::*;
use std::ffi::{c_char, c_void};
use std::os::raw::c_int;

pub type FnGoldyClearError = unsafe extern "C" fn();

pub type FnGoldyComputePipelineCreate =
    unsafe extern "C" fn(*const GoldyRuntime, *const GoldyShaderModule) -> *mut GoldyComputePipeline;
pub type FnGoldyComputePipelineDestroy = unsafe extern "C" fn(*mut GoldyComputePipeline);

pub type FnGoldyContextCreate = unsafe extern "C" fn(*const GoldyRuntime) -> *mut GoldyContext;
pub type FnGoldyContextDestroy = unsafe extern "C" fn(*mut GoldyContext);
pub type FnGoldyContextLeaseRenderTarget = unsafe extern "C" fn(
    *const GoldyContext,
    u32,
    u32,
    GoldyTextureFormat,
    bool,
    GoldyDepthFormat,
) -> *mut GoldySchemeRenderTargetLease;

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
pub type FnGoldySchemeComputeNodeWithParam = unsafe extern "C" fn(*mut GoldyScheme, u32) -> GoldyResult;
pub type FnGoldySchemeComputeNodeDispatch = unsafe extern "C" fn(*mut GoldyScheme, u32, u32, u32) -> GoldyResult;
pub type FnGoldySchemeSubmit = unsafe extern "C" fn(*mut GoldyScheme, *mut *mut GoldySchemeSubmission) -> GoldyResult;
pub type FnGoldySchemeSubmissionDestroy = unsafe extern "C" fn(*mut GoldySchemeSubmission);
pub type FnGoldySchemeSubmissionIsSettled = unsafe extern "C" fn(*const GoldySchemeSubmission) -> bool;
pub type FnGoldySchemeSubmissionWaitUntilSettled = unsafe extern "C" fn(*const GoldySchemeSubmission) -> GoldyResult;

pub type FnGoldySchemeSubmissionTake =
    unsafe extern "C" fn(*mut GoldySchemeSubmission, *const GoldyParcel) -> *mut GoldyHostView;
pub type FnGoldySchemeSubmissionTakeTexture =
    unsafe extern "C" fn(*mut GoldySchemeSubmission, *const GoldyTexture) -> *mut GoldyHostView;
pub type FnGoldyHostViewLen = unsafe extern "C" fn(*const GoldyHostView) -> u64;
pub type FnGoldyHostViewData = unsafe extern "C" fn(*const GoldyHostView) -> *const u8;
pub type FnGoldyHostViewCopy = unsafe extern "C" fn(*const GoldyHostView, *mut u8, usize) -> GoldyResult;
pub type FnGoldyHostViewDestroy = unsafe extern "C" fn(*mut GoldyHostView);
pub type FnGoldyMemoryExchangeCreate = unsafe extern "C" fn(*const GoldyContext) -> *mut GoldyMemoryExchange;
pub type FnGoldyMemoryExchangeDestroy = unsafe extern "C" fn(*mut GoldyMemoryExchange);
pub type FnGoldyMemoryExchangeBindDeposit = unsafe extern "C" fn(
    *const GoldyMemoryExchange,
    *mut GoldyScheme,
    *const GoldyDepositTarget,
) -> *mut GoldyDepositTransaction;
pub type FnGoldyDepositTransactionDestroy = unsafe extern "C" fn(*mut GoldyDepositTransaction);
pub type FnGoldyDepositTransactionCapacity = unsafe extern "C" fn(*const GoldyDepositTransaction) -> u64;
pub type FnGoldyDepositTransactionId = unsafe extern "C" fn(*const GoldyDepositTransaction) -> u32;
pub type FnGoldyDepositTransactionWrite =
    unsafe extern "C" fn(*const GoldyDepositTransaction, u64, *const u8, usize) -> GoldyResult;

pub type FnGoldySchemeRenderTargetLeaseDestroy = unsafe extern "C" fn(*mut GoldySchemeRenderTargetLease);
pub type FnGoldySchemeRenderPassBegin = unsafe extern "C" fn(
    *mut GoldyScheme,
    *const c_char,
    *const GoldySchemeRenderTargetLease,
    GoldyTargetLoad,
    GoldyColor,
) -> GoldyResult;
pub type FnGoldySchemeRenderPassWithParcel =
    unsafe extern "C" fn(*mut GoldyScheme, *const GoldyParcel, GoldyNodeAccess) -> GoldyResult;
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
pub type FnGoldyRuntimeAcquireTexture = unsafe extern "C" fn(
    *mut GoldyRuntime,
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

pub type FnGoldyRuntimeAdapterId = unsafe extern "C" fn(*const GoldyRuntime) -> u32;
pub type FnGoldyRuntimeDestroy = unsafe extern "C" fn(*mut GoldyRuntime);
pub type FnGoldyRuntimeHasLibrary = unsafe extern "C" fn(*const GoldyRuntime, *const c_char) -> bool;
pub type FnGoldyRuntimeIsValid = unsafe extern "C" fn(*const GoldyRuntime) -> bool;

pub type FnGoldyGetLastError = unsafe extern "C" fn() -> *const c_char;

pub type FnGoldyInstanceAdapterCount = unsafe extern "C" fn(*const GoldyInstance) -> u32;
pub type FnGoldyInstanceBackendType = unsafe extern "C" fn(*const GoldyInstance) -> GoldyBackendType;
pub type FnGoldyInstanceCreate = unsafe extern "C" fn() -> *mut GoldyInstance;
pub type FnGoldyInstanceCreateDeviceForAdapter = unsafe extern "C" fn(*const GoldyInstance, u32) -> *mut GoldyRuntime;
pub type FnGoldyInstanceDestroy = unsafe extern "C" fn(*mut GoldyInstance);
pub type FnGoldyInstanceGetAdapter =
    unsafe extern "C" fn(*const GoldyInstance, u32, *mut GoldyAdapterInfo) -> GoldyResult;

pub type FnGoldyRenderPipelineCreate = unsafe extern "C" fn(
    *const GoldyRuntime,
    *const GoldyShaderModule,
    *const GoldyShaderModule,
    *const GoldyRenderPipelineDesc,
) -> *mut GoldyRenderPipeline;
pub type FnGoldyRenderPipelineDestroy = unsafe extern "C" fn(*mut GoldyRenderPipeline);

pub type FnGoldySamplerCreate = unsafe extern "C" fn(*const GoldyRuntime, *const GoldySamplerDesc) -> *mut GoldySampler;
pub type FnGoldySamplerCreateDefault = unsafe extern "C" fn(*const GoldyRuntime) -> *mut GoldySampler;
pub type FnGoldySamplerDestroy = unsafe extern "C" fn(*mut GoldySampler);

pub type FnGoldyShaderBuiltinVertexColor2d = unsafe extern "C" fn() -> *const c_char;
pub type FnGoldyShaderCreate = unsafe extern "C" fn(*const GoldyRuntime, *const c_char) -> *mut GoldyShaderModule;
pub type FnGoldyShaderDestroy = unsafe extern "C" fn(*mut GoldyShaderModule);

pub type FnGoldyRuntimeAcquireBuffer =
    unsafe extern "C" fn(*mut GoldyRuntime, u64, GoldyBufferKind, u32, *const u8, usize) -> *mut GoldyBuffer;
pub type FnGoldyRecordBuilderCreate = unsafe extern "C" fn() -> *mut GoldyRecordBuilder;
pub type FnGoldyRecordBuilderDestroy = unsafe extern "C" fn(*mut GoldyRecordBuilder);
pub type FnGoldyRecordBuilderEmplace =
    unsafe extern "C" fn(*mut GoldyRecordBuilder, *const c_char, *const u8, usize, u64, u32) -> u32;
pub type FnGoldyRecordBuilderBuild =
    unsafe extern "C" fn(*mut GoldyRecordBuilder, *mut GoldyRuntime) -> *mut GoldyBuffer;
pub type FnGoldyBufferDestroy = unsafe extern "C" fn(*mut GoldyBuffer);
pub type FnGoldyBufferByteSize = unsafe extern "C" fn(*const GoldyBuffer) -> u64;
pub type FnGoldyBufferUnitCount = unsafe extern "C" fn(*const GoldyBuffer) -> u32;
pub type FnGoldyBufferUnitByteSize = unsafe extern "C" fn(*const GoldyBuffer, u32) -> u64;
pub type FnGoldyBufferField = unsafe extern "C" fn(*const GoldyBuffer, u32) -> *mut GoldyParcel;
pub type FnGoldyTextureByteSize = unsafe extern "C" fn(*const GoldyTexture) -> u64;
pub type FnGoldyTextureDestroy = unsafe extern "C" fn(*mut GoldyTexture);
pub type FnGoldyParcelByteSize = unsafe extern "C" fn(*const GoldyParcel) -> u64;
pub type FnGoldyParcelDestroy = unsafe extern "C" fn(*mut GoldyParcel);

#[cfg(feature = "tensor")]
pub type FnGoldyRuntimeAcquireTensor =
    unsafe extern "C" fn(*mut GoldyRuntime, GoldyTensorDType, u32, *const u32, *const u8, usize) -> *mut GoldyTensor;
#[cfg(feature = "tensor")]
pub type FnGoldyTensorDestroy = unsafe extern "C" fn(*mut GoldyTensor);
#[cfg(feature = "tensor")]
pub type FnGoldyTensorDtype = unsafe extern "C" fn(*const GoldyTensor) -> GoldyTensorDType;
#[cfg(feature = "tensor")]
pub type FnGoldyTensorShape = unsafe extern "C" fn(*const GoldyTensor, *mut GoldyTensorShape) -> GoldyResult;
#[cfg(feature = "tensor")]
pub type FnGoldyTensorKernelsCreate = unsafe extern "C" fn(*mut GoldyRuntime) -> *mut GoldyTensorKernels;
#[cfg(feature = "tensor")]
pub type FnGoldyTensorKernelsDestroy = unsafe extern "C" fn(*mut GoldyTensorKernels);
#[cfg(feature = "tensor")]
pub type FnGoldyTensorAdd = unsafe extern "C" fn(
    *mut GoldyTensorKernels,
    *mut GoldyScheme,
    *const c_char,
    *const GoldyTensor,
    *const GoldyTensor,
) -> *mut GoldyTensor;
#[cfg(feature = "tensor")]
pub type FnGoldyTensorMatmul = unsafe extern "C" fn(
    *mut GoldyTensorKernels,
    *mut GoldyScheme,
    *const c_char,
    *const GoldyTensor,
    *const GoldyTensor,
) -> *mut GoldyTensor;
#[cfg(feature = "tensor")]
pub type FnGoldyTensorFillF32 = unsafe extern "C" fn(
    *mut GoldyTensorKernels,
    *mut GoldyScheme,
    *const c_char,
    *mut GoldyTensor,
    f32,
) -> GoldyResult;
