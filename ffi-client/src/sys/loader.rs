//! Runtime loading of `goldy_ffi` via libloading (LoadLibrary / dlopen).

use super::ffi::*;
use libloading::Library;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[cfg(windows)]
#[link(name = "kernel32", kind = "dylib")]
extern "system" {
    fn SetDllDirectoryW(lp_path_name: *const u16) -> i32;
}

static GOLDY_FFI: OnceLock<GoldyFfi> = OnceLock::new();

pub(crate) fn lib() -> &'static GoldyFfi {
    GOLDY_FFI.get_or_init(|| GoldyFfi::load().unwrap_or_else(|e| panic!("failed to load goldy_ffi: {e}")))
}

pub(crate) struct GoldyFfi {
    _library: Library,
    pub goldy_clear_error: FnGoldyClearError,
    pub goldy_compute_pipeline_create: FnGoldyComputePipelineCreate,
    pub goldy_compute_pipeline_destroy: FnGoldyComputePipelineDestroy,
    pub goldy_context_create: FnGoldyContextCreate,
    pub goldy_context_destroy: FnGoldyContextDestroy,
    pub goldy_device_adapter_id: FnGoldyDeviceAdapterId,
    pub goldy_device_destroy: FnGoldyDeviceDestroy,
    pub goldy_device_has_library: FnGoldyDeviceHasLibrary,
    pub goldy_device_is_valid: FnGoldyDeviceIsValid,
    pub goldy_get_last_error: FnGoldyGetLastError,
    pub goldy_instance_adapter_count: FnGoldyInstanceAdapterCount,
    pub goldy_instance_backend_type: FnGoldyInstanceBackendType,
    pub goldy_instance_create: FnGoldyInstanceCreate,
    pub goldy_instance_create_device_for_adapter: FnGoldyInstanceCreateDeviceForAdapter,
    pub goldy_instance_destroy: FnGoldyInstanceDestroy,
    pub goldy_instance_get_adapter: FnGoldyInstanceGetAdapter,
    pub goldy_render_pipeline_create: FnGoldyRenderPipelineCreate,
    pub goldy_render_pipeline_destroy: FnGoldyRenderPipelineDestroy,
    pub goldy_sampler_create: FnGoldySamplerCreate,
    pub goldy_sampler_create_default: FnGoldySamplerCreateDefault,
    pub goldy_sampler_destroy: FnGoldySamplerDestroy,
    pub goldy_shader_builtin_vertex_color_2d: FnGoldyShaderBuiltinVertexColor2d,
    pub goldy_shader_create: FnGoldyShaderCreate,
    pub goldy_shader_destroy: FnGoldyShaderDestroy,
    pub goldy_retained_pool_acquire_buffer: FnGoldyRetainedPoolAcquireBuffer,
    pub goldy_retained_pool_create: FnGoldyRetainedPoolCreate,
    pub goldy_retained_pool_destroy: FnGoldyRetainedPoolDestroy,
    pub goldy_record_builder_create: FnGoldyRecordBuilderCreate,
    pub goldy_record_builder_destroy: FnGoldyRecordBuilderDestroy,
    pub goldy_record_builder_emplace: FnGoldyRecordBuilderEmplace,
    pub goldy_record_builder_build: FnGoldyRecordBuilderBuild,
    pub goldy_buffer_destroy: FnGoldyBufferDestroy,
    pub goldy_buffer_byte_size: FnGoldyBufferByteSize,
    pub goldy_buffer_unit_count: FnGoldyBufferUnitCount,
    pub goldy_buffer_unit_byte_size: FnGoldyBufferUnitByteSize,
    pub goldy_buffer_unit_read_to_cpu: FnGoldyBufferUnitReadToCpu,
    pub goldy_buffer_field: FnGoldyBufferField,
    pub goldy_texture_byte_size: FnGoldyTextureByteSize,
    pub goldy_texture_destroy: FnGoldyTextureDestroy,
    pub goldy_parcel_byte_size: FnGoldyParcelByteSize,
    pub goldy_parcel_destroy: FnGoldyParcelDestroy,
    pub goldy_scheme_create: FnGoldySchemeCreate,
    pub goldy_scheme_destroy: FnGoldySchemeDestroy,
    pub goldy_scheme_len: FnGoldySchemeLen,
    pub goldy_scheme_is_dirty: FnGoldySchemeIsDirty,
    pub goldy_scheme_replay_stats: FnGoldySchemeReplayStats,
    pub goldy_scheme_compute_node_begin: FnGoldySchemeComputeNodeBegin,
    pub goldy_scheme_compute_node_with_parcel: FnGoldySchemeComputeNodeWithParcel,
    pub goldy_scheme_compute_node_with_buffer_unit: FnGoldySchemeComputeNodeWithBufferUnit,
    pub goldy_scheme_compute_node_with_param: FnGoldySchemeComputeNodeWithParam,
    pub goldy_scheme_compute_node_dispatch: FnGoldySchemeComputeNodeDispatch,
    pub goldy_scheme_submit: FnGoldySchemeSubmit,
    pub goldy_scheme_submission_destroy: FnGoldySchemeSubmissionDestroy,
    pub goldy_scheme_submission_timeline_value: FnGoldySchemeSubmissionTimelineValue,
    pub goldy_scheme_submission_wait: FnGoldySchemeSubmissionWait,
    pub goldy_scheme_grant_read: FnGoldySchemeGrantRead,
    pub goldy_read_grant_destroy: FnGoldyReadGrantDestroy,
    pub goldy_read_grant_byte_size: FnGoldyReadGrantByteSize,
    pub goldy_read_grant_consume: FnGoldyReadGrantConsume,
    pub goldy_scheme_lease_render_target: FnGoldySchemeLeaseRenderTarget,
    pub goldy_scheme_render_target_lease_destroy: FnGoldySchemeRenderTargetLeaseDestroy,
    pub goldy_scheme_render_pass_begin: FnGoldySchemeRenderPassBegin,
    pub goldy_scheme_render_pass_with_buffer_unit: FnGoldySchemeRenderPassWithBufferUnit,
    pub goldy_scheme_render_pass_with_parcel: FnGoldySchemeRenderPassWithParcel,
    pub goldy_scheme_render_pass_clear: FnGoldySchemeRenderPassClear,
    pub goldy_scheme_render_pass_clear_depth: FnGoldySchemeRenderPassClearDepth,
    pub goldy_scheme_render_pass_set_pipeline: FnGoldySchemeRenderPassSetPipeline,
    pub goldy_scheme_render_pass_set_vertex_buffer_parcel: FnGoldySchemeRenderPassSetVertexBufferParcel,
    pub goldy_scheme_render_pass_set_index_buffer: FnGoldySchemeRenderPassSetIndexBuffer,
    pub goldy_scheme_render_pass_draw: FnGoldySchemeRenderPassDraw,
    pub goldy_scheme_render_pass_draw_indexed: FnGoldySchemeRenderPassDrawIndexed,
    pub goldy_scheme_render_pass_draw_fullscreen: FnGoldySchemeRenderPassDrawFullscreen,
    pub goldy_scheme_render_pass_finish: FnGoldySchemeRenderPassFinish,
    pub goldy_scheme_copy_to_texture: FnGoldySchemeCopyToTexture,
    pub goldy_scheme_grant_read_texture: FnGoldySchemeGrantReadTexture,
    pub goldy_retained_pool_acquire_texture: FnGoldyRetainedPoolAcquireTexture,
    pub goldy_present_lease_destroy: FnGoldyPresentLeaseDestroy,
    pub goldy_surface_exchange_destroy: FnGoldySurfaceExchangeDestroy,
    pub goldy_surface_exchange_width: FnGoldySurfaceExchangeWidth,
    pub goldy_surface_exchange_height: FnGoldySurfaceExchangeHeight,
    pub goldy_surface_exchange_format: FnGoldySurfaceExchangeFormat,
    pub goldy_surface_exchange_generation: FnGoldySurfaceExchangeGeneration,
    pub goldy_surface_exchange_resize: FnGoldySurfaceExchangeResize,
    pub goldy_surface_exchange_lease: FnGoldySurfaceExchangeLease,
    pub goldy_surface_exchange_bind_render_target: FnGoldySurfaceExchangeBindRenderTarget,
    pub goldy_surface_exchange_bind: FnGoldySurfaceExchangeBind,
    pub goldy_surface_exchange_bind_destination: FnGoldySurfaceExchangeBindDestination,
    pub goldy_transaction_destroy: FnGoldyTransactionDestroy,
    pub goldy_transaction_binding_id: FnGoldyTransactionBindingId,
    pub goldy_transaction_generation: FnGoldyTransactionGeneration,
    pub goldy_transaction_claim: FnGoldyTransactionClaim,
    pub goldy_claim_destroy: FnGoldyClaimDestroy,
    pub goldy_claim_consume: FnGoldyClaimConsume,
    pub goldy_claim_discard: FnGoldyClaimDiscard,
    #[cfg(windows)]
    pub goldy_surface_exchange_create_win32: FnGoldySurfaceExchangeCreateWin32,
    #[cfg(target_os = "macos")]
    pub goldy_surface_exchange_create_appkit: FnGoldySurfaceExchangeCreateAppkit,
    #[cfg(target_os = "linux")]
    pub goldy_surface_exchange_create_wayland: FnGoldySurfaceExchangeCreateWayland,
}

impl GoldyFfi {
    fn load() -> Result<Self, String> {
        let lib_path = find_library()?;
        #[cfg(windows)]
        set_dll_directory_for_dependencies(&lib_path)?;

        let library =
            unsafe { Library::new(&lib_path) }.map_err(|e| format!("failed to open {}: {e}", lib_path.display()))?;

        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let symbol: libloading::Symbol<$ty> = unsafe {
                    library
                        .get($name.as_bytes())
                        .map_err(|e| format!("missing symbol {}: {e}", $name))?
                };
                *symbol
            }};
        }

        Ok(Self {
            goldy_clear_error: sym!("goldy_clear_error", FnGoldyClearError),
            goldy_compute_pipeline_create: sym!("goldy_compute_pipeline_create", FnGoldyComputePipelineCreate),
            goldy_compute_pipeline_destroy: sym!("goldy_compute_pipeline_destroy", FnGoldyComputePipelineDestroy),
            goldy_context_create: sym!("goldy_context_create", FnGoldyContextCreate),
            goldy_context_destroy: sym!("goldy_context_destroy", FnGoldyContextDestroy),
            goldy_device_adapter_id: sym!("goldy_device_adapter_id", FnGoldyDeviceAdapterId),
            goldy_device_destroy: sym!("goldy_device_destroy", FnGoldyDeviceDestroy),
            goldy_device_has_library: sym!("goldy_device_has_library", FnGoldyDeviceHasLibrary),
            goldy_device_is_valid: sym!("goldy_device_is_valid", FnGoldyDeviceIsValid),
            goldy_get_last_error: sym!("goldy_get_last_error", FnGoldyGetLastError),
            goldy_instance_adapter_count: sym!("goldy_instance_adapter_count", FnGoldyInstanceAdapterCount),
            goldy_instance_backend_type: sym!("goldy_instance_backend_type", FnGoldyInstanceBackendType),
            goldy_instance_create: sym!("goldy_instance_create", FnGoldyInstanceCreate),
            goldy_instance_create_device_for_adapter: sym!(
                "goldy_instance_create_device_for_adapter",
                FnGoldyInstanceCreateDeviceForAdapter
            ),
            goldy_instance_destroy: sym!("goldy_instance_destroy", FnGoldyInstanceDestroy),
            goldy_instance_get_adapter: sym!("goldy_instance_get_adapter", FnGoldyInstanceGetAdapter),
            goldy_render_pipeline_create: sym!("goldy_render_pipeline_create", FnGoldyRenderPipelineCreate),
            goldy_render_pipeline_destroy: sym!("goldy_render_pipeline_destroy", FnGoldyRenderPipelineDestroy),
            goldy_sampler_create: sym!("goldy_sampler_create", FnGoldySamplerCreate),
            goldy_sampler_create_default: sym!("goldy_sampler_create_default", FnGoldySamplerCreateDefault),
            goldy_sampler_destroy: sym!("goldy_sampler_destroy", FnGoldySamplerDestroy),
            goldy_shader_builtin_vertex_color_2d: sym!(
                "goldy_shader_builtin_vertex_color_2d",
                FnGoldyShaderBuiltinVertexColor2d
            ),
            goldy_shader_create: sym!("goldy_shader_create", FnGoldyShaderCreate),
            goldy_shader_destroy: sym!("goldy_shader_destroy", FnGoldyShaderDestroy),
            goldy_retained_pool_acquire_buffer: sym!(
                "goldy_retained_pool_acquire_buffer",
                FnGoldyRetainedPoolAcquireBuffer
            ),
            goldy_retained_pool_create: sym!("goldy_retained_pool_create", FnGoldyRetainedPoolCreate),
            goldy_retained_pool_destroy: sym!("goldy_retained_pool_destroy", FnGoldyRetainedPoolDestroy),
            goldy_record_builder_create: sym!("goldy_record_builder_create", FnGoldyRecordBuilderCreate),
            goldy_record_builder_destroy: sym!("goldy_record_builder_destroy", FnGoldyRecordBuilderDestroy),
            goldy_record_builder_emplace: sym!("goldy_record_builder_emplace", FnGoldyRecordBuilderEmplace),
            goldy_record_builder_build: sym!("goldy_record_builder_build", FnGoldyRecordBuilderBuild),
            goldy_buffer_destroy: sym!("goldy_buffer_destroy", FnGoldyBufferDestroy),
            goldy_buffer_byte_size: sym!("goldy_buffer_byte_size", FnGoldyBufferByteSize),
            goldy_buffer_unit_count: sym!("goldy_buffer_unit_count", FnGoldyBufferUnitCount),
            goldy_buffer_unit_byte_size: sym!("goldy_buffer_unit_byte_size", FnGoldyBufferUnitByteSize),
            goldy_buffer_unit_read_to_cpu: sym!("goldy_buffer_unit_read_to_cpu", FnGoldyBufferUnitReadToCpu),
            goldy_buffer_field: sym!("goldy_buffer_field", FnGoldyBufferField),
            goldy_texture_byte_size: sym!("goldy_texture_byte_size", FnGoldyTextureByteSize),
            goldy_texture_destroy: sym!("goldy_texture_destroy", FnGoldyTextureDestroy),
            goldy_parcel_byte_size: sym!("goldy_parcel_byte_size", FnGoldyParcelByteSize),
            goldy_parcel_destroy: sym!("goldy_parcel_destroy", FnGoldyParcelDestroy),
            goldy_scheme_create: sym!("goldy_scheme_create", FnGoldySchemeCreate),
            goldy_scheme_destroy: sym!("goldy_scheme_destroy", FnGoldySchemeDestroy),
            goldy_scheme_len: sym!("goldy_scheme_len", FnGoldySchemeLen),
            goldy_scheme_is_dirty: sym!("goldy_scheme_is_dirty", FnGoldySchemeIsDirty),
            goldy_scheme_replay_stats: sym!("goldy_scheme_replay_stats", FnGoldySchemeReplayStats),
            goldy_scheme_compute_node_begin: sym!("goldy_scheme_compute_node_begin", FnGoldySchemeComputeNodeBegin),
            goldy_scheme_compute_node_with_parcel: sym!(
                "goldy_scheme_compute_node_with_parcel",
                FnGoldySchemeComputeNodeWithParcel
            ),
            goldy_scheme_compute_node_with_buffer_unit: sym!(
                "goldy_scheme_compute_node_with_buffer_unit",
                FnGoldySchemeComputeNodeWithBufferUnit
            ),
            goldy_scheme_compute_node_with_param: sym!(
                "goldy_scheme_compute_node_with_param",
                FnGoldySchemeComputeNodeWithParam
            ),
            goldy_scheme_compute_node_dispatch: sym!(
                "goldy_scheme_compute_node_dispatch",
                FnGoldySchemeComputeNodeDispatch
            ),
            goldy_scheme_submit: sym!("goldy_scheme_submit", FnGoldySchemeSubmit),
            goldy_scheme_submission_destroy: sym!("goldy_scheme_submission_destroy", FnGoldySchemeSubmissionDestroy),
            goldy_scheme_submission_timeline_value: sym!(
                "goldy_scheme_submission_timeline_value",
                FnGoldySchemeSubmissionTimelineValue
            ),
            goldy_scheme_submission_wait: sym!("goldy_scheme_submission_wait", FnGoldySchemeSubmissionWait),
            goldy_scheme_grant_read: sym!("goldy_scheme_grant_read", FnGoldySchemeGrantRead),
            goldy_read_grant_destroy: sym!("goldy_read_grant_destroy", FnGoldyReadGrantDestroy),
            goldy_read_grant_byte_size: sym!("goldy_read_grant_byte_size", FnGoldyReadGrantByteSize),
            goldy_read_grant_consume: sym!("goldy_read_grant_consume", FnGoldyReadGrantConsume),
            goldy_scheme_lease_render_target: sym!("goldy_scheme_lease_render_target", FnGoldySchemeLeaseRenderTarget),
            goldy_scheme_render_target_lease_destroy: sym!(
                "goldy_scheme_render_target_lease_destroy",
                FnGoldySchemeRenderTargetLeaseDestroy
            ),
            goldy_scheme_render_pass_begin: sym!("goldy_scheme_render_pass_begin", FnGoldySchemeRenderPassBegin),
            goldy_scheme_render_pass_with_buffer_unit: sym!(
                "goldy_scheme_render_pass_with_buffer_unit",
                FnGoldySchemeRenderPassWithBufferUnit
            ),
            goldy_scheme_render_pass_with_parcel: sym!(
                "goldy_scheme_render_pass_with_parcel",
                FnGoldySchemeRenderPassWithParcel
            ),
            goldy_scheme_render_pass_clear: sym!("goldy_scheme_render_pass_clear", FnGoldySchemeRenderPassClear),
            goldy_scheme_render_pass_clear_depth: sym!(
                "goldy_scheme_render_pass_clear_depth",
                FnGoldySchemeRenderPassClearDepth
            ),
            goldy_scheme_render_pass_set_pipeline: sym!(
                "goldy_scheme_render_pass_set_pipeline",
                FnGoldySchemeRenderPassSetPipeline
            ),
            goldy_scheme_render_pass_set_vertex_buffer_parcel: sym!(
                "goldy_scheme_render_pass_set_vertex_buffer_parcel",
                FnGoldySchemeRenderPassSetVertexBufferParcel
            ),
            goldy_scheme_render_pass_set_index_buffer: sym!(
                "goldy_scheme_render_pass_set_index_buffer",
                FnGoldySchemeRenderPassSetIndexBuffer
            ),
            goldy_scheme_render_pass_draw: sym!("goldy_scheme_render_pass_draw", FnGoldySchemeRenderPassDraw),
            goldy_scheme_render_pass_draw_indexed: sym!(
                "goldy_scheme_render_pass_draw_indexed",
                FnGoldySchemeRenderPassDrawIndexed
            ),
            goldy_scheme_render_pass_draw_fullscreen: sym!(
                "goldy_scheme_render_pass_draw_fullscreen",
                FnGoldySchemeRenderPassDrawFullscreen
            ),
            goldy_scheme_render_pass_finish: sym!("goldy_scheme_render_pass_finish", FnGoldySchemeRenderPassFinish),
            goldy_scheme_copy_to_texture: sym!("goldy_scheme_copy_to_texture", FnGoldySchemeCopyToTexture),
            goldy_scheme_grant_read_texture: sym!("goldy_scheme_grant_read_texture", FnGoldySchemeGrantReadTexture),
            goldy_retained_pool_acquire_texture: sym!(
                "goldy_retained_pool_acquire_texture",
                FnGoldyRetainedPoolAcquireTexture
            ),
            goldy_present_lease_destroy: sym!("goldy_present_lease_destroy", FnGoldyPresentLeaseDestroy),
            goldy_surface_exchange_destroy: sym!("goldy_surface_exchange_destroy", FnGoldySurfaceExchangeDestroy),
            goldy_surface_exchange_width: sym!("goldy_surface_exchange_width", FnGoldySurfaceExchangeWidth),
            goldy_surface_exchange_height: sym!("goldy_surface_exchange_height", FnGoldySurfaceExchangeHeight),
            goldy_surface_exchange_format: sym!("goldy_surface_exchange_format", FnGoldySurfaceExchangeFormat),
            goldy_surface_exchange_generation: sym!(
                "goldy_surface_exchange_generation",
                FnGoldySurfaceExchangeGeneration
            ),
            goldy_surface_exchange_resize: sym!("goldy_surface_exchange_resize", FnGoldySurfaceExchangeResize),
            goldy_surface_exchange_lease: sym!("goldy_surface_exchange_lease", FnGoldySurfaceExchangeLease),
            goldy_surface_exchange_bind_render_target: sym!(
                "goldy_surface_exchange_bind_render_target",
                FnGoldySurfaceExchangeBindRenderTarget
            ),
            goldy_surface_exchange_bind: sym!("goldy_surface_exchange_bind", FnGoldySurfaceExchangeBind),
            goldy_surface_exchange_bind_destination: sym!(
                "goldy_surface_exchange_bind_destination",
                FnGoldySurfaceExchangeBindDestination
            ),
            goldy_transaction_destroy: sym!("goldy_transaction_destroy", FnGoldyTransactionDestroy),
            goldy_transaction_binding_id: sym!("goldy_transaction_binding_id", FnGoldyTransactionBindingId),
            goldy_transaction_generation: sym!("goldy_transaction_generation", FnGoldyTransactionGeneration),
            goldy_transaction_claim: sym!("goldy_transaction_claim", FnGoldyTransactionClaim),
            goldy_claim_destroy: sym!("goldy_claim_destroy", FnGoldyClaimDestroy),
            goldy_claim_consume: sym!("goldy_claim_consume", FnGoldyClaimConsume),
            goldy_claim_discard: sym!("goldy_claim_discard", FnGoldyClaimDiscard),
            #[cfg(windows)]
            goldy_surface_exchange_create_win32: sym!(
                "goldy_surface_exchange_create_win32",
                FnGoldySurfaceExchangeCreateWin32
            ),
            #[cfg(target_os = "macos")]
            goldy_surface_exchange_create_appkit: sym!(
                "goldy_surface_exchange_create_appkit",
                FnGoldySurfaceExchangeCreateAppkit
            ),
            #[cfg(target_os = "linux")]
            goldy_surface_exchange_create_wayland: sym!(
                "goldy_surface_exchange_create_wayland",
                FnGoldySurfaceExchangeCreateWayland
            ),
            _library: library,
        })
    }
}

fn library_filename() -> &'static str {
    if cfg!(target_os = "macos") {
        "libgoldy_ffi.dylib"
    } else if cfg!(target_os = "windows") {
        "goldy_ffi.dll"
    } else {
        "libgoldy_ffi.so"
    }
}

fn find_library() -> Result<PathBuf, String> {
    if let Ok(path) = env::var("GOLDY_FFI_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!("GOLDY_FFI_PATH does not exist: {}", path.display()));
    }

    if let Some(dir) = option_env!("GOLDY_FFI_LIB_DIR") {
        let path = PathBuf::from(dir).join(library_filename());
        if path.exists() {
            return Ok(path);
        }
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            let path = dir.join(library_filename());
            if path.exists() {
                return Ok(path);
            }
        }
    }

    Err(format!(
        "{} not found. Build it with `cargo build -p goldy-ffi` or set GOLDY_FFI_PATH.",
        library_filename()
    ))
}

#[cfg(windows)]
fn set_dll_directory_for_dependencies(lib_path: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    let Some(dir) = lib_path.parent() else {
        return Ok(());
    };
    let mut wide: Vec<u16> = dir.as_os_str().encode_wide().collect();
    wide.push(0);
    let ok = unsafe { SetDllDirectoryW(wide.as_ptr()) };
    if ok == 0 {
        return Err(format!(
            "SetDllDirectoryW failed for {}: {}",
            dir.display(),
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn set_dll_directory_for_dependencies(_lib_path: &Path) -> Result<(), String> {
    Ok(())
}
