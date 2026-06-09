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
    pub goldy_buffer_access: FnGoldyBufferAccess,
    pub goldy_buffer_create: FnGoldyBufferCreate,
    pub goldy_buffer_create_with_data: FnGoldyBufferCreateWithData,
    pub goldy_buffer_create_with_data_stride: FnGoldyBufferCreateWithDataStride,
    pub goldy_buffer_destroy: FnGoldyBufferDestroy,
    pub goldy_buffer_read_to_cpu: FnGoldyBufferReadToCpu,
    pub goldy_buffer_resource_index: FnGoldyBufferResourceIndex,
    pub goldy_buffer_size: FnGoldyBufferSize,
    pub goldy_buffer_write: FnGoldyBufferWrite,
    pub goldy_clear_error: FnGoldyClearError,
    pub goldy_compute_pipeline_create: FnGoldyComputePipelineCreate,
    pub goldy_compute_pipeline_destroy: FnGoldyComputePipelineDestroy,
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
    pub goldy_render_target_buffer_size: FnGoldyRenderTargetBufferSize,
    pub goldy_render_target_create: FnGoldyRenderTargetCreate,
    pub goldy_render_target_create_with_depth: FnGoldyRenderTargetCreateWithDepth,
    pub goldy_render_target_destroy: FnGoldyRenderTargetDestroy,
    pub goldy_render_target_format: FnGoldyRenderTargetFormat,
    pub goldy_render_target_has_depth: FnGoldyRenderTargetHasDepth,
    pub goldy_render_target_height: FnGoldyRenderTargetHeight,
    pub goldy_render_target_read_to_buffer: FnGoldyRenderTargetReadToBuffer,
    pub goldy_render_target_width: FnGoldyRenderTargetWidth,
    pub goldy_sampler_create: FnGoldySamplerCreate,
    pub goldy_sampler_create_default: FnGoldySamplerCreateDefault,
    pub goldy_sampler_destroy: FnGoldySamplerDestroy,
    pub goldy_shader_builtin_vertex_color_2d: FnGoldyShaderBuiltinVertexColor2d,
    pub goldy_shader_create: FnGoldyShaderCreate,
    pub goldy_shader_destroy: FnGoldyShaderDestroy,
    pub goldy_surface_acquire: FnGoldySurfaceAcquire,
    #[cfg(target_os = "macos")]
    pub goldy_surface_create_appkit: FnGoldySurfaceCreateAppkit,
    #[cfg(windows)]
    pub goldy_surface_create_win32: FnGoldySurfaceCreateWin32,
    pub goldy_surface_destroy: FnGoldySurfaceDestroy,
    pub goldy_surface_format: FnGoldySurfaceFormat,
    pub goldy_surface_frame_height: FnGoldySurfaceFrameHeight,
    pub goldy_surface_frame_width: FnGoldySurfaceFrameWidth,
    pub goldy_surface_height: FnGoldySurfaceHeight,
    pub goldy_surface_present: FnGoldySurfacePresent,
    pub goldy_surface_resize: FnGoldySurfaceResize,
    pub goldy_surface_submit_graph_to_frame: FnGoldySurfaceSubmitGraphToFrame,
    pub goldy_surface_width: FnGoldySurfaceWidth,
    pub goldy_task_graph_clear: FnGoldyTaskGraphClear,
    pub goldy_task_graph_copy_render_target_to_swapchain: FnGoldyTaskGraphCopyRenderTargetToSwapchain,
    pub goldy_task_graph_create: FnGoldyTaskGraphCreate,
    pub goldy_task_graph_declare_swapchain_output: FnGoldyTaskGraphDeclareSwapchainOutput,
    pub goldy_task_graph_compute_node_begin: FnGoldyTaskGraphComputeNodeBegin,
    pub goldy_task_graph_compute_node_bind_buffer: FnGoldyTaskGraphComputeNodeBindBuffer,
    pub goldy_task_graph_compute_node_bind_resources_raw: FnGoldyTaskGraphComputeNodeBindResourcesRaw,
    pub goldy_task_graph_compute_node_dispatch: FnGoldyTaskGraphComputeNodeDispatch,
    pub goldy_task_graph_destroy: FnGoldyTaskGraphDestroy,
    pub goldy_task_graph_dispatch: FnGoldyTaskGraphDispatch,
    pub goldy_task_graph_write_buffer: FnGoldyTaskGraphWriteBuffer,
    pub goldy_task_graph_render_pass_begin: FnGoldyTaskGraphRenderPassBegin,
    pub goldy_task_graph_render_pass_bind_buffer: FnGoldyTaskGraphRenderPassBindBuffer,
    pub goldy_task_graph_render_pass_bind_resources: FnGoldyTaskGraphRenderPassBindResources,
    pub goldy_task_graph_render_pass_bind_resources_typed: FnGoldyTaskGraphRenderPassBindResourcesTyped,
    pub goldy_task_graph_render_pass_clear: FnGoldyTaskGraphRenderPassClear,
    pub goldy_task_graph_render_pass_clear_depth: FnGoldyTaskGraphRenderPassClearDepth,
    pub goldy_task_graph_render_pass_draw: FnGoldyTaskGraphRenderPassDraw,
    pub goldy_task_graph_render_pass_draw_fullscreen: FnGoldyTaskGraphRenderPassDrawFullscreen,
    pub goldy_task_graph_render_pass_draw_indexed: FnGoldyTaskGraphRenderPassDrawIndexed,
    pub goldy_task_graph_render_pass_finish: FnGoldyTaskGraphRenderPassFinish,
    pub goldy_task_graph_render_pass_set_index_buffer: FnGoldyTaskGraphRenderPassSetIndexBuffer,
    pub goldy_task_graph_render_pass_set_pipeline: FnGoldyTaskGraphRenderPassSetPipeline,
    pub goldy_task_graph_render_pass_set_vertex_buffer: FnGoldyTaskGraphRenderPassSetVertexBuffer,
    pub goldy_task_graph_render_pass_set_vertex_buffer_offset: FnGoldyTaskGraphRenderPassSetVertexBufferOffset,
    pub goldy_texture_create: FnGoldyTextureCreate,
    pub goldy_texture_destroy: FnGoldyTextureDestroy,
    pub goldy_texture_format: FnGoldyTextureFormat,
    pub goldy_texture_height: FnGoldyTextureHeight,
    pub goldy_texture_width: FnGoldyTextureWidth,
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
            goldy_buffer_access: sym!("goldy_buffer_access", FnGoldyBufferAccess),
            goldy_buffer_create: sym!("goldy_buffer_create", FnGoldyBufferCreate),
            goldy_buffer_create_with_data: sym!("goldy_buffer_create_with_data", FnGoldyBufferCreateWithData),
            goldy_buffer_create_with_data_stride: sym!(
                "goldy_buffer_create_with_data_stride",
                FnGoldyBufferCreateWithDataStride
            ),
            goldy_buffer_destroy: sym!("goldy_buffer_destroy", FnGoldyBufferDestroy),
            goldy_buffer_read_to_cpu: sym!("goldy_buffer_read_to_cpu", FnGoldyBufferReadToCpu),
            goldy_buffer_resource_index: sym!("goldy_buffer_resource_index", FnGoldyBufferResourceIndex),
            goldy_buffer_size: sym!("goldy_buffer_size", FnGoldyBufferSize),
            goldy_buffer_write: sym!("goldy_buffer_write", FnGoldyBufferWrite),
            goldy_clear_error: sym!("goldy_clear_error", FnGoldyClearError),
            goldy_compute_pipeline_create: sym!("goldy_compute_pipeline_create", FnGoldyComputePipelineCreate),
            goldy_compute_pipeline_destroy: sym!("goldy_compute_pipeline_destroy", FnGoldyComputePipelineDestroy),
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
            goldy_render_target_buffer_size: sym!("goldy_render_target_buffer_size", FnGoldyRenderTargetBufferSize),
            goldy_render_target_create: sym!("goldy_render_target_create", FnGoldyRenderTargetCreate),
            goldy_render_target_create_with_depth: sym!(
                "goldy_render_target_create_with_depth",
                FnGoldyRenderTargetCreateWithDepth
            ),
            goldy_render_target_destroy: sym!("goldy_render_target_destroy", FnGoldyRenderTargetDestroy),
            goldy_render_target_format: sym!("goldy_render_target_format", FnGoldyRenderTargetFormat),
            goldy_render_target_has_depth: sym!("goldy_render_target_has_depth", FnGoldyRenderTargetHasDepth),
            goldy_render_target_height: sym!("goldy_render_target_height", FnGoldyRenderTargetHeight),
            goldy_render_target_read_to_buffer: sym!(
                "goldy_render_target_read_to_buffer",
                FnGoldyRenderTargetReadToBuffer
            ),
            goldy_render_target_width: sym!("goldy_render_target_width", FnGoldyRenderTargetWidth),
            goldy_sampler_create: sym!("goldy_sampler_create", FnGoldySamplerCreate),
            goldy_sampler_create_default: sym!("goldy_sampler_create_default", FnGoldySamplerCreateDefault),
            goldy_sampler_destroy: sym!("goldy_sampler_destroy", FnGoldySamplerDestroy),
            goldy_shader_builtin_vertex_color_2d: sym!(
                "goldy_shader_builtin_vertex_color_2d",
                FnGoldyShaderBuiltinVertexColor2d
            ),
            goldy_shader_create: sym!("goldy_shader_create", FnGoldyShaderCreate),
            goldy_shader_destroy: sym!("goldy_shader_destroy", FnGoldyShaderDestroy),
            goldy_surface_acquire: sym!("goldy_surface_acquire", FnGoldySurfaceAcquire),
            #[cfg(target_os = "macos")]
            goldy_surface_create_appkit: sym!("goldy_surface_create_appkit", FnGoldySurfaceCreateAppkit),
            #[cfg(windows)]
            goldy_surface_create_win32: sym!("goldy_surface_create_win32", FnGoldySurfaceCreateWin32),
            goldy_surface_destroy: sym!("goldy_surface_destroy", FnGoldySurfaceDestroy),
            goldy_surface_format: sym!("goldy_surface_format", FnGoldySurfaceFormat),
            goldy_surface_frame_height: sym!("goldy_surface_frame_height", FnGoldySurfaceFrameHeight),
            goldy_surface_frame_width: sym!("goldy_surface_frame_width", FnGoldySurfaceFrameWidth),
            goldy_surface_height: sym!("goldy_surface_height", FnGoldySurfaceHeight),
            goldy_surface_present: sym!("goldy_surface_present", FnGoldySurfacePresent),
            goldy_surface_resize: sym!("goldy_surface_resize", FnGoldySurfaceResize),
            goldy_surface_submit_graph_to_frame: sym!(
                "goldy_surface_submit_graph_to_frame",
                FnGoldySurfaceSubmitGraphToFrame
            ),
            goldy_surface_width: sym!("goldy_surface_width", FnGoldySurfaceWidth),
            goldy_task_graph_clear: sym!("goldy_task_graph_clear", FnGoldyTaskGraphClear),
            goldy_task_graph_copy_render_target_to_swapchain: sym!(
                "goldy_task_graph_copy_render_target_to_swapchain",
                FnGoldyTaskGraphCopyRenderTargetToSwapchain
            ),
            goldy_task_graph_create: sym!("goldy_task_graph_create", FnGoldyTaskGraphCreate),
            goldy_task_graph_declare_swapchain_output: sym!(
                "goldy_task_graph_declare_swapchain_output",
                FnGoldyTaskGraphDeclareSwapchainOutput
            ),
            goldy_task_graph_compute_node_begin: sym!(
                "goldy_task_graph_compute_node_begin",
                FnGoldyTaskGraphComputeNodeBegin
            ),
            goldy_task_graph_compute_node_bind_buffer: sym!(
                "goldy_task_graph_compute_node_bind_buffer",
                FnGoldyTaskGraphComputeNodeBindBuffer
            ),
            goldy_task_graph_compute_node_bind_resources_raw: sym!(
                "goldy_task_graph_compute_node_bind_resources_raw",
                FnGoldyTaskGraphComputeNodeBindResourcesRaw
            ),
            goldy_task_graph_compute_node_dispatch: sym!(
                "goldy_task_graph_compute_node_dispatch",
                FnGoldyTaskGraphComputeNodeDispatch
            ),
            goldy_task_graph_destroy: sym!("goldy_task_graph_destroy", FnGoldyTaskGraphDestroy),
            goldy_task_graph_dispatch: sym!("goldy_task_graph_dispatch", FnGoldyTaskGraphDispatch),
            goldy_task_graph_write_buffer: sym!("goldy_task_graph_write_buffer", FnGoldyTaskGraphWriteBuffer),
            goldy_task_graph_render_pass_begin: sym!(
                "goldy_task_graph_render_pass_begin",
                FnGoldyTaskGraphRenderPassBegin
            ),
            goldy_task_graph_render_pass_bind_buffer: sym!(
                "goldy_task_graph_render_pass_bind_buffer",
                FnGoldyTaskGraphRenderPassBindBuffer
            ),
            goldy_task_graph_render_pass_bind_resources: sym!(
                "goldy_task_graph_render_pass_bind_resources",
                FnGoldyTaskGraphRenderPassBindResources
            ),
            goldy_task_graph_render_pass_bind_resources_typed: sym!(
                "goldy_task_graph_render_pass_bind_resources_typed",
                FnGoldyTaskGraphRenderPassBindResourcesTyped
            ),
            goldy_task_graph_render_pass_clear: sym!(
                "goldy_task_graph_render_pass_clear",
                FnGoldyTaskGraphRenderPassClear
            ),
            goldy_task_graph_render_pass_clear_depth: sym!(
                "goldy_task_graph_render_pass_clear_depth",
                FnGoldyTaskGraphRenderPassClearDepth
            ),
            goldy_task_graph_render_pass_draw: sym!(
                "goldy_task_graph_render_pass_draw",
                FnGoldyTaskGraphRenderPassDraw
            ),
            goldy_task_graph_render_pass_draw_fullscreen: sym!(
                "goldy_task_graph_render_pass_draw_fullscreen",
                FnGoldyTaskGraphRenderPassDrawFullscreen
            ),
            goldy_task_graph_render_pass_draw_indexed: sym!(
                "goldy_task_graph_render_pass_draw_indexed",
                FnGoldyTaskGraphRenderPassDrawIndexed
            ),
            goldy_task_graph_render_pass_finish: sym!(
                "goldy_task_graph_render_pass_finish",
                FnGoldyTaskGraphRenderPassFinish
            ),
            goldy_task_graph_render_pass_set_index_buffer: sym!(
                "goldy_task_graph_render_pass_set_index_buffer",
                FnGoldyTaskGraphRenderPassSetIndexBuffer
            ),
            goldy_task_graph_render_pass_set_pipeline: sym!(
                "goldy_task_graph_render_pass_set_pipeline",
                FnGoldyTaskGraphRenderPassSetPipeline
            ),
            goldy_task_graph_render_pass_set_vertex_buffer: sym!(
                "goldy_task_graph_render_pass_set_vertex_buffer",
                FnGoldyTaskGraphRenderPassSetVertexBuffer
            ),
            goldy_task_graph_render_pass_set_vertex_buffer_offset: sym!(
                "goldy_task_graph_render_pass_set_vertex_buffer_offset",
                FnGoldyTaskGraphRenderPassSetVertexBufferOffset
            ),
            goldy_texture_create: sym!("goldy_texture_create", FnGoldyTextureCreate),
            goldy_texture_destroy: sym!("goldy_texture_destroy", FnGoldyTextureDestroy),
            goldy_texture_format: sym!("goldy_texture_format", FnGoldyTextureFormat),
            goldy_texture_height: sym!("goldy_texture_height", FnGoldyTextureHeight),
            goldy_texture_width: sym!("goldy_texture_width", FnGoldyTextureWidth),
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
