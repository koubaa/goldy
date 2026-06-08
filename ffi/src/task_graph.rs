//! FFI bindings for [`goldy::TaskGraph`] and offscreen render-pass recording.

use crate::buffer::GoldyBuffer;
use crate::device::GoldyDevice;
use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::pipeline::GoldyRenderPipeline;
use crate::render_target::GoldyRenderTarget;
use crate::types::{GoldyColor, GoldyIndexFormat, GoldyNodeAccess};
use goldy::task_graph::{NodeAccess, RenderPassRecord, SwapchainOutputHandle, TaskGraph};
use std::ffi::CStr;
use std::ptr;

/// Opaque handle to a Goldy TaskGraph.
pub struct GoldyTaskGraph {
    pub(crate) inner: TaskGraph,
    active_pass: Option<RenderPassRecord>,
}

/// Opaque token returned by [`goldy_task_graph_declare_swapchain_output`].
///
/// Carries no data; exists for type safety at the C ABI boundary.
#[repr(C)]
pub struct GoldySwapchainOutput {
    _private: [u8; 0],
}

fn intern_label(label: *const libc::c_char) -> Result<&'static str, GoldyResult> {
    if label.is_null() {
        set_last_error("Task graph label is null");
        return Err(GoldyResult::NullPointer);
    }
    let s = unsafe { CStr::from_ptr(label) };
    let s = s.to_str().map_err(|e| {
        set_last_error_from_anyhow(&anyhow::anyhow!("Invalid UTF-8 label: {e}"));
        GoldyResult::InvalidArgument
    })?;
    Ok(Box::leak(s.to_string().into_boxed_str()))
}

fn node_access(access: GoldyNodeAccess) -> NodeAccess {
    match access {
        GoldyNodeAccess::Read => NodeAccess::Read,
        GoldyNodeAccess::Write => NodeAccess::Write,
        GoldyNodeAccess::ReadWrite => NodeAccess::ReadWrite,
    }
}

fn active_pass_mut(graph: &mut GoldyTaskGraph) -> Result<&mut RenderPassRecord, GoldyResult> {
    graph.active_pass.as_mut().ok_or_else(|| {
        set_last_error("No render pass is being recorded; call goldy_task_graph_render_pass_begin first");
        GoldyResult::InvalidArgument
    })
}

/// Create a new task graph.
#[no_mangle]
pub extern "C" fn goldy_task_graph_create() -> *mut GoldyTaskGraph {
    Box::into_raw(Box::new(GoldyTaskGraph {
        inner: TaskGraph::new(),
        active_pass: None,
    }))
}

/// Destroy a task graph.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_destroy(graph: *mut GoldyTaskGraph) {
    if !graph.is_null() {
        drop(Box::from_raw(graph));
    }
}

/// Reset the graph to empty while retaining internal capacity.
///
/// # Safety
/// The graph pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_clear(graph: *mut GoldyTaskGraph) -> GoldyResult {
    if graph.is_null() {
        return GoldyResult::NullPointer;
    }
    (*graph).inner.clear();
    (*graph).active_pass = None;
    GoldyResult::Ok
}

/// Begin recording an offscreen render pass on `target`.
///
/// Only one render pass may be open at a time per graph.
///
/// # Safety
/// All pointers must be valid. `target` must outlive the graph recording session.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_begin(
    graph: *mut GoldyTaskGraph,
    label: *const libc::c_char,
    target: *const GoldyRenderTarget,
) -> GoldyResult {
    if graph.is_null() || target.is_null() {
        return GoldyResult::NullPointer;
    }
    if (*graph).active_pass.is_some() {
        set_last_error("A render pass is already being recorded");
        return GoldyResult::InvalidArgument;
    }
    let label = match intern_label(label) {
        Ok(l) => l,
        Err(e) => return e,
    };
    (*graph).active_pass = Some(RenderPassRecord::new(label, &(*target).inner));
    GoldyResult::Ok
}

/// Declare a graph dependency on a buffer for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_bind_buffer(
    graph: *mut GoldyTaskGraph,
    buffer: *const GoldyBuffer,
    access: GoldyNodeAccess,
) -> GoldyResult {
    if graph.is_null() || buffer.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.bind_buffer(&(*buffer).inner, node_access(access));
    GoldyResult::Ok
}

/// Clear the color attachment in the active render pass.
///
/// # Safety
/// The graph pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_clear(
    graph: *mut GoldyTaskGraph,
    color: GoldyColor,
) -> GoldyResult {
    if graph.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.clear(color.into());
    GoldyResult::Ok
}

/// Clear the depth attachment in the active render pass.
///
/// # Safety
/// The graph pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_clear_depth(
    graph: *mut GoldyTaskGraph,
    depth: f32,
) -> GoldyResult {
    if graph.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.clear_depth(depth);
    GoldyResult::Ok
}

/// Set the render pipeline for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_set_pipeline(
    graph: *mut GoldyTaskGraph,
    pipeline: *const GoldyRenderPipeline,
) -> GoldyResult {
    if graph.is_null() || pipeline.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.set_pipeline(&(*pipeline).inner);
    GoldyResult::Ok
}

/// Bind a vertex buffer slot for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_set_vertex_buffer(
    graph: *mut GoldyTaskGraph,
    slot: u32,
    buffer: *const GoldyBuffer,
) -> GoldyResult {
    if graph.is_null() || buffer.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.set_vertex_buffer(slot, &(*buffer).inner);
    GoldyResult::Ok
}

/// Bind a vertex buffer slot with a byte offset for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_set_vertex_buffer_offset(
    graph: *mut GoldyTaskGraph,
    slot: u32,
    buffer: *const GoldyBuffer,
    offset: u64,
) -> GoldyResult {
    if graph.is_null() || buffer.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.set_vertex_buffer_offset(slot, &(*buffer).inner, offset);
    GoldyResult::Ok
}

/// Bind an index buffer for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_set_index_buffer(
    graph: *mut GoldyTaskGraph,
    buffer: *const GoldyBuffer,
    format: GoldyIndexFormat,
) -> GoldyResult {
    if graph.is_null() || buffer.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.set_index_buffer(&(*buffer).inner, format.into());
    GoldyResult::Ok
}

/// Draw non-indexed primitives in the active render pass.
///
/// # Safety
/// The graph pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_draw(
    graph: *mut GoldyTaskGraph,
    first_vertex: u32,
    vertex_count: u32,
    first_instance: u32,
    instance_count: u32,
) -> GoldyResult {
    if graph.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.draw(first_vertex, vertex_count, first_instance, instance_count);
    GoldyResult::Ok
}

/// Draw indexed primitives in the active render pass.
///
/// # Safety
/// The graph pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_draw_indexed(
    graph: *mut GoldyTaskGraph,
    first_index: u32,
    index_count: u32,
    base_vertex: i32,
    first_instance: u32,
    instance_count: u32,
) -> GoldyResult {
    if graph.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.draw_indexed(first_index, index_count, base_vertex, first_instance, instance_count);
    GoldyResult::Ok
}

/// Draw a fullscreen triangle (3 vertices, 1 instance) in the active render pass.
///
/// # Safety
/// The graph pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_draw_fullscreen(
    graph: *mut GoldyTaskGraph,
) -> GoldyResult {
    if graph.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.draw_fullscreen();
    GoldyResult::Ok
}

/// Bind shader resource slots from buffers for the active render pass.
///
/// # Safety
/// All pointers must be valid. `buffers` must contain `buffer_count` elements.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_bind_resources(
    graph: *mut GoldyTaskGraph,
    buffers: *const *const GoldyBuffer,
    buffer_count: u32,
) -> GoldyResult {
    if graph.is_null() || (buffer_count > 0 && buffers.is_null()) {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let buffer_refs: Vec<&goldy::Buffer> = (0..buffer_count as usize)
        .filter_map(|i| {
            let buf_ptr = *buffers.add(i);
            if buf_ptr.is_null() {
                None
            } else {
                Some(&(*buf_ptr).inner)
            }
        })
        .collect();
    pass.bind_resources(&buffer_refs);
    GoldyResult::Ok
}

/// Finalize the active render pass and append it to the graph.
///
/// # Safety
/// The graph pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_finish(graph: *mut GoldyTaskGraph) -> GoldyResult {
    if graph.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match (*graph).active_pass.take() {
        Some(p) => p,
        None => {
            set_last_error("No render pass is being recorded");
            return GoldyResult::InvalidArgument;
        }
    };
    pass.commit(&mut (*graph).inner);
    GoldyResult::Ok
}

/// Declare that this graph will copy to the swapchain at submit time.
///
/// Returns an opaque token passed to [`goldy_task_graph_copy_render_target_to_swapchain`].
/// Phase 2 surface submit uses the same graph with a `SwapchainOutput` binding.
///
/// # Safety
/// The graph pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_declare_swapchain_output(
    graph: *mut GoldyTaskGraph,
) -> *mut GoldySwapchainOutput {
    if graph.is_null() {
        return ptr::null_mut();
    }
    (*graph).inner.declare_swapchain_output();
    Box::into_raw(Box::new(GoldySwapchainOutput { _private: [] }))
}

/// Add a render-target → swapchain blit node to the graph.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_copy_render_target_to_swapchain(
    graph: *mut GoldyTaskGraph,
    src: *const GoldyRenderTarget,
    _swapchain: *const GoldySwapchainOutput,
) -> GoldyResult {
    if graph.is_null() || src.is_null() || _swapchain.is_null() {
        return GoldyResult::NullPointer;
    }
    (*graph)
        .inner
        .copy_render_target_to_swapchain(&(*src).inner, SwapchainOutputHandle);
    GoldyResult::Ok
}

/// Analyze the graph, submit GPU work, and block until complete.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_dispatch(
    graph: *mut GoldyTaskGraph,
    device: *const GoldyDevice,
) -> GoldyResult {
    if graph.is_null() || device.is_null() {
        return GoldyResult::NullPointer;
    }
    if (*graph).active_pass.is_some() {
        set_last_error("Cannot dispatch while a render pass is being recorded; call render_pass_finish first");
        return GoldyResult::InvalidArgument;
    }

    let ctx = match (*device).inner.create_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            set_last_error(format!("{e}"));
            return GoldyResult::GpuError;
        }
    };

    match (*graph).inner.dispatch(&ctx) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}
