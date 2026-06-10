//! FFI bindings for [`goldy::TaskGraph`] and offscreen render-pass recording.

use crate::buffer::GoldyBuffer;
use crate::buffer_pool::GoldyBufferView;
use crate::compute::GoldyComputePipeline;
use crate::device::GoldyDevice;
use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::pipeline::GoldyRenderPipeline;
use crate::render_target::GoldyRenderTarget;
use crate::retained_pool::GoldyParcel;
use crate::types::{GoldyColor, GoldyIndexFormat, GoldyNodeAccess};
use goldy::task_graph::{ComputeNodeRecord, NodeAccess, RenderPassRecord, SwapchainOutputHandle, TaskGraph};
use goldy::types::{ResourceCategory, ResourceHandle};
use goldy::MosaicSlot;
use std::ffi::CStr;
use std::ptr;

/// Opaque handle to a Goldy TaskGraph.
pub struct GoldyTaskGraph {
    pub(crate) inner: TaskGraph,
    active_pass: Option<RenderPassRecord>,
    active_compute: Option<ComputeNodeRecord>,
    /// Per-graph sentinel returned by [`goldy_task_graph_declare_swapchain_output`] (no heap alloc).
    swapchain_token: GoldySwapchainOutput,
    /// Keeps render-pass label strings alive for nodes in `inner` (cleared with the graph).
    labels: Vec<String>,
}

impl GoldyTaskGraph {
    pub(crate) fn has_active_render_pass(&self) -> bool {
        self.active_pass.is_some()
    }

    fn has_active_recorder(&self) -> bool {
        self.active_pass.is_some() || self.active_compute.is_some()
    }

    fn intern_label(&mut self, label: &str) -> &'static str {
        self.labels.push(label.to_string());
        let s = self.labels.last().unwrap();
        // SAFETY: `labels` is cleared in `goldy_task_graph_clear` alongside `inner`, and dropped
        // with the graph. Node labels in `inner` never outlive this storage.
        unsafe { std::mem::transmute::<&str, &'static str>(s.as_str()) }
    }
}

/// Opaque token returned by [`goldy_task_graph_declare_swapchain_output`].
///
/// Carries no data; exists for type safety at the C ABI boundary.
#[repr(C)]
pub struct GoldySwapchainOutput {
    _private: [u8; 0],
}

fn parse_label(label: *const libc::c_char) -> Result<String, GoldyResult> {
    if label.is_null() {
        set_last_error("Task graph label is null");
        return Err(GoldyResult::NullPointer);
    }
    let s = unsafe { CStr::from_ptr(label) };
    s.to_str().map(|s| s.to_string()).map_err(|e| {
        set_last_error_from_anyhow(&anyhow::anyhow!("Invalid UTF-8 label: {e}"));
        GoldyResult::InvalidArgument
    })
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

fn active_compute_mut(graph: &mut GoldyTaskGraph) -> Result<&mut ComputeNodeRecord, GoldyResult> {
    graph.active_compute.as_mut().ok_or_else(|| {
        set_last_error("No compute node is being recorded; call goldy_task_graph_compute_node_begin first");
        GoldyResult::InvalidArgument
    })
}

/// Create a new task graph.
#[no_mangle]
pub extern "C" fn goldy_task_graph_create() -> *mut GoldyTaskGraph {
    Box::into_raw(Box::new(GoldyTaskGraph {
        inner: TaskGraph::new(),
        active_pass: None,
        active_compute: None,
        swapchain_token: GoldySwapchainOutput { _private: [] },
        labels: Vec::new(),
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

/// Number of task nodes recorded in the graph (for tests and diagnostics).
///
/// # Safety
/// The graph pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_len(graph: *const GoldyTaskGraph) -> u32 {
    if graph.is_null() {
        return 0;
    }
    (*graph).inner.len() as u32
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
    (*graph).active_compute = None;
    (*graph).labels.clear();
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
    let label = match parse_label(label) {
        Ok(l) => (*graph).intern_label(&l),
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

/// Declare a buffer-view dependency for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_bind_buffer_view(
    graph: *mut GoldyTaskGraph,
    view: *const GoldyBufferView,
    access: GoldyNodeAccess,
) -> GoldyResult {
    if graph.is_null() || view.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.bind_buffer_view(&(*view).inner, node_access(access));
    GoldyResult::Ok
}

/// Declare a mosaic sub-view dependency for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_bind_parcel_view(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    slot: u32,
    access: GoldyNodeAccess,
) -> GoldyResult {
    if graph.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.bind_buffer_view((*parcel).inner.view(MosaicSlot(slot)), node_access(access));
    GoldyResult::Ok
}

/// Declare a graph dependency on a retained parcel for the active render pass.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_bind_parcel(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    access: GoldyNodeAccess,
) -> GoldyResult {
    if graph.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.bind_parcel(&(*parcel).inner, node_access(access));
    GoldyResult::Ok
}

/// Bind typed resource handles (category + index pairs) for the active render pass.
///
/// `indices` is a flat array of u32 values: `[category0, index0, category1, index1, ...]`.
/// Use `GoldyResourceCategory::Scattered` (0) for buffer views.
///
/// # Safety
/// All pointers must be valid. `indices` must contain `handle_count * 2` elements.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_bind_resources_typed(
    graph: *mut GoldyTaskGraph,
    indices: *const u32,
    handle_count: u32,
) -> GoldyResult {
    if graph.is_null() || (handle_count > 0 && indices.is_null()) {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let mut handles = Vec::with_capacity(handle_count as usize);
    for i in 0..handle_count as usize {
        let category = *indices.add(i * 2);
        let index = *indices.add(i * 2 + 1);
        let cat = match category {
            0 => ResourceCategory::Scattered,
            1 => ResourceCategory::Broadcast,
            2 => ResourceCategory::StorageImage,
            3 => ResourceCategory::Texture,
            4 => ResourceCategory::Sampler,
            _ => {
                set_last_error("Invalid resource category in bind_resources_typed");
                return GoldyResult::InvalidArgument;
            }
        };
        handles.push(ResourceHandle::new(cat, index));
    }
    pass.bind_resources_typed(&handles);
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

/// Bind a vertex buffer slot from a retained buffer parcel for the active render pass.
///
/// # Safety
/// All pointers must be valid. `parcel` must be a non-mosaic buffer parcel.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_render_pass_set_vertex_buffer_parcel(
    graph: *mut GoldyTaskGraph,
    slot: u32,
    parcel: *const GoldyParcel,
) -> GoldyResult {
    if graph.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let pass = match active_pass_mut(&mut *graph) {
        Ok(p) => p,
        Err(e) => return e,
    };
    pass.set_vertex_buffer(slot, &(*parcel).inner);
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
pub unsafe extern "C" fn goldy_task_graph_render_pass_draw_fullscreen(graph: *mut GoldyTaskGraph) -> GoldyResult {
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

/// Begin recording a compute dispatch node.
///
/// Only one recorder (render pass or compute node) may be open at a time per graph.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_compute_node_begin(
    graph: *mut GoldyTaskGraph,
    label: *const libc::c_char,
    pipeline: *const GoldyComputePipeline,
) -> GoldyResult {
    if graph.is_null() || pipeline.is_null() {
        return GoldyResult::NullPointer;
    }
    if (*graph).has_active_recorder() {
        set_last_error("Another graph node is already being recorded");
        return GoldyResult::InvalidArgument;
    }
    let label = match parse_label(label) {
        Ok(l) => (*graph).intern_label(&l),
        Err(e) => return e,
    };
    (*graph).active_compute = Some(ComputeNodeRecord::new(label, &(*pipeline).inner));
    GoldyResult::Ok
}

/// Declare a buffer dependency for the active compute node.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_compute_node_bind_buffer(
    graph: *mut GoldyTaskGraph,
    buffer: *const GoldyBuffer,
    access: GoldyNodeAccess,
) -> GoldyResult {
    if graph.is_null() || buffer.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *graph) {
        Ok(n) => n,
        Err(e) => return e,
    };
    node.bind_buffer(&(*buffer).inner, node_access(access));
    GoldyResult::Ok
}

/// Declare a buffer-view dependency for the active compute node.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_compute_node_bind_buffer_view(
    graph: *mut GoldyTaskGraph,
    view: *const GoldyBufferView,
    access: GoldyNodeAccess,
) -> GoldyResult {
    if graph.is_null() || view.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *graph) {
        Ok(n) => n,
        Err(e) => return e,
    };
    node.bind_buffer_view(&(*view).inner, node_access(access));
    GoldyResult::Ok
}

/// Declare a mosaic sub-view dependency for the active compute node.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_compute_node_bind_parcel_view(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    slot: u32,
    access: GoldyNodeAccess,
) -> GoldyResult {
    if graph.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *graph) {
        Ok(n) => n,
        Err(e) => return e,
    };
    node.bind_buffer_view((*parcel).inner.view(MosaicSlot(slot)), node_access(access));
    GoldyResult::Ok
}

/// Declare a graph dependency on a retained parcel for the active compute node.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_compute_node_bind_parcel(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    access: GoldyNodeAccess,
) -> GoldyResult {
    if graph.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *graph) {
        Ok(n) => n,
        Err(e) => return e,
    };
    node.bind_parcel(&(*parcel).inner, node_access(access));
    GoldyResult::Ok
}

/// Set bindless resource slot indices for the active compute node.
///
/// # Safety
/// All pointers must be valid. `indices` must contain `count` elements.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_compute_node_bind_resources_raw(
    graph: *mut GoldyTaskGraph,
    indices: *const u32,
    count: u32,
) -> GoldyResult {
    if graph.is_null() || (count > 0 && indices.is_null()) {
        return GoldyResult::NullPointer;
    }
    let node = match active_compute_mut(&mut *graph) {
        Ok(n) => n,
        Err(e) => return e,
    };
    let slice = if count > 0 {
        std::slice::from_raw_parts(indices, count as usize)
    } else {
        &[]
    };
    node.bind_resources_raw(slice);
    GoldyResult::Ok
}

/// Finalize the active compute node with a direct dispatch.
///
/// # Safety
/// The graph pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_compute_node_dispatch(
    graph: *mut GoldyTaskGraph,
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
) -> GoldyResult {
    if graph.is_null() {
        return GoldyResult::NullPointer;
    }
    let node = match (*graph).active_compute.take() {
        Some(n) => n,
        None => {
            set_last_error("No compute node is being recorded");
            return GoldyResult::InvalidArgument;
        }
    };
    node.commit_dispatch(&mut (*graph).inner, workgroups_x, workgroups_y, workgroups_z);
    GoldyResult::Ok
}

/// Add a CPU→GPU buffer upload node to the graph.
///
/// # Safety
/// All pointers must be valid. `data` must point to at least `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_write_buffer(
    graph: *mut GoldyTaskGraph,
    buffer: *const GoldyBuffer,
    offset: u64,
    data: *const u8,
    size: usize,
) -> GoldyResult {
    if graph.is_null() || buffer.is_null() {
        return GoldyResult::NullPointer;
    }
    if data.is_null() && size > 0 {
        return GoldyResult::NullPointer;
    }
    if (*graph).has_active_recorder() {
        set_last_error("Cannot add write_buffer while recording a pass or compute node");
        return GoldyResult::InvalidArgument;
    }

    let bytes = if size > 0 {
        std::slice::from_raw_parts(data, size).to_vec()
    } else {
        Vec::new()
    };
    (*graph).inner.write_buffer(&(*buffer).inner, offset, bytes);
    GoldyResult::Ok
}

/// Add a CPU→GPU upload node targeting a retained buffer parcel.
///
/// # Safety
/// All pointers must be valid. `data` must point to at least `size` bytes when non-null.
#[no_mangle]
pub unsafe extern "C" fn goldy_task_graph_write_parcel(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    offset: u64,
    data: *const u8,
    size: usize,
) -> GoldyResult {
    if graph.is_null() || parcel.is_null() {
        return GoldyResult::NullPointer;
    }
    if data.is_null() && size > 0 {
        return GoldyResult::NullPointer;
    }
    if (*graph).has_active_recorder() {
        set_last_error("Cannot add write_parcel while recording a pass or compute node");
        return GoldyResult::InvalidArgument;
    }

    let bytes = if size > 0 {
        std::slice::from_raw_parts(data, size).to_vec()
    } else {
        Vec::new()
    };
    match (*graph).inner.write_parcel(&(*parcel).inner, offset, bytes) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

/// Declare that this graph will copy to the swapchain at submit time.
///
/// Returns a pointer to a per-graph sentinel (not heap-allocated). The pointer is
/// valid until the graph is destroyed and must be passed to
/// [`goldy_task_graph_copy_render_target_to_swapchain`].
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
    ptr::addr_of_mut!((*graph).swapchain_token)
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
    if (*graph).has_active_recorder() {
        set_last_error("Cannot dispatch while recording a pass or compute node");
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
