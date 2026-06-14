//! Raw C bindings loaded from `goldy_ffi` at runtime via libloading.
#![allow(non_camel_case_types, non_snake_case, dead_code, improper_ctypes)]

mod ffi;
mod loader;
mod types;

pub use types::*;

use loader::lib;

pub unsafe fn goldy_clear_error() {
    (lib().goldy_clear_error)()
}

pub unsafe fn goldy_compute_pipeline_create(
    device: *const GoldyDevice,
    shader: *const GoldyShaderModule,
) -> *mut GoldyComputePipeline {
    (lib().goldy_compute_pipeline_create)(device, shader)
}

pub unsafe fn goldy_compute_pipeline_destroy(pipeline: *mut GoldyComputePipeline) {
    (lib().goldy_compute_pipeline_destroy)(pipeline)
}

pub unsafe fn goldy_context_create(device: *const GoldyDevice) -> *mut GoldyContext {
    (lib().goldy_context_create)(device)
}

pub unsafe fn goldy_context_destroy(ctx: *mut GoldyContext) {
    (lib().goldy_context_destroy)(ctx)
}

pub unsafe fn goldy_device_adapter_id(device: *const GoldyDevice) -> u32 {
    (lib().goldy_device_adapter_id)(device)
}

pub unsafe fn goldy_device_destroy(device: *mut GoldyDevice) {
    (lib().goldy_device_destroy)(device)
}

pub unsafe fn goldy_device_has_library(device: *const GoldyDevice, name: *const std::ffi::c_char) -> bool {
    (lib().goldy_device_has_library)(device, name)
}

pub unsafe fn goldy_device_is_valid(device: *const GoldyDevice) -> bool {
    (lib().goldy_device_is_valid)(device)
}

pub unsafe fn goldy_get_last_error() -> *const std::ffi::c_char {
    (lib().goldy_get_last_error)()
}

pub unsafe fn goldy_retained_pool_create(device: *const GoldyDevice) -> *mut GoldyRetainedPool {
    (lib().goldy_retained_pool_create)(device)
}

pub unsafe fn goldy_retained_pool_destroy(pool: *mut GoldyRetainedPool) {
    (lib().goldy_retained_pool_destroy)(pool)
}

pub unsafe fn goldy_retained_pool_acquire_buffer(
    pool: *mut GoldyRetainedPool,
    size: u64,
    access: GoldyBufferKind,
    element_stride: u32,
    data: *const u8,
    data_size: usize,
) -> *mut GoldyParcel {
    (lib().goldy_retained_pool_acquire_buffer)(pool, size, access, element_stride, data, data_size)
}

pub unsafe fn goldy_mosaic_builder_create() -> *mut GoldyMosaicBuilder {
    (lib().goldy_mosaic_builder_create)()
}

pub unsafe fn goldy_mosaic_builder_destroy(builder: *mut GoldyMosaicBuilder) {
    (lib().goldy_mosaic_builder_destroy)(builder)
}

pub unsafe fn goldy_mosaic_builder_emplace(
    builder: *mut GoldyMosaicBuilder,
    data: *const u8,
    data_size: usize,
    element_count: u64,
    element_stride: u32,
) -> u32 {
    (lib().goldy_mosaic_builder_emplace)(builder, data, data_size, element_count, element_stride)
}

pub unsafe fn goldy_mosaic_builder_build(
    builder: *mut GoldyMosaicBuilder,
    pool: *mut GoldyRetainedPool,
) -> *mut GoldyParcel {
    (lib().goldy_mosaic_builder_build)(builder, pool)
}

pub unsafe fn goldy_parcel_destroy(parcel: *mut GoldyParcel) {
    (lib().goldy_parcel_destroy)(parcel)
}

pub unsafe fn goldy_parcel_byte_size(parcel: *const GoldyParcel) -> u64 {
    (lib().goldy_parcel_byte_size)(parcel)
}

pub unsafe fn goldy_parcel_mosaic_view_resource_index(
    parcel: *const GoldyParcel,
    slot: u32,
    access: GoldyResourceAccess,
) -> u32 {
    (lib().goldy_parcel_mosaic_view_resource_index)(parcel, slot, access)
}

pub unsafe fn goldy_parcel_mosaic_view_read_to_cpu(
    parcel: *const GoldyParcel,
    slot: u32,
    device: *const GoldyDevice,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    (lib().goldy_parcel_mosaic_view_read_to_cpu)(parcel, slot, device, output, output_size)
}

pub unsafe fn goldy_parcel_mosaic_view_size(parcel: *const GoldyParcel, slot: u32) -> u64 {
    (lib().goldy_parcel_mosaic_view_size)(parcel, slot)
}

pub unsafe fn goldy_instance_adapter_count(instance: *const GoldyInstance) -> u32 {
    (lib().goldy_instance_adapter_count)(instance)
}

pub unsafe fn goldy_instance_backend_type(instance: *const GoldyInstance) -> GoldyBackendType {
    (lib().goldy_instance_backend_type)(instance)
}

pub unsafe fn goldy_instance_create() -> *mut GoldyInstance {
    (lib().goldy_instance_create)()
}

pub unsafe fn goldy_instance_create_device_for_adapter(
    instance: *const GoldyInstance,
    adapter_id: u32,
) -> *mut GoldyDevice {
    (lib().goldy_instance_create_device_for_adapter)(instance, adapter_id)
}

pub unsafe fn goldy_instance_destroy(instance: *mut GoldyInstance) {
    (lib().goldy_instance_destroy)(instance)
}

pub unsafe fn goldy_instance_get_adapter(
    instance: *const GoldyInstance,
    index: u32,
    info: *mut GoldyAdapterInfo,
) -> GoldyResult {
    (lib().goldy_instance_get_adapter)(instance, index, info)
}

pub unsafe fn goldy_render_pipeline_create(
    device: *const GoldyDevice,
    vertex_shader: *const GoldyShaderModule,
    fragment_shader: *const GoldyShaderModule,
    desc: *const GoldyRenderPipelineDesc,
) -> *mut GoldyRenderPipeline {
    (lib().goldy_render_pipeline_create)(device, vertex_shader, fragment_shader, desc)
}

pub unsafe fn goldy_render_pipeline_destroy(pipeline: *mut GoldyRenderPipeline) {
    (lib().goldy_render_pipeline_destroy)(pipeline)
}

pub unsafe fn goldy_render_target_buffer_size(target: *const GoldyRenderTarget) -> usize {
    (lib().goldy_render_target_buffer_size)(target)
}

pub unsafe fn goldy_render_target_create(
    device: *const GoldyDevice,
    width: u32,
    height: u32,
    format: GoldyTextureFormat,
) -> *mut GoldyRenderTarget {
    (lib().goldy_render_target_create)(device, width, height, format)
}

pub unsafe fn goldy_render_target_create_with_depth(
    device: *const GoldyDevice,
    width: u32,
    height: u32,
    format: GoldyTextureFormat,
    depth_format: GoldyDepthFormat,
) -> *mut GoldyRenderTarget {
    (lib().goldy_render_target_create_with_depth)(device, width, height, format, depth_format)
}

pub unsafe fn goldy_render_target_destroy(target: *mut GoldyRenderTarget) {
    (lib().goldy_render_target_destroy)(target)
}

pub unsafe fn goldy_render_target_format(target: *const GoldyRenderTarget) -> GoldyTextureFormat {
    (lib().goldy_render_target_format)(target)
}

pub unsafe fn goldy_render_target_has_depth(target: *const GoldyRenderTarget) -> bool {
    (lib().goldy_render_target_has_depth)(target)
}

pub unsafe fn goldy_render_target_height(target: *const GoldyRenderTarget) -> u32 {
    (lib().goldy_render_target_height)(target)
}

pub unsafe fn goldy_render_target_read_to_buffer(
    target: *const GoldyRenderTarget,
    buffer: *mut u8,
    size: usize,
) -> GoldyResult {
    (lib().goldy_render_target_read_to_buffer)(target, buffer, size)
}

pub unsafe fn goldy_render_target_width(target: *const GoldyRenderTarget) -> u32 {
    (lib().goldy_render_target_width)(target)
}

pub unsafe fn goldy_sampler_create(device: *const GoldyDevice, desc: *const GoldySamplerDesc) -> *mut GoldySampler {
    (lib().goldy_sampler_create)(device, desc)
}

pub unsafe fn goldy_sampler_create_default(device: *const GoldyDevice) -> *mut GoldySampler {
    (lib().goldy_sampler_create_default)(device)
}

pub unsafe fn goldy_sampler_destroy(sampler: *mut GoldySampler) {
    (lib().goldy_sampler_destroy)(sampler)
}

pub unsafe fn goldy_shader_builtin_vertex_color_2d() -> *const std::ffi::c_char {
    (lib().goldy_shader_builtin_vertex_color_2d)()
}

pub unsafe fn goldy_shader_create(
    device: *const GoldyDevice,
    source: *const std::ffi::c_char,
) -> *mut GoldyShaderModule {
    (lib().goldy_shader_create)(device, source)
}

pub unsafe fn goldy_shader_destroy(shader: *mut GoldyShaderModule) {
    (lib().goldy_shader_destroy)(shader)
}

pub unsafe fn goldy_surface_acquire(surface: *const GoldySurface) -> *mut GoldySurfaceFrame {
    (lib().goldy_surface_acquire)(surface)
}

#[cfg(target_os = "macos")]
pub unsafe fn goldy_surface_create_appkit(
    device: *const GoldyDevice,
    ns_view: *mut std::ffi::c_void,
) -> *mut GoldySurface {
    (lib().goldy_surface_create_appkit)(device, ns_view)
}

#[cfg(windows)]
pub unsafe fn goldy_surface_create_win32(device: *const GoldyDevice, hwnd: *mut std::ffi::c_void) -> *mut GoldySurface {
    (lib().goldy_surface_create_win32)(device, hwnd)
}

pub unsafe fn goldy_surface_destroy(surface: *mut GoldySurface) {
    (lib().goldy_surface_destroy)(surface)
}

pub unsafe fn goldy_surface_format(surface: *const GoldySurface) -> GoldyTextureFormat {
    (lib().goldy_surface_format)(surface)
}

pub unsafe fn goldy_surface_frame_height(frame: *const GoldySurfaceFrame) -> u32 {
    (lib().goldy_surface_frame_height)(frame)
}

pub unsafe fn goldy_surface_frame_width(frame: *const GoldySurfaceFrame) -> u32 {
    (lib().goldy_surface_frame_width)(frame)
}

pub unsafe fn goldy_surface_height(surface: *const GoldySurface) -> u32 {
    (lib().goldy_surface_height)(surface)
}

pub unsafe fn goldy_surface_present(surface: *const GoldySurface, frame: *mut GoldySurfaceFrame) -> GoldyResult {
    (lib().goldy_surface_present)(surface, frame)
}

pub unsafe fn goldy_surface_resize(surface: *mut GoldySurface, width: u32, height: u32) -> GoldyResult {
    (lib().goldy_surface_resize)(surface, width, height)
}

pub unsafe fn goldy_surface_submit_graph_to_frame(
    surface: *const GoldySurface,
    graph: *mut GoldyTaskGraph,
    frame: *mut GoldySurfaceFrame,
) -> GoldyResult {
    (lib().goldy_surface_submit_graph_to_frame)(surface, graph, frame)
}

pub unsafe fn goldy_surface_width(surface: *const GoldySurface) -> u32 {
    (lib().goldy_surface_width)(surface)
}

pub unsafe fn goldy_task_graph_clear(graph: *mut GoldyTaskGraph) -> GoldyResult {
    (lib().goldy_task_graph_clear)(graph)
}

pub unsafe fn goldy_task_graph_copy_render_target_to_swapchain(
    graph: *mut GoldyTaskGraph,
    src: *const GoldyRenderTarget,
    swapchain: *const GoldySwapchainOutput,
) -> GoldyResult {
    (lib().goldy_task_graph_copy_render_target_to_swapchain)(graph, src, swapchain)
}

pub unsafe fn goldy_task_graph_create() -> *mut GoldyTaskGraph {
    (lib().goldy_task_graph_create)()
}

pub unsafe fn goldy_task_graph_declare_swapchain_output(graph: *mut GoldyTaskGraph) -> *mut GoldySwapchainOutput {
    (lib().goldy_task_graph_declare_swapchain_output)(graph)
}

pub unsafe fn goldy_task_graph_destroy(graph: *mut GoldyTaskGraph) {
    (lib().goldy_task_graph_destroy)(graph)
}

pub unsafe fn goldy_task_graph_dispatch(graph: *mut GoldyTaskGraph, device: *const GoldyDevice) -> GoldyResult {
    (lib().goldy_task_graph_dispatch)(graph, device)
}

pub unsafe fn goldy_task_graph_compute_node_begin(
    graph: *mut GoldyTaskGraph,
    label: *const std::ffi::c_char,
    pipeline: *const GoldyComputePipeline,
) -> GoldyResult {
    (lib().goldy_task_graph_compute_node_begin)(graph, label, pipeline)
}

pub unsafe fn goldy_task_graph_compute_node_bind_parcel(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    access: GoldyNodeAccess,
) -> GoldyResult {
    (lib().goldy_task_graph_compute_node_bind_parcel)(graph, parcel, access)
}

pub unsafe fn goldy_task_graph_compute_node_bind_parcel_view(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    slot: u32,
    access: GoldyNodeAccess,
) -> GoldyResult {
    (lib().goldy_task_graph_compute_node_bind_parcel_view)(graph, parcel, slot, access)
}

pub unsafe fn goldy_task_graph_compute_node_bind_resources_raw(
    graph: *mut GoldyTaskGraph,
    indices: *const u32,
    count: u32,
) -> GoldyResult {
    (lib().goldy_task_graph_compute_node_bind_resources_raw)(graph, indices, count)
}

pub unsafe fn goldy_task_graph_compute_node_dispatch(
    graph: *mut GoldyTaskGraph,
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
) -> GoldyResult {
    (lib().goldy_task_graph_compute_node_dispatch)(graph, workgroups_x, workgroups_y, workgroups_z)
}

pub unsafe fn goldy_task_graph_write_parcel(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    offset: u64,
    data: *const u8,
    size: usize,
) -> GoldyResult {
    (lib().goldy_task_graph_write_parcel)(graph, parcel, offset, data, size)
}

pub unsafe fn goldy_task_graph_render_pass_begin(
    graph: *mut GoldyTaskGraph,
    label: *const std::ffi::c_char,
    target: *const GoldyRenderTarget,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_begin)(graph, label, target)
}

pub unsafe fn goldy_task_graph_render_pass_bind_parcel(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    access: GoldyNodeAccess,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_bind_parcel)(graph, parcel, access)
}

pub unsafe fn goldy_task_graph_render_pass_bind_parcel_view(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    slot: u32,
    access: GoldyNodeAccess,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_bind_parcel_view)(graph, parcel, slot, access)
}

pub unsafe fn goldy_task_graph_render_pass_bind_resources_typed(
    graph: *mut GoldyTaskGraph,
    indices: *const u32,
    handle_count: u32,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_bind_resources_typed)(graph, indices, handle_count)
}

pub unsafe fn goldy_task_graph_render_pass_clear(graph: *mut GoldyTaskGraph, color: GoldyColor) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_clear)(graph, color)
}

pub unsafe fn goldy_task_graph_render_pass_clear_depth(graph: *mut GoldyTaskGraph, depth: f32) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_clear_depth)(graph, depth)
}

pub unsafe fn goldy_task_graph_render_pass_draw(
    graph: *mut GoldyTaskGraph,
    first_vertex: u32,
    vertex_count: u32,
    first_instance: u32,
    instance_count: u32,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_draw)(graph, first_vertex, vertex_count, first_instance, instance_count)
}

pub unsafe fn goldy_task_graph_render_pass_draw_fullscreen(graph: *mut GoldyTaskGraph) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_draw_fullscreen)(graph)
}

pub unsafe fn goldy_task_graph_render_pass_draw_indexed(
    graph: *mut GoldyTaskGraph,
    first_index: u32,
    index_count: u32,
    base_vertex: std::os::raw::c_int,
    first_instance: u32,
    instance_count: u32,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_draw_indexed)(
        graph,
        first_index,
        index_count,
        base_vertex,
        first_instance,
        instance_count,
    )
}

pub unsafe fn goldy_task_graph_render_pass_finish(graph: *mut GoldyTaskGraph) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_finish)(graph)
}

pub unsafe fn goldy_task_graph_render_pass_set_index_buffer(
    graph: *mut GoldyTaskGraph,
    parcel: *const GoldyParcel,
    format: GoldyIndexFormat,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_set_index_buffer)(graph, parcel, format)
}

pub unsafe fn goldy_task_graph_render_pass_set_pipeline(
    graph: *mut GoldyTaskGraph,
    pipeline: *const GoldyRenderPipeline,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_set_pipeline)(graph, pipeline)
}

pub unsafe fn goldy_task_graph_render_pass_set_vertex_buffer_parcel(
    graph: *mut GoldyTaskGraph,
    slot: u32,
    parcel: *const GoldyParcel,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_set_vertex_buffer_parcel)(graph, slot, parcel)
}

pub unsafe fn goldy_scheme_create(ctx: *const GoldyContext) -> *mut GoldyScheme {
    (lib().goldy_scheme_create)(ctx)
}

pub unsafe fn goldy_scheme_destroy(scheme: *mut GoldyScheme) {
    (lib().goldy_scheme_destroy)(scheme)
}

pub unsafe fn goldy_scheme_len(scheme: *const GoldyScheme) -> u32 {
    (lib().goldy_scheme_len)(scheme)
}

pub unsafe fn goldy_scheme_is_dirty(scheme: *const GoldyScheme) -> bool {
    (lib().goldy_scheme_is_dirty)(scheme)
}

pub unsafe fn goldy_scheme_replay_stats(scheme: *const GoldyScheme, out_stats: *mut GoldyReplayStats) -> GoldyResult {
    (lib().goldy_scheme_replay_stats)(scheme, out_stats)
}

pub unsafe fn goldy_scheme_compute_node_begin(
    scheme: *mut GoldyScheme,
    label: *const std::ffi::c_char,
    pipeline: *const GoldyComputePipeline,
) -> GoldyResult {
    (lib().goldy_scheme_compute_node_begin)(scheme, label, pipeline)
}

pub unsafe fn goldy_scheme_compute_node_declare_parcel(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
    node_access: GoldyNodeAccess,
    resource_access: GoldyResourceAccess,
) -> GoldyResult {
    (lib().goldy_scheme_compute_node_declare_parcel)(scheme, parcel, node_access, resource_access)
}

pub unsafe fn goldy_scheme_compute_node_declare_parcel_view(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
    slot: u32,
    node_access: GoldyNodeAccess,
    resource_access: GoldyResourceAccess,
) -> GoldyResult {
    (lib().goldy_scheme_compute_node_declare_parcel_view)(scheme, parcel, slot, node_access, resource_access)
}

pub unsafe fn goldy_scheme_compute_node_dispatch(
    scheme: *mut GoldyScheme,
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
) -> GoldyResult {
    (lib().goldy_scheme_compute_node_dispatch)(scheme, workgroups_x, workgroups_y, workgroups_z)
}

pub unsafe fn goldy_scheme_submit(scheme: *mut GoldyScheme, out_frame: *mut *mut GoldySchemeFrame) -> GoldyResult {
    (lib().goldy_scheme_submit)(scheme, out_frame)
}

pub unsafe fn goldy_scheme_frame_destroy(frame: *mut GoldySchemeFrame) {
    (lib().goldy_scheme_frame_destroy)(frame)
}

pub unsafe fn goldy_scheme_frame_timeline_value(frame: *const GoldySchemeFrame) -> u64 {
    (lib().goldy_scheme_frame_timeline_value)(frame)
}

pub unsafe fn goldy_scheme_frame_wait(ctx: *const GoldyContext, frame: *const GoldySchemeFrame) -> GoldyResult {
    (lib().goldy_scheme_frame_wait)(ctx, frame)
}

pub unsafe fn goldy_scheme_grant_read(scheme: *mut GoldyScheme, parcel: *const GoldyParcel) -> *mut GoldyReadGrant {
    (lib().goldy_scheme_grant_read)(scheme, parcel)
}

pub unsafe fn goldy_read_grant_destroy(grant: *mut GoldyReadGrant) {
    (lib().goldy_read_grant_destroy)(grant)
}

pub unsafe fn goldy_read_grant_byte_size(grant: *const GoldyReadGrant) -> u64 {
    (lib().goldy_read_grant_byte_size)(grant)
}

pub unsafe fn goldy_read_grant_read(
    grant: *const GoldyReadGrant,
    frame: *const GoldySchemeFrame,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    (lib().goldy_read_grant_read)(grant, frame, output, output_size)
}
