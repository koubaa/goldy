//! Raw C bindings loaded from `goldy_ffi` at runtime via libloading.
#![allow(non_camel_case_types, non_snake_case, dead_code, improper_ctypes)]

mod ffi;
mod loader;
mod types;

pub use types::*;

use loader::lib;

pub unsafe fn goldy_buffer_access(buffer: *const GoldyBuffer) -> GoldyBufferKind {
    (lib().goldy_buffer_access)(buffer)
}

pub unsafe fn goldy_buffer_create(device: *const GoldyDevice, size: u64, access: GoldyBufferKind) -> *mut GoldyBuffer {
    (lib().goldy_buffer_create)(device, size, access)
}

pub unsafe fn goldy_buffer_create_with_data(
    device: *const GoldyDevice,
    data: *const u8,
    size: usize,
    access: GoldyBufferKind,
) -> *mut GoldyBuffer {
    (lib().goldy_buffer_create_with_data)(device, data, size, access)
}

pub unsafe fn goldy_buffer_create_with_data_stride(
    device: *const GoldyDevice,
    data: *const u8,
    size: usize,
    access: GoldyBufferKind,
    element_stride: u32,
) -> *mut GoldyBuffer {
    (lib().goldy_buffer_create_with_data_stride)(device, data, size, access, element_stride)
}

pub unsafe fn goldy_buffer_destroy(buffer: *mut GoldyBuffer) {
    (lib().goldy_buffer_destroy)(buffer)
}

pub unsafe fn goldy_buffer_size(buffer: *const GoldyBuffer) -> u64 {
    (lib().goldy_buffer_size)(buffer)
}

pub unsafe fn goldy_buffer_read_to_cpu(
    buffer: *const GoldyBuffer,
    device: *const GoldyDevice,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    (lib().goldy_buffer_read_to_cpu)(buffer, device, output, output_size)
}

pub unsafe fn goldy_buffer_resource_index(buffer: *const GoldyBuffer, access: GoldyResourceAccess) -> u32 {
    (lib().goldy_buffer_resource_index)(buffer, access)
}

pub unsafe fn goldy_buffer_write(buffer: *const GoldyBuffer, offset: u64, data: *const u8, size: usize) -> GoldyResult {
    (lib().goldy_buffer_write)(buffer, offset, data, size)
}

pub unsafe fn goldy_clear_error() {
    (lib().goldy_clear_error)()
}

pub unsafe fn goldy_compute_encoder_bind_resources(
    encoder: *mut GoldyComputeEncoder,
    buffers: *const *const GoldyBuffer,
    buffer_count: u32,
) {
    (lib().goldy_compute_encoder_bind_resources)(encoder, buffers, buffer_count)
}

pub unsafe fn goldy_compute_encoder_create() -> *mut GoldyComputeEncoder {
    (lib().goldy_compute_encoder_create)()
}

pub unsafe fn goldy_compute_encoder_destroy(encoder: *mut GoldyComputeEncoder) {
    (lib().goldy_compute_encoder_destroy)(encoder)
}

pub unsafe fn goldy_compute_encoder_dispatch(
    encoder: *mut GoldyComputeEncoder,
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
) {
    (lib().goldy_compute_encoder_dispatch)(encoder, workgroups_x, workgroups_y, workgroups_z)
}

pub unsafe fn goldy_compute_encoder_execute(
    encoder: *const GoldyComputeEncoder,
    device: *const GoldyDevice,
) -> GoldyResult {
    (lib().goldy_compute_encoder_execute)(encoder, device)
}

pub unsafe fn goldy_compute_encoder_set_pipeline(
    encoder: *mut GoldyComputeEncoder,
    pipeline: *const GoldyComputePipeline,
) {
    (lib().goldy_compute_encoder_set_pipeline)(encoder, pipeline)
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

pub unsafe fn goldy_task_graph_compute_node_bind_buffer(
    graph: *mut GoldyTaskGraph,
    buffer: *const GoldyBuffer,
    access: GoldyNodeAccess,
) -> GoldyResult {
    (lib().goldy_task_graph_compute_node_bind_buffer)(graph, buffer, access)
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

pub unsafe fn goldy_task_graph_write_buffer(
    graph: *mut GoldyTaskGraph,
    buffer: *const GoldyBuffer,
    offset: u64,
    data: *const u8,
    size: usize,
) -> GoldyResult {
    (lib().goldy_task_graph_write_buffer)(graph, buffer, offset, data, size)
}

pub unsafe fn goldy_task_graph_render_pass_begin(
    graph: *mut GoldyTaskGraph,
    label: *const std::ffi::c_char,
    target: *const GoldyRenderTarget,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_begin)(graph, label, target)
}

pub unsafe fn goldy_task_graph_render_pass_bind_buffer(
    graph: *mut GoldyTaskGraph,
    buffer: *const GoldyBuffer,
    access: GoldyNodeAccess,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_bind_buffer)(graph, buffer, access)
}

pub unsafe fn goldy_task_graph_render_pass_bind_resources(
    graph: *mut GoldyTaskGraph,
    buffers: *const *const GoldyBuffer,
    buffer_count: u32,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_bind_resources)(graph, buffers, buffer_count)
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
    buffer: *const GoldyBuffer,
    format: GoldyIndexFormat,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_set_index_buffer)(graph, buffer, format)
}

pub unsafe fn goldy_task_graph_render_pass_set_pipeline(
    graph: *mut GoldyTaskGraph,
    pipeline: *const GoldyRenderPipeline,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_set_pipeline)(graph, pipeline)
}

pub unsafe fn goldy_task_graph_render_pass_set_vertex_buffer(
    graph: *mut GoldyTaskGraph,
    slot: u32,
    buffer: *const GoldyBuffer,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_set_vertex_buffer)(graph, slot, buffer)
}

pub unsafe fn goldy_task_graph_render_pass_set_vertex_buffer_offset(
    graph: *mut GoldyTaskGraph,
    slot: u32,
    buffer: *const GoldyBuffer,
    offset: u64,
) -> GoldyResult {
    (lib().goldy_task_graph_render_pass_set_vertex_buffer_offset)(graph, slot, buffer, offset)
}

pub unsafe fn goldy_texture_create(
    device: *const GoldyDevice,
    width: u32,
    height: u32,
    format: GoldyTextureFormat,
    access: GoldyTextureKind,
    flags: GoldyTextureFlags,
) -> *mut GoldyTexture {
    (lib().goldy_texture_create)(device, width, height, format, access, flags)
}

pub unsafe fn goldy_texture_destroy(texture: *mut GoldyTexture) {
    (lib().goldy_texture_destroy)(texture)
}

pub unsafe fn goldy_texture_format(texture: *const GoldyTexture) -> GoldyTextureFormat {
    (lib().goldy_texture_format)(texture)
}

pub unsafe fn goldy_texture_height(texture: *const GoldyTexture) -> u32 {
    (lib().goldy_texture_height)(texture)
}

pub unsafe fn goldy_texture_width(texture: *const GoldyTexture) -> u32 {
    (lib().goldy_texture_width)(texture)
}
