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
) -> *mut GoldyBuffer {
    (lib().goldy_retained_pool_acquire_buffer)(pool, size, access, element_stride, data, data_size)
}

pub unsafe fn goldy_record_builder_create() -> *mut GoldyRecordBuilder {
    (lib().goldy_record_builder_create)()
}

pub unsafe fn goldy_record_builder_destroy(builder: *mut GoldyRecordBuilder) {
    (lib().goldy_record_builder_destroy)(builder)
}

pub unsafe fn goldy_record_builder_emplace(
    builder: *mut GoldyRecordBuilder,
    name: *const std::ffi::c_char,
    data: *const u8,
    data_size: usize,
    element_count: u64,
    element_stride: u32,
) -> u32 {
    (lib().goldy_record_builder_emplace)(builder, name, data, data_size, element_count, element_stride)
}

pub unsafe fn goldy_record_builder_build(
    builder: *mut GoldyRecordBuilder,
    pool: *mut GoldyRetainedPool,
) -> *mut GoldyBuffer {
    (lib().goldy_record_builder_build)(builder, pool)
}

pub unsafe fn goldy_buffer_destroy(buffer: *mut GoldyBuffer) {
    (lib().goldy_buffer_destroy)(buffer)
}

pub unsafe fn goldy_buffer_byte_size(buffer: *const GoldyBuffer) -> u64 {
    (lib().goldy_buffer_byte_size)(buffer)
}

pub unsafe fn goldy_buffer_unit_count(buffer: *const GoldyBuffer) -> u32 {
    (lib().goldy_buffer_unit_count)(buffer)
}

pub unsafe fn goldy_buffer_unit_byte_size(buffer: *const GoldyBuffer, unit: u32) -> u64 {
    (lib().goldy_buffer_unit_byte_size)(buffer, unit)
}

pub unsafe fn goldy_buffer_unit_read_to_cpu(
    buffer: *const GoldyBuffer,
    unit: u32,
    device: *const GoldyDevice,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    (lib().goldy_buffer_unit_read_to_cpu)(buffer, unit, device, output, output_size)
}

pub unsafe fn goldy_buffer_field(buffer: *const GoldyBuffer, unit: u32) -> *mut GoldyParcel {
    (lib().goldy_buffer_field)(buffer, unit)
}

pub unsafe fn goldy_texture_destroy(texture: *mut GoldyTexture) {
    (lib().goldy_texture_destroy)(texture)
}

pub unsafe fn goldy_texture_byte_size(texture: *const GoldyTexture) -> u64 {
    (lib().goldy_texture_byte_size)(texture)
}

pub unsafe fn goldy_parcel_destroy(parcel: *mut GoldyParcel) {
    (lib().goldy_parcel_destroy)(parcel)
}

pub unsafe fn goldy_parcel_byte_size(parcel: *const GoldyParcel) -> u64 {
    (lib().goldy_parcel_byte_size)(parcel)
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

pub unsafe fn goldy_scheme_compute_node_with_parcel(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
    node_access: GoldyNodeAccess,
) -> GoldyResult {
    (lib().goldy_scheme_compute_node_with_parcel)(scheme, parcel, node_access)
}

pub unsafe fn goldy_scheme_compute_node_with_buffer_unit(
    scheme: *mut GoldyScheme,
    buffer: *const GoldyBuffer,
    unit: u32,
    node_access: GoldyNodeAccess,
) -> GoldyResult {
    (lib().goldy_scheme_compute_node_with_buffer_unit)(scheme, buffer, unit, node_access)
}

pub unsafe fn goldy_scheme_compute_node_with_param(scheme: *mut GoldyScheme, value: u32) -> GoldyResult {
    (lib().goldy_scheme_compute_node_with_param)(scheme, value)
}

pub unsafe fn goldy_scheme_compute_node_dispatch(
    scheme: *mut GoldyScheme,
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
) -> GoldyResult {
    (lib().goldy_scheme_compute_node_dispatch)(scheme, workgroups_x, workgroups_y, workgroups_z)
}

pub unsafe fn goldy_scheme_submit(
    scheme: *mut GoldyScheme,
    out_submission: *mut *mut GoldySchemeSubmission,
) -> GoldyResult {
    (lib().goldy_scheme_submit)(scheme, out_submission)
}

pub unsafe fn goldy_scheme_submission_destroy(submission: *mut GoldySchemeSubmission) {
    (lib().goldy_scheme_submission_destroy)(submission)
}

pub unsafe fn goldy_scheme_submission_timeline_value(submission: *const GoldySchemeSubmission) -> u64 {
    (lib().goldy_scheme_submission_timeline_value)(submission)
}

pub unsafe fn goldy_scheme_submission_wait(
    ctx: *const GoldyContext,
    submission: *const GoldySchemeSubmission,
) -> GoldyResult {
    (lib().goldy_scheme_submission_wait)(ctx, submission)
}

pub unsafe fn goldy_scheme_grant_read(scheme: *mut GoldyScheme, buffer: *const GoldyBuffer) -> *mut GoldyReadGrant {
    (lib().goldy_scheme_grant_read)(scheme, buffer)
}

pub unsafe fn goldy_read_grant_destroy(grant: *mut GoldyReadGrant) {
    (lib().goldy_read_grant_destroy)(grant)
}

pub unsafe fn goldy_read_grant_byte_size(grant: *const GoldyReadGrant) -> u64 {
    (lib().goldy_read_grant_byte_size)(grant)
}

pub unsafe fn goldy_read_grant_consume(
    grant: *const GoldyReadGrant,
    submission: *const GoldySchemeSubmission,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    (lib().goldy_read_grant_consume)(grant, submission, output, output_size)
}

pub unsafe fn goldy_scheme_lease_render_target(
    scheme: *mut GoldyScheme,
    width: u32,
    height: u32,
    format: GoldyTextureFormat,
    has_depth: bool,
    depth_format: GoldyDepthFormat,
) -> *mut GoldySchemeRenderTargetLease {
    (lib().goldy_scheme_lease_render_target)(scheme, width, height, format, has_depth, depth_format)
}

pub unsafe fn goldy_scheme_render_target_lease_destroy(lease: *mut GoldySchemeRenderTargetLease) {
    (lib().goldy_scheme_render_target_lease_destroy)(lease)
}

pub unsafe fn goldy_scheme_render_pass_begin(
    scheme: *mut GoldyScheme,
    label: *const std::ffi::c_char,
    lease: *const GoldySchemeRenderTargetLease,
) -> GoldyResult {
    (lib().goldy_scheme_render_pass_begin)(scheme, label, lease)
}

pub unsafe fn goldy_scheme_render_pass_with_buffer_unit(
    scheme: *mut GoldyScheme,
    buffer: *const GoldyBuffer,
    unit: u32,
    access: GoldyNodeAccess,
) -> GoldyResult {
    (lib().goldy_scheme_render_pass_with_buffer_unit)(scheme, buffer, unit, access)
}

pub unsafe fn goldy_scheme_render_pass_with_parcel(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
    access: GoldyNodeAccess,
) -> GoldyResult {
    (lib().goldy_scheme_render_pass_with_parcel)(scheme, parcel, access)
}

pub unsafe fn goldy_scheme_render_pass_clear(scheme: *mut GoldyScheme, color: GoldyColor) -> GoldyResult {
    (lib().goldy_scheme_render_pass_clear)(scheme, color)
}

pub unsafe fn goldy_scheme_render_pass_clear_depth(scheme: *mut GoldyScheme, depth: f32) -> GoldyResult {
    (lib().goldy_scheme_render_pass_clear_depth)(scheme, depth)
}

pub unsafe fn goldy_scheme_render_pass_set_pipeline(
    scheme: *mut GoldyScheme,
    pipeline: *const GoldyRenderPipeline,
) -> GoldyResult {
    (lib().goldy_scheme_render_pass_set_pipeline)(scheme, pipeline)
}

pub unsafe fn goldy_scheme_render_pass_set_vertex_buffer_parcel(
    scheme: *mut GoldyScheme,
    slot: u32,
    parcel: *const GoldyParcel,
) -> GoldyResult {
    (lib().goldy_scheme_render_pass_set_vertex_buffer_parcel)(scheme, slot, parcel)
}

pub unsafe fn goldy_scheme_render_pass_set_index_buffer(
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
    format: GoldyIndexFormat,
) -> GoldyResult {
    (lib().goldy_scheme_render_pass_set_index_buffer)(scheme, parcel, format)
}

pub unsafe fn goldy_scheme_render_pass_draw(
    scheme: *mut GoldyScheme,
    first_vertex: u32,
    vertex_count: u32,
    first_instance: u32,
    instance_count: u32,
) -> GoldyResult {
    (lib().goldy_scheme_render_pass_draw)(scheme, first_vertex, vertex_count, first_instance, instance_count)
}

pub unsafe fn goldy_scheme_render_pass_draw_indexed(
    scheme: *mut GoldyScheme,
    first_index: u32,
    index_count: u32,
    base_vertex: i32,
    first_instance: u32,
    instance_count: u32,
) -> GoldyResult {
    (lib().goldy_scheme_render_pass_draw_indexed)(
        scheme,
        first_index,
        index_count,
        base_vertex,
        first_instance,
        instance_count,
    )
}

pub unsafe fn goldy_scheme_render_pass_draw_fullscreen(scheme: *mut GoldyScheme) -> GoldyResult {
    (lib().goldy_scheme_render_pass_draw_fullscreen)(scheme)
}

pub unsafe fn goldy_scheme_render_pass_finish(scheme: *mut GoldyScheme) -> GoldyResult {
    (lib().goldy_scheme_render_pass_finish)(scheme)
}

pub unsafe fn goldy_scheme_copy_to_texture(
    scheme: *mut GoldyScheme,
    src_lease: *const GoldySchemeRenderTargetLease,
    dst_texture: *const GoldyTexture,
) -> GoldyResult {
    (lib().goldy_scheme_copy_to_texture)(scheme, src_lease, dst_texture)
}

pub unsafe fn goldy_scheme_grant_read_texture(
    scheme: *mut GoldyScheme,
    texture: *const GoldyTexture,
) -> *mut GoldyReadGrant {
    (lib().goldy_scheme_grant_read_texture)(scheme, texture)
}

pub unsafe fn goldy_retained_pool_acquire_texture(
    pool: *mut GoldyRetainedPool,
    width: u32,
    height: u32,
    format: GoldyTextureFormat,
    access: GoldyTextureKind,
    flags: GoldyTextureFlags,
    data: *const u8,
    data_size: usize,
) -> *mut GoldyTexture {
    (lib().goldy_retained_pool_acquire_texture)(pool, width, height, format, access, flags, data, data_size)
}

pub unsafe fn goldy_present_lease_destroy(lease: *mut GoldyPresentLease) {
    (lib().goldy_present_lease_destroy)(lease)
}

pub unsafe fn goldy_surface_exchange_destroy(exchange: *mut GoldySurfaceExchange) {
    (lib().goldy_surface_exchange_destroy)(exchange)
}

pub unsafe fn goldy_surface_exchange_width(exchange: *const GoldySurfaceExchange) -> u32 {
    (lib().goldy_surface_exchange_width)(exchange)
}

pub unsafe fn goldy_surface_exchange_height(exchange: *const GoldySurfaceExchange) -> u32 {
    (lib().goldy_surface_exchange_height)(exchange)
}

pub unsafe fn goldy_surface_exchange_format(exchange: *const GoldySurfaceExchange) -> GoldyTextureFormat {
    (lib().goldy_surface_exchange_format)(exchange)
}

pub unsafe fn goldy_surface_exchange_generation(exchange: *const GoldySurfaceExchange) -> u64 {
    (lib().goldy_surface_exchange_generation)(exchange)
}

pub unsafe fn goldy_surface_exchange_resize(
    exchange: *mut GoldySurfaceExchange,
    width: u32,
    height: u32,
) -> GoldyResult {
    (lib().goldy_surface_exchange_resize)(exchange, width, height)
}

pub unsafe fn goldy_surface_exchange_lease(exchange: *const GoldySurfaceExchange) -> *mut GoldyPresentLease {
    (lib().goldy_surface_exchange_lease)(exchange)
}

pub unsafe fn goldy_surface_exchange_bind_render_target(
    exchange: *const GoldySurfaceExchange,
    scheme: *mut GoldyScheme,
    src_lease: *const GoldySchemeRenderTargetLease,
) -> *mut GoldyTransaction {
    (lib().goldy_surface_exchange_bind_render_target)(exchange, scheme, src_lease)
}

pub unsafe fn goldy_surface_exchange_bind(
    exchange: *const GoldySurfaceExchange,
    scheme: *mut GoldyScheme,
    source: *const GoldyTexture,
) -> *mut GoldyTransaction {
    (lib().goldy_surface_exchange_bind)(exchange, scheme, source)
}

pub unsafe fn goldy_surface_exchange_bind_destination(
    exchange: *const GoldySurfaceExchange,
    scheme: *mut GoldyScheme,
    out: *mut GoldySurfaceExchangeBindDestinationOut,
) -> GoldyResult {
    (lib().goldy_surface_exchange_bind_destination)(exchange, scheme, out)
}

pub unsafe fn goldy_transaction_destroy(transaction: *mut GoldyTransaction) {
    (lib().goldy_transaction_destroy)(transaction)
}

pub unsafe fn goldy_transaction_binding_id(transaction: *const GoldyTransaction) -> u32 {
    (lib().goldy_transaction_binding_id)(transaction)
}

pub unsafe fn goldy_transaction_generation(transaction: *const GoldyTransaction) -> u64 {
    (lib().goldy_transaction_generation)(transaction)
}

pub unsafe fn goldy_transaction_claim(
    transaction: *const GoldyTransaction,
    submission: *mut GoldySchemeSubmission,
) -> *mut GoldyClaim {
    (lib().goldy_transaction_claim)(transaction, submission)
}

pub unsafe fn goldy_claim_destroy(claim: *mut GoldyClaim) {
    (lib().goldy_claim_destroy)(claim)
}

pub unsafe fn goldy_claim_consume(claim: *mut GoldyClaim) -> GoldyResult {
    (lib().goldy_claim_consume)(claim)
}

pub unsafe fn goldy_claim_discard(claim: *mut GoldyClaim) -> GoldyResult {
    (lib().goldy_claim_discard)(claim)
}

#[cfg(windows)]
pub unsafe fn goldy_surface_exchange_create_win32(
    ctx: *const GoldyContext,
    hwnd: *mut std::ffi::c_void,
    depth: u32,
) -> *mut GoldySurfaceExchange {
    (lib().goldy_surface_exchange_create_win32)(ctx, hwnd, depth)
}

#[cfg(target_os = "macos")]
pub unsafe fn goldy_surface_exchange_create_appkit(
    ctx: *const GoldyContext,
    ns_view: *mut std::ffi::c_void,
    depth: u32,
) -> *mut GoldySurfaceExchange {
    (lib().goldy_surface_exchange_create_appkit)(ctx, ns_view, depth)
}

#[cfg(target_os = "linux")]
pub unsafe fn goldy_surface_exchange_create_wayland(
    ctx: *const GoldyContext,
    display: *mut std::ffi::c_void,
    surface: *mut std::ffi::c_void,
    depth: u32,
) -> *mut GoldySurfaceExchange {
    (lib().goldy_surface_exchange_create_wayland)(ctx, display, surface, depth)
}
