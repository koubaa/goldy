//! FFI bindings for CommandEncoder.

use crate::bind_group::GoldyBindGroup;
use crate::buffer::GoldyBuffer;
use crate::pipeline::GoldyRenderPipeline;
use crate::types::{GoldyColor, GoldyIndexFormat};

/// Opaque handle to a Goldy CommandEncoder.
pub struct GoldyCommandEncoder {
    pub(crate) inner: goldy::CommandEncoder,
}

/// Create a new command encoder.
#[no_mangle]
pub extern "C" fn goldy_encoder_create() -> *mut GoldyCommandEncoder {
    Box::into_raw(Box::new(GoldyCommandEncoder {
        inner: goldy::CommandEncoder::new(),
    }))
}

/// Destroy a command encoder without rendering.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_encoder_destroy(encoder: *mut GoldyCommandEncoder) {
    if !encoder.is_null() {
        drop(Box::from_raw(encoder));
    }
}

/// Clear the color render target.
///
/// # Safety
/// The encoder pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_encoder_clear(encoder: *mut GoldyCommandEncoder, color: GoldyColor) {
    if encoder.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_render_pass();
    pass.clear(color.into());
}

/// Clear the depth buffer.
///
/// # Safety
/// The encoder pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_encoder_clear_depth(encoder: *mut GoldyCommandEncoder, depth: f32) {
    if encoder.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_render_pass();
    pass.clear_depth(depth);
}

/// Set the render pipeline.
///
/// # Safety
/// Both pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_encoder_set_pipeline(
    encoder: *mut GoldyCommandEncoder,
    pipeline: *const GoldyRenderPipeline,
) {
    if encoder.is_null() || pipeline.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_render_pass();
    pass.set_pipeline(&(*pipeline).inner);
}

/// Set a vertex buffer.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_encoder_set_vertex_buffer(
    encoder: *mut GoldyCommandEncoder,
    slot: u32,
    buffer: *const GoldyBuffer,
) {
    if encoder.is_null() || buffer.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_render_pass();
    pass.set_vertex_buffer(slot, &(*buffer).inner);
}

/// Set a vertex buffer with offset.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_encoder_set_vertex_buffer_offset(
    encoder: *mut GoldyCommandEncoder,
    slot: u32,
    buffer: *const GoldyBuffer,
    offset: u64,
) {
    if encoder.is_null() || buffer.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_render_pass();
    pass.set_vertex_buffer_offset(slot, &(*buffer).inner, offset);
}

/// Set a bind group.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_encoder_set_bind_group(
    encoder: *mut GoldyCommandEncoder,
    index: u32,
    bind_group: *const GoldyBindGroup,
) {
    if encoder.is_null() || bind_group.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_render_pass();
    pass.set_bind_group(index, &(*bind_group).inner);
}

/// Set an index buffer.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_encoder_set_index_buffer(
    encoder: *mut GoldyCommandEncoder,
    buffer: *const GoldyBuffer,
    format: GoldyIndexFormat,
) {
    if encoder.is_null() || buffer.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_render_pass();
    pass.set_index_buffer(&(*buffer).inner, format.into());
}

/// Draw primitives.
///
/// # Safety
/// The encoder pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_encoder_draw(
    encoder: *mut GoldyCommandEncoder,
    vertex_start: u32,
    vertex_count: u32,
    instance_start: u32,
    instance_count: u32,
) {
    if encoder.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_render_pass();
    pass.draw(
        vertex_start..(vertex_start + vertex_count),
        instance_start..(instance_start + instance_count),
    );
}

/// Draw indexed primitives.
///
/// # Safety
/// The encoder pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_encoder_draw_indexed(
    encoder: *mut GoldyCommandEncoder,
    index_start: u32,
    index_count: u32,
    base_vertex: i32,
    instance_start: u32,
    instance_count: u32,
) {
    if encoder.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_render_pass();
    pass.draw_indexed(
        index_start..(index_start + index_count),
        base_vertex,
        instance_start..(instance_start + instance_count),
    );
}
