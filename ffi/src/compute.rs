//! FFI bindings for ComputePipeline and ComputeEncoder.

use crate::device::GoldyDevice;
use crate::error::{set_last_error_from_anyhow, GoldyResult};
use crate::shader::GoldyShaderModule;
use std::ptr;

/// Opaque handle to a Goldy ComputePipeline.
pub struct GoldyComputePipeline {
    pub(crate) inner: goldy::ComputePipeline,
}

/// Opaque handle to a Goldy ComputeEncoder.
pub struct GoldyComputeEncoder {
    pub(crate) inner: goldy::ComputeEncoder,
}

/// Create a new compute pipeline.
///
/// Returns a pointer to the pipeline, or null on failure.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_compute_pipeline_create(
    device: *const GoldyDevice,
    compute_shader: *const GoldyShaderModule,
) -> *mut GoldyComputePipeline {
    if device.is_null() || compute_shader.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device or shader is null"));
        return ptr::null_mut();
    }

    match goldy::ComputePipeline::new(&(*device).inner, &(*compute_shader).inner) {
        Ok(pipeline) => Box::into_raw(Box::new(GoldyComputePipeline { inner: pipeline })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Destroy a compute pipeline.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_compute_pipeline_destroy(pipeline: *mut GoldyComputePipeline) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}

/// Create a new compute encoder.
#[no_mangle]
pub extern "C" fn goldy_compute_encoder_create() -> *mut GoldyComputeEncoder {
    Box::into_raw(Box::new(GoldyComputeEncoder {
        inner: goldy::ComputeEncoder::new(),
    }))
}

/// Destroy a compute encoder without dispatching.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_compute_encoder_destroy(encoder: *mut GoldyComputeEncoder) {
    if !encoder.is_null() {
        drop(Box::from_raw(encoder));
    }
}

/// Set the compute pipeline.
///
/// # Safety
/// Both pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_compute_encoder_set_pipeline(
    encoder: *mut GoldyComputeEncoder,
    pipeline: *const GoldyComputePipeline,
) {
    if encoder.is_null() || pipeline.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_compute_pass();
    pass.set_pipeline(&(*pipeline).inner);
}

/// Set push constants for compute resource binding.
///
/// Pass the buffers whose indices should be pushed to the shader.
/// The indices are pushed in order, so `buffers[0]` becomes index 0,
/// `buffers[1]` becomes index 1, etc.
///
/// # Safety
/// All pointers must be valid. The buffers array must contain buffer_count elements.
#[no_mangle]
pub unsafe extern "C" fn goldy_compute_encoder_set_push_constants(
    encoder: *mut GoldyComputeEncoder,
    buffers: *const *const crate::buffer::GoldyBuffer,
    buffer_count: u32,
) {
    if encoder.is_null() || (buffer_count > 0 && buffers.is_null()) {
        return;
    }

    // Convert array of buffer pointers to slice of Buffer references
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

    let mut pass = (*encoder).inner.begin_compute_pass();
    pass.set_push_constants(&buffer_refs);
}

/// Dispatch compute workgroups.
///
/// # Safety
/// The encoder pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_compute_encoder_dispatch(
    encoder: *mut GoldyComputeEncoder,
    workgroups_x: u32,
    workgroups_y: u32,
    workgroups_z: u32,
) {
    if encoder.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_compute_pass();
    pass.dispatch(workgroups_x, workgroups_y, workgroups_z);
}

/// Execute the compute commands on the device.
///
/// # Safety
/// Both pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_compute_encoder_execute(
    encoder: *const GoldyComputeEncoder,
    device: *const GoldyDevice,
) -> GoldyResult {
    if encoder.is_null() || device.is_null() {
        return GoldyResult::NullPointer;
    }

    match (*encoder).inner.dispatch(&(*device).inner) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}
