//! FFI bindings for ComputePipeline.

use crate::device::GoldyDevice;
use crate::error::set_last_error_from_anyhow;
use crate::shader::GoldyShaderModule;
use std::ptr;

/// Opaque handle to a Goldy ComputePipeline.
pub struct GoldyComputePipeline {
    pub(crate) inner: goldy::ComputePipeline,
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
