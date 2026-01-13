//! FFI bindings for ComputePipeline and ComputeEncoder.

use crate::bind_group::{GoldyBindGroup, GoldyBindGroupLayout};
use crate::device::GoldyDevice;
use crate::error::{set_last_error_from_anyhow, GoldyResult};
use crate::shader::GoldyShaderModule;
use std::ptr;
use std::slice;

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
    bind_group_layouts: *const *const GoldyBindGroupLayout,
    bind_group_layout_count: u32,
) -> *mut GoldyComputePipeline {
    if device.is_null() || compute_shader.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device or shader is null"));
        return ptr::null_mut();
    }

    // Collect bind group layouts
    let layouts: Vec<&goldy::BindGroupLayout> =
        if bind_group_layout_count > 0 && !bind_group_layouts.is_null() {
            slice::from_raw_parts(bind_group_layouts, bind_group_layout_count as usize)
                .iter()
                .map(|&ptr| &(*ptr).inner)
                .collect()
        } else {
            vec![]
        };

    let layout_refs: Vec<&goldy::BindGroupLayout> = layouts.iter().copied().collect();

    let desc = goldy::ComputePipelineDesc {
        bind_group_layouts: &layout_refs,
    };

    match goldy::ComputePipeline::new(&(*device).inner, &(*compute_shader).inner, &desc) {
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

/// Set a bind group for compute.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_compute_encoder_set_bind_group(
    encoder: *mut GoldyComputeEncoder,
    index: u32,
    bind_group: *const GoldyBindGroup,
) {
    if encoder.is_null() || bind_group.is_null() {
        return;
    }
    let mut pass = (*encoder).inner.begin_compute_pass();
    pass.set_bind_group(index, &(*bind_group).inner);
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
