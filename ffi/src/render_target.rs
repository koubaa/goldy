//! FFI bindings for RenderTarget.

use crate::device::GoldyDevice;
use crate::encoder::GoldyCommandEncoder;
use crate::error::{set_last_error_from_anyhow, GoldyResult};
use crate::types::{GoldyDepthFormat, GoldyTextureFormat};
use std::ptr;
use std::slice;

/// Opaque handle to a Goldy RenderTarget.
pub struct GoldyRenderTarget {
    pub(crate) inner: goldy::RenderTarget,
}

/// Create a new render target without depth buffer.
///
/// Returns a pointer to the render target, or null on failure.
///
/// # Safety
/// The device pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_target_create(
    device: *const GoldyDevice,
    width: u32,
    height: u32,
    format: GoldyTextureFormat,
) -> *mut GoldyRenderTarget {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }
    
    match goldy::RenderTarget::new(&(*device).inner, width, height, format.into()) {
        Ok(target) => Box::into_raw(Box::new(GoldyRenderTarget { inner: target })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Create a new render target with depth buffer.
///
/// Returns a pointer to the render target, or null on failure.
///
/// # Safety
/// The device pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_target_create_with_depth(
    device: *const GoldyDevice,
    width: u32,
    height: u32,
    color_format: GoldyTextureFormat,
    depth_format: GoldyDepthFormat,
) -> *mut GoldyRenderTarget {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }
    
    match goldy::RenderTarget::new_with_depth(
        &(*device).inner,
        width,
        height,
        color_format.into(),
        Some(depth_format.into()),
    ) {
        Ok(target) => Box::into_raw(Box::new(GoldyRenderTarget { inner: target })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Destroy a render target.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_target_destroy(target: *mut GoldyRenderTarget) {
    if !target.is_null() {
        drop(Box::from_raw(target));
    }
}

/// Get the render target width.
///
/// # Safety
/// The target pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_target_width(target: *const GoldyRenderTarget) -> u32 {
    if target.is_null() {
        return 0;
    }
    (*target).inner.width()
}

/// Get the render target height.
///
/// # Safety
/// The target pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_target_height(target: *const GoldyRenderTarget) -> u32 {
    if target.is_null() {
        return 0;
    }
    (*target).inner.height()
}

/// Get the render target format.
///
/// # Safety
/// The target pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_target_format(target: *const GoldyRenderTarget) -> GoldyTextureFormat {
    if target.is_null() {
        return GoldyTextureFormat::Rgba8Unorm;
    }
    (*target).inner.format().into()
}

/// Check if the render target has a depth buffer.
///
/// # Safety
/// The target pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_target_has_depth(target: *const GoldyRenderTarget) -> bool {
    if target.is_null() {
        return false;
    }
    (*target).inner.has_depth()
}

/// Get the buffer size in bytes.
///
/// # Safety
/// The target pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_target_buffer_size(target: *const GoldyRenderTarget) -> usize {
    if target.is_null() {
        return 0;
    }
    (*target).inner.buffer_size()
}

/// Render commands to the target.
///
/// This consumes the encoder.
///
/// # Safety
/// Both pointers must be valid.
/// The encoder is consumed and must not be used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_target_render(
    target: *const GoldyRenderTarget,
    encoder: *mut GoldyCommandEncoder,
) -> GoldyResult {
    if target.is_null() || encoder.is_null() {
        return GoldyResult::NullPointer;
    }
    
    let encoder = Box::from_raw(encoder);
    match (*target).inner.render(encoder.inner) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

/// Read the rendered pixels to a CPU buffer.
///
/// # Safety
/// The target pointer must be valid.
/// The output pointer must point to a buffer of at least `goldy_render_target_buffer_size()` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_target_read_to_buffer(
    target: *const GoldyRenderTarget,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    if target.is_null() || output.is_null() {
        return GoldyResult::NullPointer;
    }
    
    let required_size = (*target).inner.buffer_size();
    if output_size < required_size {
        set_last_error_from_anyhow(&anyhow::anyhow!(
            "Output buffer too small: {} < {}",
            output_size,
            required_size
        ));
        return GoldyResult::InvalidArgument;
    }
    
    let output_slice = slice::from_raw_parts_mut(output, required_size);
    match (*target).inner.read_to_buffer(output_slice) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

