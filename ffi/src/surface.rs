//! FFI bindings for Surface and SurfaceFrame.
//!
//! Note: Surface creation requires platform-specific window handles.
//! This module provides a minimal FFI for surface operations.

use crate::device::GoldyDevice;
use crate::encoder::GoldyCommandEncoder;
use crate::error::{set_last_error_from_anyhow, GoldyResult};
use crate::types::GoldyTextureFormat;
use std::ptr;

/// Opaque handle to a Goldy Surface.
///
/// Surface creation is platform-specific and typically done through
/// higher-level bindings that can pass window handles.
pub struct GoldySurface {
    pub(crate) inner: goldy::Surface,
}

/// Opaque handle to a Goldy SurfaceFrame.
pub struct GoldySurfaceFrame {
    pub(crate) inner: goldy::SurfaceFrame,
}

// Note: Surface creation is complex due to platform-specific window handles.
// For now, we provide the operations assuming the Surface is created by
// platform-specific code that can obtain raw window handles.

/// Destroy a surface.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_destroy(surface: *mut GoldySurface) {
    if !surface.is_null() {
        drop(Box::from_raw(surface));
    }
}

/// Get the surface width.
///
/// # Safety
/// The surface pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_width(surface: *const GoldySurface) -> u32 {
    if surface.is_null() {
        return 0;
    }
    (*surface).inner.width()
}

/// Get the surface height.
///
/// # Safety
/// The surface pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_height(surface: *const GoldySurface) -> u32 {
    if surface.is_null() {
        return 0;
    }
    (*surface).inner.height()
}

/// Get the surface format.
///
/// # Safety
/// The surface pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_format(surface: *const GoldySurface) -> GoldyTextureFormat {
    if surface.is_null() {
        return GoldyTextureFormat::Bgra8UnormSrgb;
    }
    (*surface).inner.format().into()
}

/// Resize the surface.
///
/// # Safety
/// The surface pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_resize(
    surface: *mut GoldySurface,
    width: u32,
    height: u32,
) -> GoldyResult {
    if surface.is_null() {
        return GoldyResult::NullPointer;
    }

    match (*surface).inner.resize(width, height) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

/// Acquire the next frame from the surface.
///
/// Returns a pointer to the frame, or null on failure.
///
/// # Safety
/// The surface pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_acquire(
    surface: *const GoldySurface,
) -> *mut GoldySurfaceFrame {
    if surface.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Surface is null"));
        return ptr::null_mut();
    }

    match (*surface).inner.acquire() {
        Ok(frame) => Box::into_raw(Box::new(GoldySurfaceFrame { inner: frame })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Present a frame to the surface.
///
/// This consumes the frame.
///
/// # Safety
/// Both pointers must be valid.
/// The frame is consumed and must not be used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_present(
    surface: *const GoldySurface,
    frame: *mut GoldySurfaceFrame,
) -> GoldyResult {
    if surface.is_null() || frame.is_null() {
        return GoldyResult::NullPointer;
    }

    let frame = Box::from_raw(frame);
    match (*surface).inner.present(frame.inner) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

/// Render commands to a frame.
///
/// This consumes the encoder.
///
/// # Safety
/// Both pointers must be valid.
/// The encoder is consumed and must not be used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_frame_render(
    frame: *const GoldySurfaceFrame,
    encoder: *mut GoldyCommandEncoder,
) -> GoldyResult {
    if frame.is_null() || encoder.is_null() {
        return GoldyResult::NullPointer;
    }

    let encoder = Box::from_raw(encoder);
    match (*frame).inner.render(encoder.inner) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

/// Get the frame width.
///
/// # Safety
/// The frame pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_frame_width(frame: *const GoldySurfaceFrame) -> u32 {
    if frame.is_null() {
        return 0;
    }
    (*frame).inner.width()
}

/// Get the frame height.
///
/// # Safety
/// The frame pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_frame_height(frame: *const GoldySurfaceFrame) -> u32 {
    if frame.is_null() {
        return 0;
    }
    (*frame).inner.height()
}

// Platform-specific surface creation would go here.
// For Windows, we'd need HWND; for macOS, NSView/CAMetalLayer; for Linux, Wayland/X11 handles.
// These are typically better handled at the C# level using raw-window-handle compatible types.

#[cfg(windows)]
mod windows_surface {
    use super::*;
    use std::ffi::c_void;

    /// Create a surface from a Win32 HWND.
    ///
    /// # Safety
    /// The device pointer and hwnd must be valid.
    #[no_mangle]
    pub unsafe extern "C" fn goldy_surface_create_win32(
        _device: *const GoldyDevice,
        _hwnd: *mut c_void,
    ) -> *mut GoldySurface {
        // This would require implementing raw-window-handle traits for HWND
        // For now, return null - actual implementation needs platform-specific code
        set_last_error_from_anyhow(&anyhow::anyhow!(
            "Win32 surface creation not yet implemented in FFI"
        ));
        ptr::null_mut()
    }
}
