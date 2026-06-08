//! Cross-platform surface creation for winit windows (Rust-only helper).
//!
//! C/C++ clients use platform entry points from the generated header (e.g. `goldy_surface_create_win32` on Windows).
//! Enabled with the `winit` feature on `goldy-ffi`.

use crate::device::GoldyDevice;
use crate::error::set_last_error_from_anyhow;
use crate::surface::GoldySurface;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ptr;

/// Create a swapchain surface for `window` using an existing FFI device pointer.
///
/// Returns null on failure; call [`crate::goldy_get_last_error`] for details.
pub fn goldy_surface_from_winit_window<W: HasWindowHandle + HasDisplayHandle>(
    device: *const GoldyDevice,
    window: &W,
) -> *mut GoldySurface {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device pointer is null"));
        return ptr::null_mut();
    }
    let device = unsafe { &(*device).inner };
    let ctx = match device.create_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            crate::error::set_last_error(format!("{e}"));
            return ptr::null_mut();
        }
    };
    match goldy::Surface::new(&ctx, window) {
        Ok(surface) => Box::into_raw(Box::new(GoldySurface { inner: surface })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}
