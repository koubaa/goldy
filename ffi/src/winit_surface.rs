//! Create a [`GoldySurface`](crate::surface::GoldySurface) from any winit window.
//!
//! This uses Goldy's `Surface::new` internally. Exposed for `goldy-ffi` examples; the
//! stable C API provides `goldy_surface_create_win32` on Windows instead.

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
    match goldy::Surface::new(unsafe { &(*device).inner }, window) {
        Ok(surface) => Box::into_raw(Box::new(GoldySurface { inner: surface })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}
