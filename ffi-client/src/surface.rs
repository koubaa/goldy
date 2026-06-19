use crate::device::Device;
use crate::error::{check, non_null, Result};
use crate::sys::{self, GoldySurface, GoldySurfaceFrame};
use crate::types::TextureFormat;
use std::ffi::c_void;

/// A window swapchain surface.
pub struct Surface {
    ptr: *mut GoldySurface,
}

impl Surface {
    /// Create a surface from a Win32 `HWND` (`goldy_surface_create_win32`).
    ///
    /// # Safety
    /// `hwnd` must be a valid window handle that outlives the surface.
    #[cfg(windows)]
    pub unsafe fn from_win32(device: &Device, hwnd: *mut c_void) -> Result<Self> {
        let ptr = non_null(sys::goldy_surface_create_win32(device.as_ptr(), hwnd))?;
        Ok(Self { ptr })
    }

    /// Create a surface from an AppKit `NSView` pointer (`goldy_surface_create_appkit`).
    ///
    /// # Safety
    /// `ns_view` must be a valid view pointer that outlives the surface.
    #[cfg(target_os = "macos")]
    pub unsafe fn from_appkit(device: &Device, ns_view: *mut c_void) -> Result<Self> {
        let ptr = non_null(sys::goldy_surface_create_appkit(device.as_ptr(), ns_view))?;
        Ok(Self { ptr })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width(), self.height())
    }

    pub fn width(&self) -> u32 {
        unsafe { sys::goldy_surface_width(self.ptr) }
    }

    pub fn height(&self) -> u32 {
        unsafe { sys::goldy_surface_height(self.ptr) }
    }

    pub fn format(&self) -> TextureFormat {
        unsafe { sys::goldy_surface_format(self.ptr) }.into()
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        check(unsafe { sys::goldy_surface_resize(self.ptr, width, height) })
    }

    /// Begin the next frame (acquire swapchain image).
    pub fn begin(&self) -> Result<Frame> {
        let ptr = non_null(unsafe { sys::goldy_surface_acquire(self.ptr) })?;
        Ok(Frame { surface: self.ptr, ptr })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_surface_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// An acquired swapchain frame.
pub struct Frame {
    surface: *const GoldySurface,
    ptr: *mut GoldySurfaceFrame,
}

impl Frame {
    pub fn present(self) -> Result<()> {
        check(unsafe { sys::goldy_surface_present(self.surface, self.into_raw()) })
    }

    fn into_raw(self) -> *mut GoldySurfaceFrame {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let frame = self.ptr;
            self.ptr = std::ptr::null_mut();
            let _ = unsafe { Box::from_raw(frame) };
        }
    }
}
