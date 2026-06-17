use crate::context::Context;
use crate::error::{check, non_null_expect, Result};
use crate::scheme::PresentLease;
use crate::sys::{self, GoldySwapchainPool};
use crate::types::TextureFormat;

/// Pool of OS swapchain drawables for present-on-scheme.
pub struct SwapchainPool {
    ptr: *mut GoldySwapchainPool,
}

impl SwapchainPool {
    #[cfg(windows)]
    pub fn from_win32(ctx: &Context, hwnd: *mut std::ffi::c_void, depth: u32) -> Result<Self> {
        let ptr = non_null_expect(unsafe { sys::goldy_swapchain_pool_create_win32(ctx.as_ptr(), hwnd, depth) });
        Ok(Self { ptr })
    }

    #[cfg(target_os = "macos")]
    pub fn from_appkit(ctx: &Context, ns_view: *mut std::ffi::c_void, depth: u32) -> Result<Self> {
        let ptr = non_null_expect(unsafe { sys::goldy_swapchain_pool_create_appkit(ctx.as_ptr(), ns_view, depth) });
        Ok(Self { ptr })
    }

    #[cfg(target_os = "linux")]
    pub fn from_wayland(
        ctx: &Context,
        display: *mut std::ffi::c_void,
        surface: *mut std::ffi::c_void,
        depth: u32,
    ) -> Result<Self> {
        let ptr = non_null_expect(unsafe {
            sys::goldy_swapchain_pool_create_wayland(ctx.as_ptr(), display, surface, depth)
        });
        Ok(Self { ptr })
    }

    pub fn size(&self) -> (u32, u32) {
        (
            unsafe { sys::goldy_swapchain_pool_width(self.ptr) },
            unsafe { sys::goldy_swapchain_pool_height(self.ptr) },
        )
    }

    pub fn format(&self) -> TextureFormat {
        unsafe { sys::goldy_swapchain_pool_format(self.ptr).into() }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        check(unsafe { sys::goldy_swapchain_pool_resize(self.ptr, width, height) })
    }

    pub fn lease(&self) -> Result<PresentLease> {
        let ptr = non_null_expect(unsafe { sys::goldy_swapchain_pool_lease(self.ptr) });
        Ok(PresentLease::from_ptr(ptr))
    }
}

impl Drop for SwapchainPool {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_swapchain_pool_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
