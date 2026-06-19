//! FFI bindings for [`goldy::SwapchainPool`] and [`goldy::PresentLease`].

use crate::context::GoldyContext;
use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::types::GoldyTextureFormat;
use goldy::swapchain_pool::{PresentLease, SwapchainPool};
use std::ptr;

/// Opaque handle to a swapchain pool.
pub struct GoldySwapchainPool {
    pub(crate) inner: SwapchainPool,
}

/// Opaque handle to a stable present lease from a swapchain pool.
pub struct GoldyPresentLease {
    pub(crate) inner: PresentLease,
}

/// Destroy a swapchain pool.
///
/// # Safety
/// `pool` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_swapchain_pool_destroy(pool: *mut GoldySwapchainPool) {
    if !pool.is_null() {
        drop(Box::from_raw(pool));
    }
}

/// Acquire a stable present lease from `pool`.
///
/// Returns a heap-allocated lease handle; destroy with [`goldy_present_lease_destroy`].
/// The lease identity remains valid until the pool is destroyed.
///
/// # Safety
/// `pool` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_swapchain_pool_lease(pool: *const GoldySwapchainPool) -> *mut GoldyPresentLease {
    if pool.is_null() {
        set_last_error("SwapchainPool pointer is null");
        return ptr::null_mut();
    }
    let lease = (*pool).inner.lease();
    Box::into_raw(Box::new(GoldyPresentLease { inner: lease }))
}

/// Destroy a present lease handle.
///
/// Does not remove the lease from the pool; the backing remains until the pool is dropped.
///
/// # Safety
/// `lease` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_present_lease_destroy(lease: *mut GoldyPresentLease) {
    if !lease.is_null() {
        drop(Box::from_raw(lease));
    }
}

/// Current swapchain drawable width.
///
/// # Safety
/// `pool` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_swapchain_pool_width(pool: *const GoldySwapchainPool) -> u32 {
    if pool.is_null() {
        return 0;
    }
    (*pool).inner.width()
}

/// Current swapchain drawable height.
///
/// # Safety
/// `pool` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_swapchain_pool_height(pool: *const GoldySwapchainPool) -> u32 {
    if pool.is_null() {
        return 0;
    }
    (*pool).inner.height()
}

/// Swapchain surface format.
///
/// # Safety
/// `pool` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_swapchain_pool_format(pool: *const GoldySwapchainPool) -> GoldyTextureFormat {
    if pool.is_null() {
        return GoldyTextureFormat::Bgra8UnormSrgb;
    }
    (*pool).inner.format().into()
}

/// Resize the underlying swapchain (structural edit — rebuild scheme nodes).
///
/// # Safety
/// `pool` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_swapchain_pool_resize(
    pool: *mut GoldySwapchainPool,
    width: u32,
    height: u32,
) -> GoldyResult {
    if pool.is_null() {
        return GoldyResult::NullPointer;
    }
    match (*pool).inner.resize(width, height) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error_from_anyhow(&e);
            GoldyResult::GpuError
        }
    }
}

#[cfg(target_os = "macos")]
mod appkit_pool {
    use super::*;
    use raw_window_handle::{
        AppKitDisplayHandle, AppKitWindowHandle, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    struct AppKitWindow {
        ns_view: NonNull<c_void>,
    }

    impl HasWindowHandle for AppKitWindow {
        fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            let handle = AppKitWindowHandle::new(self.ns_view);
            Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(RawWindowHandle::AppKit(handle)) })
        }
    }

    impl HasDisplayHandle for AppKitWindow {
        fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            let handle = AppKitDisplayHandle::new();
            Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(RawDisplayHandle::AppKit(handle)) })
        }
    }

    /// Create a swapchain pool from an AppKit `NSView` pointer.
    ///
    /// # Safety
    /// `ctx` must be valid. `ns_view` must be a valid `NSView*` for the window's content view.
    #[no_mangle]
    pub unsafe extern "C" fn goldy_swapchain_pool_create_appkit(
        ctx: *const GoldyContext,
        ns_view: *mut c_void,
        depth: u32,
    ) -> *mut GoldySwapchainPool {
        if ctx.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Context pointer is null"));
            return ptr::null_mut();
        }
        if ns_view.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("NSView pointer is null"));
            return ptr::null_mut();
        }
        let ns_view = match NonNull::new(ns_view) {
            Some(v) => v,
            None => {
                set_last_error_from_anyhow(&anyhow::anyhow!("NSView pointer is null"));
                return ptr::null_mut();
            }
        };
        let window = AppKitWindow { ns_view };
        match SwapchainPool::new(&(*ctx).inner, &window, depth) {
            Ok(pool) => Box::into_raw(Box::new(GoldySwapchainPool { inner: pool })),
            Err(e) => {
                set_last_error_from_anyhow(&e);
                ptr::null_mut()
            }
        }
    }
}

#[cfg(windows)]
mod windows_pool {
    use super::*;
    use raw_window_handle::{
        HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
    };
    use std::ffi::c_void;
    use std::num::NonZeroIsize;

    struct Win32Window {
        hwnd: NonZeroIsize,
    }

    impl HasWindowHandle for Win32Window {
        fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            let handle = Win32WindowHandle::new(self.hwnd);
            Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
        }
    }

    impl HasDisplayHandle for Win32Window {
        fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            let handle = WindowsDisplayHandle::new();
            Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(RawDisplayHandle::Windows(handle)) })
        }
    }

    /// Create a swapchain pool from a Win32 HWND.
    ///
    /// # Safety
    /// `ctx` must be valid. `hwnd` must be a valid Win32 window handle.
    #[no_mangle]
    pub unsafe extern "C" fn goldy_swapchain_pool_create_win32(
        ctx: *const GoldyContext,
        hwnd: *mut c_void,
        depth: u32,
    ) -> *mut GoldySwapchainPool {
        if ctx.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Context pointer is null"));
            return ptr::null_mut();
        }
        if hwnd.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("HWND is null"));
            return ptr::null_mut();
        }
        let hwnd_nonzero = match NonZeroIsize::new(hwnd as isize) {
            Some(h) => h,
            None => {
                set_last_error_from_anyhow(&anyhow::anyhow!("HWND is zero"));
                return ptr::null_mut();
            }
        };
        let window = Win32Window { hwnd: hwnd_nonzero };
        match SwapchainPool::new(&(*ctx).inner, &window, depth) {
            Ok(pool) => Box::into_raw(Box::new(GoldySwapchainPool { inner: pool })),
            Err(e) => {
                set_last_error_from_anyhow(&e);
                ptr::null_mut()
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod wayland_pool {
    use super::*;
    use raw_window_handle::{
        HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    struct WaylandWindow {
        display: NonNull<c_void>,
        surface: NonNull<c_void>,
    }

    impl HasWindowHandle for WaylandWindow {
        fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            let handle = WaylandWindowHandle::new(self.surface);
            Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(RawWindowHandle::Wayland(handle)) })
        }
    }

    impl HasDisplayHandle for WaylandWindow {
        fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            let handle = WaylandDisplayHandle::new(self.display);
            Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(RawDisplayHandle::Wayland(handle)) })
        }
    }

    /// Create a swapchain pool from Wayland `wl_display` and `wl_surface` pointers.
    ///
    /// # Safety
    /// `ctx` must be valid. `display` and `surface` must be valid Wayland handles.
    #[no_mangle]
    pub unsafe extern "C" fn goldy_swapchain_pool_create_wayland(
        ctx: *const GoldyContext,
        display: *mut c_void,
        surface: *mut c_void,
        depth: u32,
    ) -> *mut GoldySwapchainPool {
        if ctx.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Context pointer is null"));
            return ptr::null_mut();
        }
        if display.is_null() || surface.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Wayland display or surface pointer is null"));
            return ptr::null_mut();
        }
        let display = match NonNull::new(display) {
            Some(d) => d,
            None => {
                set_last_error_from_anyhow(&anyhow::anyhow!("Wayland display pointer is null"));
                return ptr::null_mut();
            }
        };
        let surface = match NonNull::new(surface) {
            Some(s) => s,
            None => {
                set_last_error_from_anyhow(&anyhow::anyhow!("Wayland surface pointer is null"));
                return ptr::null_mut();
            }
        };
        let window = WaylandWindow { display, surface };
        match SwapchainPool::new(&(*ctx).inner, &window, depth) {
            Ok(pool) => Box::into_raw(Box::new(GoldySwapchainPool { inner: pool })),
            Err(e) => {
                set_last_error_from_anyhow(&e);
                ptr::null_mut()
            }
        }
    }
}
