//! FFI bindings for Surface and SurfaceFrame.
//!
//! Note: Surface creation requires platform-specific window handles.
//! This module provides a minimal FFI for surface operations.

#[cfg(windows)]
use crate::device::GoldyDevice;
use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::task_graph::GoldyTaskGraph;
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
    pub(crate) inner: Option<goldy::Frame>,
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
pub unsafe extern "C" fn goldy_surface_resize(surface: *mut GoldySurface, width: u32, height: u32) -> GoldyResult {
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
pub unsafe extern "C" fn goldy_surface_acquire(surface: *const GoldySurface) -> *mut GoldySurfaceFrame {
    if surface.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Surface is null"));
        return ptr::null_mut();
    }

    match (*surface).inner.acquire() {
        Ok(frame) => Box::into_raw(Box::new(GoldySurfaceFrame { inner: Some(frame) })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Submit a task graph that writes to an already-acquired swapchain frame.
///
/// The graph must include [`goldy_task_graph_declare_swapchain_output`] and
/// [`goldy_task_graph_copy_render_target_to_swapchain`] (or another swapchain
/// binding). Updates `frame` in place with any parcel stamp targets from the graph.
///
/// # Safety
/// All pointers must be valid. No render pass may be open on `graph`.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_submit_graph_to_frame(
    surface: *const GoldySurface,
    graph: *mut GoldyTaskGraph,
    frame: *mut GoldySurfaceFrame,
) -> GoldyResult {
    if surface.is_null() || graph.is_null() || frame.is_null() {
        return GoldyResult::NullPointer;
    }
    if (*graph).has_active_render_pass() {
        set_last_error(
            "Cannot submit graph while a render pass is being recorded; call render_pass_finish first",
        );
        return GoldyResult::InvalidArgument;
    }

    let mut frame_box = Box::from_raw(frame);
    let Some(goldy_frame) = frame_box.inner.take() else {
        set_last_error("Surface frame already consumed");
        let _ = Box::into_raw(frame_box);
        return GoldyResult::InvalidArgument;
    };
    match (*surface)
        .inner
        .submit_graph_to_frame(&mut (*graph).inner, goldy_frame)
    {
        Ok(updated) => {
            frame_box.inner = Some(updated);
            let _ = Box::into_raw(frame_box);
            GoldyResult::Ok
        }
        Err(e) => {
            set_last_error_from_anyhow(&e);
            frame_box.inner = None;
            let _ = Box::into_raw(frame_box);
            GoldyResult::GpuError
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

    let mut frame = Box::from_raw(frame);
    let Some(goldy_frame) = frame.inner.take() else {
        set_last_error("Surface frame already consumed");
        let _ = Box::into_raw(frame);
        return GoldyResult::InvalidArgument;
    };
    match (*surface).inner.present(goldy_frame) {
        Ok(_) => GoldyResult::Ok,
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
    (*frame).inner.as_ref().map(|f| f.width()).unwrap_or(0)
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
    (*frame).inner.as_ref().map(|f| f.height()).unwrap_or(0)
}

// Platform-specific surface creation

#[cfg(target_os = "macos")]
mod appkit_surface {
    use super::*;
    use crate::device::GoldyDevice;
    use raw_window_handle::{
        AppKitDisplayHandle, AppKitWindowHandle, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
        RawWindowHandle,
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

    /// Create a surface from an AppKit `NSView` pointer.
    ///
    /// # Safety
    /// - `device` must be valid.
    /// - `ns_view` must be a valid `NSView*` for the window's content view.
    /// - The view must outlive the surface.
    #[no_mangle]
    pub unsafe extern "C" fn goldy_surface_create_appkit(
        device: *const GoldyDevice,
        ns_view: *mut c_void,
    ) -> *mut GoldySurface {
        if device.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Device pointer is null"));
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
        let device = &(*device).inner;
        let ctx = match device.create_context() {
            Ok(ctx) => ctx,
            Err(e) => {
                set_last_error(format!("{e}"));
                return ptr::null_mut();
            }
        };
        match goldy::Surface::new(&ctx, &window) {
            Ok(surface) => Box::into_raw(Box::new(GoldySurface { inner: surface })),
            Err(e) => {
                set_last_error_from_anyhow(&e);
                ptr::null_mut()
            }
        }
    }
}

#[cfg(windows)]
mod windows_surface {
    use super::*;
    use raw_window_handle::{
        HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
    };
    use std::ffi::c_void;
    use std::num::NonZeroIsize;

    /// Wrapper struct to hold Win32 window handles for surface creation.
    struct Win32Window {
        hwnd: NonZeroIsize,
    }

    impl HasWindowHandle for Win32Window {
        fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            let handle = Win32WindowHandle::new(self.hwnd);
            // hinstance is optional for surface creation
            Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
        }
    }

    impl HasDisplayHandle for Win32Window {
        fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            let handle = WindowsDisplayHandle::new();
            Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(RawDisplayHandle::Windows(handle)) })
        }
    }

    /// Create a surface from a Win32 HWND.
    ///
    /// # Arguments
    /// * `device` - A valid Goldy device pointer
    /// * `hwnd` - A Win32 HWND (window handle)
    ///
    /// # Returns
    /// A pointer to the created surface, or null on failure.
    /// Call `goldy_get_last_error()` for error details on failure.
    ///
    /// # Safety
    /// - The device pointer must be valid and not null.
    /// - The hwnd must be a valid Win32 window handle.
    /// - The window must remain valid for the lifetime of the surface.
    #[no_mangle]
    pub unsafe extern "C" fn goldy_surface_create_win32(
        device: *const GoldyDevice,
        hwnd: *mut c_void,
    ) -> *mut GoldySurface {
        if device.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Device pointer is null"));
            return ptr::null_mut();
        }

        if hwnd.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("HWND is null"));
            return ptr::null_mut();
        }

        // Convert the void pointer to NonZeroIsize
        let hwnd_isize = hwnd as isize;
        let hwnd_nonzero = match NonZeroIsize::new(hwnd_isize) {
            Some(h) => h,
            None => {
                set_last_error_from_anyhow(&anyhow::anyhow!("HWND is zero"));
                return ptr::null_mut();
            }
        };

        let window = Win32Window { hwnd: hwnd_nonzero };

        let device = &(*device).inner;
        let ctx = match device.create_context() {
            Ok(ctx) => ctx,
            Err(e) => {
                set_last_error(format!("{e}"));
                return ptr::null_mut();
            }
        };
        match goldy::Surface::new(&ctx, &window) {
            Ok(surface) => Box::into_raw(Box::new(GoldySurface { inner: surface })),
            Err(e) => {
                set_last_error_from_anyhow(&e);
                ptr::null_mut()
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod wayland_surface {
    use super::*;
    use crate::device::GoldyDevice;
    use raw_window_handle::{
        HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle,
        WaylandWindowHandle,
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

    /// Create a surface from Wayland `wl_display` and `wl_surface` pointers.
    ///
    /// # Safety
    /// - `device` must be valid.
    /// - `display` and `surface` must be valid Wayland handles for the window.
    /// - They must outlive the surface.
    #[no_mangle]
    pub unsafe extern "C" fn goldy_surface_create_wayland(
        device: *const GoldyDevice,
        display: *mut c_void,
        surface: *mut c_void,
    ) -> *mut GoldySurface {
        if device.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Device pointer is null"));
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
        let device = &(*device).inner;
        let ctx = match device.create_context() {
            Ok(ctx) => ctx,
            Err(e) => {
                set_last_error(format!("{e}"));
                return ptr::null_mut();
            }
        };
        match goldy::Surface::new(&ctx, &window) {
            Ok(s) => Box::into_raw(Box::new(GoldySurface { inner: s })),
            Err(e) => {
                set_last_error_from_anyhow(&e);
                ptr::null_mut()
            }
        }
    }
}
