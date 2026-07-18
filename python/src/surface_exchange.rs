//! Python wrapper for [`goldy::SurfaceExchange`].

use crate::error::IntoPyResult;
use crate::exchange::PyTransaction;
use crate::scheme::{PyContext, PyPresentLease, PyScheme, PySchemeRenderTargetLease};
use crate::texture::PyTexture;
use crate::types::PyTextureFormat;
use goldy::{SurfaceConfig, SurfaceExchange};
use pyo3::prelude::*;

/// Window-surface exchange for present-on-scheme.
#[pyclass(name = "SurfaceExchange", module = "goldy", unsendable)]
pub struct PySurfaceExchange {
    pub(crate) inner: SurfaceExchange,
}

#[pymethods]
impl PySurfaceExchange {
    /// Create a surface exchange from a GLFW window and submission context.
    #[staticmethod]
    fn from_glfw(ctx: &PyContext, glfw_window: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = glfw_window.py();
        let glfw = py.import("glfw")?;

        #[cfg(target_os = "windows")]
        let exchange = {
            let get_win32_window = glfw.getattr("get_win32_window")?;
            let hwnd: isize = get_win32_window.call1((glfw_window,))?.extract()?;
            let window_wrapper = Win32WindowWrapper {
                hwnd: hwnd as *mut std::ffi::c_void,
            };
            SurfaceExchange::new_with_depth(&ctx.inner, &window_wrapper, 3, SurfaceConfig::default())
                .into_py_result()?
        };

        #[cfg(target_os = "linux")]
        let exchange = {
            let get_wayland_window = glfw.getattr("get_wayland_window")?;
            let get_wayland_display = glfw.getattr("get_wayland_display")?;
            let wl_surface: isize = get_wayland_window.call1((glfw_window,))?.extract()?;
            let wl_display: isize = get_wayland_display.call1(())?.extract()?;
            if wl_surface == 0 || wl_display == 0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Wayland handles unavailable — run under a Wayland session",
                ));
            }
            let window_wrapper = WaylandWindowWrapper {
                display: wl_display as *mut std::ffi::c_void,
                surface: wl_surface as *mut std::ffi::c_void,
            };
            SurfaceExchange::new_with_depth(&ctx.inner, &window_wrapper, 3, SurfaceConfig::default())
                .into_py_result()?
        };

        #[cfg(target_os = "macos")]
        let exchange = {
            let get_cocoa_window = glfw.getattr("get_cocoa_window")?;
            let ns_window: isize = get_cocoa_window.call1((glfw_window,))?.extract()?;
            let ns_view = unsafe {
                use objc2::runtime::AnyObject;
                let window = ns_window as *mut AnyObject;
                let content_view: *mut AnyObject = objc2::msg_send![window, contentView];
                content_view as *mut std::ffi::c_void
            };
            let window_wrapper = CocoaWindowWrapper { ns_view };
            SurfaceExchange::new_with_depth(&ctx.inner, &window_wrapper, 3, SurfaceConfig::default())
                .into_py_result()?
        };

        Ok(PySurfaceExchange { inner: exchange })
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    #[getter]
    fn format(&self) -> PyTextureFormat {
        self.inner.format().into()
    }

    fn generation(&self) -> u64 {
        self.inner.generation()
    }

    fn resize(&mut self, width: u32, height: u32) -> PyResult<()> {
        self.inner.resize(width, height).into_py_result()
    }

    fn lease(&self) -> PyResult<PyPresentLease> {
        Ok(PyPresentLease {
            inner: self.inner.lease(),
        })
    }

    fn bind_render_target(&self, scheme: &PyScheme, source: &PySchemeRenderTargetLease) -> PyResult<PyTransaction> {
        scheme.ensure_no_active_recorder()?;
        let transaction = self
            .inner
            .bind_render_target(&mut scheme.inner.borrow_mut(), &source.inner)
            .into_py_result()?;
        Ok(PyTransaction { inner: transaction })
    }

    fn bind(&self, scheme: &PyScheme, source: &PyTexture) -> PyResult<PyTransaction> {
        scheme.ensure_no_active_recorder()?;
        let transaction = self
            .inner
            .bind(&mut scheme.inner.borrow_mut(), &source.inner)
            .into_py_result()?;
        Ok(PyTransaction { inner: transaction })
    }

    fn bind_destination(&self, scheme: &PyScheme) -> PyResult<(PyPresentLease, PyTransaction)> {
        scheme.ensure_no_active_recorder()?;
        let (lease, transaction) = self
            .inner
            .bind_destination(&mut scheme.inner.borrow_mut())
            .into_py_result()?;
        Ok((PyPresentLease { inner: lease }, PyTransaction { inner: transaction }))
    }

    fn __repr__(&self) -> String {
        let (w, h) = self.inner.size();
        format!("SurfaceExchange({w}x{h}, {:?})", self.inner.format())
    }
}

#[cfg(target_os = "windows")]
struct Win32WindowWrapper {
    hwnd: *mut std::ffi::c_void,
}
#[cfg(target_os = "windows")]
unsafe impl Send for Win32WindowWrapper {}
#[cfg(target_os = "windows")]
unsafe impl Sync for Win32WindowWrapper {}
#[cfg(target_os = "windows")]
impl raw_window_handle::HasWindowHandle for Win32WindowWrapper {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let handle =
            raw_window_handle::Win32WindowHandle::new(std::num::NonZeroIsize::new(self.hwnd as isize).unwrap());
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Win32(handle)) })
    }
}
#[cfg(target_os = "windows")]
impl raw_window_handle::HasDisplayHandle for Win32WindowWrapper {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        let handle = raw_window_handle::WindowsDisplayHandle::new();
        Ok(unsafe {
            raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Windows(handle))
        })
    }
}

#[cfg(target_os = "linux")]
struct WaylandWindowWrapper {
    display: *mut std::ffi::c_void,
    surface: *mut std::ffi::c_void,
}
#[cfg(target_os = "linux")]
unsafe impl Send for WaylandWindowWrapper {}
#[cfg(target_os = "linux")]
unsafe impl Sync for WaylandWindowWrapper {}
#[cfg(target_os = "linux")]
impl raw_window_handle::HasWindowHandle for WaylandWindowWrapper {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let handle =
            raw_window_handle::WaylandWindowHandle::new(std::ptr::NonNull::new(self.surface).expect("wayland surface"));
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Wayland(handle)) })
    }
}
#[cfg(target_os = "linux")]
impl raw_window_handle::HasDisplayHandle for WaylandWindowWrapper {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        let handle = raw_window_handle::WaylandDisplayHandle::new(
            std::ptr::NonNull::new(self.display).expect("wayland display"),
        );
        Ok(unsafe {
            raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Wayland(handle))
        })
    }
}

#[cfg(target_os = "macos")]
struct CocoaWindowWrapper {
    ns_view: *mut std::ffi::c_void,
}
#[cfg(target_os = "macos")]
unsafe impl Send for CocoaWindowWrapper {}
#[cfg(target_os = "macos")]
unsafe impl Sync for CocoaWindowWrapper {}
#[cfg(target_os = "macos")]
impl raw_window_handle::HasWindowHandle for CocoaWindowWrapper {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let handle = raw_window_handle::AppKitWindowHandle::new(std::ptr::NonNull::new(self.ns_view).unwrap());
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::AppKit(handle)) })
    }
}
#[cfg(target_os = "macos")]
impl raw_window_handle::HasDisplayHandle for CocoaWindowWrapper {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        let handle = raw_window_handle::AppKitDisplayHandle::new();
        Ok(
            unsafe {
                raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::AppKit(handle))
            },
        )
    }
}
