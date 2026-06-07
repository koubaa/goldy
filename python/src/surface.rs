//! Python wrapper for Surface (windowed rendering).

use crate::device::PyDevice;
use crate::encoder::PyCommandEncoder;
use crate::error::IntoPyResult;
use crate::types::PyTextureFormat;
use pyo3::prelude::*;

/// A GPU surface for zero-copy presentation to a window.
///
/// Unlike RenderTarget, a Surface presents directly to the display
/// without any CPU-side copies. This is the optimal path for windowed
/// rendering.
///
/// Example with GLFW:
///     >>> import glfw
///     >>> import goldy
///     >>>
///     >>> glfw.init()
///     >>> glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # No OpenGL
///     >>> window = glfw.create_window(800, 600, "Goldy", None, None)
///     >>>
///     >>> instance = goldy.Instance()
///     >>> device = instance.request_adapter().request_device()
///     >>> surface = goldy.Surface.from_glfw(device, window)
///     >>>
///     >>> while not glfw.window_should_close(window):
///     ...     frame = surface.acquire()
///     ...     encoder = goldy.CommandEncoder()
///     ...     with encoder.begin_render_pass() as rp:
///     ...         rp.clear(goldy.Color.CORNFLOWER_BLUE)
///     ...     frame.render(encoder)
///     ...     surface.present(frame)
///     ...     glfw.poll_events()
#[pyclass(name = "Surface", module = "goldy")]
pub struct PySurface {
    inner: goldy::Surface,
}

#[pymethods]
impl PySurface {
    /// Create a surface from a GLFW window.
    ///
    /// Args:
    ///     device: The GPU device.
    ///     glfw_window: A GLFW window handle (from glfw.create_window).
    ///
    /// Note: The window must be created with glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    #[staticmethod]
    fn from_glfw(device: &PyDevice, glfw_window: &Bound<'_, PyAny>) -> PyResult<Self> {
        let context = device.inner.create_context().into_py_result()?;

        // GLFW windows in Python expose the native handle via ctypes
        // We need to get the raw window handle

        let py = glfw_window.py();

        // Import glfw to get native window handle
        let glfw = py.import("glfw")?;

        // Get the native window handle based on platform
        #[cfg(target_os = "windows")]
        let surface = {
            // On Windows, get HWND
            let get_win32_window = glfw.getattr("get_win32_window")?;
            let hwnd: isize = get_win32_window.call1((glfw_window,))?.extract()?;

            // Create a wrapper that implements HasWindowHandle
            let window_wrapper = Win32WindowWrapper {
                hwnd: hwnd as *mut std::ffi::c_void,
            };
            goldy::Surface::new(&context, &window_wrapper).into_py_result()?
        };

        #[cfg(target_os = "linux")]
        let surface = {
            // On Linux/X11, get the X11 window and display
            let get_x11_window = glfw.getattr("get_x11_window")?;
            let get_x11_display = glfw.getattr("get_x11_display")?;

            let x11_window: u64 = get_x11_window.call1((glfw_window,))?.extract()?;
            let x11_display: isize = get_x11_display.call1(())?.extract()?;

            let window_wrapper = X11WindowWrapper {
                window: x11_window as u32,
                display: x11_display as *mut std::ffi::c_void,
            };
            goldy::Surface::new(&context, &window_wrapper).into_py_result()?
        };

        #[cfg(target_os = "macos")]
        let surface = {
            // On macOS, get the Cocoa window and then its contentView
            // GLFW returns NSWindow, but raw_window_handle needs NSView
            let get_cocoa_window = glfw.getattr("get_cocoa_window")?;
            let ns_window: isize = get_cocoa_window.call1((glfw_window,))?.extract()?;

            // Get contentView from NSWindow using Objective-C runtime
            let ns_view = unsafe {
                use objc2::runtime::AnyObject;

                let window = ns_window as *mut AnyObject;
                // Call [window contentView] to get the NSView
                let content_view: *mut AnyObject = objc2::msg_send![window, contentView];
                content_view as *mut std::ffi::c_void
            };

            let window_wrapper = CocoaWindowWrapper { ns_view };
            goldy::Surface::new(&context, &window_wrapper).into_py_result()?
        };

        Ok(PySurface { inner: surface })
    }

    /// Get the surface width.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Get the surface height.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    /// Get the surface format.
    #[getter]
    fn format(&self) -> PyTextureFormat {
        self.inner.format().into()
    }

    /// Resize the surface (call when window is resized).
    fn resize(&mut self, width: u32, height: u32) -> PyResult<()> {
        self.inner.resize(width, height).into_py_result()
    }

    /// Acquire a frame for rendering.
    ///
    /// Returns a SurfaceFrame that you can render to and then present.
    fn acquire(&mut self) -> PyResult<PySurfaceFrame> {
        let frame = self.inner.acquire().into_py_result()?;
        Ok(PySurfaceFrame { inner: Some(frame) })
    }

    /// Present a rendered frame to the display.
    fn present(&mut self, frame: &mut PySurfaceFrame) -> PyResult<()> {
        let frame = frame
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Frame already presented"))?;
        self.inner.present(frame).map(|_| ()).into_py_result()
    }

    fn __repr__(&self) -> String {
        format!(
            "Surface({}x{}, {:?})",
            self.inner.width(),
            self.inner.height(),
            self.inner.format()
        )
    }
}

/// A frame acquired from a surface, ready for rendering.
#[pyclass(name = "SurfaceFrame", module = "goldy")]
pub struct PySurfaceFrame {
    inner: Option<goldy::Frame>,
}

#[pymethods]
impl PySurfaceFrame {
    /// Render commands to this frame.
    fn render(&mut self, encoder: &PyCommandEncoder) -> PyResult<()> {
        let frame = self
            .inner
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Frame already consumed"))?;

        let commands = encoder.take_inner();
        frame.render(commands).into_py_result()
    }

    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "SurfaceFrame(pending)".to_string()
        } else {
            "SurfaceFrame(consumed)".to_string()
        }
    }
}

// Platform-specific window handle wrappers

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
        let raw = raw_window_handle::RawWindowHandle::Win32(handle);
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(raw) })
    }
}

#[cfg(target_os = "windows")]
impl raw_window_handle::HasDisplayHandle for Win32WindowWrapper {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        let handle = raw_window_handle::WindowsDisplayHandle::new();
        let raw = raw_window_handle::RawDisplayHandle::Windows(handle);
        Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(raw) })
    }
}

#[cfg(target_os = "linux")]
struct X11WindowWrapper {
    window: u32,
    display: *mut std::ffi::c_void,
}

#[cfg(target_os = "linux")]
unsafe impl Send for X11WindowWrapper {}
#[cfg(target_os = "linux")]
unsafe impl Sync for X11WindowWrapper {}

#[cfg(target_os = "linux")]
impl raw_window_handle::HasWindowHandle for X11WindowWrapper {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let mut handle = raw_window_handle::XlibWindowHandle::new(self.window as _);
        let raw = raw_window_handle::RawWindowHandle::Xlib(handle);
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(raw) })
    }
}

#[cfg(target_os = "linux")]
impl raw_window_handle::HasDisplayHandle for X11WindowWrapper {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        let handle = raw_window_handle::XlibDisplayHandle::new(std::ptr::NonNull::new(self.display), 0);
        let raw = raw_window_handle::RawDisplayHandle::Xlib(handle);
        Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(raw) })
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
        let raw = raw_window_handle::RawWindowHandle::AppKit(handle);
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(raw) })
    }
}

#[cfg(target_os = "macos")]
impl raw_window_handle::HasDisplayHandle for CocoaWindowWrapper {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        let handle = raw_window_handle::AppKitDisplayHandle::new();
        let raw = raw_window_handle::RawDisplayHandle::AppKit(handle);
        Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(raw) })
    }
}
