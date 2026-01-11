//! Surface presentation API.
//!
//! This module provides zero-copy GPU presentation to windows.
//! Use `Surface` to render directly to a window without CPU readback.
//!
//! # Example
//!
//! ```rust,no_run
//! use rag::{Instance, DeviceType, Surface, TextureFormat, CommandEncoder, Color};
//! use winit::window::Window;
//!
//! # fn example(window: &Window) -> anyhow::Result<()> {
//! let instance = Instance::new()?;
//! let device = instance.create_device(DeviceType::DiscreteGpu)?;
//! let mut surface = Surface::new(&device, window)?;
//!
//! // In your render loop:
//! let frame = surface.acquire()?;
//! 
//! let mut encoder = CommandEncoder::new();
//! {
//!     let mut pass = encoder.begin_render_pass();
//!     pass.clear(Color::CORNFLOWER_BLUE);
//!     // ... draw commands ...
//! }
//! 
//! frame.render(encoder)?;
//! surface.present(frame)?;
//! # Ok(())
//! # }
//! ```

use crate::backend::{GpuBackend, SurfaceHandle, SwapchainImageHandle};
use crate::device::Device;
use crate::encoder::CommandEncoder;
use crate::types::TextureFormat;
use anyhow::Result;
use raw_window_handle::{HasWindowHandle, HasDisplayHandle};
use std::sync::{Arc, Mutex};

/// A GPU surface for zero-copy presentation to a window.
///
/// Unlike `RenderTarget`, a `Surface` presents directly to the display
/// without any CPU-side copies. This is the optimal path for windowed
/// rendering.
pub struct Surface {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    handle: SurfaceHandle,
    width: u32,
    height: u32,
}

/// A frame acquired from a surface, ready for rendering.
///
/// When you're done rendering to this frame, call `Surface::present()`
/// to display it.
pub struct SurfaceFrame {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    surface_handle: SurfaceHandle,
    image_handle: SwapchainImageHandle,
    width: u32,
    height: u32,
}

impl Surface {
    /// Create a new surface for the given window.
    ///
    /// # Arguments
    ///
    /// * `device` - The GPU device to use for rendering
    /// * `window` - The window to present to (must implement `HasWindowHandle + HasDisplayHandle`)
    ///
    /// # Platform Support
    ///
    /// - **Windows**: Uses `VK_KHR_win32_surface`
    /// - **Linux**: Uses `VK_KHR_wayland_surface` (X11 not supported)
    /// - **macOS**: Uses native Metal with `CAMetalLayer`
    pub fn new<W>(device: &Device, window: &W) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        let handle = {
            let mut backend = device.backend.lock().unwrap();
            backend.create_surface(device.handle, window, window)?
        };

        let (width, height) = {
            let backend = device.backend.lock().unwrap();
            backend.surface_size(handle)
        };

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
            width,
            height,
        })
    }

    /// Acquire the next frame to render to.
    ///
    /// This blocks until a frame is available from the swapchain.
    /// The returned `SurfaceFrame` can be rendered to and then presented.
    pub fn acquire(&self) -> Result<SurfaceFrame> {
        let image_handle = {
            let mut backend = self.backend.lock().unwrap();
            backend.surface_acquire(self.handle)?
        };

        Ok(SurfaceFrame {
            backend: Arc::clone(&self.backend),
            surface_handle: self.handle,
            image_handle,
            width: self.width,
            height: self.height,
        })
    }

    /// Present a rendered frame to the screen.
    ///
    /// This submits the frame to be displayed and returns immediately.
    /// The actual display may happen asynchronously depending on vsync settings.
    pub fn present(&self, frame: SurfaceFrame) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.surface_present(frame.surface_handle, frame.image_handle)
    }

    /// Resize the surface.
    ///
    /// Call this when the window is resized. This recreates the swapchain
    /// with the new dimensions.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(()); // Ignore zero-sized resize (minimized)
        }

        {
            let mut backend = self.backend.lock().unwrap();
            backend.surface_resize(self.handle, width, height)?;
        }

        self.width = width;
        self.height = height;
        Ok(())
    }

    /// Get the current surface dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the surface width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the surface height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the swapchain texture format.
    ///
    /// Use this to set `RenderPipelineDesc::target_format` when rendering
    /// to this surface. The format is determined by the GPU and display
    /// during surface creation.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use rag::{Surface, RenderPipelineDesc};
    /// # fn example(surface: &Surface) {
    /// let desc = RenderPipelineDesc {
    ///     target_format: surface.format(),
    ///     ..Default::default()
    /// };
    /// # }
    /// ```
    pub fn format(&self) -> TextureFormat {
        let backend = self.backend.lock().unwrap();
        backend.surface_format(self.handle)
    }

    /// Validate that a pipeline is compatible with this surface.
    ///
    /// Returns `Ok(())` if the pipeline's target format matches the surface format,
    /// or an error with a helpful message if they don't match.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use rag::{Surface, RenderPipelineDesc, TextureFormat};
    /// # fn example(surface: &Surface, desc: &RenderPipelineDesc) -> anyhow::Result<()> {
    /// surface.validate_pipeline_format(desc.target_format)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate_pipeline_format(&self, pipeline_format: TextureFormat) -> Result<()> {
        let surface_format = self.format();
        if pipeline_format != surface_format {
            anyhow::bail!(
                "Pipeline format mismatch: pipeline uses {:?} but surface uses {:?}.\n\
                 Set RenderPipelineDesc::target_format = surface.format() to fix this.",
                pipeline_format,
                surface_format
            );
        }
        Ok(())
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_surface(self.handle);
    }
}

impl SurfaceFrame {
    /// Render commands to this frame.
    ///
    /// This executes the commands recorded in the encoder to the swapchain image.
    pub fn render(&self, encoder: CommandEncoder) -> Result<()> {
        let commands = encoder.finish();
        
        let mut backend = self.backend.lock().unwrap();
        backend.surface_render(self.surface_handle, self.image_handle, &commands)
    }

    /// Get the frame width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the frame height.
    pub fn height(&self) -> u32 {
        self.height
    }
}
