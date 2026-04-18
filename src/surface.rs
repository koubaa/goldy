//! Surface presentation API.
//!
//! This module provides zero-copy GPU presentation to windows.
//! Use `Surface` to render directly to a window without CPU readback.
//!
//! # Render-command path (graphics pipelines)
//!
//! ```rust,no_run
//! use goldy::{Instance, DeviceType, Surface, TextureFormat, CommandEncoder, Color};
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
//! frame.present()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Compute path (e.g. Ekrano)
//!
//! ```rust,no_run,ignore
//! let frame = surface.acquire()?;
//! // frame.texture() returns the swapchain image as a Texture that can be
//! // used with compute passes, the compute graph, or render_to_texture.
//! renderer.render_to_texture(&device, &scene, frame.texture(), &params)?;
//! frame.present()?;
//! ```

use crate::backend::{GpuBackend, SurfaceHandle, SwapchainImageHandle};
use crate::device::Device;
use crate::encoder::CommandEncoder;
use crate::texture::Texture;
use crate::types::{PresentMode, SurfaceConfig, TextureFormat};
use anyhow::Result;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
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
/// After acquiring a frame, you can either:
/// - Use `render()` to execute render commands (graphics pipeline path)
/// - Use `texture()` to get the frame's texture for compute or external rendering
///
/// When done, call `present()` to display the frame. Dropping without
/// presenting will clean up resources but the frame won't be shown.
pub struct SurfaceFrame {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    surface_handle: SurfaceHandle,
    image_handle: SwapchainImageHandle,
    texture: Option<Texture>,
    width: u32,
    height: u32,
    presented: bool,
}

impl Surface {
    /// Create a new surface for the given window.
    ///
    /// Uses default configuration (Auto present mode, no depth buffer).
    pub fn new<W>(device: &Device, window: &W) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        Self::new_with_config(device, window, SurfaceConfig::default())
    }

    /// Create a new surface with an optional depth buffer for 3D rendering.
    pub fn new_with_depth<W>(
        device: &Device,
        window: &W,
        depth_format: Option<crate::types::DepthFormat>,
    ) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        Self::new_with_config(
            device,
            window,
            SurfaceConfig {
                depth_format,
                ..Default::default()
            },
        )
    }

    /// Create a new surface with full configuration.
    pub fn new_with_config<W>(device: &Device, window: &W, config: SurfaceConfig) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        let handle = {
            let mut backend = device.backend.lock().unwrap();
            backend.create_surface(device.handle, window, window, config.depth_format)?
        };

        let (width, height) = {
            let backend = device.backend.lock().unwrap();
            backend.surface_size(handle)
        };

        // Apply present mode if non-default
        if config.present_mode != PresentMode::Auto {
            let mut backend = device.backend.lock().unwrap();
            backend.surface_set_present_mode(handle, config.present_mode)?;
        }

        tracing::debug!(
            width,
            height,
            ?config.depth_format,
            ?config.present_mode,
            "Surface created"
        );

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
    /// The returned `SurfaceFrame` exposes the frame's texture and must
    /// be presented (or dropped) before acquiring the next frame.
    pub fn acquire(&self) -> Result<SurfaceFrame> {
        let (image_handle, texture) = {
            let mut backend = self.backend.lock().unwrap();
            let image_handle = backend.surface_acquire(self.handle)?;

            let (w, h) = backend.surface_size(self.handle);
            let format = backend.surface_format(self.handle);

            let texture = backend
                .surface_frame_texture(self.handle)
                .map(|tex_handle| {
                    Texture::borrowed(Arc::clone(&self.backend), tex_handle, w, h, format)
                });

            (image_handle, texture)
        };

        // Update cached dimensions from the backend (may change after resize)
        let (w, h) = {
            let backend = self.backend.lock().unwrap();
            backend.surface_size(self.handle)
        };

        Ok(SurfaceFrame {
            backend: Arc::clone(&self.backend),
            surface_handle: self.handle,
            image_handle,
            texture,
            width: w,
            height: h,
            presented: false,
        })
    }

    /// Present a rendered frame to the screen (legacy API).
    ///
    /// Prefer using `frame.present()` instead, which consumes the frame
    /// and provides better ergonomics.
    pub fn present(&self, mut frame: SurfaceFrame) -> Result<()> {
        frame.do_present()
    }

    /// Resize the surface.
    ///
    /// Call this when the window is resized. This recreates the swapchain
    /// with the new dimensions.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        tracing::debug!(width, height, "Resizing surface");
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
    pub fn format(&self) -> TextureFormat {
        let backend = self.backend.lock().unwrap();
        backend.surface_format(self.handle)
    }

    /// Set the present mode (vsync strategy).
    pub fn set_present_mode(&mut self, mode: PresentMode) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.surface_set_present_mode(self.handle, mode)
    }

    /// Get the current present mode.
    pub fn present_mode(&self) -> PresentMode {
        let backend = self.backend.lock().unwrap();
        backend.surface_present_mode(self.handle)
    }

    /// Validate that a pipeline is compatible with this surface.
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
        tracing::debug!(
            width = self.width,
            height = self.height,
            "Destroying surface"
        );
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_surface(self.handle);
    }
}

impl SurfaceFrame {
    /// Get the frame's texture for rendering.
    ///
    /// The returned texture is the swapchain image, registered for bindless
    /// access. You can pass it to compute shaders, the compute graph, or
    /// `render_to_texture`. Call `present()` when done.
    ///
    /// Returns `None` if the backend does not support exposing the frame
    /// texture (e.g. Vulkan/DX12 — not yet implemented).
    pub fn texture(&self) -> Option<&Texture> {
        self.texture.as_ref()
    }

    /// Render commands to this frame (graphics pipeline path).
    ///
    /// This executes the commands recorded in the encoder to the swapchain image.
    pub fn render(&self, encoder: CommandEncoder) -> Result<()> {
        let commands = encoder.finish();

        let mut backend = self.backend.lock().unwrap();
        backend.surface_render(self.surface_handle, self.image_handle, &commands)
    }

    /// Present this frame to the display.
    ///
    /// Consumes the frame — the borrow checker prevents use-after-present.
    pub fn present(mut self) -> Result<()> {
        self.do_present()
    }

    fn do_present(&mut self) -> Result<()> {
        if self.presented {
            return Ok(());
        }
        self.presented = true;

        // Drop the borrowed texture before presenting so it isn't accessed
        // after the drawable is returned to the layer.
        self.texture = None;

        let mut backend = self.backend.lock().unwrap();
        backend.surface_present(self.surface_handle, self.image_handle)
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

impl Drop for SurfaceFrame {
    fn drop(&mut self) {
        if !self.presented {
            // Frame was dropped without present — clean up the acquired drawable
            let _ = self.do_present();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::device::Device;

    // Mock window for testing Surface API
    struct MockWindow {
        width: u32,
        height: u32,
    }

    impl MockWindow {
        fn new(width: u32, height: u32) -> Self {
            Self { width, height }
        }

        fn size(&self) -> (u32, u32) {
            (self.width, self.height)
        }
    }

    impl raw_window_handle::HasWindowHandle for MockWindow {
        fn window_handle(
            &self,
        ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            Ok(unsafe {
                raw_window_handle::WindowHandle::borrow_raw(
                    raw_window_handle::RawWindowHandle::Web(
                        raw_window_handle::WebWindowHandle::new(0),
                    ),
                )
            })
        }
    }

    impl raw_window_handle::HasDisplayHandle for MockWindow {
        fn display_handle(
            &self,
        ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            Ok(unsafe {
                raw_window_handle::DisplayHandle::borrow_raw(
                    raw_window_handle::RawDisplayHandle::Web(
                        raw_window_handle::WebDisplayHandle::new(),
                    ),
                )
            })
        }
    }

    fn create_test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn create_test_device_with_format(format: TextureFormat) -> Device {
        let mut backend = MockBackend::new();
        backend.set_default_surface_format(format);
        Device::from_backend(Box::new(backend)).unwrap()
    }

    #[test]
    fn test_surface_size() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        assert_eq!(window.size(), (800, 600));
        let surface = Surface::new(&device, &window).unwrap();

        assert_eq!(surface.width(), 800);
        assert_eq!(surface.height(), 600);
        assert_eq!(surface.size(), (800, 600));
    }

    #[test]
    fn test_surface_format_default() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        assert_eq!(surface.format(), TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_surface_with_depth() {
        use crate::types::DepthFormat;

        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface =
            Surface::new_with_depth(&device, &window, Some(DepthFormat::Depth24Plus)).unwrap();

        assert_eq!(surface.width(), 800);
        assert_eq!(surface.height(), 600);
    }

    #[test]
    fn test_surface_format_custom() {
        let device = create_test_device_with_format(TextureFormat::Rgba8Unorm);
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        assert_eq!(surface.format(), TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_surface_resize() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let mut surface = Surface::new(&device, &window).unwrap();

        surface.resize(1920, 1080).unwrap();

        assert_eq!(surface.width(), 1920);
        assert_eq!(surface.height(), 1080);
        assert_eq!(surface.size(), (1920, 1080));
    }

    #[test]
    fn test_surface_resize_ignores_zero_dimensions() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let mut surface = Surface::new(&device, &window).unwrap();

        surface.resize(0, 0).unwrap();

        assert_eq!(surface.width(), 800);
        assert_eq!(surface.height(), 600);
    }

    #[test]
    fn test_surface_resize_ignores_zero_width() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let mut surface = Surface::new(&device, &window).unwrap();

        surface.resize(0, 600).unwrap();

        assert_eq!(surface.width(), 800);
        assert_eq!(surface.height(), 600);
    }

    #[test]
    fn test_surface_resize_ignores_zero_height() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let mut surface = Surface::new(&device, &window).unwrap();

        surface.resize(800, 0).unwrap();

        assert_eq!(surface.width(), 800);
        assert_eq!(surface.height(), 600);
    }

    #[test]
    fn test_surface_acquire_and_present_new_api() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        let frame = surface.acquire().unwrap();

        assert_eq!(frame.width(), 800);
        assert_eq!(frame.height(), 600);

        // Frame should have a texture (mock backend supports it)
        assert!(frame.texture().is_some());

        frame.present().unwrap();
    }

    #[test]
    fn test_surface_acquire_and_present_legacy_api() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        let frame = surface.acquire().unwrap();
        surface.present(frame).unwrap();
    }

    #[test]
    fn test_surface_frame_texture_has_correct_dimensions() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        let frame = surface.acquire().unwrap();
        let texture = frame.texture().unwrap();

        // Mock backend creates surfaces with 800x600
        assert_eq!(texture.width(), 800);
        assert_eq!(texture.height(), 600);

        frame.present().unwrap();
    }

    #[test]
    fn test_surface_frame_render_and_present() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        let frame = surface.acquire().unwrap();

        let mut encoder = crate::encoder::CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(crate::types::Color::RED);
        }

        frame.render(encoder).unwrap();
        frame.present().unwrap();
    }

    #[test]
    fn test_surface_depth_frame_render() {
        use crate::types::DepthFormat;

        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface =
            Surface::new_with_depth(&device, &window, Some(DepthFormat::Depth32Float)).unwrap();

        let frame = surface.acquire().unwrap();

        let mut encoder = crate::encoder::CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(crate::types::Color::CORNFLOWER_BLUE);
            pass.clear_depth(1.0);
        }

        frame.render(encoder).unwrap();
        frame.present().unwrap();
    }

    #[test]
    fn test_surface_multiple_acquire_present() {
        let device = create_test_device();
        let window = MockWindow::new(640, 480);
        let surface = Surface::new(&device, &window).unwrap();

        for _ in 0..5 {
            let frame = surface.acquire().unwrap();
            frame.present().unwrap();
        }
    }

    #[test]
    fn test_validate_pipeline_format_matching() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        let result = surface.validate_pipeline_format(TextureFormat::Bgra8UnormSrgb);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_pipeline_format_mismatch() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        let result = surface.validate_pipeline_format(TextureFormat::Rgba8Unorm);
        assert!(result.is_err());

        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Pipeline format mismatch"));
        assert!(msg.contains("Rgba8Unorm"));
        assert!(msg.contains("Bgra8UnormSrgb"));
    }

    #[test]
    fn test_surface_validate_custom_format() {
        let device = create_test_device_with_format(TextureFormat::Rgba8Unorm);
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        assert!(surface
            .validate_pipeline_format(TextureFormat::Rgba8Unorm)
            .is_ok());

        assert!(surface
            .validate_pipeline_format(TextureFormat::Bgra8UnormSrgb)
            .is_err());
    }

    #[test]
    fn test_surface_with_config() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new_with_config(
            &device,
            &window,
            SurfaceConfig {
                present_mode: PresentMode::Immediate,
                depth_format: None,
            },
        )
        .unwrap();

        assert_eq!(surface.width(), 800);
        assert_eq!(surface.height(), 600);
    }

    #[test]
    fn test_surface_frame_drop_without_present() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        // Acquire and drop without presenting — should not panic
        let _frame = surface.acquire().unwrap();
        // frame is dropped here, triggering cleanup
    }
}
