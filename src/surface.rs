//! Surface presentation API.
//!
//! This module provides zero-copy GPU presentation to windows.
//! Use `Surface` to render directly to a window without CPU readback.

use crate::backend::{FrameToken, GpuBackend, SurfaceHandle};
use crate::context::Context as GpuContext;
use crate::timeline::TimelineValue;
use crate::tracy_frame_mark;
use crate::tracy_zone;
use crate::types::{PresentMode, SurfaceConfig, TextureFormat};
use crate::vram_allocator::DeferredPayload;
use crate::Texture;
use anyhow::Result;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::{Arc, Mutex};

/// A GPU surface for zero-copy presentation to a window.
pub(crate) struct Surface {
    context: GpuContext,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    ctx_handle: crate::backend::ContextHandle,
    handle: SurfaceHandle,
    width: u32,
    height: u32,
}

/// A frame acquired from a surface — explicit bracket for render/compute + present.
pub(crate) struct Frame {
    context: GpuContext,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    token: FrameToken,
    texture: Option<Texture>,
    presented: bool,
    /// Timeline returned by [`GpuBackend::submit_frame`] for this bracket.
    submit_tv: Option<TimelineValue>,
    /// Resources (e.g. transient textures) that must outlive the frame's GPU work.
    /// Deferred to the VramAllocator ring at present time. Uses a Mutex so
    /// submit_compute can push to it via &self without requiring &mut Frame.
    keepalive: Mutex<DeferredPayload>,
}

impl Surface {
    #[cfg(test)]
    pub fn new<W>(context: &GpuContext, window: &W) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        Self::new_with_config(context, window, SurfaceConfig::default())
    }

    /// Create a surface bound to `context`'s submission timeline.
    ///
    /// The same [`GpuContext`] must be used for frame submission (`Frame::submit`,
    /// `Frame::present`) and for [`GpuContext::poll_signals`] / reclamation on this
    /// surface. Creating the surface on one context while submitting or polling on
    /// another leaves `gpu_progress()` and swapchain signals on mismatched clocks.
    pub fn new_with_config<W>(context: &GpuContext, window: &W, config: SurfaceConfig) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        let device = context.device();
        let handle = {
            let mut backend = device.inner.backend.lock().unwrap();
            backend.create_surface(device.inner.handle, window, window, config.depth_format)?
        };

        let (width, height) = {
            let backend = device.inner.backend.lock().unwrap();
            backend.surface_size(handle)
        };

        if config.present_mode != PresentMode::Auto {
            let mut backend = device.inner.backend.lock().unwrap();
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
            context: context.clone(),
            backend: Arc::clone(&device.inner.backend),
            ctx_handle: context.backend_handle(),
            handle,
            width,
            height,
        })
    }

    /// Begin the next frame (acquire swapchain image and open the frame bracket).
    pub fn begin(&self) -> Result<Frame> {
        let _tz = tracy_zone!("surface.begin");
        let (token, texture_handle, w, h, format) = {
            let mut backend = self.backend.lock().unwrap();
            let (tok, th) = backend.begin_frame(self.handle, self.ctx_handle)?;
            let (w, h) = backend.surface_size(self.handle);
            let format = backend.surface_format(self.handle);
            (tok, th, w, h, format)
        };

        let texture = Some(Texture::borrowed(
            self.context.device(),
            Arc::clone(&self.backend),
            texture_handle,
            w,
            h,
            format,
        ));

        Ok(Frame {
            context: self.context.clone(),
            backend: Arc::clone(&self.backend),
            token,
            texture,
            presented: false,
            submit_tv: None,
            keepalive: Mutex::new(DeferredPayload::new()),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        tracing::debug!(width, height, "Resizing surface");
        let mut backend = self.backend.lock().unwrap();
        backend.surface_resize(self.handle, width, height)?;
        // Read back the dimensions actually used by the backend. surface_resize may clamp
        // the requested extents to the surface's Vulkan/DX12/Metal capability limits, or
        // bail out early when the clamped extent matches the current swapchain. Storing the
        // actual backend dimensions here keeps Surface.width/height consistent with the
        // underlying swapchain — preventing render targets from being sized at a different
        // resolution than the swapchain scratch texture.
        let (actual_w, actual_h) = backend.surface_size(self.handle);
        self.width = actual_w;
        self.height = actual_h;
        Ok(())
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn format(&self) -> TextureFormat {
        let backend = self.backend.lock().unwrap();
        backend.surface_format(self.handle)
    }

    pub fn set_present_mode(&mut self, mode: PresentMode) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.surface_set_present_mode(self.handle, mode)
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        tracing::debug!(width = self.width, height = self.height, "Destroying surface");
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_surface(self.handle);
    }
}

impl Frame {
    pub fn texture(&self) -> &Texture {
        self.texture
            .as_ref()
            .expect("swapchain texture is only cleared after present")
    }

    /// Timeline already associated with this frame's GPU submit, if any.
    ///
    /// For scheme present this is the present-partition timeline (including
    /// `copy_texture_to_present`), which is when copy sources such as `out_image`
    /// finish being read — earlier than the display present timeline.
    pub(crate) fn submit_timeline(&self) -> Option<TimelineValue> {
        self.submit_tv
    }

    /// Record that GPU work for this frame was already submitted at `tv`.
    ///
    /// Used by scheme present when the present partition submits outside
    /// [`Self::submit_frame`]. First stamp wins; does not call the backend.
    pub(crate) fn note_submit_timeline(&mut self, tv: TimelineValue) {
        if self.submit_tv.is_none() {
            self.submit_tv = Some(tv);
        }
    }

    /// Submit recorded GPU work for this frame. Does not present.
    ///
    /// Safe to call once per frame before [`Self::present`].
    pub fn submit_frame(&mut self) -> Result<TimelineValue> {
        let _tz = tracy_zone!("frame.submit_frame");
        if let Some(tv) = self.submit_tv {
            return Ok(tv);
        }
        let mut backend = self.backend.lock().unwrap();
        let tv = backend.submit_frame(&self.token)?;
        self.submit_tv = Some(tv);
        Ok(tv)
    }

    /// Submit recorded work and present on this thread.
    ///
    /// Returns the easement-expiry timeline value (present/copy completion on the owning
    /// context), not the compute submit timeline.
    pub fn present(mut self) -> Result<TimelineValue> {
        self.do_present_sync()
    }

    fn do_present_sync(&mut self) -> Result<TimelineValue> {
        let _tz = tracy_zone!("frame.present");
        if self.presented {
            return Ok(self.submit_tv.unwrap_or_else(|| self.context.gpu_progress()));
        }
        self.presented = true;
        let _ = self.texture.take();
        let submit_tv = self.submit_frame()?;
        let backend_mutex = &self.backend;
        let present_tv = {
            let work = {
                let mut backend = backend_mutex.lock().unwrap();
                backend.take_present_gpu_work(self.token, submit_tv)?
            };
            let finish = {
                let _gpu = tracy_zone!("frame.present.gpu");
                work.run()?
            };
            let mut backend = backend_mutex.lock().unwrap();
            backend.finish_present(finish, submit_tv)?
        };
        self.apply_frame_bookkeeping(submit_tv)?;
        Ok(present_tv)
    }

    fn apply_frame_bookkeeping(&self, submit_tv: TimelineValue) -> Result<()> {
        tracy_frame_mark!();
        let keepalive = std::mem::take(&mut *self.keepalive.lock().unwrap());
        if !keepalive.is_empty() {
            self.context.defer_release(submit_tv, keepalive);
        }
        Ok(())
    }

    /// In-flight slot index for the compute/scratch texture bound this frame.
    ///
    /// Present-lease retention keys must use this, not [`Self::image_index`], because
    /// on Vulkan the WSI swapchain image and the shader-target scratch texture are
    /// indexed independently.
    pub fn frame_slot(&self) -> u32 {
        self.token.frame_slot
    }

    /// Abandon this frame without presenting.
    ///
    /// Marks the frame as already-presented so the `Drop` impl does not trigger
    /// an implicit swapchain present. Use this to cancel a frame when submission
    /// fails after the swapchain image was acquired but before work was submitted.
    pub(crate) fn cancel(mut self) {
        self.presented = true;
        // Drop self — with `presented = true` the Drop impl is a no-op.
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        if !self.presented {
            let _ = self.do_present_sync();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::device::Device;

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
        fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            Ok(unsafe {
                raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Web(
                    raw_window_handle::WebWindowHandle::new(0),
                ))
            })
        }
    }

    impl raw_window_handle::HasDisplayHandle for MockWindow {
        fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            Ok(unsafe {
                raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Web(
                    raw_window_handle::WebDisplayHandle::new(),
                ))
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
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        assert_eq!(window.size(), (800, 600));
        let surface = Surface::new(&ctx, &window).unwrap();

        assert_eq!(surface.size(), (800, 600));
    }

    #[test]
    fn test_surface_format_default() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        assert_eq!(surface.format(), TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_surface_with_depth_config() {
        use crate::types::DepthFormat;

        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new_with_config(
            &ctx,
            &window,
            SurfaceConfig {
                depth_format: Some(DepthFormat::Depth24Plus),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(surface.size(), (800, 600));
    }

    #[test]
    fn test_surface_format_custom() {
        let device = create_test_device_with_format(TextureFormat::Rgba8Unorm);
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        assert_eq!(surface.format(), TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_surface_resize() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let mut surface = Surface::new(&ctx, &window).unwrap();

        surface.resize(1920, 1080).unwrap();

        assert_eq!(surface.size(), (1920, 1080));
    }

    #[test]
    fn test_surface_resize_ignores_zero_dimensions() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let mut surface = Surface::new(&ctx, &window).unwrap();

        surface.resize(0, 0).unwrap();

        assert_eq!(surface.size(), (800, 600));
    }

    #[test]
    fn test_surface_begin_and_present() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        let frame = surface.begin().unwrap();

        assert_eq!(frame.texture().width(), 800);
        assert_eq!(frame.texture().height(), 600);

        frame.present().unwrap();
    }

    #[test]
    fn test_surface_multiple_begin_present() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(640, 480);
        let surface = Surface::new(&ctx, &window).unwrap();

        for _ in 0..5 {
            let frame = surface.begin().unwrap();
            frame.present().unwrap();
        }
    }

    #[test]
    fn test_surface_with_config() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new_with_config(
            &ctx,
            &window,
            SurfaceConfig {
                present_mode: PresentMode::Immediate,
                depth_format: None,
            },
        )
        .unwrap();

        assert_eq!(surface.size(), (800, 600));
    }

    #[test]
    fn test_surface_frame_drop_without_present() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        let _frame = surface.begin().unwrap();
    }
}
