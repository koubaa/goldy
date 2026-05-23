//! Surface presentation API.
//!
//! This module provides zero-copy GPU presentation to windows.
//! Use `Surface` to render directly to a window without CPU readback.

use crate::backend::{FrameToken, GpuBackend, SurfaceHandle};
use crate::device::Device;
use crate::encoder::CommandEncoder;
use crate::task_graph::TaskGraph;
use crate::texture::Texture;
use crate::timeline::TimelineValue;
use crate::tracy_frame_mark;
use crate::tracy_zone;
use crate::types::{PresentMode, SurfaceConfig, TextureFormat};
use crate::vram_allocator::DeferredPayload;
use anyhow::{Context, Result};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A GPU surface for zero-copy presentation to a window.
pub struct Surface {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    device_handle: crate::backend::DeviceHandle,
    handle: SurfaceHandle,
    width: u32,
    height: u32,
}

/// A frame acquired from a surface — explicit bracket for render/compute + present.
pub struct Frame {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    device_handle: crate::backend::DeviceHandle,
    token: FrameToken,
    texture: Option<Texture>,
    width: u32,
    height: u32,
    presented: bool,
    /// Resources (e.g. transient textures) that must outlive the frame's GPU work.
    /// Deferred to the VramAllocator ring at present time. Uses a Mutex so
    /// submit_compute can push to it via &self without requiring &mut Frame.
    keepalive: Mutex<DeferredPayload>,
}

impl Surface {
    pub fn new<W>(device: &Device, window: &W) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        Self::new_with_config(device, window, SurfaceConfig::default())
    }

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

    pub fn new_with_config<W>(device: &Device, window: &W, config: SurfaceConfig) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
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
            _device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            device_handle: device.inner.handle,
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
            let (tok, th) = backend.begin_frame(self.handle)?;
            let (w, h) = backend.surface_size(self.handle);
            let format = backend.surface_format(self.handle);
            (tok, th, w, h, format)
        };

        let texture = Some(Texture::borrowed(
            Arc::clone(&self.backend),
            texture_handle,
            w,
            h,
            format,
        ));

        Ok(Frame {
            _device: self._device.clone(),
            backend: Arc::clone(&self.backend),
            device_handle: self.device_handle,
            token,
            texture,
            width: w,
            height: h,
            presented: false,
            keepalive: Mutex::new(DeferredPayload::new()),
        })
    }

    /// Acquire the next frame (legacy name for [`Surface::begin`]).
    pub fn acquire(&self) -> Result<Frame> {
        self.begin()
    }

    /// Present a rendered frame (legacy API — prefer [`Frame::present`]).
    pub fn present(&self, mut frame: Frame) -> Result<TimelineValue> {
        frame.do_present()
    }

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

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> TextureFormat {
        let backend = self.backend.lock().unwrap();
        backend.surface_format(self.handle)
    }

    pub fn set_present_mode(&mut self, mode: PresentMode) -> Result<()> {
        let mut backend = self.backend.lock().unwrap();
        backend.surface_set_present_mode(self.handle, mode)
    }

    pub fn present_mode(&self) -> PresentMode {
        let backend = self.backend.lock().unwrap();
        backend.surface_present_mode(self.handle)
    }

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

impl Frame {
    pub fn texture(&self) -> &Texture {
        self.texture
            .as_ref()
            .expect("swapchain texture is only cleared after present")
    }

    pub fn render(&self, encoder: CommandEncoder) -> Result<()> {
        let commands = encoder.finish();
        let mut backend = self.backend.lock().unwrap();
        backend.record_render(&self.token, &commands)
    }

    /// Record analyzed compute / transfer work for this frame (e.g. compute into the swapchain).
    ///
    /// Graphs containing transient buffers and/or transient textures are resolved
    /// automatically using the device-owned placement heap. The caller does not need
    /// to call [`Device::submit`](crate::Device::submit) for graphs with transients;
    /// this method handles the full resolution transparently.
    pub fn submit_compute(&self, graph: &mut TaskGraph) -> Result<()> {
        let _tz = tracy_zone!("frame.submit_compute");
        let partitions = if !graph.has_transient_resources() {
            graph.compile_partitioned_commands()
        } else {
            // Resolve transient textures first (may be needed even without transient buffers).
            let (tex_keepalive, tex_handles) = graph.allocate_transient_textures(&self._device)?;

            let parts = if graph.has_transient_buffers() {
                self.resolve_and_compile_transient_buffers(graph, tex_handles)?
            } else {
                // Transient textures only — lower the IR and compile partitioned.
                let resolved_ir = TaskGraph::lower_transient_textures(graph.ir(), &tex_handles)?;
                graph.compile_resolved_to_partitioned_commands(&resolved_ir)
            };

            // Stash textures so they survive until do_present defers them via VramAllocator.
            if !tex_keepalive.is_empty() {
                let mut keepalive = self.keepalive.lock().unwrap();
                for tex in tex_keepalive {
                    keepalive.push(tex);
                }
            }

            parts
        };

        self.submit_partitions(partitions)
    }

    /// Submit a list of partitioned command slices to the backend.
    ///
    /// All partitions except the last are submitted immediately via
    /// [`GpuBackend::submit_standalone`], allowing the GPU to begin executing
    /// early (coarse) work while the CPU records the final partition. The last
    /// partition is deferred to present via [`GpuBackend::record_gpu_work`].
    ///
    /// Same-queue in-order execution guarantees that early partitions complete
    /// before the final partition begins on the GPU. Memory visibility across
    /// submissions is covered by the backend's cross-submission acquire barrier.
    fn submit_partitions(&self, partitions: Vec<Vec<crate::backend::GpuCommand>>) -> Result<()> {
        let n = partitions.len();
        let mut backend = self.backend.lock().unwrap();
        for (i, partition) in partitions.into_iter().enumerate() {
            if i < n - 1 {
                // Early partitions: submit immediately so the GPU starts executing.
                let _tz = tracy_zone!("frame.submit_partition_early");
                backend.submit_standalone(self.device_handle, &partition)?;
            } else {
                // Final partition: defer to present so it synchronises with the swapchain.
                let _tz = tracy_zone!("frame.submit_partition_late");
                backend.record_gpu_work(&self.token, &partition)?;
            }
        }
        Ok(())
    }

    /// Acquire a placement-heap region, create BufferViews at colored offsets, lower the
    /// graph IR, and compile it to a flat GPU command stream.
    ///
    /// Mirrors the logic in `Device::submit_with_placement_heap` but targets the
    /// surface-frame path: views are attached to the heap ring entry *unstamped*;
    /// `do_present` stamps all pending regions via `stamp_all_pending`.
    fn resolve_and_compile_transient_buffers(
        &self,
        graph: &mut TaskGraph,
        tex_handles: HashMap<u32, crate::backend::TextureHandle>,
    ) -> Result<Vec<Vec<crate::backend::GpuCommand>>> {
        use crate::buffer::BufferView;
        use crate::placement_heap::PlacementHeap;

        let (total_size, base_align, layout) = graph.transient_heap_size_and_layout()?;
        let alloc_size = (total_size + base_align - 1).max(256);

        let mut heap_guard = self._device.inner.placement_heap.lock().unwrap();

        // Lazily create or grow the placement heap (same logic as Device::submit).
        if heap_guard.is_none() {
            let cap = (256 * 1024 * 1024u64)
                .max(alloc_size * crate::device::Device::DEFAULT_PIPELINE_DEPTH);
            *heap_guard = Some(
                PlacementHeap::with_capacity(&self._device, cap)
                    .context("failed to create device placement heap")?,
            );
        }
        let heap = heap_guard.as_mut().unwrap();

        let progress = self._device.gpu_progress();
        heap.reclaim(progress);

        let raw_offset = match heap.acquire(alloc_size) {
            Some(off) => off,
            None => {
                let progress2 = self._device.gpu_progress();
                heap.reclaim(progress2);
                heap.acquire(alloc_size).ok_or_else(|| {
                    anyhow::anyhow!(
                        "PlacementHeap exhausted: need {} bytes, cap={}, in_flight={}",
                        alloc_size,
                        heap.capacity(),
                        heap.in_flight_bytes(),
                    )
                })?
            }
        };
        let base_offset = raw_offset.div_ceil(base_align) * base_align;

        let buf = heap.buffer();

        let mut views: Vec<BufferView> = Vec::with_capacity(layout.len());
        let mut bindless_map: HashMap<u32, (u32, u32)> = HashMap::with_capacity(layout.len());

        for spec in graph.transient_specs() {
            let offset = base_offset + layout[&spec.id];
            let view_stride = spec.stride.max(1);
            let view = buf.create_view(offset, spec.size, Some(view_stride))?;
            let uav = view.bindless_index().unwrap_or(u32::MAX);
            let srv = view.bindless_srv_index().unwrap_or(uav);
            bindless_map.insert(spec.id, (uav, srv));
            views.push(view);
        }

        let range_map = TaskGraph::transient_buffer_range_map_with_base(
            buf,
            &layout,
            graph.transient_specs(),
            base_offset,
        );
        let mut resolved_ir =
            graph.lower_transient_buffers_with_bindless(&range_map, &bindless_map)?;

        // Also lower transient textures if present.
        if !tex_handles.is_empty() {
            resolved_ir = TaskGraph::lower_transient_textures(&resolved_ir, &tex_handles)?;
        }

        let partitions = graph.compile_resolved_to_partitioned_commands(&resolved_ir);

        // Defer views via keepalive so they outlive the GPU work.
        // do_present drains keepalive via device.defer_release(tv, ...).
        // This keeps PlacementHeap ring entries free of view ownership (#150).
        {
            let mut keepalive = self.keepalive.lock().unwrap();
            for view in views {
                keepalive.push(view);
            }
        }

        Ok(partitions)
    }

    /// Submit all work and present. Returns the GPU timeline value when this frame completes.
    pub fn present(mut self) -> Result<TimelineValue> {
        self.do_present()
    }

    fn do_present(&mut self) -> Result<TimelineValue> {
        let _tz = tracy_zone!("frame.present");
        if self.presented {
            let backend = self.backend.lock().unwrap();
            return Ok(backend.gpu_progress(self.device_handle));
        }
        self.presented = true;
        let _ = self.texture.take();
        let token = self.token;
        let tv = {
            let mut backend = self.backend.lock().unwrap();
            backend.end_frame(token)?
        };
        tracy_frame_mark!();
        // Stamp any pending placement heap regions with the present timeline.
        if let Ok(mut heap_guard) = self._device.inner.placement_heap.lock() {
            if let Some(ref mut heap) = *heap_guard {
                heap.stamp_all_pending(tv);
            }
        }
        // Defer any keepalive resources (e.g. transient textures from submit_compute)
        // until the GPU retires this frame's timeline.
        let keepalive = std::mem::take(&mut *self.keepalive.lock().unwrap());
        if !keepalive.is_empty() {
            self._device.defer_release(tv, keepalive);
        }
        Ok(tv)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        if !self.presented {
            let _ = self.do_present();
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
    fn test_surface_begin_and_present() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        let frame = surface.begin().unwrap();

        assert_eq!(frame.width(), 800);
        assert_eq!(frame.height(), 600);
        assert_eq!(frame.texture().width(), 800);
        assert_eq!(frame.texture().height(), 600);

        frame.present().unwrap();
    }

    #[test]
    fn test_surface_present_legacy() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        let frame = surface.begin().unwrap();
        surface.present(frame).unwrap();
    }

    #[test]
    fn test_surface_frame_render_and_present() {
        let device = create_test_device();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&device, &window).unwrap();

        let frame = surface.begin().unwrap();

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

        let frame = surface.begin().unwrap();

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
    fn test_surface_multiple_begin_present() {
        let device = create_test_device();
        let window = MockWindow::new(640, 480);
        let surface = Surface::new(&device, &window).unwrap();

        for _ in 0..5 {
            let frame = surface.begin().unwrap();
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

        let _frame = surface.begin().unwrap();
    }
}
