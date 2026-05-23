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

    /// Compile, auto-partition, and submit a graph that writes to the swapchain
    /// output, deferring surface acquisition until after early partitions are
    /// already executing on the GPU.
    ///
    /// 1. Compiles the schedule (cached) and partitions the graph.
    /// 2. Emits and submits early partitions as standalone — GPU starts coarse work.
    /// 3. Calls `self.begin()` (deferred acquire) after ~200 µs of CPU work.
    /// 4. Lowers `SwapchainOutput` → concrete `TextureHandle` + patches UAV index.
    /// 5. Records the final partition deferred to present.
    /// 6. Returns the [`Frame`] for the caller to present.
    ///
    /// The graph **must** contain at least one swapchain-output binding
    /// (declared via [`TaskGraph::declare_swapchain_output`] and bound
    /// via [`NodeBuilder::bind_swapchain_output`](crate::NodeBuilder::bind_swapchain_output)).
    /// Transient buffers and
    /// textures are fully resolved before partitioning.
    pub fn submit_graph(&self, graph: &mut TaskGraph) -> Result<Frame> {
        let _tz = tracy_zone!("surface.submit_graph");

        if !graph.has_swapchain_output() {
            anyhow::bail!(
                "Surface::submit_graph: graph contains no SwapchainOutput binding; \
                 use TaskGraph::declare_swapchain_output + NodeBuilder::bind_swapchain_output, \
                 or render to a texture and copy in a separate graph"
            );
        }

        // ── Step 1: resolve transient resources into a concrete IR ──────────
        // Transient views and textures are now owned by the PlacementHeap across frames.
        let resolved_ir = self.resolve_ir_for_submit_graph(graph)?;

        // ── Step 2: get schedule + swapchain split wave ──────────────────────
        // Schedule is keyed on self.ir (pre-lowering fingerprint) and cached.
        let (schedule, split_wave) = graph.schedule_and_split_wave();

        // ── Step 3: emit early commands from resolved IR ─────────────────────
        // Early waves contain no SwapchainOutput bindings by construction.
        let early_cmds = crate::task_graph::analysis::emit_waves_to_commands(
            &resolved_ir,
            &schedule.waves[..split_wave],
        );

        // ── Step 4: submit early commands ────────────────────────────────────
        if !early_cmds.is_empty() {
            let mut backend = self.backend.lock().unwrap();
            let _tz = tracy_zone!("surface.submit_partition_early");
            backend.submit_standalone(self.device_handle, &early_cmds)?;
        }

        // ── Step 5: deferred surface acquire ────────────────────────────────
        let frame = {
            let _tz = tracy_zone!("surface.deferred_acquire");
            self.begin()?
        };

        // ── Step 6: Transient views and textures are owned by PlacementHeap. ───
        // No keepalive push needed — the heap holds BufferViews and Textures until
        // they are evicted or invalidated. Ring stamping happens in Frame::do_present.

        // ── Step 7: lower SwapchainOutput and emit final partition ───────────
        let swapchain_tex = frame.texture();
        let sc_handle = swapchain_tex.handle;
        let uav_index = swapchain_tex
            .bindless_index()
            .context("swapchain texture has no UAV bindless index")?;

        let final_ir = TaskGraph::lower_swapchain_output(&resolved_ir, sc_handle, uav_index);
        let final_cmds = crate::task_graph::analysis::emit_waves_to_commands(
            &final_ir,
            &schedule.waves[split_wave..],
        );

        // ── Step 8: record final partition deferred to present ───────────────
        {
            let mut backend = self.backend.lock().unwrap();
            let _tz = tracy_zone!("surface.submit_partition_late");
            backend.record_gpu_work(&frame.token, &final_cmds)?;
        }

        Ok(frame)
    }

    /// Resolve transient resources in `graph` into a concrete [`GraphIR`].
    ///
    /// Transient `BufferView`s and `Texture`s are owned by the device's
    /// [`PlacementHeap`] across frames (stable slot ID cache). No keepalive
    /// collections are returned; the heap manages lifetimes directly.
    ///
    /// The IR still contains `ResourceId::SwapchainOutput`; that is lowered
    /// separately after `surface.begin()` in [`submit_graph`].
    fn resolve_ir_for_submit_graph(
        &self,
        graph: &mut TaskGraph,
    ) -> Result<crate::task_graph::GraphIR> {
        use crate::placement_heap::PlacementHeap;
        use crate::task_graph::TaskGraph;

        if !graph.has_transient_resources() {
            return Ok(graph.ir().clone());
        }

        // ── Initialise / reclaim the placement heap ──────────────────────────
        let (total_size, base_align, layout) = if graph.has_transient_buffers() {
            let (ts, ba, lay) = graph.transient_heap_size_and_layout()?;
            (ts, ba, Some(lay))
        } else {
            (0, 1, None)
        };
        let alloc_size = if layout.is_some() {
            (total_size + base_align - 1).max(256)
        } else {
            0
        };

        let mut heap_guard = self._device.inner.placement_heap.lock().unwrap();
        if heap_guard.is_none() {
            let cap = (256 * 1024 * 1024u64)
                .max(alloc_size * crate::device::Device::DEFAULT_PIPELINE_DEPTH);
            *heap_guard = Some(
                PlacementHeap::with_capacity(&self._device, cap)
                    .context("failed to create device placement heap")?,
            );
        }
        let heap = heap_guard.as_mut().unwrap();

        // ── Resolve transient textures via the heap's texture cache ──────────
        let tex_handles = graph.resolve_transient_textures_with_heap(&self._device, heap)?;

        if layout.is_none() {
            // Texture-only path (no transient buffers).
            let resolved_ir = TaskGraph::lower_transient_textures(graph.ir(), &tex_handles)?;
            return Ok(resolved_ir);
        }
        let layout = layout.unwrap();

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

        // ── Resolve transient buffer views via the heap's view cache ─────────
        let mut bindless_map: HashMap<u32, (u32, u32)> =
            HashMap::with_capacity(graph.transient_specs().len());
        for spec in graph.transient_specs() {
            let offset = base_offset + layout[&spec.id];
            let view_stride = spec.stride.max(1);
            let (uav, srv, _hit) =
                heap.get_or_create_view(spec.id, offset, spec.size, view_stride, &self._device)?;
            bindless_map.insert(spec.id, (uav, srv));
        }

        let buf = heap.buffer();
        let range_map = TaskGraph::transient_buffer_range_map_with_base(
            buf,
            &layout,
            graph.transient_specs(),
            base_offset,
        );
        let mut resolved_ir =
            graph.lower_transient_buffers_with_bindless(&range_map, &bindless_map)?;
        if !tex_handles.is_empty() {
            resolved_ir = TaskGraph::lower_transient_textures(&resolved_ir, &tex_handles)?;
        }

        Ok(resolved_ir)
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
