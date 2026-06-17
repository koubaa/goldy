//! Surface presentation API.
//!
//! This module provides zero-copy GPU presentation to windows.
//! Use `Surface` to render directly to a window without CPU readback.

use crate::backend::{FrameToken, GpuBackend, SurfaceHandle};
use crate::context::Context as GpuContext;
use crate::task_graph::TaskGraph;
use crate::texture::Texture;
use crate::timeline::TimelineValue;
use crate::tracy_frame_mark;
use crate::tracy_zone;
use crate::types::{PresentMode, ResourceAccess, SurfaceConfig, TextureFormat};
use crate::vram_allocator::DeferredPayload;
use anyhow::{Context, Result};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::{Arc, Mutex};

/// A GPU surface for zero-copy presentation to a window.
pub struct Surface {
    context: GpuContext,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    ctx_handle: crate::backend::ContextHandle,
    handle: SurfaceHandle,
    width: u32,
    height: u32,
}

/// A frame acquired from a surface — explicit bracket for render/compute + present.
pub struct Frame {
    context: GpuContext,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    ctx_handle: crate::backend::ContextHandle,
    token: FrameToken,
    texture: Option<Texture>,
    width: u32,
    height: u32,
    presented: bool,
    /// Timeline returned by [`GpuBackend::submit_frame`] for this bracket.
    submit_tv: Option<TimelineValue>,
    /// Resources (e.g. transient textures) that must outlive the frame's GPU work.
    /// Deferred to the VramAllocator ring at present time. Uses a Mutex so
    /// submit_compute can push to it via &self without requiring &mut Frame.
    keepalive: Mutex<DeferredPayload>,
    /// Parcel stamp cells moved from a [`TaskGraph`] at [`Surface::submit_graph`] time;
    /// applied at [`Self::submit_frame`].
    stamp_targets: Vec<Arc<crate::parcel::ParcelStamp>>,
}

impl Surface {
    /// Create a surface for any type that exposes window/display handles (e.g. winit, SDL).
    ///
    /// This only requires [`HasWindowHandle`] and [`HasDisplayHandle`] — it is not tied to a
    /// particular window toolkit. The stable C ABI (`libgoldy_ffi`) currently exposes separate
    /// platform entry points instead (`goldy_surface_create_win32`, `goldy_surface_create_appkit`),
    /// which forces FFI clients and examples to extract raw handles themselves. That coupling
    /// should be loosened over time (e.g. a portable surface-create descriptor in the C header)
    /// so `goldy-ffi-client` can offer the same constructor shape without window-toolkit code
    /// in application examples.
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
            Arc::clone(&self.backend),
            texture_handle,
            w,
            h,
            format,
        ));

        Ok(Frame {
            context: self.context.clone(),
            backend: Arc::clone(&self.backend),
            ctx_handle: self.ctx_handle,
            token,
            texture,
            width: w,
            height: h,
            presented: false,
            submit_tv: None,
            keepalive: Mutex::new(DeferredPayload::new()),
            stamp_targets: Vec::new(),
        })
    }

    /// Acquire the next frame (legacy name for [`Surface::begin`]).
    pub fn acquire(&self) -> Result<Frame> {
        self.begin()
    }

    /// Present a rendered frame (legacy API — prefer [`Frame::present`]).
    pub fn present(&self, mut frame: Frame) -> Result<TimelineValue> {
        frame.do_present_sync()
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

    /// How many swapchain drawables are in-flight (acquired or presented, not yet returned).
    pub fn pending_acquire_count(&self) -> u32 {
        let backend = self.backend.lock().unwrap();
        backend.pending_acquire_count(self.handle)
    }

    /// Compile, auto-partition, and submit a graph that writes to the swapchain
    /// output, deferring surface acquisition until after early partitions are
    /// already executing on the GPU.
    ///
    /// 1. Compiles the schedule (cached) and partitions the graph.
    /// 2. Emits and submits early partitions as standalone — GPU starts coarse work.
    /// 3. Calls `self.begin()` (deferred acquire) after ~200 µs of CPU work.
    /// 4. Emits late-partition commands with swapchain resolved via `SlotResolver`.
    /// 5. Records the final partition deferred to present.
    /// 6. Returns the [`Frame`] for the caller to present.
    ///
    /// The graph **must** contain at least one swapchain-output binding
    /// (declared via [`TaskGraph::declare_swapchain_output`] and bound
    /// via [`NodeBuilder::bind_swapchain_output`](crate::NodeBuilder::bind_swapchain_output)).
    pub fn submit_graph(&self, graph: &mut TaskGraph) -> Result<Frame> {
        let _tz = tracy_zone!("surface.submit_graph");

        if !graph.has_swapchain_output() {
            anyhow::bail!(
                "Surface::submit_graph: graph contains no SwapchainOutput binding; \
                 use TaskGraph::declare_swapchain_output + NodeBuilder::bind_swapchain_output, \
                 or render to a texture and copy in a separate graph"
            );
        }

        // ── Step 1: schedule + swapchain split wave ──────────────────────────
        let (schedule, split_wave) = {
            let _tz = tracy_zone!("surface.submit_graph.schedule");
            graph.schedule_and_split_wave()
        };

        // Derive the node-to-wave map once from the cached schedule. This is
        // O(N) and is reused by transient_heap_size_and_layout and
        // resolve_transient_textures_with_heap, avoiding two redundant
        // build_edges + schedule_waves passes per frame.
        let node_waves = {
            let _tz = tracy_zone!("surface.submit_graph.node_to_wave_map");
            crate::task_graph::analysis::node_to_wave_map(&schedule, graph.ir().nodes.len())
        };

        // ── Step 2: build the transient SlotResolver ─────────────────────────
        let resolver = {
            let _tz = tracy_zone!("surface.submit_graph.build_resolver");
            self.build_resolver_for_submit_graph(graph, &node_waves)?
        };

        // ── Step 3–4: emit + submit early partition ──────────────────────────
        if split_wave > 0 {
            let early_waves = &schedule.waves[..split_wave];
            let mut backend = self.backend.lock().unwrap();
            let _tz = tracy_zone!("surface.submit_partition_early");
            if crate::task_graph::analysis::waves_contain_render_pass(graph.ir(), early_waves) {
                let early_g = crate::task_graph::analysis::emit_graph_commands_for_waves(
                    graph.ir(),
                    early_waves,
                    resolver.as_ref(),
                );
                if !early_g.is_empty() {
                    backend.submit_graph(self.ctx_handle, &early_g, None)?;
                }
            } else {
                let early_cmds =
                    crate::task_graph::analysis::emit_waves_to_commands(graph.ir(), early_waves, resolver.as_ref());
                if !early_cmds.is_empty() {
                    backend.submit_standalone(self.ctx_handle, &early_cmds, None)?;
                }
            }
        }

        // ── Step 5: deferred surface acquire ─────────────────────────────────
        let mut frame = {
            let _tz = tracy_zone!("surface.deferred_acquire");
            self.begin()?
        };

        // ── Step 6: add swapchain to the resolver ────────────────────────────
        let swapchain_tex = frame.texture();
        let sc_handle = swapchain_tex.handle;
        let uav_index = swapchain_tex
            .resource_index(ResourceAccess::Write)
            .context("swapchain texture has no UAV resource index")?;

        let mut full_resolver = resolver.unwrap_or_default();
        full_resolver.swapchain = Some(crate::task_graph::ResolvedSwapchain {
            handle: sc_handle,
            uav_index,
        });

        // ── Step 7: emit late commands with full resolver ────────────────────
        let final_cmds = crate::task_graph::analysis::emit_waves_to_commands(
            graph.ir(),
            &schedule.waves[split_wave..],
            Some(&full_resolver),
        );

        // ── Step 8: record final partition deferred to present ───────────────
        {
            let mut backend = self.backend.lock().unwrap();
            let _tz = tracy_zone!("surface.submit_partition_late");
            backend.record_gpu_work(&frame.token, &final_cmds)?;
        }

        frame.stamp_targets = graph.take_stamp_targets();
        Ok(frame)
    }

    /// Compile, auto-partition, and submit a graph that writes to an already-acquired frame.
    ///
    /// This preserves `SwapchainOutput` as an abstract graph resource while allowing callers to
    /// request the WSI image at the beginning of frame recording. Early partitions still run before
    /// the swapchain-touching partition; the late partition is resolved against `frame`.
    pub fn submit_graph_to_frame(&self, graph: &mut TaskGraph, mut frame: Frame) -> Result<Frame> {
        let _tz = tracy_zone!("surface.submit_graph_to_frame");

        if frame.token.surface != self.handle {
            anyhow::bail!("Surface::submit_graph_to_frame: frame belongs to a different surface");
        }
        if graph.is_empty() {
            return Ok(frame);
        }
        if !graph.has_swapchain_output() {
            anyhow::bail!(
                "Surface::submit_graph_to_frame: graph contains no SwapchainOutput binding; \
                 use TaskGraph::declare_swapchain_output + copy/bind to SwapchainOutput"
            );
        }

        let (schedule, split_wave) = {
            let _tz = tracy_zone!("surface.submit_graph_to_frame.schedule");
            graph.schedule_and_split_wave()
        };
        let node_waves = {
            let _tz = tracy_zone!("surface.submit_graph_to_frame.node_to_wave_map");
            crate::task_graph::analysis::node_to_wave_map(&schedule, graph.ir().nodes.len())
        };
        let resolver = {
            let _tz = tracy_zone!("surface.submit_graph_to_frame.build_resolver");
            self.build_resolver_for_submit_graph(graph, &node_waves)?
        };

        let swapchain_tex = frame.texture();
        let sc_handle = swapchain_tex.handle;
        let uav_index = swapchain_tex
            .resource_index(ResourceAccess::Write)
            .context("swapchain texture has no UAV resource index")?;

        let mut full_resolver = resolver.unwrap_or_default();
        full_resolver.swapchain = Some(crate::task_graph::ResolvedSwapchain {
            handle: sc_handle,
            uav_index,
        });

        // The drawable is already acquired — record early and late partitions into the
        // same frame command buffer instead of submitting early work standalone.
        // A separate early CB forces gpu_idle=false on the late partition uploads
        // (staging-belt slow path on every frame).
        let mut backend = self.backend.lock().unwrap();
        let _tz = tracy_zone!("surface.submit_graph_to_frame.record");

        if split_wave > 0 {
            let early_waves = &schedule.waves[..split_wave];
            if crate::task_graph::analysis::waves_contain_render_pass(graph.ir(), early_waves) {
                let early_g = crate::task_graph::analysis::emit_graph_commands_for_waves(
                    graph.ir(),
                    early_waves,
                    Some(&full_resolver),
                );
                if !early_g.is_empty() {
                    backend.submit_graph(self.ctx_handle, &early_g, None)?;
                }
            } else {
                let early_cmds =
                    crate::task_graph::analysis::emit_waves_to_commands(graph.ir(), early_waves, Some(&full_resolver));
                if !early_cmds.is_empty() {
                    backend.record_gpu_work(&frame.token, &early_cmds)?;
                }
            }
        }

        let final_cmds = crate::task_graph::analysis::emit_waves_to_commands(
            graph.ir(),
            &schedule.waves[split_wave..],
            Some(&full_resolver),
        );
        if !final_cmds.is_empty() {
            backend.record_gpu_work(&frame.token, &final_cmds)?;
        }

        frame.stamp_targets = graph.take_stamp_targets();
        Ok(frame)
    }

    /// Build a [`SlotResolver`] for transient resources (buffers + textures).
    ///
    /// Returns `None` when the graph has no transients (no resolution needed).
    /// The resolver does **not** include swapchain — that is filled after
    /// `surface.begin()`.
    fn build_resolver_for_submit_graph(
        &self,
        graph: &mut TaskGraph,
        node_waves: &[u32],
    ) -> Result<Option<crate::task_graph::SlotResolver>> {
        use crate::context::Context as GpuContext;
        use crate::placement_heap::PlacementHeap;
        use crate::task_graph::{ResolvedTransientBuffer, ResolvedTransientTexture, SlotResolver};

        if !graph.has_transient_resources() {
            return Ok(None);
        }

        let mut resolver = SlotResolver::new();

        // ── Compute layout (needed to know alloc_size for configure_pages) ─────
        let (alloc_size, base_align, layout_opt) = if graph.has_transient_buffers() {
            let (ts, ba, lay) = graph.transient_heap_size_and_layout(node_waves)?;
            let sz = (ts + ba - 1).max(256);
            (sz, ba, Some(lay))
        } else {
            (0u64, 1u64, None)
        };

        // ── Initialise the placement heap ────────────────────────────────────────
        let device = self.context.device();
        // Query GPU-retired timeline before locking the heap to avoid acquiring
        // the backend lock while the heap guard is held.
        let retired_timeline = device.timeline_retired();
        let mut heap_guard = self.context.inner.placement_heap.lock().unwrap();
        if heap_guard.is_none() {
            let cap = (256 * 1024 * 1024u64).max(alloc_size * GpuContext::DEFAULT_PIPELINE_DEPTH);
            *heap_guard =
                Some(PlacementHeap::with_capacity(device, cap).context("failed to create context placement heap")?);
        }
        let heap = heap_guard.as_mut().unwrap();

        // ── Switch to paged mode FIRST (may invalidate_all; must happen before ──
        // ── texture resolution so caches are clean when textures are created)  ──
        let depth = GpuContext::DEFAULT_PIPELINE_DEPTH as usize;
        if layout_opt.is_some() {
            let _tz = tracy_zone!("surface.build_resolver.configure_pages");
            heap.configure_pages(alloc_size, depth, device)?;
        }

        // ── Resolve transient textures ───────────────────────────────────────────
        let tex_page_slot = heap.current_page_slot();
        let tex_handles = {
            let _tz = tracy_zone!("surface.build_resolver.resolve_textures");
            graph.resolve_transient_textures_with_heap(device, heap, node_waves, tex_page_slot, retired_timeline)?
        };
        for (id, handle) in &tex_handles {
            resolver
                .textures
                .insert(*id, ResolvedTransientTexture { handle: *handle });
        }

        if layout_opt.is_none() {
            return Ok(Some(resolver));
        }
        let layout = layout_opt.unwrap();

        // ── Get this frame's deterministic page offset ───────────────────────────
        let raw_offset = {
            let _tz = tracy_zone!("surface.build_resolver.advance_page");
            heap.advance_page()
        };
        let base_offset = raw_offset.div_ceil(base_align) * base_align;

        // ── Populate view cache and fill resolver ────────────────────────────────
        let buf_handle = heap.buffer().handle;
        {
            let _tz = tracy_zone!("surface.build_resolver.create_views");
            for spec in graph.transient_specs() {
                let offset = base_offset + layout[&spec.id];
                let view_stride = spec.stride.max(1);
                let (uav, srv, _hit) = heap.get_or_create_view(spec.id, offset, spec.size, view_stride, device)?;
                resolver.buffers.insert(
                    spec.id,
                    ResolvedTransientBuffer {
                        parent: buf_handle,
                        offset,
                        len: spec.size,
                        uav_index: uav,
                        srv_index: srv,
                    },
                );
            }
        }

        Ok(Some(resolver))
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
        if !self.stamp_targets.is_empty() {
            crate::task_graph::apply_stamp_targets(
                &self.stamp_targets,
                self.ctx_handle,
                &self.context.device().inner,
                tv,
            );
            self.context.advance_high_water_timeline(tv);
        }
        self.submit_tv = Some(tv);
        Ok(tv)
    }

    /// Submit recorded work and present on this thread.
    ///
    /// Returns the **submit** timeline (compute completion), not a separate present signal.
    pub fn present(mut self) -> Result<TimelineValue> {
        self.do_present_sync()
    }

    fn do_present_sync(&mut self) -> Result<TimelineValue> {
        let _tz = tracy_zone!("frame.present");
        if self.presented {
            return Ok(self.submit_tv.unwrap_or_else(|| {
                let backend = self.backend.lock().unwrap();
                backend.gpu_progress(self.ctx_handle)
            }));
        }
        self.presented = true;
        let _ = self.texture.take();
        let submit_tv = self.submit_frame()?;
        {
            let mut backend = self.backend.lock().unwrap();
            backend.present_frame(self.token, submit_tv)?;
        }
        self.apply_frame_bookkeeping(submit_tv)?;
        Ok(submit_tv)
    }

    fn apply_frame_bookkeeping(&self, submit_tv: TimelineValue) -> Result<()> {
        tracy_frame_mark!();
        if let Ok(mut heap_guard) = self.context.inner.placement_heap.lock() {
            if let Some(ref mut heap) = *heap_guard {
                heap.stamp_all_pending(submit_tv);
            }
        }
        let keepalive = std::mem::take(&mut *self.keepalive.lock().unwrap());
        if !keepalive.is_empty() {
            self.context.defer_release(submit_tv, keepalive);
        }
        Ok(())
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Swapchain image index acquired by [`Surface::begin`] for this frame.
    pub fn image_index(&self) -> u32 {
        self.token.image as u32
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

        assert_eq!(surface.width(), 800);
        assert_eq!(surface.height(), 600);
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

        assert_eq!(surface.width(), 800);
        assert_eq!(surface.height(), 600);
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

        assert_eq!(surface.width(), 1920);
        assert_eq!(surface.height(), 1080);
        assert_eq!(surface.size(), (1920, 1080));
    }

    #[test]
    fn test_surface_resize_ignores_zero_dimensions() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let mut surface = Surface::new(&ctx, &window).unwrap();

        surface.resize(0, 0).unwrap();

        assert_eq!(surface.width(), 800);
        assert_eq!(surface.height(), 600);
    }

    #[test]
    fn test_surface_begin_and_present() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

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
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        let frame = surface.begin().unwrap();
        surface.present(frame).unwrap();
    }

    #[test]
    fn test_surface_graph_render_and_present() {
        use crate::render_target::RenderTarget;

        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();
        let scene_rt = RenderTarget::new(&device, surface.width(), surface.height(), surface.format()).unwrap();

        let mut graph = TaskGraph::new();
        let mut pass = graph.render_pass("clear", &scene_rt);
        pass.clear(crate::types::Color::RED);
        pass.finish_recorded();
        let swapchain = graph.declare_swapchain_output();
        graph.copy_render_target_to_swapchain(&scene_rt, swapchain);

        let frame = surface.submit_graph(&mut graph).unwrap();
        frame.present().unwrap();
    }

    #[test]
    fn test_surface_graph_render_with_depth_rt() {
        use crate::render_target::RenderTarget;
        use crate::types::DepthFormat;

        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();
        let scene_rt = RenderTarget::new_with_depth(
            &device,
            surface.width(),
            surface.height(),
            surface.format(),
            Some(DepthFormat::Depth32Float),
        )
        .unwrap();

        let mut graph = TaskGraph::new();
        let mut pass = graph.render_pass("depth_clear", &scene_rt);
        pass.clear(crate::types::Color::CORNFLOWER_BLUE);
        pass.clear_depth(1.0);
        pass.finish_recorded();
        let swapchain = graph.declare_swapchain_output();
        graph.copy_render_target_to_swapchain(&scene_rt, swapchain);

        let frame = surface.submit_graph(&mut graph).unwrap();
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
    fn test_validate_pipeline_format_matching() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        let result = surface.validate_pipeline_format(TextureFormat::Bgra8UnormSrgb);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_pipeline_format_mismatch() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        let result = surface.validate_pipeline_format(TextureFormat::Rgba8Unorm);
        assert!(result.is_err());
    }

    #[test]
    fn test_surface_validate_custom_format() {
        let device = create_test_device_with_format(TextureFormat::Rgba8Unorm);
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        assert!(surface.validate_pipeline_format(TextureFormat::Rgba8Unorm).is_ok());

        assert!(surface.validate_pipeline_format(TextureFormat::Bgra8UnormSrgb).is_err());
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

        assert_eq!(surface.width(), 800);
        assert_eq!(surface.height(), 600);
    }

    #[test]
    fn test_surface_frame_drop_without_present() {
        let device = create_test_device();
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        let _frame = surface.begin().unwrap();
    }

    #[test]
    fn submit_graph_stamps_bound_parcel_at_submit_frame() {
        use std::sync::Arc;

        use crate::compute::ComputePipeline;
        use crate::retained_pool::RetainedPool;
        use crate::shader::ShaderModule;
        use crate::task_graph::NodeAccess;
        use crate::types::{BufferFlags, BufferKind};

        let device = Arc::new(create_test_device());
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        let mut pool = RetainedPool::new(device.clone());
        let parcel = pool
            .acquire_buffer(256, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();
        assert!(parcel.last_referenced().is_empty());

        let shader = ShaderModule::from_slang(&device, "void main() {}").unwrap();
        let pipeline = ComputePipeline::new(&device, &shader).unwrap();

        let mut graph = TaskGraph::new();
        let sc = graph.declare_swapchain_output();
        graph
            .node("fine", &pipeline)
            .with_parcel(&parcel, NodeAccess::ReadWrite)
            .with_swapchain_output(sc, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let mut frame = surface.submit_graph(&mut graph).unwrap();
        let tv = frame.submit_frame().unwrap();
        assert_eq!(parcel.last_referenced_on(ctx.backend_handle()), Some(tv));
        let _ = frame.present();
    }

    #[test]
    fn submit_graph_to_frame_stamps_bound_parcel_at_submit_frame() {
        use std::sync::Arc;

        use crate::compute::ComputePipeline;
        use crate::retained_pool::RetainedPool;
        use crate::shader::ShaderModule;
        use crate::task_graph::NodeAccess;
        use crate::types::{BufferFlags, BufferKind};

        let device = Arc::new(create_test_device());
        let ctx = device.create_context().unwrap();
        let window = MockWindow::new(800, 600);
        let surface = Surface::new(&ctx, &window).unwrap();

        let mut pool = RetainedPool::new(device.clone());
        let parcel = pool
            .acquire_buffer(256, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();

        let shader = ShaderModule::from_slang(&device, "void main() {}").unwrap();
        let pipeline = ComputePipeline::new(&device, &shader).unwrap();

        let mut graph = TaskGraph::new();
        let sc = graph.declare_swapchain_output();
        graph
            .node("fine", &pipeline)
            .with_parcel(&parcel, NodeAccess::ReadWrite)
            .with_swapchain_output(sc, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let acquired = surface.begin().unwrap();
        let mut frame = surface.submit_graph_to_frame(&mut graph, acquired).unwrap();
        let tv = frame.submit_frame().unwrap();
        assert_eq!(parcel.last_referenced_on(ctx.backend_handle()), Some(tv));
        let _ = frame.present();
    }
}
