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
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

/// A GPU surface for zero-copy presentation to a window.
pub struct Surface {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    device_handle: crate::backend::DeviceHandle,
    handle: SurfaceHandle,
    width: u32,
    height: u32,
    present_thread: Arc<PresentThread>,
}

/// A present operation that completed (or is completing) on a persistent worker thread.
///
/// Returned by [`Frame::present_async`]. Call [`PendingPresent::finish`] to wait for
/// the compositor hand-off and to obtain the frame's **submit** timeline value.
///
/// GPU compute for the frame is already submitted on the calling thread before this
/// is returned; `finish()` only waits for the optional background present worker.
pub struct PendingPresent {
    inner: PendingPresentInner,
}

enum PendingPresentInner {
    /// Present completes on the worker; recv reports errors only.
    Channel {
        rx: mpsc::Receiver<Result<()>>,
        submit_tv: TimelineValue,
    },
    /// Present already executed synchronously (fallback / already-presented guard).
    AlreadyDone(TimelineValue),
}

impl PendingPresent {
    /// Block until the present worker completes (if any) and return the submit timeline.
    ///
    /// The returned value is the timeline from [`Frame::submit_frame`] on the render
    /// thread, not a separate present-only signal.
    pub fn finish(self) -> Result<TimelineValue> {
        let _tz = tracy_zone!("pending_present.finish");
        match self.inner {
            PendingPresentInner::Channel { rx, submit_tv } => {
                let _rz = tracy_zone!("pending_present.recv");
                rx.recv()
                    .map_err(|_| {
                        anyhow::anyhow!("present worker thread dropped sender (panicked?)")
                    })??;
                Ok(submit_tv)
            }
            PendingPresentInner::AlreadyDone(tv) => Ok(tv),
        }
    }

    /// Returns `true` if the result is already available without blocking.
    ///
    /// Note: `std::sync::mpsc` does not support peeking without consuming, so
    /// this is conservative — it may return `false` even when the result is ready.
    /// Prefer calling `finish()` directly; with the persistent thread design the
    /// recv is typically < 1 µs regardless.
    pub fn is_done(&self) -> bool {
        match &self.inner {
            PendingPresentInner::Channel { .. } => false,
            PendingPresentInner::AlreadyDone(_) => true,
        }
    }
}

// -----------------------------------------------------------------------
// PresentThread — persistent worker that parks between frames
// -----------------------------------------------------------------------

type PresentWork = (
    Arc<Mutex<Box<dyn GpuBackend>>>,
    FrameToken,
    TimelineValue,
    Device,
    DeferredPayload,
    mpsc::Sender<Result<()>>,
);

/// A persistent thread dedicated to OS present calls.
///
/// Lives on `Surface` and is reused across frames.  The thread parks (sleeps)
/// between frames and wakes on `mpsc::recv` — zero allocation, zero thread
/// create/destroy overhead per frame.
pub(crate) struct PresentThread {
    tx: mpsc::Sender<PresentWork>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

impl PresentThread {
    /// Spawn the persistent present worker.
    pub(crate) fn new() -> Self {
        let (tx, rx) = mpsc::channel::<PresentWork>();
        let join_handle = std::thread::Builder::new()
            .name("goldy-present".into())
            .spawn(move || {
                Self::worker_loop(rx);
            })
            .expect("failed to spawn goldy-present thread");
        Self {
            tx,
            join_handle: Some(join_handle),
        }
    }

    fn worker_loop(rx: mpsc::Receiver<PresentWork>) {
        while let Ok((backend, token, submit_tv, device, keepalive, reply_tx)) = rx.recv() {
            let _tz = tracy_zone!("present_thread.do_present");
            let result = do_present_only(backend, token, submit_tv, device, keepalive);
            let _ = reply_tx.send(result);
        }
    }

    /// Offload compositor present to the worker. Compute must already be submitted.
    pub(crate) fn submit_present(
        &self,
        backend: Arc<Mutex<Box<dyn GpuBackend>>>,
        token: FrameToken,
        submit_tv: TimelineValue,
        device: Device,
        keepalive: DeferredPayload,
    ) -> PendingPresent {
        let (reply_tx, reply_rx) = mpsc::channel();
        if self
            .tx
            .send((
                backend.clone(),
                token,
                submit_tv,
                device.clone(),
                keepalive,
                reply_tx,
            ))
            .is_err()
        {
            tracing::warn!("present worker dead, falling back to synchronous present");
            let _ = do_present_only(backend, token, submit_tv, device, DeferredPayload::new());
            return PendingPresent {
                inner: PendingPresentInner::AlreadyDone(submit_tv),
            };
        }
        PendingPresent {
            inner: PendingPresentInner::Channel {
                rx: reply_rx,
                submit_tv,
            },
        }
    }
}

impl Drop for PresentThread {
    fn drop(&mut self) {
        // Drop the sender to signal the worker to exit, then join.
        drop(std::mem::replace(&mut self.tx, mpsc::channel().0));
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
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
    /// Timeline returned by [`GpuBackend::submit_frame`] for this bracket.
    submit_tv: Option<TimelineValue>,
    /// Resources (e.g. transient textures) that must outlive the frame's GPU work.
    /// Deferred to the VramAllocator ring at present time. Uses a Mutex so
    /// submit_compute can push to it via &self without requiring &mut Frame.
    keepalive: Mutex<DeferredPayload>,
    /// Shared reference to the persistent present worker thread.
    present_thread: Arc<PresentThread>,
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
            present_thread: Arc::new(PresentThread::new()),
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
            submit_tv: None,
            keepalive: Mutex::new(DeferredPayload::new()),
            present_thread: Arc::clone(&self.present_thread),
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

        // ── Step 3: emit early commands from the original IR + resolver ──────
        let early_cmds = {
            let _tz = tracy_zone!("surface.submit_graph.emit_early");
            crate::task_graph::analysis::emit_waves_to_commands(
                graph.ir(),
                &schedule.waves[..split_wave],
                resolver.as_ref(),
            )
        };

        // ── Step 4: submit early commands ────────────────────────────────────
        if !early_cmds.is_empty() {
            let mut backend = self.backend.lock().unwrap();
            let _tz = tracy_zone!("surface.submit_partition_early");
            backend.submit_standalone(self.device_handle, &early_cmds)?;
        }

        // ── Step 5: deferred surface acquire ─────────────────────────────────
        let frame = {
            let _tz = tracy_zone!("surface.deferred_acquire");
            self.begin()?
        };

        // ── Step 6: add swapchain to the resolver ────────────────────────────
        let swapchain_tex = frame.texture();
        let sc_handle = swapchain_tex.handle;
        let uav_index = swapchain_tex
            .bindless_index()
            .context("swapchain texture has no UAV bindless index")?;

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

        Ok(frame)
    }

    /// Compile, auto-partition, and submit a graph that writes to an already-acquired frame.
    ///
    /// This preserves `SwapchainOutput` as an abstract graph resource while allowing callers to
    /// request the WSI image at the beginning of frame recording. Early partitions still run before
    /// the swapchain-touching partition; the late partition is resolved against `frame`.
    pub fn submit_graph_to_frame(&self, graph: &mut TaskGraph, frame: Frame) -> Result<Frame> {
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

        let early_cmds = {
            let _tz = tracy_zone!("surface.submit_graph_to_frame.emit_early");
            crate::task_graph::analysis::emit_waves_to_commands(
                graph.ir(),
                &schedule.waves[..split_wave],
                resolver.as_ref(),
            )
        };
        if !early_cmds.is_empty() {
            let mut backend = self.backend.lock().unwrap();
            let _tz = tracy_zone!("surface.submit_graph_to_frame.partition_early");
            backend.submit_standalone(self.device_handle, &early_cmds)?;
        }

        let swapchain_tex = frame.texture();
        let sc_handle = swapchain_tex.handle;
        let uav_index = swapchain_tex
            .bindless_index()
            .context("swapchain texture has no UAV bindless index")?;

        let mut full_resolver = resolver.unwrap_or_default();
        full_resolver.swapchain = Some(crate::task_graph::ResolvedSwapchain {
            handle: sc_handle,
            uav_index,
        });

        let final_cmds = crate::task_graph::analysis::emit_waves_to_commands(
            graph.ir(),
            &schedule.waves[split_wave..],
            Some(&full_resolver),
        );

        {
            let mut backend = self.backend.lock().unwrap();
            let _tz = tracy_zone!("surface.submit_graph_to_frame.partition_late");
            backend.record_gpu_work(&frame.token, &final_cmds)?;
        }

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
        use crate::device::Device;
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
        let mut heap_guard = self._device.inner.placement_heap.lock().unwrap();
        if heap_guard.is_none() {
            let cap = (256 * 1024 * 1024u64).max(alloc_size * Device::DEFAULT_PIPELINE_DEPTH);
            *heap_guard = Some(
                PlacementHeap::with_capacity(&self._device, cap)
                    .context("failed to create device placement heap")?,
            );
        }
        let heap = heap_guard.as_mut().unwrap();

        // ── Switch to paged mode FIRST (may invalidate_all; must happen before ──
        // ── texture resolution so caches are clean when textures are created)  ──
        let depth = Device::DEFAULT_PIPELINE_DEPTH as usize;
        if layout_opt.is_some() {
            let _tz = tracy_zone!("surface.build_resolver.configure_pages");
            heap.configure_pages(alloc_size, depth, &self._device)?;
        }

        // ── Resolve transient textures ───────────────────────────────────────────
        let tex_handles = {
            let _tz = tracy_zone!("surface.build_resolver.resolve_textures");
            graph.resolve_transient_textures_with_heap(&self._device, heap, node_waves)?
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
                let (uav, srv, _hit) = heap.get_or_create_view(
                    spec.id,
                    offset,
                    spec.size,
                    view_stride,
                    &self._device,
                )?;
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

    /// Submit recorded GPU work for this frame. Does not present.
    ///
    /// Safe to call once per frame before [`Self::present`] or [`Self::present_async`].
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
    /// Returns the **submit** timeline (compute completion), not a separate present signal.
    pub fn present(mut self) -> Result<TimelineValue> {
        self.do_present_sync()
    }

    fn do_present_sync(&mut self) -> Result<TimelineValue> {
        let _tz = tracy_zone!("frame.present");
        if self.presented {
            return Ok(self.submit_tv.unwrap_or_else(|| {
                let backend = self.backend.lock().unwrap();
                backend.gpu_progress(self.device_handle)
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
        if let Ok(mut heap_guard) = self._device.inner.placement_heap.lock() {
            if let Some(ref mut heap) = *heap_guard {
                heap.stamp_all_pending(submit_tv);
            }
        }
        let keepalive = std::mem::take(&mut *self.keepalive.lock().unwrap());
        if !keepalive.is_empty() {
            self._device.defer_release(submit_tv, keepalive);
        }
        Ok(())
    }

    /// Submit recorded work on this thread, then offload only the compositor present.
    ///
    /// Returns [`PendingPresent`] carrying the submit timeline immediately; call
    /// [`PendingPresent::finish`] at the start of the next frame to wait for present errors.
    pub fn present_async(mut self) -> Result<PendingPresent> {
        let _tz = tracy_zone!("frame.present_async");
        if self.presented {
            let submit_tv = self.submit_tv.unwrap_or_else(|| {
                let backend = self.backend.lock().unwrap();
                backend.gpu_progress(self.device_handle)
            });
            return Ok(PendingPresent {
                inner: PendingPresentInner::AlreadyDone(submit_tv),
            });
        }
        self.presented = true;
        let _ = self.texture.take();

        let submit_tv = self.submit_frame()?;
        self.apply_frame_bookkeeping(submit_tv)?;

        let backend = Arc::clone(&self.backend);
        let device = self._device.clone();
        let token = self.token;
        let pending = self.present_thread.submit_present(
            backend,
            token,
            submit_tv,
            device,
            DeferredPayload::new(),
        );
        Ok(pending)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Compositor present only (compute already submitted). Used by the present worker.
fn do_present_only(
    backend: Arc<Mutex<Box<dyn crate::backend::GpuBackend>>>,
    token: crate::backend::FrameToken,
    submit_tv: TimelineValue,
    _device: Device,
    _keepalive: crate::vram_allocator::DeferredPayload,
) -> Result<()> {
    let _tz = tracy_zone!("frame.do_present_only");
    let mut b = backend.lock().unwrap();
    b.present_frame(token, submit_tv)?;
    Ok(())
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
