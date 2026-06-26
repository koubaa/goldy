//! Submission/timeline context bound to a [`Device`].
//!
//! A [`Context`] holds an `Arc` clone of the device substrate so the device
//! outlives every context. Submit, wait, signal, and reclamation APIs live here.

use crate::backend::ContextHandle;
use crate::device::Device;
use crate::error::GoldyError;
use crate::parcel::BytesByKind;
use crate::task_graph::TaskGraph;
use crate::timeline::{is_ready, ReferenceTable, TimelineValue};
use crate::transient_pool::TransientPool;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// GPU submission/timeline context for a single device.
///
/// Clone is cheap (`Arc` bump). Multiple contexts may be created per device; each
/// owns its own submission timeline (semaphore/fence/event, signal queue, and on
/// Vulkan/DX12 a fence poller). [`Device`] substrate (deletion queue, VRAM ring,
/// placement heap) stays device-scoped.
pub struct Context {
    pub(crate) inner: Arc<ContextInner>,
}

pub(crate) struct ContextInner {
    device: Device,
    handle: ContextHandle,
    timeline_reader: Option<Arc<dyn crate::backend::ContextTimelineReader>>,
    deletion_flush: Arc<dyn crate::backend::ContextDeferredDeletionFlush>,
    reclamation_scope: Arc<dyn crate::backend::ContextReclamationScope>,
    submit_session: Arc<dyn crate::backend::ContextSubmitSession>,
    high_water_timeline: AtomicU64,
    /// Per-context transient resource heap (backing buffer + cached views/textures).
    ///
    /// Scoped to the context rather than the device so that independent contexts —
    /// e.g. concurrent renders that each own a context — never share transient
    /// `BufferView`/`Texture` handles. Sharing a device-global heap across contexts
    /// caused GPU-level write-write races on the cached transient textures.
    pub(crate) placement_heap: Mutex<Option<crate::placement_heap::PlacementHeap>>,
    /// Epoch-gated transient parcel pool backing scheme-held leases.
    transient_pool: Mutex<TransientPool>,
}

impl Clone for Context {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context").finish_non_exhaustive()
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        // Drop the placement heap (and its views/buffer/textures) while the device is
        // still alive (this `ContextInner` still holds a `Device` clone). The heap's
        // resource `Drop`s route through the backend deletion queue.
        if let Ok(mut heap_guard) = self.placement_heap.lock() {
            *heap_guard = None;
        }
        // Drop the transient pool (and its parked parcels) while the device is alive.
        if let Ok(mut pool_guard) = self.transient_pool.lock() {
            *pool_guard = TransientPool::new();
        }
        // Release the cloned submission-context handle before backend destroy.
        // Backends expect sole ownership of the per-context Arc at teardown.
        self.timeline_reader.take();
        self.device.unregister_context_timeline_reader(self.handle);
        // Runs while `Context` still holds `Arc<Device>`; joins per-context pollers
        // (Vulkan/DX12) before [`DeviceInner::drop`] calls `device_wait_idle`.
        let mut backend = self.device.inner.backend.lock().unwrap();
        backend.destroy_context(self.handle);
    }
}

impl Context {
    pub(crate) fn new(device: Device) -> Result<Self, GoldyError> {
        let handle = {
            let mut backend = device.inner.backend.lock().unwrap();
            backend
                .create_context(device.inner.handle)
                .map_err(GoldyError::Backend)?
        };
        let timeline_reader = {
            let backend = device.inner.backend.lock().unwrap();
            backend
                .clone_context_timeline_reader(handle)
                .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("missing context timeline reader").into()))?
        };
        device.register_context_timeline_reader(handle, Arc::clone(&timeline_reader));
        let (deletion_flush, reclamation_scope) = {
            let backend = device.inner.backend.lock().unwrap();
            let deletion_flush = backend
                .clone_context_deletion_flush(handle, Arc::clone(&device.inner.context_readers))
                .ok_or_else(|| {
                    GoldyError::Backend(anyhow::anyhow!("missing context deletion flush").into())
                })?;
            let reclamation_scope = backend.clone_context_reclamation_scope(handle);
            (deletion_flush, reclamation_scope)
        };
        let submit_session = crate::backend::LockedSubmitSession::new(
            Arc::clone(&device.inner.backend),
            handle,
        );
        Ok(Self {
            inner: Arc::new(ContextInner {
                device,
                handle,
                timeline_reader: Some(timeline_reader),
                deletion_flush,
                reclamation_scope,
                submit_session,
                high_water_timeline: AtomicU64::new(0),
                placement_heap: Mutex::new(None),
                transient_pool: Mutex::new(TransientPool::new()),
            }),
        })
    }

    /// The device this context is bound to.
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    pub(crate) fn backend_handle(&self) -> ContextHandle {
        self.inner.handle
    }

    pub(crate) fn submit_session(&self) -> &dyn crate::backend::ContextSubmitSession {
        self.inner.submit_session.as_ref()
    }

    /// Test-only access to the backend context id.
    #[doc(hidden)]
    pub fn test_backend_handle(&self) -> ContextHandle {
        self.backend_handle()
    }

    /// Run `f` with exclusive access to this context's transient parcel pool.
    pub(crate) fn with_transient_pool<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut TransientPool) -> R,
    {
        let mut pool = self.inner.transient_pool.lock().unwrap();
        f(&mut pool)
    }

    /// Bytes held outside this context's transient pool (leased or otherwise acquired).
    ///
    /// Aggregate memory telemetry for debug checking and tracing
    /// Not a synchronization primitive — use [`crate::Parcel::is_settled`] for parcel currency.
    /// The pool's internal recycle bins and pending counts are never exposed.
    pub fn transient_outstanding_bytes(&self) -> BytesByKind {
        self.with_transient_pool(|pool| pool.outstanding_bytes())
    }

    /// Total number of fresh GPU buffer allocations made by this context's transient pool.
    ///
    /// Does not increment when a retired bin entry is reused. Monotonically increasing.
    /// Useful in tests to assert that the pool's recycling path fires (alloc count stays
    /// flat across a lease reuse cycle, mirroring [`Self::transient_texture_create_count`]).
    pub fn transient_buffer_alloc_count(&self) -> usize {
        self.with_transient_pool(|pool| pool.buffer_alloc_count())
    }

    pub(crate) fn classify(&self, e: anyhow::Error) -> GoldyError {
        if self.device().is_device_lost() {
            return GoldyError::DeviceLost;
        }
        GoldyError::Backend(e)
    }

    /// Latest GPU completion counter on this context's timeline.
    pub fn gpu_progress(&self) -> TimelineValue {
        let _tz = crate::tracy_zone!("context.gpu_progress");
        let _query = crate::tracy_zone!("context.gpu_progress.query");
        self.inner
            .timeline_reader
            .as_ref()
            .expect("timeline reader")
            .gpu_progress()
    }

    /// Block until the timeline reaches at least `value`.
    pub fn wait_until(&self, value: TimelineValue) -> Result<(), GoldyError> {
        self.wait_until_context(self.inner.handle, value)
    }

    fn wait_until_context(&self, ctx: ContextHandle, value: TimelineValue) -> Result<(), GoldyError> {
        let _tz = crate::tracy_zone!("context.wait_until");
        let already_complete = if ctx == self.inner.handle {
            self.gpu_progress() >= value
        } else {
            self.inner
                .device
                .context_gpu_progress(ctx)
                .is_some_and(|p| p >= value)
        };
        let backend_mutex = &self.inner.device.inner.backend;
        if !already_complete {
            let blocking = {
                let _lock = crate::tracy_zone!("context.wait_until.lock");
                let backend = backend_mutex.lock().unwrap();
                let _prepare = crate::tracy_zone!("context.wait_until.prepare");
                backend
                    .take_timeline_blocking_wait(ctx, value)
                    .map_err(|e| self.classify(e))?
            };
            if let Some(wait) = blocking {
                let _block = crate::tracy_zone!("context.wait_until.block");
                wait.block().map_err(|e| self.classify(e))?;
            }
        }
        {
            let _lock = crate::tracy_zone!("context.wait_until.lock");
            let mut backend = backend_mutex.lock().unwrap();
            let _finish = crate::tracy_zone!("context.wait_until.finish");
            backend.finish_timeline_wait(ctx, value).map_err(|e| {
                drop(backend);
                self.classify(e)
            })
        }
    }

    /// Like [`wait_until`](Self::wait_until) but returns `Err(`[`GoldyError::SubmitTimeout`]`)` on timeout.
    pub fn wait_until_timeout(&self, value: TimelineValue, timeout_ms: u32) -> Result<(), GoldyError> {
        let ctx = self.inner.handle;
        let already_complete = self.gpu_progress() >= value;
        let backend_mutex = &self.inner.device.inner.backend;
        if !already_complete {
            let blocking = {
                let _lock = crate::tracy_zone!("context.wait_until.lock");
                let backend = backend_mutex.lock().unwrap();
                backend
                    .take_timeline_blocking_wait(ctx, value)
                    .map_err(|e| self.classify(e))?
            };
            if let Some(wait) = blocking {
                let _block = crate::tracy_zone!("context.wait_until.block");
                if !wait.block_timeout(timeout_ms).map_err(|e| self.classify(e))? {
                    return Err(GoldyError::SubmitTimeout);
                }
            }
        }
        {
            let _lock = crate::tracy_zone!("context.wait_until.lock");
            let mut backend = backend_mutex.lock().unwrap();
            backend.finish_timeline_wait(ctx, value).map_err(|e| {
                drop(backend);
                self.classify(e)
            })
        }
    }

    /// Oldest timeline ticket not yet retired by the GPU, if work is still in flight.
    pub fn peek_oldest_in_flight(&self) -> Option<TimelineValue> {
        self.inner
            .timeline_reader
            .as_ref()
            .expect("timeline reader")
            .peek_oldest_in_flight()
    }

    /// The largest [`TimelineValue`] ever returned by [`submit`](Self::submit) on this context.
    pub fn high_water_timeline(&self) -> TimelineValue {
        self.inner.high_water_timeline.load(Ordering::Relaxed)
    }

    pub(crate) fn advance_high_water_timeline(&self, tv: TimelineValue) {
        self.inner.high_water_timeline.fetch_max(tv, Ordering::Relaxed);
    }

    /// Drain pending backend signals (GPU completion, swapchain, oversubscribed).
    pub fn poll_signals(&self) -> Vec<crate::signal::Signal> {
        let progress = self.gpu_progress();
        let mut backend = self.inner.device.inner.backend.lock().unwrap();
        backend.poll_signals(self.inner.handle, progress)
    }

    /// Drain pending signals and service [`crate::signal::Signal::BoundaryCrossed`].
    pub fn poll_signals_and_service(&self) -> Vec<crate::signal::Signal> {
        let _tz = crate::tracy_zone!("context.poll_signals_and_service");
        let signals = self.poll_signals();
        let latest_boundary = signals.iter().fold(None, |latest, signal| match signal {
            crate::signal::Signal::BoundaryCrossed { epoch } => Some(latest.unwrap_or(0).max(*epoch)),
            _ => latest,
        });
        if let Some(epoch) = latest_boundary {
            self.boundary_crossed(epoch);
        }
        signals
    }

    /// Process deferred GPU deletions and reclaim VRAM-ring payloads whose epoch has retired.
    ///
    /// The device-installed deferred VRAM ring ([`Device::vram_allocator`]) is drained
    /// against [`Device::timeline_retired`] (max completed over all live contexts). Any
    /// context may call this after a `BoundaryCrossed` signal; defer/release epochs are
    /// device-global submission sequence values, so `device_retired >= epoch` proves the GPU
    /// work is done regardless of which context originally submitted the payload.
    ///
    /// The placement-heap ring is reclaimed against the signal `epoch` itself rather than
    /// `device_retired`. Placement-heap regions are stamped with the exact submission epoch
    /// that guards their contents; advancing past that epoch is sufficient to reclaim them,
    /// and using `device_retired` would over-reclaim regions whose guard epoch has not yet
    /// completed on the submitting context.
    ///
    /// Per-handle last-touch reclamation (tighter than `device_retired` for the VRAM ring)
    /// is a future optimization.
    pub fn boundary_crossed(&self, epoch: TimelineValue) {
        let _tz = crate::tracy_zone!("context.boundary_crossed");
        let retired = self.device().timeline_retired();
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.flush_pre");
            self.inner.deletion_flush.flush(retired);
            let _tz = crate::tracy_zone!("context.boundary_crossed.reclaim");
            self.inner.reclamation_scope.set_epoch(Some(epoch));
        }
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.drain_vram");
            self.device().vram_allocator().boundary_crossed(retired);
        }
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.drain_transient_pool");
            // `RetainedPool::release` parks parcels here for epoch-gated reuse (leases,
            // future scheme-held transients). Until ekrano migrates off its own VRAM
            // machinery (`ResourcePool`, `DeferredPayload` returns, pipeline cache) and
            // acquires through the transient pool, those parked buffers are not re-issued
            // — only dropped once `ready_after` retires. Without this drain at every frame
            // boundary, `release` leaks GPU heap (velato: Metal buffer heaps exhausted).
            self.with_transient_pool(|pool| pool.drain_ready(self));
        }
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.flush_post");
            self.inner.reclamation_scope.set_epoch(None);
            self.inner.deletion_flush.flush(retired);
        }
    }

    /// Pull-side reclamation using [`gpu_progress`](Self::gpu_progress).
    pub fn flush_deferred_deletions(&self) {
        let _tz = crate::tracy_zone!("context.flush_deferred_deletions");
        self.boundary_crossed(self.gpu_progress());
    }

    pub fn has_deferred_payloads(&self) -> bool {
        self.device().vram_allocator().has_deferred_payloads()
    }

    pub fn oldest_deferred_epoch(&self) -> Option<TimelineValue> {
        self.device().vram_allocator().oldest_deferred_epoch()
    }

    pub fn defer_release(&self, epoch: TimelineValue, payload: crate::vram_allocator::DeferredPayload) {
        self.device().vram_allocator().defer_release(epoch, payload);
    }

    pub fn defer_until<T: Send + 'static>(&self, epoch: TimelineValue, resource: T) {
        let mut payload = crate::vram_allocator::DeferredPayload::new();
        payload.push(resource);
        self.device().vram_allocator().defer_release(epoch, payload);
    }

    #[doc(hidden)]
    pub fn deferred_deletion_pending_count(&self) -> usize {
        let backend = self.inner.device.inner.backend.lock().unwrap();
        backend.deferred_deletion_pending_count(self.inner.handle)
    }

    #[doc(hidden)]
    pub fn in_flight_command_buffer_count(&self) -> usize {
        let backend = self.inner.device.inner.backend.lock().unwrap();
        backend.in_flight_command_buffer_count(self.inner.handle)
    }

    /// Submit a compiled [`TaskGraph`] on this context's timeline.
    pub fn submit(&self, graph: &mut TaskGraph) -> Result<TimelineValue, GoldyError> {
        if !graph.has_transient_resources() {
            let tv = graph
                .submit_with_backend(self, self.submit_session(), None, &HashMap::new(), true)
                .map_err(|e| self.classify(e))?;
            graph.apply_reference_stamps(self.backend_handle(), &self.inner.device.inner, tv);
            self.inner.high_water_timeline.fetch_max(tv, Ordering::Relaxed);
            return Ok(tv);
        }

        let tv = self
            .submit_with_placement_heap(graph, true)
            .map_err(|e| self.classify(e))?;
        graph.apply_reference_stamps(self.backend_handle(), &self.inner.device.inner, tv);
        self.inner.high_water_timeline.fetch_max(tv, Ordering::Relaxed);
        Ok(tv)
    }

    /// Like [`submit`](Self::submit) but does not block on transient GPU completion.
    pub fn submit_pipelined(&self, graph: &mut TaskGraph) -> Result<TimelineValue, GoldyError> {
        let _tz = crate::tracy_zone!("context.submit_pipelined");
        if !graph.has_transient_resources() {
            let tv = graph
                .submit_with_backend(self, self.submit_session(), None, &HashMap::new(), false)
                .map_err(|e| self.classify(e))?;
            graph.apply_reference_stamps(self.backend_handle(), &self.inner.device.inner, tv);
            self.inner.high_water_timeline.fetch_max(tv, Ordering::Relaxed);
            return Ok(tv);
        }

        let tv = self
            .submit_with_placement_heap(graph, false)
            .map_err(|e| self.classify(e))?;
        graph.apply_reference_stamps(self.backend_handle(), &self.inner.device.inner, tv);
        self.inner.high_water_timeline.fetch_max(tv, Ordering::Relaxed);
        Ok(tv)
    }

    pub fn submit_pipelined_and_retain(&self, graph: &mut TaskGraph) -> Result<TimelineValue, GoldyError> {
        if graph.has_transient_resources() {
            return self.submit_pipelined(graph);
        }
        let tv = graph
            .submit_with_backend_and_retain(self, self.submit_session())
            .map_err(|e| self.classify(e))?;
        graph.apply_reference_stamps(self.backend_handle(), &self.inner.device.inner, tv);
        self.inner.high_water_timeline.fetch_max(tv, Ordering::Relaxed);
        Ok(tv)
    }

    /// True when every context in `refs` has retired the stamped timeline values.
    ///
    /// Prefer [`crate::Parcel::is_settled`] when checking a single parcel the caller holds.
    pub(crate) fn parcel_ready(&self, refs: &ReferenceTable) -> bool {
        if refs.is_empty() {
            return true;
        }
        let device = &self.inner.device;
        let mut progress = HashMap::with_capacity(refs.len());
        for &ctx in refs.keys() {
            let p = device
                .context_gpu_progress(ctx)
                .unwrap_or_else(|| device.timeline_retired());
            progress.insert(ctx, p);
        }
        is_ready(refs, &progress)
    }

    /// Block until every context in `refs` has retired the stamped timeline values.
    pub fn wait_until_parcel_ready(&self, refs: &ReferenceTable) -> Result<(), GoldyError> {
        for (&ctx_handle, &tv) in refs {
            self.wait_until_context(ctx_handle, tv)?;
        }
        Ok(())
    }

    pub fn try_resubmit_retained(&self, key: u64) -> Result<Option<TimelineValue>, GoldyError> {
        if crate::validation_env::retained_cb_reuse_disabled() {
            return Ok(None);
        }
        let mut backend = self.inner.device.inner.backend.lock().unwrap();
        let result = backend
            .try_resubmit_retained(self.inner.handle, key, None)
            .map_err(|e| {
                drop(backend);
                self.classify(e)
            })?;
        if let Some(tv) = result {
            self.inner.high_water_timeline.fetch_max(tv, Ordering::Relaxed);
        }
        Ok(result)
    }

    pub fn dispatch(&self, graph: &mut TaskGraph) -> Result<(), GoldyError> {
        let v = self.submit(graph)?;
        self.wait_until(v)
    }

    /// Snapshot of this context's placement-heap state for diagnostics.
    ///
    /// Returns `None` if the heap hasn't been created yet (no transient-resource
    /// graphs have been submitted on this context).
    pub fn placement_heap_stats(&self) -> Option<crate::placement_heap::PlacementHeapStats> {
        let heap_guard = self.inner.placement_heap.lock().unwrap();
        heap_guard.as_ref().map(|h| h.stats())
    }

    /// Number of `BufferView`s and `Texture`s currently held in this context's placement
    /// heap caches. Returns `(cached_views, cached_textures)`.
    pub fn transient_cache_counts(&self) -> (usize, usize) {
        let heap_guard = self.inner.placement_heap.lock().unwrap();
        match heap_guard.as_ref() {
            Some(h) => (h.cached_view_count(), h.cached_texture_count()),
            None => (0, 0),
        }
    }

    /// Total number of `create_buffer_view` backend calls made by this context's placement
    /// heap view cache since initialization. Monotonically increasing.
    pub fn transient_view_create_count(&self) -> usize {
        let heap_guard = self.inner.placement_heap.lock().unwrap();
        heap_guard.as_ref().map(|h| h.view_create_count()).unwrap_or(0)
    }

    /// Total number of `Texture::new` calls made by this context's placement heap texture
    /// cache since initialization. Monotonically increasing.
    pub fn transient_texture_create_count(&self) -> usize {
        let heap_guard = self.inner.placement_heap.lock().unwrap();
        heap_guard.as_ref().map(|h| h.texture_create_count()).unwrap_or(0)
    }

    pub(crate) const DEFAULT_PIPELINE_DEPTH: u64 = 4;

    fn submit_with_placement_heap(
        &self,
        graph: &mut TaskGraph,
        wait_for_transient_completion: bool,
    ) -> anyhow::Result<TimelineValue> {
        use crate::task_graph::{ResolvedTransientBuffer, ResolvedTransientTexture, SlotResolver};
        use anyhow::Context;

        let device = self.device();
        let (schedule, _) = graph.schedule_and_split_wave();
        let node_waves = crate::task_graph::analysis::node_to_wave_map(&schedule, graph.ir().nodes.len());

        let has_buffers = graph.has_transient_buffers();

        let (alloc_size, base_align, layout_opt) = if has_buffers {
            let (ts, ba, lay) = graph.transient_heap_size_and_layout(&node_waves)?;
            let sz = (ts + ba - 1).max(256);
            (sz, ba, Some(lay))
        } else {
            (0u64, 1u64, None)
        };

        // Query GPU-retired timeline BEFORE locking the heap so we can use it
        // inside get_or_create_textures without acquiring the backend lock a
        // second time while the heap guard is held.
        let retired_timeline = device.timeline_retired();

        let mut heap_guard = self.inner.placement_heap.lock().unwrap();

        if heap_guard.is_none() {
            let cap = (256 * 1024 * 1024u64).max(alloc_size * Self::DEFAULT_PIPELINE_DEPTH);
            *heap_guard = Some(
                crate::placement_heap::PlacementHeap::with_capacity(device, cap)
                    .context("failed to create device placement heap")?,
            );
        }
        let heap = heap_guard.as_mut().unwrap();

        if has_buffers {
            let depth = Self::DEFAULT_PIPELINE_DEPTH as usize;
            heap.configure_pages(alloc_size, depth, device)?;
        }

        let mut resolver = SlotResolver::new();

        // Obtain the page slot BEFORE advance_page so that transient textures
        // are tied to the same slot as transient buffers.  Concurrent renders on
        // the same device get distinct page slots (0, 1, 2, 3, 0, …), so each
        // render writes to a different set of textures on the GPU.
        let tex_page_slot = heap.current_page_slot();
        let tex_handles =
            graph.resolve_transient_textures_with_heap(device, heap, &node_waves, tex_page_slot, retired_timeline)?;
        for (id, handle) in &tex_handles {
            resolver
                .textures
                .insert(*id, ResolvedTransientTexture { handle: *handle });
        }

        if has_buffers {
            let layout = layout_opt.unwrap();
            let raw_offset = heap.advance_page();
            let base_offset = raw_offset.div_ceil(base_align) * base_align;

            let buf_handle = heap.buffer().handle;
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

        // Submit while heap_guard is still held so that stamp_pending happens before
        // any other thread can observe last_timeline.  Releasing heap_guard before
        // stamping creates a window where a concurrent submit sees last_timeline = None
        // and evicts in-flight transient textures synchronously, causing GPU UAF.
        // Lock order is always heap → backend; no other code path reverses this.
        let tv = graph.submit_ir_with_resolver(
            self,
            self.submit_session(),
            &resolver,
            wait_for_transient_completion,
        )?;

        if let Some(heap) = heap_guard.as_mut() {
            heap.stamp_pending(tv);
        }

        Ok(tv)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::mock::MockBackend;
    use crate::device::Device;
    use crate::task_graph::TaskGraph;
    use std::sync::Arc;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn device_outlives_context() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        assert_eq!(Arc::strong_count(&device.inner), 2);
        drop(device);
        assert_eq!(ctx.gpu_progress(), 0);
        assert_eq!(Arc::strong_count(&ctx.device().inner), 1);
    }

    #[test]
    fn device_inner_dropped_only_after_context() {
        let device = test_device();
        let weak = Arc::downgrade(&device.inner);
        let ctx = device.create_context().unwrap();
        drop(device);
        assert!(weak.upgrade().is_some());
        drop(ctx);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn adapter_outlives_device() {
        let device = test_device();
        let adapter = device.adapter().clone();
        let weak = Arc::downgrade(&adapter.inner);
        drop(device);
        assert!(weak.upgrade().is_some());
    }

    #[test]
    fn context_wait_until_after_submit() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut graph = TaskGraph::new();
        let tv = ctx.submit(&mut graph).unwrap();
        ctx.wait_until(tv).unwrap();
        assert!(ctx.gpu_progress() >= tv);
    }

    #[test]
    fn high_water_timeline_starts_at_zero() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        assert_eq!(ctx.high_water_timeline(), 0);
    }

    #[test]
    fn high_water_timeline_advances_after_submit() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        assert_eq!(ctx.high_water_timeline(), 0);
        let mut graph = TaskGraph::new();
        let tv = ctx.submit(&mut graph).unwrap();
        assert!(tv > 0);
        assert_eq!(ctx.high_water_timeline(), tv);
        let tv2 = ctx.submit(&mut graph).unwrap();
        assert!(tv2 > tv);
        assert_eq!(ctx.high_water_timeline(), tv2);
    }

    #[test]
    fn defer_until_resources_are_not_dropped_before_epoch() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut graph = TaskGraph::new();
        let tv = ctx.submit(&mut graph).unwrap();

        let alive = std::sync::Arc::new(99u32);
        let weak = std::sync::Arc::downgrade(&alive);

        ctx.defer_until(tv + 100, alive);

        ctx.flush_deferred_deletions();
        assert!(
            weak.upgrade().is_some(),
            "resource should still be alive after flush at tv"
        );
    }

    #[test]
    fn defer_until_resources_dropped_after_flush_at_epoch() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut graph = TaskGraph::new();
        let tv = ctx.submit(&mut graph).unwrap();

        let alive = std::sync::Arc::new(42u32);
        let weak = std::sync::Arc::downgrade(&alive);

        ctx.defer_until(tv, alive);

        ctx.wait_until(tv).unwrap();
        ctx.flush_deferred_deletions();
        assert!(
            weak.upgrade().is_none(),
            "resource should be dropped after flush at epoch"
        );
    }

    #[test]
    fn defer_release_drops_all_on_device_drop() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut graph = TaskGraph::new();
        let tv = ctx.submit(&mut graph).unwrap();

        let alive = std::sync::Arc::new(7u32);
        let weak = std::sync::Arc::downgrade(&alive);

        ctx.defer_until(tv + 9999, alive);

        drop(ctx);
        drop(device);
        assert!(weak.upgrade().is_none(), "device drop should drain deferred resources");
    }

    #[test]
    fn independent_context_timelines() {
        let device = test_device();
        let ctx_a = device.create_context().unwrap();
        let ctx_b = device.create_context().unwrap();
        assert_eq!(ctx_a.gpu_progress(), 0);
        assert_eq!(ctx_b.gpu_progress(), 0);

        let mut graph = TaskGraph::new();
        let tv_a = ctx_a.submit(&mut graph).unwrap();
        assert!(tv_a > 0);
        assert_eq!(ctx_a.gpu_progress(), tv_a);
        assert_eq!(ctx_b.gpu_progress(), 0, "context B must not observe A's submit");
    }

    #[test]
    fn drop_one_context_leaves_other_usable() {
        let device = test_device();
        let ctx_a = device.create_context().unwrap();
        let ctx_b = device.create_context().unwrap();
        drop(ctx_a);

        let mut graph = TaskGraph::new();
        let tv = ctx_b.submit(&mut graph).unwrap();
        assert!(tv > 0);
        assert_eq!(ctx_b.gpu_progress(), tv);
    }

    /// Deferred payloads are reclaimed when `device_retired` passes their epoch, even if
    /// another context drives `boundary_crossed` with a lower per-context signal epoch.
    #[test]
    fn multi_context_deferred_reclamation_uses_device_retired() {
        let device = test_device();
        let ctx_a = device.create_context().unwrap();
        let ctx_b = device.create_context().unwrap();
        let mut graph = TaskGraph::new();

        let _tv1 = ctx_a.submit(&mut graph).unwrap();
        let _tv2 = ctx_b.submit(&mut graph).unwrap();
        let tv3 = ctx_a.submit(&mut graph).unwrap();
        assert_eq!(tv3, 3);

        let alive = std::sync::Arc::new(42u32);
        let weak = std::sync::Arc::downgrade(&alive);
        ctx_a.defer_until(tv3, alive);

        // Context B's latest boundary epoch is 2; per-context progress alone would not drain 3.
        assert_eq!(ctx_b.gpu_progress(), 2);
        assert!(device.timeline_retired() >= tv3);

        ctx_b.boundary_crossed(2);
        assert!(
            weak.upgrade().is_none(),
            "epoch 3 should reclaim when boundary_crossed uses device_retired (>= 3)"
        );
    }
}
