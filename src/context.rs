//! Submission/timeline context bound to a [`Device`].
//!
//! A [`Context`] holds an `Arc` clone of the device substrate so the device
//! outlives every context. Submit, wait, signal, and reclamation APIs live here.

use crate::backend::ContextHandle;
use crate::device::Device;
use crate::error::GoldyError;
use crate::task_graph::TaskGraph;
use crate::timeline::TimelineValue;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// GPU submission/timeline context for a single device.
///
/// Clone is cheap (`Arc` bump). Multiple contexts may be created per device;
/// in the current backend they alias the device's single submission stream.
pub struct Context {
    pub(crate) inner: Arc<ContextInner>,
}

pub(crate) struct ContextInner {
    device: Device,
    handle: ContextHandle,
    high_water_timeline: AtomicU64,
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
        Ok(Self {
            inner: Arc::new(ContextInner {
                device,
                handle,
                high_water_timeline: AtomicU64::new(0),
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

    fn classify(&self, e: anyhow::Error) -> GoldyError {
        if self.device().is_device_lost() {
            return GoldyError::DeviceLost;
        }
        GoldyError::Backend(e)
    }

    /// Latest GPU completion counter on this context's timeline.
    pub fn gpu_progress(&self) -> TimelineValue {
        let _tz = crate::tracy_zone!("context.gpu_progress");
        let backend = {
            let _lock = crate::tracy_zone!("context.gpu_progress.lock");
            self.inner.device.inner.backend.lock().unwrap()
        };
        let _query = crate::tracy_zone!("context.gpu_progress.query");
        backend.gpu_progress(self.inner.handle)
    }

    /// Block until the timeline reaches at least `value`.
    pub fn wait_until(&self, value: TimelineValue) -> Result<(), GoldyError> {
        let _tz = crate::tracy_zone!("context.wait_until");
        let mut backend = {
            let _lock = crate::tracy_zone!("context.wait_until.lock");
            self.inner.device.inner.backend.lock().unwrap()
        };
        let _backend = crate::tracy_zone!("context.wait_until.backend");
        backend.wait_until(self.inner.handle, value).map_err(|e| {
            drop(backend);
            self.classify(e)
        })
    }

    /// Like [`wait_until`](Self::wait_until) but returns `Err(`[`GoldyError::SubmitTimeout`]`)` on timeout.
    pub fn wait_until_timeout(
        &self,
        value: TimelineValue,
        timeout_ms: u32,
    ) -> Result<(), GoldyError> {
        let mut backend = self.inner.device.inner.backend.lock().unwrap();
        match backend.wait_until_timeout(self.inner.handle, value, timeout_ms) {
            Ok(true) => Ok(()),
            Ok(false) => Err(GoldyError::SubmitTimeout),
            Err(e) => {
                drop(backend);
                Err(self.classify(e))
            }
        }
    }

    /// Oldest timeline ticket not yet retired by the GPU, if work is still in flight.
    pub fn peek_oldest_in_flight(&self) -> Option<TimelineValue> {
        let backend = self.inner.device.inner.backend.lock().unwrap();
        backend.peek_oldest_in_flight(self.inner.handle)
    }

    /// The largest [`TimelineValue`] ever returned by [`submit`](Self::submit) on this context.
    pub fn high_water_timeline(&self) -> TimelineValue {
        self.inner.high_water_timeline.load(Ordering::Relaxed)
    }

    /// Drain pending backend signals (GPU completion, swapchain, oversubscribed).
    pub fn poll_signals(&self) -> Vec<crate::signal::Signal> {
        let mut backend = self.inner.device.inner.backend.lock().unwrap();
        backend.poll_signals(self.inner.handle)
    }

    /// Drain pending signals and service [`crate::signal::Signal::BoundaryCrossed`].
    pub fn poll_signals_and_service(&self) -> Vec<crate::signal::Signal> {
        let _tz = crate::tracy_zone!("context.poll_signals_and_service");
        let signals = self.poll_signals();
        let latest_boundary = signals.iter().fold(None, |latest, signal| match signal {
            crate::signal::Signal::BoundaryCrossed { epoch } => {
                Some(latest.unwrap_or(0).max(*epoch))
            }
            _ => latest,
        });
        if let Some(epoch) = latest_boundary {
            self.boundary_crossed(epoch);
        }
        signals
    }

    /// Process deferred GPU deletions and reclaim VRAM-ring payloads whose epoch has retired.
    pub fn boundary_crossed(&self, epoch: TimelineValue) {
        let _tz = crate::tracy_zone!("context.boundary_crossed");
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.flush_pre");
            let mut backend = self.inner.device.inner.backend.lock().unwrap();
            backend.flush_deferred_deletions(self.inner.handle);
        }
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.reclaim");
            let mut backend = self.inner.device.inner.backend.lock().unwrap();
            backend.set_reclamation_context(self.inner.handle, Some(epoch));
            drop(backend);
            self.device().vram_allocator().boundary_crossed(epoch);
            if let Ok(mut heap_guard) = self.inner.device.inner.placement_heap.lock() {
                if let Some(heap) = heap_guard.as_mut() {
                    heap.reclaim(epoch);
                }
            }
        }
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.flush_post");
            let mut backend = self.inner.device.inner.backend.lock().unwrap();
            backend.set_reclamation_context(self.inner.handle, None);
            backend.flush_deferred_deletions(self.inner.handle);
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

    pub fn defer_release(
        &self,
        epoch: TimelineValue,
        payload: crate::vram_allocator::DeferredPayload,
    ) {
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
            let mut backend = self.inner.device.inner.backend.lock().unwrap();
            let tv = graph
                .submit_with_backend(self, backend.as_mut(), None, &HashMap::new(), true)
                .map_err(|e| {
                    drop(backend);
                    self.classify(e)
                })?;
            self.inner
                .high_water_timeline
                .fetch_max(tv, Ordering::Relaxed);
            return Ok(tv);
        }

        let tv = self
            .submit_with_placement_heap(graph, true)
            .map_err(|e| self.classify(e))?;
        self.inner
            .high_water_timeline
            .fetch_max(tv, Ordering::Relaxed);
        Ok(tv)
    }

    /// Like [`submit`](Self::submit) but does not block on transient GPU completion.
    pub fn submit_pipelined(&self, graph: &mut TaskGraph) -> Result<TimelineValue, GoldyError> {
        let _tz = crate::tracy_zone!("context.submit_pipelined");
        if !graph.has_transient_resources() {
            let mut backend = self.inner.device.inner.backend.lock().unwrap();
            let tv = graph
                .submit_with_backend(self, backend.as_mut(), None, &HashMap::new(), false)
                .map_err(|e| {
                    drop(backend);
                    self.classify(e)
                })?;
            self.inner
                .high_water_timeline
                .fetch_max(tv, Ordering::Relaxed);
            return Ok(tv);
        }

        let tv = self
            .submit_with_placement_heap(graph, false)
            .map_err(|e| self.classify(e))?;
        self.inner
            .high_water_timeline
            .fetch_max(tv, Ordering::Relaxed);
        Ok(tv)
    }

    pub fn submit_pipelined_and_retain(
        &self,
        graph: &mut TaskGraph,
    ) -> Result<TimelineValue, GoldyError> {
        if graph.has_transient_resources() {
            return self.submit_pipelined(graph);
        }
        let mut backend = self.inner.device.inner.backend.lock().unwrap();
        let tv = graph
            .submit_with_backend_and_retain(self, backend.as_mut())
            .map_err(|e| {
                drop(backend);
                self.classify(e)
            })?;
        self.inner
            .high_water_timeline
            .fetch_max(tv, Ordering::Relaxed);
        Ok(tv)
    }

    pub fn try_resubmit_retained(&self, key: u64) -> Result<Option<TimelineValue>, GoldyError> {
        let mut backend = self.inner.device.inner.backend.lock().unwrap();
        let result = backend
            .try_resubmit_retained(self.inner.handle, key)
            .map_err(|e| {
                drop(backend);
                self.classify(e)
            })?;
        if let Some(tv) = result {
            self.inner
                .high_water_timeline
                .fetch_max(tv, Ordering::Relaxed);
        }
        Ok(result)
    }

    pub fn dispatch(&self, graph: &mut TaskGraph) -> Result<(), GoldyError> {
        let v = self.submit(graph)?;
        self.wait_until(v)
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
        let node_waves =
            crate::task_graph::analysis::node_to_wave_map(&schedule, graph.ir().nodes.len());

        let has_buffers = graph.has_transient_buffers();

        let (alloc_size, base_align, layout_opt) = if has_buffers {
            let (ts, ba, lay) = graph.transient_heap_size_and_layout(&node_waves)?;
            let sz = (ts + ba - 1).max(256);
            (sz, ba, Some(lay))
        } else {
            (0u64, 1u64, None)
        };

        let mut heap_guard = device.inner.placement_heap.lock().unwrap();

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

        let tex_handles = graph.resolve_transient_textures_with_heap(device, heap, &node_waves)?;
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
                let (uav, srv, _hit) =
                    heap.get_or_create_view(spec.id, offset, spec.size, view_stride, device)?;

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

        drop(heap_guard);

        let mut backend = device.inner.backend.lock().unwrap();
        let tv = graph.submit_ir_with_resolver(
            self,
            backend.as_mut(),
            &resolver,
            wait_for_transient_completion,
        )?;
        drop(backend);

        let mut heap_guard = device.inner.placement_heap.lock().unwrap();
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
        assert!(
            weak.upgrade().is_none(),
            "device drop should drain deferred resources"
        );
    }
}
