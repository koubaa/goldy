//! Submission/timeline context bound to a [`Device`].
//!
//! A [`Context`] holds an `Arc` clone of the device substrate so the device
//! outlives every context. Submit, wait, signal, and reclamation APIs live here.

use crate::backend::ContextHandle;
use crate::device::Device;
use crate::error::GoldyError;
use crate::parcel::BytesByKind;
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
    deletion_flush: Option<Arc<dyn crate::backend::ContextDeferredDeletionFlush>>,
    gpu_progress: Option<Arc<dyn crate::backend::ContextGpuProgress>>,
    reclamation_scope: Option<Arc<dyn crate::backend::ContextReclamationScope>>,
    submit_session: Option<Arc<dyn crate::backend::ContextSubmitSession>>,
    high_water_timeline: AtomicU64,
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
        // Drop the transient pool (and its parked parcels) while the device is alive.
        if let Ok(mut pool_guard) = self.transient_pool.lock() {
            *pool_guard = TransientPool::new();
        }
        // Release cloned per-context backend handles before teardown.
        // Backends expect sole ownership of the per-context Arc at teardown.
        self.deletion_flush.take();
        self.gpu_progress.take();
        self.reclamation_scope.take();
        self.submit_session.take();
        // Runs while `Context` still holds `Arc<Device>`; joins per-context pollers
        // (Vulkan/DX12) before [`DeviceInner::drop`] calls `device_wait_idle`.
        crate::backend::destroy_context(&self.device.inner.backend, self.handle);
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
        let (deletion_flush, reclamation_scope) = {
            let backend = device.inner.backend.lock().unwrap();
            let deletion_flush = backend
                .clone_context_deletion_flush(handle)
                .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("missing context deletion flush")))?;
            let reclamation_scope = backend.clone_context_reclamation_scope(handle);
            (deletion_flush, reclamation_scope)
        };
        let (submit_session, gpu_progress) = {
            let backend = device.inner.backend.lock().unwrap();
            (
                backend.clone_context_submit_session(handle, Arc::clone(&device.inner.backend)),
                backend.clone_context_gpu_progress(handle),
            )
        };
        Ok(Self {
            inner: Arc::new(ContextInner {
                device,
                handle,
                deletion_flush: Some(deletion_flush),
                gpu_progress,
                reclamation_scope: Some(reclamation_scope),
                submit_session: Some(submit_session),
                high_water_timeline: AtomicU64::new(0),
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
        self.inner.submit_session.as_ref().expect("submit session").as_ref()
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

    /// Acquire a one-submission texture from this context's transient pool.
    pub fn acquire_transient_texture(
        &self,
        width: u32,
        height: u32,
        format: crate::types::TextureFormat,
        access: crate::types::TextureKind,
        flags: crate::types::TextureFlags,
    ) -> anyhow::Result<crate::Texture> {
        self.with_transient_pool(|pool| pool.acquire_texture(self, width, height, format, access, flags))
    }

    /// Return a transient texture to this context's pool for epoch-gated reuse.
    pub fn return_transient_texture(&self, texture: crate::Texture) {
        let ready_after = texture.last_referenced();
        self.with_transient_pool(|pool| pool.return_texture(texture, ready_after));
    }

    /// Drop all parked transient textures (Metal resize purge).
    pub fn clear_transient_textures(&self) {
        self.with_transient_pool(|pool| pool.clear_textures());
    }

    /// Acquire a one-submission buffer from this context's transient pool.
    pub fn acquire_transient_buffer(
        &self,
        size: u64,
        kind: crate::types::BufferKind,
        flags: crate::types::BufferFlags,
        element_stride: Option<u32>,
    ) -> anyhow::Result<crate::parcel::Buffer> {
        self.with_transient_pool(|pool| pool.acquire_whole_buffer(self, size, kind, flags, element_stride))
    }

    /// Return a transient buffer to this context's pool for epoch-gated reuse.
    pub fn return_transient_buffer(&self, buf: crate::parcel::Buffer) {
        let ready_after = buf.last_referenced();
        match buf.into_transient_parcel() {
            Ok(parcel) => {
                self.with_transient_pool(|pool| pool.return_buffer_parcel(parcel, ready_after));
            }
            Err(e) => {
                tracing::warn!("return_transient_buffer: dropping non-binneable buffer: {e}");
            }
        }
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
    /// flat across a lease reuse cycle, mirroring [`Self::transient_texture_alloc_count`]).
    pub fn transient_buffer_alloc_count(&self) -> usize {
        self.with_transient_pool(|pool| pool.buffer_alloc_count())
    }

    /// Total number of fresh GPU texture allocations made by this context's transient pool.
    ///
    /// Does not increment when a retired bin entry is reused. Monotonically increasing.
    pub fn transient_texture_alloc_count(&self) -> usize {
        self.with_transient_pool(|pool| pool.texture_alloc_count())
    }

    pub(crate) fn classify(&self, e: anyhow::Error) -> GoldyError {
        if self.device().is_device_lost() {
            return GoldyError::DeviceLost;
        }
        GoldyError::Backend(e)
    }

    /// Latest GPU completion counter on this context's timeline.
    pub(crate) fn gpu_progress(&self) -> TimelineValue {
        let _tz = crate::tracy_zone!("context.gpu_progress");
        let _query = crate::tracy_zone!("context.gpu_progress.query");
        if let Some(progress) = &self.inner.gpu_progress {
            return progress.gpu_progress();
        }
        self.inner
            .device
            .inner
            .backend
            .lock()
            .unwrap()
            .gpu_progress(self.inner.handle)
    }

    /// Block until the timeline reaches at least `value`.
    pub(crate) fn wait_until(&self, value: TimelineValue) -> Result<(), GoldyError> {
        self.wait_until_context(self.inner.handle, value)
    }

    fn wait_until_context(&self, ctx: ContextHandle, value: TimelineValue) -> Result<(), GoldyError> {
        let _tz = crate::tracy_zone!("context.wait_until");
        let progress = self.gpu_progress();
        let already_complete = if ctx == self.inner.handle {
            progress >= value
        } else {
            self.inner.device.context_gpu_progress(ctx).is_some_and(|p| p >= value)
        };
        let backend_mutex = &self.inner.device.inner.backend;
        if !already_complete {
            let submission_wait = {
                let _lock = crate::tracy_zone!("context.wait_until.lock");
                let backend = backend_mutex.lock().unwrap();
                backend
                    .take_timeline_submission_epoch_wait(ctx, value)
                    .map_err(|e| self.classify(e))?
            };
            if let Some(wait) = submission_wait {
                let _sw = crate::tracy_zone!("context.wait_until.submission_worker");
                wait.wait().map_err(|e| self.classify(e))?;
            }
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
            })?;
        }
        Ok(())
    }

    /// Like [`wait_until`](Self::wait_until) but returns `Err(`[`GoldyError::SubmitTimeout`]`)` on timeout.
    pub(crate) fn wait_until_timeout(&self, value: TimelineValue, timeout_ms: u32) -> Result<(), GoldyError> {
        let ctx = self.inner.handle;
        let already_complete = self.gpu_progress() >= value;
        let backend_mutex = &self.inner.device.inner.backend;
        if !already_complete {
            let submission_wait = {
                let _lock = crate::tracy_zone!("context.wait_until.lock");
                let backend = backend_mutex.lock().unwrap();
                backend
                    .take_timeline_submission_epoch_wait(ctx, value)
                    .map_err(|e| self.classify(e))?
            };
            if let Some(wait) = submission_wait {
                let _sw = crate::tracy_zone!("context.wait_until.submission_worker");
                wait.wait().map_err(|e| self.classify(e))?;
            }
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
            let _finish = crate::tracy_zone!("context.wait_until.finish");
            backend.finish_timeline_wait(ctx, value).map_err(|e| {
                drop(backend);
                self.classify(e)
            })
        }
    }

    /// Block until every submission scheduled on this context has retired.
    ///
    /// Use before destroying surfaces or other resources that may still be
    /// referenced by in-flight work. Prefer [`crate::Submission::wait_until_settled`]
    /// when waiting for a specific receipt.
    pub fn wait_until_idle(&self) -> Result<(), GoldyError> {
        let hw = self.high_water_timeline();
        if hw == 0 {
            return Ok(());
        }
        self.wait_until(hw)
    }

    /// The largest [`TimelineValue`] ever returned by a scheme submit on this context.
    pub(crate) fn high_water_timeline(&self) -> TimelineValue {
        self.inner.high_water_timeline.load(Ordering::Relaxed)
    }

    pub(crate) fn advance_high_water_timeline(&self, tv: TimelineValue) {
        self.inner.high_water_timeline.fetch_max(tv, Ordering::Relaxed);
    }

    /// Drain pending backend signals (GPU completion, swapchain, oversubscribed).
    ///
    /// Boundary-crossed events are included for internal servicing only; prefer
    /// [`Self::poll_signals_and_service`] which reclaims and returns client-visible signals.
    pub(crate) fn poll_signals_queued(&self) -> Vec<crate::signal::QueuedSignal> {
        let progress = self.gpu_progress();
        let mut backend = self.inner.device.inner.backend.lock().unwrap();
        backend.poll_signals(self.inner.handle, progress)
    }

    /// Drain pending backend signals (swapchain, oversubscribed). Boundary-crossed is omitted.
    pub fn poll_signals(&self) -> Vec<crate::signal::Signal> {
        self.poll_signals_queued()
            .into_iter()
            .filter_map(|s| match s {
                crate::signal::QueuedSignal::Client(c) => Some(c),
                crate::signal::QueuedSignal::BoundaryCrossed { .. } => None,
            })
            .collect()
    }

    /// Drain pending signals, service boundary-crossed reclamation, return client-visible signals.
    pub fn poll_signals_and_service(&self) -> Vec<crate::signal::Signal> {
        let _tz = crate::tracy_zone!("context.poll_signals_and_service");
        let queued = self.poll_signals_queued();
        let latest_boundary = queued.iter().fold(None, |latest, signal| match signal {
            crate::signal::QueuedSignal::BoundaryCrossed { epoch } => Some(latest.unwrap_or(0).max(*epoch)),
            _ => latest,
        });
        if let Some(epoch) = latest_boundary {
            self.boundary_crossed(epoch);
        }
        queued
            .into_iter()
            .filter_map(|s| match s {
                crate::signal::QueuedSignal::Client(c) => Some(c),
                crate::signal::QueuedSignal::BoundaryCrossed { .. } => None,
            })
            .collect()
    }

    /// Process deferred GPU deletions and reclaim VRAM-ring payloads whose epoch has retired.
    ///
    /// The device-installed deferred VRAM ring ([`Device::vram_allocator`]) is drained
    /// against [`Device::timeline_retired`] (max completed over all live contexts). Any
    /// context may call this after a `BoundaryCrossed` signal; defer/release epochs are
    /// device-global submission sequence values, so `device_retired >= epoch` proves the GPU
    /// work is done regardless of which context originally submitted the payload.
    ///
    /// Per-handle last-touch reclamation (tighter than `device_retired` for the VRAM ring)
    /// is a future optimization.
    pub(crate) fn boundary_crossed(&self, epoch: TimelineValue) {
        self.boundary_crossed_inner(epoch, self.device().timeline_retired());
    }

    fn boundary_crossed_inner(&self, epoch: TimelineValue, vram_retire: TimelineValue) {
        let _tz = crate::tracy_zone!("context.boundary_crossed");
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.flush_pre");
            self.inner.deletion_flush.as_ref().expect("deletion flush").flush();
        }
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.reclaim");
            self.inner
                .reclamation_scope
                .as_ref()
                .expect("reclamation scope")
                .set_epoch(Some(epoch));
        }
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.drain_vram");
            self.device().vram_allocator().boundary_crossed(vram_retire);
        }
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.drain_transient_pool");
            // `RetainedPool::release` parks parcels here for epoch-gated reuse (leases,
            // future scheme-held transients). When callers park buffers here but do not
            // acquire through the transient pool, those parked buffers are not re-issued
            // — only dropped once `ready_after` retires. Without this drain at every frame
            // boundary, `release` leaks GPU heap (e.g. Metal buffer heaps exhausted).
            self.with_transient_pool(|pool| pool.drain_ready(self));
        }
        {
            let _tz = crate::tracy_zone!("context.boundary_crossed.flush_post");
            self.inner
                .reclamation_scope
                .as_ref()
                .expect("reclamation scope")
                .set_epoch(None);
            self.inner.deletion_flush.as_ref().expect("deletion flush").flush();
        }
    }

    /// Pull-side reclamation using this context's current GPU progress.
    ///
    /// Uses per-context fence progress for the VRAM ring (not the device-wide
    /// retired timeline), so device-queue work (e.g. texture upload on DIRECT)
    /// cannot advance reclamation past in-flight compute submits on this
    /// context's queue.
    pub fn flush_deferred_deletions(&self) {
        let _tz = crate::tracy_zone!("context.flush_deferred_deletions");
        let progress = self.gpu_progress();
        self.boundary_crossed_inner(progress, progress);
    }

    pub fn has_deferred_payloads(&self) -> bool {
        self.device().vram_allocator().has_deferred_payloads()
    }

    pub fn defer_release(&self, epoch: TimelineValue, payload: crate::vram_allocator::DeferredPayload) {
        self.device().vram_allocator().defer_release(epoch, payload);
    }

    #[cfg(test)]
    pub(crate) fn defer_until<T: Send + 'static>(&self, epoch: TimelineValue, resource: T) {
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

    /// True when every context in `refs` has retired the stamped timeline values.
    ///
    /// Prefer [`crate::Parcel::is_settled`] when checking a single parcel the caller holds.
    pub(crate) fn parcel_ready(&self, refs: &ReferenceTable) -> bool {
        if refs.is_empty() {
            return true;
        }
        let device = &self.inner.device;
        let mut progress = HashMap::with_capacity(refs.len());
        for ctx in refs.keys() {
            let p = device
                .context_gpu_progress(ctx)
                .unwrap_or(crate::timeline::CONTEXT_DESTROYED_PROGRESS);
            progress.insert(ctx, p);
        }
        is_ready(refs, &progress)
    }

    pub fn try_resubmit_retained(&self, key: u64) -> Result<Option<TimelineValue>, GoldyError> {
        if crate::validation_env::retained_cb_reuse_disabled() {
            return Ok(None);
        }
        let result = self
            .submit_session()
            .try_resubmit_retained(self.inner.handle, key, None)
            .map_err(|e| self.classify(e))?;
        if let Some(tv) = result {
            self.inner.high_water_timeline.fetch_max(tv, Ordering::Relaxed);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::mock::MockBackend;
    use crate::device::Device;
    use crate::test_support::scheme_advance_timeline;
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
    fn context_wait_until_after_scheme_submit() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let tv = scheme_advance_timeline(&ctx);
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
    fn high_water_timeline_advances_after_scheme_submit() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        assert_eq!(ctx.high_water_timeline(), 0);
        let tv = scheme_advance_timeline(&ctx);
        assert!(tv > 0);
        assert_eq!(ctx.high_water_timeline(), tv);
        let tv2 = scheme_advance_timeline(&ctx);
        assert!(tv2 > tv);
        assert_eq!(ctx.high_water_timeline(), tv2);
    }
}
