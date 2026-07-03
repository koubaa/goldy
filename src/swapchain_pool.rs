//! Swapchain pool — N-backed present leases for retained schemes.
//!
//! A [`SwapchainPool`] wraps a [`Surface`] and supplies drawable backings for
//! [`PresentLease`] handles acquired via [`SwapchainPool::lease`]. The scheme
//! records the lease once; each [`crate::Scheme::submit`] acquires the next
//! drawable and resolves it through the present partition retention path.
//!
//! When [`SwapchainPoolOptions::speculative_acquire`] is set, the host calls
//! [`crate::PresentGrant::speculate_next_acquire_after_present`] after present scanout
//! (and after any async present ack) so the render thread's subsequent `submit` avoids a
//! synchronous acquire.

use crate::backend::TextureHandle;
use crate::context::Context;
use crate::surface::{Frame as SurfaceFrame, Surface};
use crate::types::{ResourceAccess, SurfaceConfig};
use anyhow::Result;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

/// Resolved present slot: slot id, acquired frame, UAV index, texture handle.
pub(crate) type ResolvedPresentSlotData = (u32, SurfaceFrame, u32, TextureHandle);

pub(crate) struct SwapchainPoolInner {
    /// Context for acquire-capacity waits on TID_PRESENT (tech debt: pool should own this policy).
    ctx: Context,
    surface: RwLock<Surface>,
    /// Client-stated max in-flight drawables (present pipeline depth).
    depth: u32,
    /// When true, [`crate::PresentGrant::speculate_next_acquire_after_present`] may acquire
    /// the next drawable for the following submit.
    pub(crate) speculative_acquire: bool,
    /// Serializes depth checks and in-flight acquire reservations (render + present threads).
    acquire_mutex: Mutex<()>,
    /// Acquires that passed the depth gate but have not finished `Surface::begin()` yet.
    /// Counted in capacity checks so `begin()` blocking waits do not need to hold `acquire_mutex`.
    pending_acquire_reservations: AtomicU32,
    /// Drawable acquired speculatively after present for the next submit.
    speculative_acquire_slot: Mutex<Option<ResolvedPresentSlotData>>,
    /// Swapchain rebuild in progress — blocks post-ack speculative acquire on TID_PRESENT
    /// and aborts in-flight `Surface::begin()` blocking waits (DX12 resize deadlock).
    rebuilding: Arc<AtomicBool>,
    /// Monotonic count of present easements whose WSI handoff has begun (after copy epoch).
    presents_begun: AtomicU64,
    /// Monotonic count of present easements whose WSI handoff has completed.
    presents_completed: AtomicU64,
    present_lifecycle_notify: Arc<(Mutex<()>, Condvar)>,
    /// Set during rebuild quiesce so blocked lifecycle waits can bail out.
    present_wait_abort: AtomicBool,
}

/// Construction options for [`SwapchainPool`].
#[derive(Clone, Debug)]
pub struct SwapchainPoolOptions {
    /// Max concurrent acquired drawables. Use `2` when [`Self::speculative_acquire`]
    /// is enabled so one drawable can be in-flight for present while another is
    /// stashed for the next submit.
    pub depth: u32,
    pub config: SurfaceConfig,
    /// Acquire the next drawable on TID_PRESENT after present, for the following submit.
    pub speculative_acquire: bool,
}

impl Default for SwapchainPoolOptions {
    fn default() -> Self {
        Self {
            depth: 1,
            config: SurfaceConfig::default(),
            speculative_acquire: false,
        }
    }
}

/// Pool of OS swapchain drawables for present-on-scheme.
///
/// Construct once per window; call [`Self::lease`] to obtain a stable
/// [`PresentLease`] identity for scheme recording.
pub struct SwapchainPool {
    inner: Arc<SwapchainPoolInner>,
}

impl Clone for SwapchainPool {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Stable scheme-scoped name for a swapchain drawable lease.
///
/// The physical backing rotates per submission; the lease id is recorded once
/// in the scheme IR as `ResourceId::PresentLease`.
pub struct PresentLease {
    pub(crate) id: u32,
    pub(crate) pool: Arc<SwapchainPoolInner>,
}

impl SwapchainPool {
    /// Create a swapchain pool bound to `window` on `ctx`.
    ///
    /// `depth` is the client's stated in-flight frame count (used for pool
    /// policy; the OS may provide fewer or more swapchain images).
    pub fn new<W>(ctx: &Context, window: &W, depth: u32) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        Self::new_with_config(ctx, window, depth, SurfaceConfig::default())
    }

    /// Create with explicit surface configuration.
    pub fn new_with_config<W>(ctx: &Context, window: &W, depth: u32, config: SurfaceConfig) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        Self::new_with_options(
            ctx,
            window,
            SwapchainPoolOptions {
                depth,
                config,
                ..SwapchainPoolOptions::default()
            },
        )
    }

    /// Create with explicit pool and surface options.
    pub fn new_with_options<W>(ctx: &Context, window: &W, options: SwapchainPoolOptions) -> Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        let surface = Surface::new_with_config(ctx, window, options.config)?;
        Ok(Self {
            inner: Arc::new(SwapchainPoolInner {
                ctx: ctx.clone(),
                surface: RwLock::new(surface),
                depth: options.depth.max(1),
                speculative_acquire: options.speculative_acquire,
                acquire_mutex: Mutex::new(()),
                pending_acquire_reservations: AtomicU32::new(0),
                speculative_acquire_slot: Mutex::new(None),
                rebuilding: Arc::new(AtomicBool::new(false)),
                presents_begun: AtomicU64::new(0),
                presents_completed: AtomicU64::new(0),
                present_lifecycle_notify: Arc::new((Mutex::new(()), Condvar::new())),
                present_wait_abort: AtomicBool::new(false),
            }),
        })
    }

    /// Acquire a stable present lease handle (v1: one lease per pool).
    pub fn lease(&self) -> PresentLease {
        PresentLease {
            id: 0,
            pool: Arc::clone(&self.inner),
        }
    }

    /// Current drawable extent.
    pub fn size(&self) -> (u32, u32) {
        let surface = self.inner.surface.read().unwrap();
        surface.size()
    }

    pub fn width(&self) -> u32 {
        self.size().0
    }

    pub fn height(&self) -> u32 {
        self.size().1
    }

    pub fn format(&self) -> crate::types::TextureFormat {
        let surface = self.inner.surface.read().unwrap();
        surface.format()
    }

    /// Client-stated present pipeline depth (max concurrent acquired drawables).
    pub fn depth(&self) -> u32 {
        self.inner.depth
    }

    /// How many swapchain drawables are in-flight (acquired or presented, not yet returned).
    pub fn pending_acquire_count(&self) -> u32 {
        let surface = self.inner.surface.read().unwrap();
        surface.pending_acquire_count()
    }

    /// Present easements whose WSI handoff has begun (after copy submit epoch).
    pub fn presents_begun(&self) -> u64 {
        self.inner.presents_begun.load(Ordering::Acquire)
    }

    /// Present easements whose WSI handoff has completed.
    pub fn presents_completed(&self) -> u64 {
        self.inner.presents_completed.load(Ordering::Acquire)
    }

    /// Block until at least `count` present easements have begun WSI handoff.
    ///
    /// Returns `false` when aborted by an in-progress swapchain rebuild.
    pub fn wait_present_began(&self, count: u64) -> bool {
        if count == 0 {
            return true;
        }
        Self::wait_lifecycle_counter(
            &self.inner,
            &self.inner.presents_begun,
            count,
        )
    }

    /// Block until at least `count` present easements have completed WSI handoff.
    ///
    /// Returns `false` when aborted by an in-progress swapchain rebuild.
    pub fn wait_present_completed(&self, count: u64) -> bool {
        if count == 0 {
            return true;
        }
        Self::wait_lifecycle_counter(
            &self.inner,
            &self.inner.presents_completed,
            count,
        )
    }

    fn wait_lifecycle_counter(pool: &Arc<SwapchainPoolInner>, counter: &AtomicU64, target: u64) -> bool {
        const POLL_MS: u64 = 2;
        while counter.load(Ordering::Acquire) < target {
            if pool.present_wait_abort.load(Ordering::Acquire) {
                tracing::debug!(
                    target: "goldy::swapchain_pool",
                    target,
                    "present lifecycle wait aborted for rebuild"
                );
                return false;
            }
            let mut guard = pool.present_lifecycle_notify.0.lock().unwrap();
            while counter.load(Ordering::Acquire) < target {
                if pool.present_wait_abort.load(Ordering::Acquire) {
                    return false;
                }
                let (next_guard, timeout) = pool
                    .present_lifecycle_notify
                    .1
                    .wait_timeout(guard, std::time::Duration::from_millis(POLL_MS))
                    .unwrap();
                guard = next_guard;
                if timeout.timed_out() && counter.load(Ordering::Acquire) < target {
                    drop(guard);
                    break;
                }
            }
        }
        true
    }

    pub(crate) fn publish_present_began(pool: &Arc<SwapchainPoolInner>) {
        pool.presents_begun.fetch_add(1, Ordering::Release);
        let guard = pool.present_lifecycle_notify.0.lock().unwrap();
        pool.present_lifecycle_notify.1.notify_all();
        drop(guard);
    }

    pub(crate) fn publish_present_completed(pool: &Arc<SwapchainPoolInner>) {
        pool.presents_completed.fetch_add(1, Ordering::Release);
        let guard = pool.present_lifecycle_notify.0.lock().unwrap();
        pool.present_lifecycle_notify.1.notify_all();
        drop(guard);
    }

    /// Non-blocking check whether a render-thread early acquire is likely cheap.
    ///
    /// When `min_present_began > 0`, returns false until at least that many present easements
    /// have begun WSI handoff — avoids blocking `Surface::begin()` on flip-index waits before
    /// the previous frame's present has started on TID_PRESENT.
    pub fn ready_for_acquire(&self, min_present_began: u64) -> bool {
        if min_present_began > 0 && self.presents_begun() < min_present_began {
            return false;
        }
        !Self::rebuilding(&self.inner) && Self::has_acquire_capacity(&self.inner)
    }

    /// True when an overlap-phase early acquire stashed a drawable for the next submit.
    pub fn has_stashed_drawable(&self) -> bool {
        self.inner.speculative_acquire_slot.lock().unwrap().is_some()
    }

    /// Acquire and stash the next drawable when [`Self::ready_for_acquire`] is true.
    pub fn try_early_acquire(&self, min_present_began: u64) -> Result<bool> {
        if !self.ready_for_acquire(min_present_began) {
            return Ok(false);
        }
        if Self::rebuilding(&self.inner) {
            return Ok(false);
        }
        // Overlap stash only when no drawable is in flight; otherwise we climb to
        // pending=2 and DXGI frame-latency wait blocks the next sync acquire.
        if self.pending_acquire_count() > 0 {
            return Ok(false);
        }
        match Self::acquire_slot(&self.inner) {
            Ok(slot) => {
                Self::stash_speculative_acquire(&self.inner, slot);
                Ok(true)
            }
            Err(e) => {
                tracing::debug!(
                    target: "goldy::swapchain_pool",
                    error = %e,
                    "early acquire skipped"
                );
                Ok(false)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn test_signal_present_began(&self) {
        Self::publish_present_began(&self.inner);
    }

    #[cfg(test)]
    pub(crate) fn test_signal_present_completed(&self) {
        Self::publish_present_completed(&self.inner);
    }

    /// Block until an acquire slot would succeed (effective in-flight `<` [`depth`](Self::depth)).
    ///
    /// Includes in-progress `Surface::begin()` reservations. On flip-model backends (DX12/Vulkan)
    /// stays counted until its return fence retires. Blocks on the device sync fence when
    /// a pending return is registered, then polls `ctx` once to process deferred returns.
    pub fn wait_for_acquire_capacity(&self, ctx: &crate::Context) {
        self.wait_until_pending_below(self.depth(), ctx);
    }

    /// Block until the next submit can acquire without wedging DXGI frame latency.
    ///
    /// When the submit path will take a stashed drawable, requires `effective < depth`.
    /// When it must perform a synchronous acquire (no stash), requires all drawables
    /// returned (`effective == 0`) so `Surface::begin()` does not block after saturation.
    pub fn wait_for_submit_acquire(&self, ctx: &crate::Context, using_stash: bool) {
        let threshold = if using_stash { self.depth() } else { 1 };
        self.wait_until_pending_below(threshold, ctx);
    }

    /// Block until every acquired drawable has been returned (`pending_acquire_count == 0`).
    ///
    /// Required before swapchain rebuild when speculative acquire or depth>1 may leave
    /// drawables counted even though the present ack has already been consumed.
    fn wait_for_all_drawables_returned(&self, ctx: &crate::Context) {
        self.wait_until_pending_below(1, ctx);
    }

    /// Quiesce drawables and wait for GPU returns before swapchain rebuild.
    ///
    /// `sent_present_tokens` is the number of present tokens dispatched to the async
    /// present host since startup (velato's `present_sent_seq`). Waits until that many
    /// easements have completed WSI handoff, including tokens still queued on the
    /// present channel, so `ResizeBuffers` does not race an in-flight present.
    ///
    /// Leaves [`SwapchainPoolInner::rebuilding`] set until [`Self::resize`] or
    /// [`Self::set_present_mode`] completes so a post-ack speculative acquire on TID_PRESENT
    /// cannot stash a drawable after this drain.
    pub fn sync_before_rebuild(&self, ctx: &crate::Context, sent_present_tokens: u64) {
        let _tz = crate::tracy_zone!("goldy.swapchain_pool.sync_before_rebuild");
        if sent_present_tokens > 0 {
            self.wait_present_completed(sent_present_tokens);
        }
        Self::begin_rebuild_quiesce(&self.inner);
        let device = ctx.device();
        if let Err(e) = {
            let mut backend = device.inner.backend.lock().unwrap();
            backend.device_wait_idle(device.inner.handle)
        } {
            tracing::warn!(
                target: "goldy::swapchain_pool",
                error = %e,
                "device_wait_idle before swapchain rebuild failed"
            );
        }
        ctx.poll_signals_and_service();
        self.wait_for_all_drawables_returned(ctx);
    }

    fn wait_until_pending_below(&self, threshold: u32, ctx: &crate::Context) {
        Self::wait_until_pending_below_inner(&self.inner, threshold, ctx);
    }

    /// Resize the underlying swapchain (structural edit — rebuild scheme nodes).
    pub fn resize(&self, width: u32, height: u32) -> Result<()> {
        Self::begin_rebuild_quiesce(&self.inner);
        let result = {
            let mut surface = self.inner.surface.write().unwrap();
            surface.resize(width, height)
        };
        Self::end_rebuild(&self.inner);
        result
    }

    pub fn set_present_mode(&self, mode: crate::types::PresentMode) -> Result<()> {
        Self::begin_rebuild_quiesce(&self.inner);
        let result = {
            let mut surface = self.inner.surface.write().unwrap();
            surface.set_present_mode(mode)
        };
        Self::end_rebuild(&self.inner);
        result
    }

    /// Block speculative acquire, wait for an in-flight acquire, and drain the stash.
    fn begin_rebuild_quiesce(pool: &Arc<SwapchainPoolInner>) {
        pool.present_wait_abort.store(true, Ordering::Release);
        {
            let guard = pool.present_lifecycle_notify.0.lock().unwrap();
            pool.present_lifecycle_notify.1.notify_all();
            drop(guard);
        }
        pool.rebuilding.store(true, Ordering::Release);
        let _guard = pool.acquire_mutex.lock().unwrap();
        while pool.pending_acquire_reservations.load(Ordering::Acquire) > 0 {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        Self::drain_speculative_acquire(pool);
    }

    fn end_rebuild(pool: &Arc<SwapchainPoolInner>) {
        pool.rebuilding.store(false, Ordering::Release);
        pool.present_wait_abort.store(false, Ordering::Release);
        {
            let guard = pool.present_lifecycle_notify.0.lock().unwrap();
            pool.present_lifecycle_notify.1.notify_all();
            drop(guard);
        }
    }

    fn rebuilding(pool: &Arc<SwapchainPoolInner>) -> bool {
        pool.rebuilding.load(Ordering::Acquire)
    }

    fn effective_pending_acquire_count(pool: &SwapchainPoolInner) -> u32 {
        let surface = pool.surface.read().unwrap();
        surface.pending_acquire_count() + pool.pending_acquire_reservations.load(Ordering::Acquire)
    }

    pub(crate) fn acquire_slot(pool: &Arc<SwapchainPoolInner>) -> Result<ResolvedPresentSlotData> {
        {
            let _guard = pool.acquire_mutex.lock().unwrap();
            if Self::rebuilding(pool) {
                anyhow::bail!("swapchain pool rebuild in progress");
            }
            let in_flight = Self::effective_pending_acquire_count(pool);
            if in_flight >= pool.depth {
                anyhow::bail!(
                    "swapchain pool at present depth {} ({} drawable(s) acquired, not yet returned)",
                    pool.depth,
                    in_flight
                );
            }
            pool.pending_acquire_reservations.fetch_add(1, Ordering::Release);
        }

        if Self::rebuilding(pool) {
            pool.pending_acquire_reservations.fetch_sub(1, Ordering::Release);
            anyhow::bail!("swapchain pool rebuild in progress");
        }

        // `begin()` may block on DXGI waitable / fence / flip-model index waits; keep that
        // outside `acquire_mutex` so the render thread is not serialized behind TID_PRESENT.
        let begin_result = {
            let surface = pool.surface.read().unwrap();
            surface.begin()
        };
        pool.pending_acquire_reservations.fetch_sub(1, Ordering::Release);
        let frame = begin_result?;
        let slot_id = frame.frame_slot();
        let (handle, uav_index) = {
            let tex = frame.texture();
            let uav_index = tex
                .resource_index(ResourceAccess::Write)
                .ok_or_else(|| anyhow::anyhow!("swapchain texture has no UAV resource index"))?;
            (tex.gpu_handle(), uav_index)
        };
        Ok((slot_id, frame, uav_index, handle))
    }

    /// Take a speculatively acquired slot, or acquire synchronously.
    pub(crate) fn resolve_present_slot(pool: &Arc<SwapchainPoolInner>) -> Result<ResolvedPresentSlotData> {
        if let Some(slot) = pool.speculative_acquire_slot.lock().unwrap().take() {
            return Ok(slot);
        }
        Self::acquire_slot(pool)
    }

    /// After present on TID_PRESENT: try to acquire and stash for the next submit.
    ///
    /// Non-blocking on pool capacity: if `pending_acquire_count >= depth`, returns immediately
    /// and the render thread acquires synchronously on the next submit. Never blocks on the
    /// capacity spin loop — TID_PRESENT must return to its present channel promptly.
    pub(crate) fn speculative_acquire_after_present(pool: &Arc<SwapchainPoolInner>) {
        let _tz = crate::tracy_zone!("goldy.swapchain_pool.speculative_acquire_after_present");
        if Self::rebuilding(pool) {
            return;
        }
        if !Self::has_acquire_capacity(pool) {
            tracing::debug!(
                target: "goldy::swapchain_pool",
                pending = Self::effective_pending_acquire_count(pool),
                depth = pool.depth,
                "speculative present acquire skipped: pool at capacity"
            );
            return;
        }
        if Self::rebuilding(pool) {
            return;
        }
        match Self::acquire_slot(pool) {
            Ok(slot) => {
                if Self::rebuilding(pool) {
                    slot.1.cancel();
                    return;
                }
                Self::stash_speculative_acquire(pool, slot);
            }
            Err(e) => {
                tracing::debug!(
                    target: "goldy::swapchain_pool",
                    error = %e,
                    "speculative present acquire failed; submit will acquire synchronously"
                );
            }
        }
    }

    fn has_acquire_capacity(pool: &Arc<SwapchainPoolInner>) -> bool {
        Self::effective_pending_acquire_count(pool) < pool.depth
    }

    #[allow(dead_code, reason = "render-thread rebuild waits; speculate uses has_acquire_capacity")]
    fn wait_for_acquire_capacity_inner(pool: &Arc<SwapchainPoolInner>) {
        Self::wait_until_pending_below_inner(pool, pool.depth, &pool.ctx);
    }

    fn wait_until_pending_below_inner(pool: &Arc<SwapchainPoolInner>, threshold: u32, ctx: &Context) {
        while Self::effective_pending_acquire_count(pool) >= threshold {
            let return_fence = {
                let surface = pool.surface.read().unwrap();
                surface.peek_oldest_pending_swapchain_return()
            };
            if let Some(return_fence) = return_fence {
                let _tz = crate::tracy_zone!("goldy.swapchain_pool.blocking_return_wait");
                let surface = pool.surface.read().unwrap();
                if let Err(e) = surface.blocking_wait_swapchain_return(ctx, return_fence) {
                    tracing::warn!(
                        target: "goldy::swapchain_pool",
                        ?return_fence,
                        error = %e,
                        "blocking swapchain return wait failed; falling back to poll"
                    );
                }
            } else {
                std::thread::yield_now();
            }
            ctx.poll_signals_and_service();
        }
    }

    /// Stash a drawable acquired after present for the next submit.
    pub(crate) fn stash_speculative_acquire(pool: &Arc<SwapchainPoolInner>, slot: ResolvedPresentSlotData) {
        let mut guard = pool.speculative_acquire_slot.lock().unwrap();
        if let Some((_, old_frame, _, _)) = guard.take() {
            tracing::warn!(
                target: "goldy::swapchain_pool",
                "discarding unconsumed speculative present acquire"
            );
            old_frame.cancel();
        }
        *guard = Some(slot);
    }

    fn drain_speculative_acquire(pool: &Arc<SwapchainPoolInner>) {
        if let Some((_, frame, _, _)) = pool.speculative_acquire_slot.lock().unwrap().take() {
            frame.cancel();
        }
    }

    #[cfg(test)]
    pub(crate) fn has_speculative_acquire(pool: &Arc<SwapchainPoolInner>) -> bool {
        pool.speculative_acquire_slot.lock().unwrap().is_some()
    }
}
