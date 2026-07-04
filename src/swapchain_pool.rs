//! Swapchain pool — N-backed present leases for retained schemes.
//!
//! A [`SwapchainPool`] wraps a [`Surface`] and supplies drawable backings for
//! [`PresentLease`] handles acquired via [`SwapchainPool::lease`]. The scheme
//! records the lease once; each [`crate::Scheme::submit`] acquires the next
//! drawable and resolves it through the present partition retention path.
//!
//! Hosts may call [`SwapchainPool::try_early_acquire`] during the overlap phase
//! (while TID_PRESENT presents the previous frame) so the next submit can take a
//! stashed drawable via [`resolve_present_slot`](SwapchainPool::resolve_present_slot).

use crate::backend::TextureHandle;
use crate::context::Context;
use crate::surface::{Frame as SurfaceFrame, Surface};
use crate::types::{ResourceAccess, SurfaceConfig};
use anyhow::Result;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

/// Resolved present slot: slot id, acquired frame, UAV index, texture handle.
pub(crate) type ResolvedPresentSlotData = (u32, SurfaceFrame, u32, TextureHandle);

pub(crate) struct SwapchainPoolInner {
    surface: RwLock<Surface>,
    /// Client-stated max in-flight drawables (present pipeline depth).
    depth: u32,
    /// Serializes depth checks and in-flight acquire reservations (render + present threads).
    acquire_mutex: Mutex<()>,
    /// Acquires that passed the depth gate but have not finished `Surface::begin()` yet.
    /// Counted in capacity checks so `begin()` blocking waits do not need to hold `acquire_mutex`.
    pending_acquire_reservations: AtomicU32,
    /// Drawable acquired during overlap for the next submit.
    stashed_drawable_slot: Mutex<Option<ResolvedPresentSlotData>>,
    /// Swapchain rebuild in progress — blocks overlap early acquire and aborts in-flight
    /// `Surface::begin()` blocking waits (DX12 resize deadlock).
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
    /// Max concurrent acquired drawables. Use `2` on DXGI/Metal so one drawable can be
    /// in-flight for present while another is stashed for the next submit.
    pub depth: u32,
    pub config: SurfaceConfig,
}

impl Default for SwapchainPoolOptions {
    fn default() -> Self {
        Self {
            depth: 1,
            config: SurfaceConfig::default(),
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
                surface: RwLock::new(surface),
                depth: options.depth.max(1),
                acquire_mutex: Mutex::new(()),
                pending_acquire_reservations: AtomicU32::new(0),
                stashed_drawable_slot: Mutex::new(None),
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
    pub fn presents_began(&self) -> u64 {
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
        Self::wait_lifecycle_counter(&self.inner, &self.inner.presents_begun, count)
    }

    /// Block until at least `count` present easements have completed WSI handoff.
    ///
    /// Returns `false` when aborted by an in-progress swapchain rebuild.
    pub fn wait_present_completed(&self, count: u64) -> bool {
        if count == 0 {
            return true;
        }
        Self::wait_lifecycle_counter(&self.inner, &self.inner.presents_completed, count)
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
        if min_present_began > 0 && self.presents_began() < min_present_began {
            return false;
        }
        !Self::rebuilding(&self.inner) && Self::has_acquire_capacity(&self.inner)
    }

    /// True when an overlap-phase early acquire stashed a drawable for the next submit.
    pub fn has_stashed_drawable(&self) -> bool {
        self.inner.stashed_drawable_slot.lock().unwrap().is_some()
    }

    /// Acquire and stash the next drawable when [`Self::ready_for_acquire`] is true.
    ///
    /// `ctx` is polled once before the `pending_acquire_count` gate below so a present's
    /// completion handler (queued asynchronously off-thread) is reconciled into the atomic
    /// counter before we read it. Without this, `pending_acquire_count` only gets drained
    /// inside a *blocking* wait (see [`Self::wait_until_pending_below_inner`]), so this
    /// non-blocking check can read a one-frame-stale "still in flight" value and skip a
    /// stash it didn't need to — pushing the acquire onto the next frame's synchronous,
    /// blocking `resolve_present_slot` path instead. That produces an every-other-frame
    /// alternation between a cheap stash hit and a full blocking `nextDrawable` call.
    ///
    /// Only stash when no drawable is in flight, on every backend — including Metal.
    ///
    /// Tried relaxing this to a plain capacity check (`< depth`) on Metal so a second
    /// drawable could be acquired while the first is still presenting: this made things
    /// *worse* (~15ms/frame in `nextDrawable`). `CAMetalLayer.nextDrawable()` (with the
    /// default `displaySyncEnabled`) is vsync-paced independent of `pending_acquire_count`
    /// bookkeeping — it can block up to a full vsync interval regardless of nominal
    /// capacity if called too soon after the previous acquire. This call site runs
    /// eagerly, synchronously, on the render thread immediately after handing frame N's
    /// present token to `TID_PRESENT` — essentially no wall-clock time has elapsed since
    /// the prior acquire — so climbing to `pending == 2` here just moves the vsync stall
    /// earlier and onto the render thread instead of avoiding it. Do not relax this gate
    /// without also pacing the call (e.g. only attempting it after real CPU/GPU work has
    /// elapsed, or moving it back to run after present has actually retired, as the now
    /// removed `speculative_acquire_after_present` did on `TID_PRESENT`).
    pub fn try_early_acquire(&self, ctx: &Context, min_present_began: u64) -> Result<bool> {
        ctx.poll_signals_and_service();
        if !self.ready_for_acquire(min_present_began) {
            return Ok(false);
        }
        if Self::rebuilding(&self.inner) {
            return Ok(false);
        }
        if self.pending_acquire_count() > 0 {
            return Ok(false);
        }
        match Self::acquire_slot(&self.inner) {
            Ok(slot) => {
                Self::stash_drawable(&self.inner, slot);
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
    /// When a drawable is already stashed, returns immediately. Otherwise requires all
    /// drawables returned (`effective == 0`) before a blocking acquire. Re-checks stash
    /// each iteration so a post-present acquire on `TID_PRESENT` unblocks a wait started
    /// with `using_stash == false`.
    pub fn wait_for_submit_acquire(&self, ctx: &crate::Context, _using_stash: bool) {
        while !self.has_stashed_drawable() {
            if Self::effective_pending_acquire_count(&self.inner) < 1 {
                break;
            }
            // One poll iteration — do not call `wait_until_pending_below(1)` here; it
            // blocks until pending==0 and cannot observe a stash landing at pending==2.
            let return_fence = {
                let surface = self.inner.surface.read().unwrap();
                surface.peek_oldest_pending_swapchain_return()
            };
            if let Some(return_fence) = return_fence {
                let _tz = crate::tracy_zone!("goldy.swapchain_pool.blocking_return_wait");
                let surface = self.inner.surface.read().unwrap();
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

    /// Block until every acquired drawable has been returned (`pending_acquire_count == 0`).
    ///
    /// Required before swapchain rebuild when depth>1 may leave drawables counted even
    /// though the present ack has already been consumed.
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
    /// [`Self::set_present_mode`] completes so overlap early acquire cannot stash a
    /// drawable after this drain.
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

    /// Block overlap early acquire, wait for an in-flight acquire, and drain the stash.
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
        Self::drain_stashed_drawable(pool);
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

    /// Take a stashed drawable from overlap early acquire, or acquire synchronously.
    pub(crate) fn resolve_present_slot(pool: &Arc<SwapchainPoolInner>) -> Result<ResolvedPresentSlotData> {
        if let Some(slot) = pool.stashed_drawable_slot.lock().unwrap().take() {
            return Ok(slot);
        }
        Self::acquire_slot(pool)
    }

    fn has_acquire_capacity(pool: &Arc<SwapchainPoolInner>) -> bool {
        Self::effective_pending_acquire_count(pool) < pool.depth
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

    /// After present has been consumed on `TID_PRESENT`: try to acquire and stash the next
    /// drawable for the following submit.
    ///
    /// Unlike [`Self::try_early_acquire`] (called from the render thread immediately after
    /// dispatching the present token — before this frame's present has had any chance to
    /// retire), this runs on `TID_PRESENT` *after* [`complete_scheduled_present`] for the
    /// current frame, so real wall-clock time has elapsed and any vsync-paced block inside
    /// `Surface::begin()` (e.g. `CAMetalLayer.nextDrawable()`, which blocks up to a full
    /// vsync interval independent of nominal capacity) lands on the present thread — which
    /// has nothing else to do until the next present token arrives — instead of stalling
    /// the render thread's CPU work for frame N+1.
    ///
    /// Non-blocking on pool capacity: if `pending_acquire_count >= depth`, returns
    /// immediately and the render thread falls back to a synchronous acquire on the next
    /// submit. Never blocks on the capacity check itself — only the (possibly vsync-paced)
    /// `acquire_slot` call below can block, and only once capacity is confirmed available.
    pub(crate) fn try_acquire_after_present(pool: &Arc<SwapchainPoolInner>) {
        let _tz = crate::tracy_zone!("goldy.swapchain_pool.acquire_after_present");
        if Self::rebuilding(pool) {
            return;
        }
        if !Self::has_acquire_capacity(pool) {
            return;
        }
        match Self::acquire_slot(pool) {
            Ok(slot) => {
                if Self::rebuilding(pool) {
                    slot.1.cancel();
                    return;
                }
                Self::stash_drawable(pool, slot);
            }
            Err(e) => {
                tracing::debug!(
                    target: "goldy::swapchain_pool",
                    error = %e,
                    "post-present acquire skipped; submit will acquire synchronously"
                );
            }
        }
    }

    /// Stash a drawable acquired during overlap for the next submit.
    pub(crate) fn stash_drawable(pool: &Arc<SwapchainPoolInner>, slot: ResolvedPresentSlotData) {
        let mut guard = pool.stashed_drawable_slot.lock().unwrap();
        if let Some((_, old_frame, _, _)) = guard.take() {
            tracing::warn!(
                target: "goldy::swapchain_pool",
                "discarding unconsumed stashed drawable"
            );
            old_frame.cancel();
        }
        *guard = Some(slot);
    }

    fn drain_stashed_drawable(pool: &Arc<SwapchainPoolInner>) {
        if let Some((_, frame, _, _)) = pool.stashed_drawable_slot.lock().unwrap().take() {
            frame.cancel();
        }
    }
}
