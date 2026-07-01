//! Swapchain pool — N-backed present leases for retained schemes.
//!
//! A [`SwapchainPool`] wraps a [`Surface`] and supplies drawable backings for
//! [`PresentLease`] handles acquired via [`SwapchainPool::lease`]. The scheme
//! records the lease once; each [`crate::Scheme::submit`] acquires the next
//! drawable and resolves it through the present partition retention path.
//!
//! When [`SwapchainPoolOptions::speculative_acquire`] is set, [`crate::Grant::consume`] on a
//! [`crate::PresentGrant`] may stash the next drawable on the pool so the render thread's subsequent
//! `submit` avoids a synchronous acquire.

use crate::backend::TextureHandle;
use crate::context::Context;
use crate::surface::{Frame as SurfaceFrame, Surface};
use crate::types::{ResourceAccess, SurfaceConfig};
use anyhow::Result;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::{Arc, Mutex, RwLock};

/// Resolved present slot: slot id, acquired frame, UAV index, texture handle.
pub(crate) type ResolvedPresentSlotData = (u32, SurfaceFrame, u32, TextureHandle);

pub(crate) struct SwapchainPoolInner {
    surface: RwLock<Surface>,
    /// Client-stated max in-flight drawables (present pipeline depth).
    depth: u32,
    /// When true, [`crate::Grant::consume`] on a [`crate::PresentGrant`] may acquire the next drawable
    /// for the following submit.
    pub(crate) speculative_acquire: bool,
    /// Serializes depth checks and synchronous acquires (render + present threads).
    acquire_mutex: Mutex<()>,
    /// Drawable acquired speculatively by [`crate::Grant::consume`] on a [`crate::PresentGrant`] for the next submit.
    speculative_acquire_slot: Mutex<Option<ResolvedPresentSlotData>>,
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
                speculative_acquire: options.speculative_acquire,
                acquire_mutex: Mutex::new(()),
                speculative_acquire_slot: Mutex::new(None),
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

    /// Block until an acquire slot would succeed (`pending_acquire_count` `<` [`depth`](Self::depth)).
    ///
    /// On flip-model backends (DX12/Vulkan) present ack alone is insufficient: the drawable
    /// stays counted until its return fence retires. Blocks on the device sync fence when
    /// a pending return is registered, then polls `ctx` once to process deferred returns.
    pub fn wait_for_acquire_capacity(&self, ctx: &crate::Context) {
        self.wait_until_pending_below(self.depth(), ctx);
    }

    /// Block until every acquired drawable has been returned (`pending_acquire_count == 0`).
    ///
    /// Required before swapchain rebuild when speculative acquire or depth>1 may leave
    /// drawables counted even though the present ack has already been consumed.
    fn wait_for_all_drawables_returned(&self, ctx: &crate::Context) {
        self.wait_until_pending_below(1, ctx);
    }

    /// Drain speculative stash and wait for all pending swapchain returns.
    pub fn sync_before_rebuild(&self, ctx: &crate::Context) {
        Self::drain_speculative_acquire(&self.inner);
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
        while self.pending_acquire_count() >= threshold {
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

    /// Resize the underlying swapchain (structural edit — rebuild scheme nodes).
    pub fn resize(&self, width: u32, height: u32) -> Result<()> {
        Self::drain_speculative_acquire(&self.inner);
        let mut surface = self.inner.surface.write().unwrap();
        surface.resize(width, height)
    }

    pub fn set_present_mode(&self, mode: crate::types::PresentMode) -> Result<()> {
        Self::drain_speculative_acquire(&self.inner);
        let mut surface = self.inner.surface.write().unwrap();
        surface.set_present_mode(mode)
    }

    pub(crate) fn acquire_slot(pool: &Arc<SwapchainPoolInner>) -> Result<ResolvedPresentSlotData> {
        let _guard = pool.acquire_mutex.lock().unwrap();
        // Read lock only: `begin()` may block on swapchain image availability without
        // preventing concurrent queries or stalling resize on a held write lock.
        {
            let surface = pool.surface.read().unwrap();
            let in_flight = surface.pending_acquire_count();
            if in_flight >= pool.depth {
                anyhow::bail!(
                    "swapchain pool at present depth {} ({} drawable(s) acquired, not yet returned)",
                    pool.depth,
                    in_flight
                );
            }
        }
        let frame = {
            let surface = pool.surface.read().unwrap();
            surface.begin()?
        };
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

    /// Stash a drawable acquired during [`crate::Grant::consume`] on a [`crate::PresentGrant`] for the next submit.
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
