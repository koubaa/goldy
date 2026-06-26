//! Swapchain pool — N-backed present leases for retained schemes.
//!
//! A [`SwapchainPool`] wraps a [`Surface`] and supplies drawable backings for
//! [`PresentLease`] handles acquired via [`SwapchainPool::lease`]. The scheme
//! records the lease once; each [`crate::Scheme::submit`] acquires the next
//! drawable and resolves it through the present partition retention path.
//!
//! When [`crate::validation_env::speculative_present_acquire_enabled`] is set,
//! [`crate::PresentGrant::consume`] may stash the next drawable on the pool so
//! the render thread's subsequent `submit` avoids a synchronous acquire.

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
    #[allow(dead_code)]
    depth: u32,
    /// Drawable acquired speculatively by [`crate::PresentGrant::consume`] for the next submit.
    speculative_acquire: Mutex<Option<ResolvedPresentSlotData>>,
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
        let surface = Surface::new_with_config(ctx, window, config)?;
        Ok(Self {
            inner: Arc::new(SwapchainPoolInner {
                surface: RwLock::new(surface),
                depth: depth.max(1),
                speculative_acquire: Mutex::new(None),
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
        // Read lock only: `begin()` may block on swapchain image availability without
        // preventing concurrent queries or stalling resize on a held write lock.
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
        if let Some(slot) = pool.speculative_acquire.lock().unwrap().take() {
            return Ok(slot);
        }
        Self::acquire_slot(pool)
    }

    /// Stash a drawable acquired during [`crate::PresentGrant::consume`] for the next submit.
    pub(crate) fn stash_speculative_acquire(pool: &Arc<SwapchainPoolInner>, slot: ResolvedPresentSlotData) {
        let mut guard = pool.speculative_acquire.lock().unwrap();
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
        if let Some((_, frame, _, _)) = pool.speculative_acquire.lock().unwrap().take() {
            frame.cancel();
        }
    }

    #[cfg(test)]
    pub(crate) fn has_speculative_acquire(pool: &Arc<SwapchainPoolInner>) -> bool {
        pool.speculative_acquire.lock().unwrap().is_some()
    }
}
