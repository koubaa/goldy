//! Swapchain pool — N-backed present leases for retained schemes.
//!
//! A [`SwapchainPool`] wraps a [`Surface`] and supplies drawable backings for
//! [`PresentLease`] handles acquired via [`SwapchainPool::lease`]. Callers may
//! acquire a concrete drawable early via [`SwapchainPool::acquire_present`] (classic
//! frame timing) or let [`crate::Scheme::submit`] defer acquire until the present
//! partition is about to run.

use crate::backend::TextureHandle;
use crate::context::Context;
use crate::surface::{Frame as SurfaceFrame, Surface};
use crate::types::{ResourceAccess, SurfaceConfig};
use anyhow::Result;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::{Arc, RwLock};

pub(crate) struct SwapchainPoolInner {
    surface: RwLock<Surface>,
    #[allow(dead_code)]
    depth: u32,
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

/// A swapchain drawable acquired before scheme submit (classic-like early acquire).
///
/// Dropping an unconsumed claim cancels the underlying [`Surface`] frame so the
/// image is not presented.
pub struct AcquiredPresent {
    lease_id: u32,
    slot_id: u32,
    handle: TextureHandle,
    uav_index: u32,
    frame: Option<SurfaceFrame>,
}

impl AcquiredPresent {
    /// Lease id this drawable fulfills ([`PresentLease`] identity).
    pub fn lease_id(&self) -> u32 {
        self.lease_id
    }

    pub(crate) fn into_parts(mut self) -> (u32, u32, TextureHandle, u32, SurfaceFrame) {
        let frame = self.frame.take().expect("AcquiredPresent frame already taken");
        let lease_id = self.lease_id;
        let slot_id = self.slot_id;
        let handle = self.handle;
        let uav_index = self.uav_index;
        (lease_id, slot_id, handle, uav_index, frame)
    }
}

impl Drop for AcquiredPresent {
    fn drop(&mut self) {
        if let Some(frame) = self.frame.take() {
            frame.cancel();
        }
    }
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

    /// Acquire the next drawable for `lease` now (blocks on DXGI / return fence).
    ///
    /// Pass the result to [`crate::Scheme::submit_with_acquired_presents`] so submit
    /// does not wait again at the present partition. Prefer this when matching
    /// classic task-graph timing (acquire at the start of the frame).
    pub fn acquire_present(&self, lease: &PresentLease) -> Result<AcquiredPresent> {
        if !Arc::ptr_eq(&lease.pool, &self.inner) {
            anyhow::bail!("PresentLease does not belong to this SwapchainPool");
        }
        let (slot_id, frame, uav_index, handle) = Self::acquire_slot(&lease.pool)?;
        Ok(AcquiredPresent {
            lease_id: lease.id,
            slot_id,
            handle,
            uav_index,
            frame: Some(frame),
        })
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
        let mut surface = self.inner.surface.write().unwrap();
        surface.resize(width, height)
    }

    pub fn set_present_mode(&self, mode: crate::types::PresentMode) -> Result<()> {
        let mut surface = self.inner.surface.write().unwrap();
        surface.set_present_mode(mode)
    }

    pub(crate) fn acquire_slot(pool: &Arc<SwapchainPoolInner>) -> Result<(u32, SurfaceFrame, u32, TextureHandle)> {
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
}
