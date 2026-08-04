//! Swapchain pool — N-backed present leases for retained schemes.
//!
//! A swapchain pool wraps a surface and supplies drawable backings for
//! [`PresentLease`] handles. Callers may acquire a concrete drawable early
//! (classic frame timing) or let [`crate::Scheme::submit`] defer acquire until
//! the present partition is about to run.

use crate::context::Context;
use crate::handles::TextureHandle;
use crate::surface::{Frame as SurfaceFrame, Surface};
use crate::types::{ResourceAccess, SurfaceConfig};
use anyhow::Result;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

#[cfg(test)]
use std::sync::atomic::AtomicBool;

pub(crate) struct SwapchainPoolInner {
    surface: RwLock<Surface>,
    #[allow(dead_code)]
    depth: u32,
    /// Advanced on resize / present-mode change so stale claims and retained
    /// physical variants cannot be reused after backing recreation.
    pub(crate) generation: Arc<AtomicU64>,
    /// Last requested extent; applied on the next acquire so interactive
    /// `WM_SIZE` storms only pay for one `ResizeBuffers` per frame.
    pending_resize: Mutex<Option<(u32, u32)>>,
    /// When true, the next [`SwapchainPool::acquire_slot`] fails once (scheme tests).
    #[cfg(test)]
    fail_next_acquire: AtomicBool,
}

impl SwapchainPoolInner {
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn bump_generation(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Logical drawable size (pending resize if any, else live swapchain).
    fn logical_size(&self) -> (u32, u32) {
        if let Ok(guard) = self.pending_resize.lock() {
            if let Some(size) = *guard {
                return size;
            }
        }
        self.surface.read().unwrap().size()
    }

    /// Apply a coalesced pending resize before acquiring a drawable.
    fn apply_pending_resize(&self) -> Result<()> {
        let pending = self.pending_resize.lock().unwrap().take();
        let Some((width, height)) = pending else {
            return Ok(());
        };
        let mut surface = self.surface.write().unwrap();
        // Generation already advanced when the pending request was recorded.
        if let Err(err) = surface.resize(width, height) {
            // Restore so a later acquire / present-mode change can retry.
            *self.pending_resize.lock().unwrap() = Some((width, height));
            return Err(err);
        }
        Ok(())
    }
}

/// Pool of OS swapchain drawables for present-on-scheme.
///
/// Constructed by [`crate::SurfaceExchange`]. Call [`Self::lease`] to obtain a
/// stable [`PresentLease`] identity for scheme recording.
pub(crate) struct SwapchainPool {
    inner: Arc<SwapchainPoolInner>,
}

/// Pool-local present lease handle.
///
/// The physical backing rotates per submission. Schemes map `(pool, id)` to a
/// scheme-unique present-lease binding so two pools that both use local id `0`
/// remain distinct.
pub struct PresentLease {
    pub(crate) id: u32,
    pub(crate) pool: Arc<SwapchainPoolInner>,
}

impl PresentLease {
    /// Shared live generation counter for this lease's pool.
    pub(crate) fn generation_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.pool.generation)
    }
}

/// A swapchain drawable acquired before scheme submit (classic-like early acquire).
///
/// Dropping an unconsumed claim cancels the underlying surface frame so the
/// image is not presented.
pub struct AcquiredPresent {
    /// Pool-local lease id (matches a [`PresentLease`], not the scheme binding id).
    lease_id: u32,
    pool: Arc<SwapchainPoolInner>,
    slot_id: u32,
    /// Pool generation at acquire time.
    generation: u64,
    handle: TextureHandle,
    uav_index: u32,
    frame: Option<SurfaceFrame>,
}

impl AcquiredPresent {
    /// Pool-local lease id this drawable fulfills ([`PresentLease`] identity).
    pub fn lease_id(&self) -> u32 {
        self.lease_id
    }

    pub(crate) fn pool(&self) -> &Arc<SwapchainPoolInner> {
        &self.pool
    }

    pub(crate) fn into_parts(mut self) -> (u32, Arc<SwapchainPoolInner>, u32, u64, TextureHandle, u32, SurfaceFrame) {
        let frame = self.frame.take().expect("AcquiredPresent frame already taken");
        let lease_id = self.lease_id;
        let pool = Arc::clone(&self.pool);
        let slot_id = self.slot_id;
        let generation = self.generation;
        let handle = self.handle;
        let uav_index = self.uav_index;
        (lease_id, pool, slot_id, generation, handle, uav_index, frame)
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
    #[cfg_attr(not(test), allow(dead_code))]
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
                generation: Arc::new(AtomicU64::new(0)),
                pending_resize: Mutex::new(None),
                #[cfg(test)]
                fail_next_acquire: AtomicBool::new(false),
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

    /// Current backing generation (advances on resize / present-mode change).
    pub fn generation(&self) -> u64 {
        self.inner.generation()
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
        let (slot_id, generation, frame, uav_index, handle) = Self::acquire_slot(&lease.pool)?;
        Ok(AcquiredPresent {
            lease_id: lease.id,
            pool: Arc::clone(&lease.pool),
            slot_id,
            generation,
            handle,
            uav_index,
            frame: Some(frame),
        })
    }

    /// Current drawable extent (includes a pending resize not yet applied).
    pub fn size(&self) -> (u32, u32) {
        self.inner.logical_size()
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

    /// Request a swapchain resize (structural edit — rebuild scheme nodes).
    ///
    /// The DXGI/CUDA rebuild is deferred until the next drawable acquire so a
    /// burst of window-size events only pays for one `ResizeBuffers` per frame.
    /// Advances the pool generation immediately so claims and retained variants
    /// from the previous backing cannot be reused.
    pub fn resize(&self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        let (old_w, old_h) = self.inner.logical_size();
        if width == old_w && height == old_h {
            return Ok(());
        }
        *self.inner.pending_resize.lock().unwrap() = Some((width, height));
        self.inner.bump_generation();
        Ok(())
    }

    pub fn set_present_mode(&self, mode: crate::types::PresentMode) -> Result<()> {
        // Present mode touches the live swapchain; apply any pending extent first.
        self.inner.apply_pending_resize()?;
        let mut surface = self.inner.surface.write().unwrap();
        surface.set_present_mode(mode)?;
        self.inner.bump_generation();
        Ok(())
    }

    /// Force the next deferred/eager acquire from this pool to fail (tests only).
    #[cfg(test)]
    pub(crate) fn fail_next_acquire(&self) {
        self.inner.fail_next_acquire.store(true, Ordering::SeqCst);
    }

    pub(crate) fn acquire_slot(pool: &Arc<SwapchainPoolInner>) -> Result<(u32, u64, SurfaceFrame, u32, TextureHandle)> {
        #[cfg(test)]
        {
            if pool.fail_next_acquire.swap(false, Ordering::SeqCst) {
                anyhow::bail!("test-injected acquire failure");
            }
        }
        // Coalesce window-size storms: apply the latest pending extent once per frame.
        pool.apply_pending_resize()?;
        let generation = pool.generation();
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
        Ok((slot_id, generation, frame, uav_index, handle))
    }
}
