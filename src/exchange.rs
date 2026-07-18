//! Erased exchange transactions and claims.
//!
//! A concrete exchange (for example [`SurfaceExchange`]) binds a source into a
//! scheme and returns a reusable [`Transaction`]. Each successful
//! [`crate::Scheme::submit`] publishes one erased [`Claim`] per transaction.
//! The claim is extracted with [`Transaction::claim`] and settled by
//! [`Claim::consume`], [`Claim::discard`], or drop.

use crate::context::Context;
use crate::error::GoldyError;
use crate::scheme::{Lease, LeaseRenderTarget, Scheme, Submission, Transaction};
use crate::surface::Frame as SurfaceFrame;
use crate::swapchain_pool::{PresentLease, SwapchainPool};
use crate::types::{PresentMode, SurfaceConfig, TextureFormat};
use crate::Texture;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// Object-safe per-submission foreign handoff.
pub(crate) trait ClaimImpl: Send {
    fn consume(self: Box<Self>) -> Result<(), GoldyError>;
    fn discard(self: Box<Self>) -> Result<(), GoldyError>;
    fn discard_best_effort(self: Box<Self>);

    #[cfg(test)]
    fn debug_submit_timeline(&self) -> Option<crate::timeline::TimelineValue> {
        None
    }
}

/// Surface present claim: owns the acquired drawable until consume/discard/drop.
pub(crate) struct SurfaceClaimImpl {
    frame: Option<SurfaceFrame>,
}

impl SurfaceClaimImpl {
    pub(crate) fn new(frame: SurfaceFrame) -> Self {
        Self { frame: Some(frame) }
    }

    #[cfg(test)]
    pub(crate) fn submit_timeline(&self) -> Option<crate::timeline::TimelineValue> {
        self.frame.as_ref().and_then(|f| f.submit_timeline())
    }
}

impl ClaimImpl for SurfaceClaimImpl {
    fn consume(mut self: Box<Self>) -> Result<(), GoldyError> {
        let frame = self
            .frame
            .take()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("surface claim already settled")))?;
        frame.present().map(|_| ()).map_err(GoldyError::Backend)
    }

    fn discard(mut self: Box<Self>) -> Result<(), GoldyError> {
        if let Some(frame) = self.frame.take() {
            frame.cancel();
        }
        Ok(())
    }

    fn discard_best_effort(mut self: Box<Self>) {
        if let Some(frame) = self.frame.take() {
            frame.cancel();
        }
    }

    #[cfg(test)]
    fn debug_submit_timeline(&self) -> Option<crate::timeline::TimelineValue> {
        self.submit_timeline()
    }
}

impl Drop for SurfaceClaimImpl {
    fn drop(&mut self) {
        // Raw claim values may be dropped on submit failure after publish construction
        // (before wrapping in Claim / Submission). Cancel so Frame::drop cannot present.
        if let Some(frame) = self.frame.take() {
            frame.cancel();
        }
    }
}

/// Erased linear claim for one submission's foreign handoff.
pub struct Claim {
    pub(crate) implementation: Option<Box<dyn ClaimImpl>>,
}

impl Claim {
    pub(crate) fn from_impl(implementation: Box<dyn ClaimImpl>) -> Self {
        Self {
            implementation: Some(implementation),
        }
    }

    /// Perform the transaction's external handoff.
    ///
    /// Terminal even when it returns an error.
    pub fn consume(mut self) -> Result<(), GoldyError> {
        self.implementation
            .take()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("claim already settled")))?
            .consume()
    }

    /// Settle without intentionally performing the useful external operation.
    ///
    /// Terminal even when it returns an error.
    pub fn discard(mut self) -> Result<(), GoldyError> {
        self.implementation
            .take()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("claim already settled")))?
            .discard()
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        if let Some(claim) = self.implementation.take() {
            claim.discard_best_effort();
        }
    }
}

impl std::fmt::Debug for Claim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Claim")
            .field("settled", &self.implementation.is_none())
            .finish()
    }
}

/// Window-surface exchange: binds a scheme source to a drawable destination.
pub struct SurfaceExchange {
    pool: SwapchainPool,
}

impl SurfaceExchange {
    /// Create a surface exchange bound to `window` on `context`.
    ///
    /// Uses an in-flight depth of 3. Prefer [`Self::new_with_depth`] to choose pacing depth.
    pub fn new<W>(context: &Context, window: &W, config: SurfaceConfig) -> Result<Self, GoldyError>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        Self::new_with_depth(context, window, 3, config)
    }

    /// Create with an explicit client in-flight frame depth.
    pub fn new_with_depth<W>(
        context: &Context,
        window: &W,
        depth: u32,
        config: SurfaceConfig,
    ) -> Result<Self, GoldyError>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        let pool = SwapchainPool::new_with_config(context, window, depth, config).map_err(GoldyError::Backend)?;
        Ok(Self { pool })
    }

    /// Stable lease for scheme recording (one lease per exchange in v1).
    ///
    /// Prefer [`Self::bind`], [`Self::bind_render_target`], or [`Self::bind_destination`]
    /// for new code; this remains for callers that need the lease handle explicitly.
    pub fn lease(&self) -> PresentLease {
        self.pool.lease()
    }

    fn ensure_unbound(&self, scheme: &Scheme) -> Result<PresentLease, GoldyError> {
        let lease = self.pool.lease();
        if scheme.has_present_grant_for(&lease) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "SurfaceExchange: lease already bound in this scheme"
            )));
        }
        Ok(lease)
    }

    /// Record a stable texture → surface copy and return an erased transaction.
    ///
    /// Does not acquire a drawable. Each surface lease may be bound at most once per
    /// scheme; a second bind for the same lease returns an error rather than
    /// appending another copy that would share one claim slot.
    pub fn bind(&self, scheme: &mut Scheme, source: &Texture) -> Result<Transaction, GoldyError> {
        let lease = self.ensure_unbound(scheme)?;
        scheme.copy_texture_to_present(source, &lease);
        Ok(scheme.grant_present(&lease).transaction())
    }

    /// Record a stable offscreen render-target → surface copy and return a transaction.
    pub fn bind_render_target(
        &self,
        scheme: &mut Scheme,
        source: &Lease<LeaseRenderTarget>,
    ) -> Result<Transaction, GoldyError> {
        let lease = self.ensure_unbound(scheme)?;
        scheme.copy_to_present(source, &lease);
        Ok(scheme.grant_present(&lease).transaction())
    }

    /// Register present without a copy: the scheme writes the drawable directly.
    ///
    /// Returns the lease for [`Scheme`] node binding (for example `with_present`) and
    /// the erased transaction for claim extraction after submit.
    pub fn bind_destination(&self, scheme: &mut Scheme) -> Result<(PresentLease, Transaction), GoldyError> {
        let lease = self.ensure_unbound(scheme)?;
        let transaction = scheme.grant_present(&lease).transaction();
        Ok((lease, transaction))
    }

    /// Resize the underlying swapchain.
    ///
    /// Advances this exchange's backing generation so claims and retained variants
    /// published under the previous generation become stale.
    pub fn resize(&self, width: u32, height: u32) -> Result<(), GoldyError> {
        self.pool.resize(width, height).map_err(GoldyError::Backend)
    }

    pub fn set_present_mode(&self, mode: PresentMode) -> Result<(), GoldyError> {
        self.pool.set_present_mode(mode).map_err(GoldyError::Backend)
    }

    pub fn size(&self) -> (u32, u32) {
        self.pool.size()
    }

    pub fn width(&self) -> u32 {
        self.pool.width()
    }

    pub fn height(&self) -> u32 {
        self.pool.height()
    }

    pub fn format(&self) -> TextureFormat {
        self.pool.format()
    }

    /// Current backing generation (advances on resize / present-mode change).
    pub fn generation(&self) -> u64 {
        self.pool.generation()
    }

    /// Acquire the next drawable now (classic early-acquire timing).
    ///
    /// Pass the result to [`Scheme::submit_with_acquired_presents`] so submit does
    /// not wait again at the present partition. Prefer deferred acquire via plain
    /// [`Scheme::submit`] unless matching classic frame-start acquire timing.
    pub fn acquire_present(&self) -> Result<crate::swapchain_pool::AcquiredPresent, GoldyError> {
        let lease = self.pool.lease();
        self.pool.acquire_present(&lease).map_err(GoldyError::Backend)
    }

    /// Test-only: force the next deferred acquire on this exchange to fail once.
    #[cfg(test)]
    pub(crate) fn fail_next_acquire(&self) {
        self.pool.fail_next_acquire();
    }
}

impl Transaction {
    /// Scheme-unique present binding id for this transaction.
    pub fn binding_id(&self) -> u32 {
        self.binding_id
    }

    /// Current backing generation for this transaction's exchange.
    ///
    /// Resize and backing recreation advance this without changing binding identity.
    pub fn generation(&self) -> u64 {
        self.generation.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Remove this transaction's claim from a successful submission.
    ///
    /// Acquisition already happened inside [`Scheme::submit`].
    /// Fails when the exchange generation no longer matches the published claim
    /// (for example after resize between submit and claim).
    pub fn claim(&self, submission: &mut Submission) -> Result<Claim, GoldyError> {
        submission.take_claim(self.scheme_id, self.key, self.binding_id, self.generation())
    }
}
