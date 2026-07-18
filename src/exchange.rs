//! Erased exchange transactions and claims.
//!
//! A concrete exchange (for example [`SurfaceExchange`]) binds a source into a
//! scheme and returns a reusable [`Transaction`]. Each successful
//! [`crate::Scheme::submit`] publishes one erased [`Claim`] per transaction.
//! The claim is extracted with [`Transaction::claim`] and settled by
//! [`Claim::consume`], [`Claim::discard`], or drop.

use crate::context::Context;
use crate::error::GoldyError;
use crate::scheme::{ClaimKey, Scheme, Submission, Transaction};
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
    pub fn new<W>(context: &Context, window: &W, config: SurfaceConfig) -> Result<Self, GoldyError>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        let pool = SwapchainPool::new_with_config(context, window, 1, config).map_err(GoldyError::Backend)?;
        Ok(Self { pool })
    }

    /// Stable lease for scheme recording (one lease per exchange in v1).
    pub fn lease(&self) -> PresentLease {
        self.pool.lease()
    }

    /// Record a stable texture → surface relationship and return an erased transaction.
    ///
    /// Does not acquire a drawable. Each surface lease may be bound at most once per
    /// scheme; a second `bind` for the same lease returns an error rather than
    /// appending another copy that would share one claim slot.
    pub fn bind(&self, scheme: &mut Scheme, source: &Texture) -> Result<Transaction, GoldyError> {
        let lease = self.pool.lease();
        if scheme.has_present_grant_for(&lease) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "SurfaceExchange::bind: lease already bound in this scheme"
            )));
        }
        scheme.copy_texture_to_present(source, &lease);
        let grant = scheme.grant_present(&lease);
        Ok(Transaction {
            scheme_id: grant.scheme_id,
            key: ClaimKey {
                present_idx: grant.grant_id(),
            },
            binding_id: grant.binding_id,
        })
    }

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

    /// Access the underlying swapchain pool (compatibility / migration).
    pub fn pool(&self) -> &SwapchainPool {
        &self.pool
    }
}

impl Transaction {
    /// Scheme-unique present binding id for this transaction.
    pub fn binding_id(&self) -> u32 {
        self.binding_id
    }

    /// Remove this transaction's claim from a successful submission.
    ///
    /// Acquisition already happened inside [`Scheme::submit`].
    pub fn claim(&self, submission: &mut Submission) -> Result<Claim, GoldyError> {
        submission.take_claim(self.scheme_id, self.key, self.binding_id)
    }
}
