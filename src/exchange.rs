//! Erased exchange transactions and claims.
//!
//! Concrete exchanges ([`SurfaceExchange`], [`MemoryExchange`]) bind a relationship into a
//! scheme and return a reusable transaction. Each successful [`crate::Scheme::submit`] may
//! publish a claim for relationships that settle outside graph execution.
//!
//! - Surface present: [`Transaction::claim`] → erased [`Claim`] → [`Claim::consume`] / discard
//! - Memory withdraw: [`WithdrawTransaction::claim`] → [`WithdrawClaim`] → [`WithdrawBytes`]
//! - Memory deposit: graph execution settles the upload; there is no claim

use crate::context::Context;
use crate::error::GoldyError;
use crate::parcel::Parcel;
use crate::scheme::{Lease, LeaseRenderTarget, Scheme, Submission, Transaction};
use crate::surface::Frame as SurfaceFrame;
use crate::swapchain_pool::{PresentLease, SwapchainPool};
use crate::texture::TextureCopyFootprint;
use crate::types::{PresentMode, SurfaceConfig, TextureFormat};
use crate::Buffer;
use crate::Texture;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ops::Deref;
use std::sync::Arc;

/// Object-safe per-submission foreign handoff (surface present today).
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

/// Erased linear claim for one submission's surface present handoff.
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
        if scheme.has_present_transaction_for(&lease) {
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
        Ok(scheme.register_present_exchange(&lease))
    }

    /// Record a stable offscreen render-target → surface copy and return a transaction.
    pub fn bind_render_target(
        &self,
        scheme: &mut Scheme,
        source: &Lease<LeaseRenderTarget>,
    ) -> Result<Transaction, GoldyError> {
        let lease = self.ensure_unbound(scheme)?;
        scheme.copy_to_present(source, &lease);
        Ok(scheme.register_present_exchange(&lease))
    }

    /// Register present without a copy: the scheme writes the drawable directly.
    ///
    /// Returns the lease for [`Scheme`] node binding (for example `with_present`) and
    /// the erased transaction for claim extraction after submit.
    pub fn bind_destination(&self, scheme: &mut Scheme) -> Result<(PresentLease, Transaction), GoldyError> {
        let lease = self.ensure_unbound(scheme)?;
        let transaction = scheme.register_present_exchange(&lease);
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
        submission.take_present_claim(self.scheme_id, self.key, self.binding_id, self.generation())
    }
}

/// CPU↔GPU memory exchange: withdrawals (readback) and deposits (upload).
#[derive(Clone)]
pub struct MemoryExchange {
    ctx: Context,
}

impl MemoryExchange {
    /// Create a memory exchange bound to `context`.
    pub fn new(ctx: &Context) -> Self {
        Self { ctx: ctx.clone() }
    }

    /// Bind a withdrawal over a buffer or texture deed parcel.
    ///
    /// Buffer parcels and texture parcels (`COPY_SRC`, storage-writable) are both accepted.
    /// Texture layout is captured at bind time. Each successful submit publishes one claim.
    pub fn bind_withdraw(&self, scheme: &mut Scheme, parcel: &Parcel) -> Result<WithdrawTransaction, GoldyError> {
        let _ = &self.ctx;
        scheme.register_withdraw(parcel)
    }

    /// Bind a deposit that copies staging bytes into a destination buffer parcel.
    ///
    /// Records copy topology once (destination offset 0 within the parcel). Each submission
    /// must [`DepositTransaction::write`] before [`Scheme::submit`]; graph execution settles
    /// the upload (no claim).
    pub fn bind_deposit_buffer(
        &self,
        scheme: &mut Scheme,
        destination: &Parcel,
        capacity: u64,
    ) -> Result<DepositTransaction, GoldyError> {
        self.bind_deposit_buffer_at(scheme, destination, 0, capacity)
    }

    /// Like [`Self::bind_deposit_buffer`], with an explicit byte offset into `destination`.
    pub fn bind_deposit_buffer_at(
        &self,
        scheme: &mut Scheme,
        destination: &Parcel,
        dst_offset: u64,
        capacity: u64,
    ) -> Result<DepositTransaction, GoldyError> {
        scheme.register_deposit_buffer(destination, dst_offset, capacity)
    }

    /// Bind a deposit that copies staging bytes into a texture region.
    ///
    /// Prefer a non-zero `src_row_pitch` (device footprint pitch) so the partition remains
    /// retainable without backend repacking.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_deposit_texture(
        &self,
        scheme: &mut Scheme,
        destination: &Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        capacity: u64,
        src_row_pitch: u32,
    ) -> Result<DepositTransaction, GoldyError> {
        scheme.register_deposit_texture(destination, x, y, width, height, capacity, src_row_pitch)
    }
}

impl std::fmt::Debug for MemoryExchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryExchange").finish_non_exhaustive()
    }
}

/// How withdraw staging contents are interpreted on the CPU.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WithdrawReadKind {
    Buffer,
    Texture(TextureCopyFootprint),
}

/// Stable withdraw relationship recorded in one [`Scheme`].
///
/// Reusable across submissions. Extract each submission's product with [`Self::claim`].
#[derive(Clone)]
pub struct WithdrawTransaction {
    pub(crate) scheme_id: u64,
    pub(crate) key: crate::scheme::ClaimKey,
    pub(crate) byte_size: u64,
    pub(crate) read_kind: WithdrawReadKind,
    pub(crate) ctx: Context,
}

impl WithdrawTransaction {
    /// Logical byte size of readable data for this withdrawal.
    pub fn byte_size(&self) -> u64 {
        self.byte_size
    }

    /// Texture copy footprint when this withdrawal reads a texture; `None` for buffers.
    pub fn texture_layout(&self) -> Option<TextureCopyFootprint> {
        match self.read_kind {
            WithdrawReadKind::Buffer => None,
            WithdrawReadKind::Texture(layout) => Some(layout),
        }
    }

    /// Remove this transaction's withdraw claim from a successful submission.
    pub fn claim(&self, submission: &mut Submission) -> Result<WithdrawClaim, GoldyError> {
        let slot = submission.take_withdraw_claim(self.scheme_id, self.key)?;
        Ok(WithdrawClaim {
            slot: Some(slot),
            byte_size: self.byte_size,
            read_kind: self.read_kind,
            ctx: self.ctx.clone(),
            ready_after: submission.timeline_value(),
        })
    }
}

impl std::fmt::Debug for WithdrawTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WithdrawTransaction")
            .field("scheme_id", &self.scheme_id)
            .field("key", &self.key)
            .field("byte_size", &self.byte_size)
            .finish_non_exhaustive()
    }
}

/// Staging handle published for one withdraw claim.
pub(crate) struct WithdrawSlot {
    pub(crate) staging: crate::backend::BufferHandle,
    pub(crate) pool: Arc<crate::scheme::WithdrawStagingPool>,
}

/// Linear claim for one submission's memory withdrawal.
pub struct WithdrawClaim {
    slot: Option<WithdrawSlot>,
    byte_size: u64,
    read_kind: WithdrawReadKind,
    ctx: Context,
    ready_after: crate::timeline::TimelineValue,
}

impl WithdrawClaim {
    /// Wait for the submission, read staging into CPU bytes, and return RAII-managed bytes.
    ///
    /// Terminal even when it returns an error (staging is recycled on failure after take).
    pub fn consume(mut self) -> Result<WithdrawBytes, GoldyError> {
        let slot = self
            .slot
            .take()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("withdraw claim already settled")))?;
        self.ctx.wait_until(self.ready_after)?;
        let byte_size = usize::try_from(self.byte_size)
            .map_err(|_| GoldyError::Backend(anyhow::anyhow!("withdraw readback byte size exceeds address space")))?;
        let mut bytes = vec![0u8; byte_size];
        let read_result = {
            let backend = self.ctx.device().inner.backend.lock().unwrap();
            match self.read_kind {
                WithdrawReadKind::Buffer => backend.read_readback_buffer(slot.staging, &mut bytes),
                WithdrawReadKind::Texture(layout) => {
                    backend.read_texture_readback_staging(slot.staging, layout, &mut bytes)
                }
            }
        };
        if let Err(e) = read_result {
            slot.pool.return_handle(slot.staging, self.ready_after);
            return Err(self.ctx.classify(e));
        }
        Ok(WithdrawBytes {
            bytes,
            handle: slot.staging,
            ready_after: self.ready_after,
            return_pool: slot.pool,
        })
    }

    /// Settle without reading bytes; recycle staging.
    pub fn discard(mut self) -> Result<(), GoldyError> {
        if let Some(slot) = self.slot.take() {
            slot.pool.return_handle(slot.staging, self.ready_after);
        }
        Ok(())
    }
}

impl Drop for WithdrawClaim {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.pool.return_handle(slot.staging, self.ready_after);
        }
    }
}

impl std::fmt::Debug for WithdrawClaim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WithdrawClaim")
            .field("settled", &self.slot.is_none())
            .field("byte_size", &self.byte_size)
            .finish_non_exhaustive()
    }
}

/// CPU-readable bytes from a consumed withdraw claim.
///
/// Dropping returns the staging buffer to the withdraw pool once the submission timeline retires.
pub struct WithdrawBytes {
    bytes: Vec<u8>,
    handle: crate::backend::BufferHandle,
    ready_after: crate::timeline::TimelineValue,
    return_pool: Arc<crate::scheme::WithdrawStagingPool>,
}

impl std::fmt::Debug for WithdrawBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WithdrawBytes")
            .field("len", &self.bytes.len())
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl Deref for WithdrawBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl WithdrawBytes {
    /// Owned copy of the readback bytes (staging still recycled on drop of this value).
    pub fn to_vec(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    /// Consume into an owned `Vec`, recycling staging immediately after copy.
    pub fn into_vec(mut self) -> Vec<u8> {
        std::mem::take(&mut self.bytes)
    }
}

impl Drop for WithdrawBytes {
    fn drop(&mut self) {
        self.return_pool.return_handle(self.handle, self.ready_after);
    }
}

/// Stable deposit relationship recorded in one [`Scheme`].
///
/// Topology (destination copy) is recorded at bind time. Each submission writes staging
/// bytes via [`Self::write`]; [`Scheme::submit`] settles the upload inside graph execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepositTransaction {
    pub(crate) scheme_id: u64,
    pub(crate) deposit_id: u32,
    pub(crate) capacity: u64,
}

impl DepositTransaction {
    /// Staging capacity declared for this deposit.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Stable declaration index within the owning [`Scheme`].
    pub fn id(&self) -> u32 {
        self.deposit_id
    }

    /// Write `data` into a settled (or newly allocated) physical staging parcel.
    ///
    /// Never waits: if every prior parcel is still in flight, allocates another.
    /// Must be called before [`Scheme::submit`] for every deposit referenced this frame.
    pub fn write(&self, scheme: &mut Scheme, offset: u64, data: &[u8]) -> Result<(), GoldyError> {
        scheme.stage_deposit(self.scheme_id, self.deposit_id, offset, data)
    }

    /// Write `data` at offset 0.
    pub fn write_bytes(&self, scheme: &mut Scheme, data: &[u8]) -> Result<(), GoldyError> {
        self.write(scheme, 0, data)
    }
}

/// Convenience: buffer-shaped destination helper for callers that hold a [`Buffer`].
impl MemoryExchange {
    /// Bind a full-buffer deposit into `destination.whole()` with `capacity` staging bytes.
    pub fn bind_deposit_into_buffer(
        &self,
        scheme: &mut Scheme,
        destination: &Buffer,
        capacity: u64,
    ) -> Result<DepositTransaction, GoldyError> {
        self.bind_deposit_buffer(scheme, destination.whole(), capacity)
    }
}
