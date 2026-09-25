//! Erased exchange transactions and claims.
//!
//! Concrete exchanges ([`SurfaceExchange`], [`MemoryExchange`]) bind a relationship into a
//! scheme and return a reusable transaction. Each successful [`crate::Scheme::submit`] may
//! produce a claim. Representation and delivery of claims are defined by the exchange:
//!
//! - Surface present: [`Transaction::claim`] → erased [`Claim`] → [`Claim::consume`] / discard,
//!   or `(&mut submission >> &transaction).take()` for the common consume path
//! - Host reads of parcels: `(&mut submission >> &parcel).take::<T>()` ([`crate::HostView`])
//! - Memory deposit: [`DepositTransaction::write`] (or `(&deposit << &data)?`) prepares an
//!   occurrence for this submission; submit claims it internally and graph execution consumes
//!   it at the copy dispatch. Recording ([`MemoryExchange::bind_deposit`]) survives across
//!   submissions; `<<` does not.

use crate::backend::BufferHandle;
use crate::buffer::StructuredBufferElement;
use crate::context::Context;
use crate::deposit_pool::DepositExchangePool;
use crate::error::GoldyError;
use crate::parcel::Parcel;
#[cfg(feature = "graphics")]
use crate::scheme::{Lease, LeaseRenderTarget, Transaction};
use crate::scheme::{Scheme, Submission};
#[cfg(feature = "graphics")]
use crate::surface::Frame as SurfaceFrame;
#[cfg(feature = "graphics")]
use crate::swapchain_pool::{PresentLease, SwapchainPool};
use crate::timeline::TimelineValue;
#[cfg(feature = "graphics")]
use crate::types::{PresentMode, SurfaceConfig, TextureFormat};
use crate::Buffer;
use crate::Texture;
#[cfg(feature = "graphics")]
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ops::{Shl, Shr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Object-safe per-submission foreign handoff (surface present today).
#[cfg(feature = "graphics")]
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
#[cfg(feature = "graphics")]
pub(crate) struct SurfaceClaimImpl {
    frame: Option<SurfaceFrame>,
}

#[cfg(feature = "graphics")]
impl SurfaceClaimImpl {
    pub(crate) fn new(frame: SurfaceFrame) -> Self {
        Self { frame: Some(frame) }
    }

    #[cfg(test)]
    pub(crate) fn submit_timeline(&self) -> Option<crate::timeline::TimelineValue> {
        self.frame.as_ref().and_then(|f| f.submit_timeline())
    }
}

#[cfg(feature = "graphics")]
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

#[cfg(feature = "graphics")]
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
#[cfg(feature = "graphics")]
pub struct Claim {
    pub(crate) implementation: Option<Box<dyn ClaimImpl>>,
}

#[cfg(feature = "graphics")]
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

#[cfg(feature = "graphics")]
impl Drop for Claim {
    fn drop(&mut self) {
        if let Some(claim) = self.implementation.take() {
            claim.discard_best_effort();
        }
    }
}

#[cfg(feature = "graphics")]
impl std::fmt::Debug for Claim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Claim")
            .field("settled", &self.implementation.is_none())
            .finish()
    }
}

/// Window-surface exchange: binds a scheme source to a drawable destination.
#[cfg(feature = "graphics")]
pub struct SurfaceExchange {
    pool: SwapchainPool,
}

#[cfg(feature = "graphics")]
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

#[cfg(feature = "graphics")]
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
    ///
    /// The common consume path is [`std::ops::Shr`] sugar: `(&mut submission >> &transaction).take()?`.
    /// This method remains for explicit multi-step settlement (`consume` / `discard`).
    pub fn claim(&self, submission: &mut Submission) -> Result<Claim, GoldyError> {
        submission.take_present_claim(self.scheme_id, self.key, self.binding_id, self.generation())
    }
}

/// Surface claim selected from a [`Submission`] by `submission >> &transaction`.
///
/// [`Self::take`] presents. Dropping a successfully selected wrapper discards the
/// claim (same as dropping a [`Claim`]). Selection does not touch any other claim
/// on the submission; the mutable borrow is required by operator semantics so the
/// submission remains available afterward.
#[cfg(feature = "graphics")]
#[must_use = "call take() to present, or drop to discard"]
pub struct PendingClaim {
    inner: Result<Claim, GoldyError>,
}

#[cfg(feature = "graphics")]
impl PendingClaim {
    /// Present the selected surface claim.
    ///
    /// Equivalent to [`Transaction::claim`] followed by [`Claim::consume`].
    /// Selection errors (wrong scheme, stale generation, already taken) and present
    /// errors both surface here.
    pub fn take(self) -> Result<(), GoldyError> {
        self.inner?.consume()
    }
}

#[cfg(feature = "graphics")]
impl std::fmt::Debug for PendingClaim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingClaim").field("ok", &self.inner.is_ok()).finish()
    }
}

#[cfg(feature = "graphics")]
impl Shr<&Transaction> for &mut Submission {
    type Output = PendingClaim;

    fn shr(self, transaction: &Transaction) -> Self::Output {
        PendingClaim {
            inner: transaction.claim(self),
        }
    }
}

/// CPU→GPU memory exchange: deposits (upload from merchant-owned host memory).
#[derive(Clone)]
pub struct MemoryExchange {
    ctx: Context,
}

impl MemoryExchange {
    /// Create a memory exchange bound to `context`.
    pub fn new(ctx: &Context) -> Self {
        Self { ctx: ctx.clone() }
    }

    /// Bind a deposit into `target`. Shape (buffer range vs texture region) is target data.
    ///
    /// Records copy topology once. Each submission must tender bytes via
    /// [`DepositTransaction::write`] or `(&deposit << &data)?` before [`Scheme::submit`];
    /// submit claims the occurrence internally and graph execution consumes it at the
    /// deposit copy dispatch.
    pub fn bind_deposit(
        &self,
        scheme: &mut Scheme,
        target: DepositTarget<'_>,
    ) -> Result<DepositTransaction, GoldyError> {
        let _ = &self.ctx;
        scheme.register_deposit(target)
    }

    /// Bind a device buffer parcel to an eager host-readable sink.
    ///
    /// The device-to-host copy is recorded in `scheme`, after prior writes to
    /// `source` according to normal graph and ledger ordering. Each successful
    /// submission therefore includes the copy; claiming the sink only waits for
    /// that submission and reads the already-populated host staging.
    pub fn bind_host_sink(&self, scheme: &mut Scheme, source: &Parcel) -> Result<HostSink, GoldyError> {
        if !std::sync::Arc::ptr_eq(&self.ctx.inner, &scheme.context().inner) {
            return Err(GoldyError::Validation(
                "MemoryExchange and Scheme belong to different contexts".into(),
            ));
        }
        scheme.register_host_sink(source)
    }
}

impl std::fmt::Debug for MemoryExchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryExchange").finish_non_exhaustive()
    }
}

/// Stable GPU-to-host memory exchange recorded in one [`Scheme`].
///
/// Unlike a direct parcel withdrawal, the readback copy is part of the recorded
/// graph. The sink has its own ledger stamp: a live [`crate::HostView`] prevents
/// a later submission from overwriting its staging, while dropping the view
/// permits replay immediately.
#[derive(Clone)]
pub struct HostSink {
    pub(crate) inner: Arc<HostSinkInner>,
}

pub(crate) struct HostSinkInner {
    pub(crate) scheme_id: u64,
    pub(crate) ctx: Context,
    pub(crate) handle: BufferHandle,
    pub(crate) byte_size: u64,
    pub(crate) stamp: Arc<crate::parcel::ParcelStamp>,
}

impl Drop for HostSinkInner {
    fn drop(&mut self) {
        self.stamp.mark_dead();
        if let Ok(mut backend) = self.ctx.runtime().inner.backend.lock() {
            backend.free_readback_buffer(self.handle);
        }
    }
}

impl HostSink {
    /// Select this sink's occurrence from `submission`.
    ///
    /// Selection is non-blocking. [`crate::PendingHostSinkRead::take`] performs
    /// the completion wait and returns the staged bytes.
    pub fn claim(&self, submission: &Submission) -> crate::PendingHostSinkRead {
        crate::PendingHostSinkRead::from_submission(submission, self)
    }

    /// Number of bytes copied into this sink on each submission.
    pub fn byte_size(&self) -> u64 {
        self.inner.byte_size
    }
}

impl std::fmt::Debug for HostSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostSink")
            .field("scheme_id", &self.inner.scheme_id)
            .field("byte_size", &self.inner.byte_size)
            .finish_non_exhaustive()
    }
}

impl Shr<&HostSink> for &mut Submission {
    type Output = crate::PendingHostSinkRead;

    fn shr(self, sink: &HostSink) -> Self::Output {
        sink.claim(self)
    }
}

/// Destination of a memory-exchange deposit (buffer range or texture region).
pub enum DepositTarget<'a> {
    /// Copy staging bytes into a buffer parcel, starting at `dst_offset` within the parcel.
    Buffer {
        destination: &'a Parcel,
        dst_offset: u64,
        capacity: u64,
    },
    /// Copy staging bytes into a texture region.
    Texture {
        destination: &'a Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        capacity: u64,
        src_row_pitch: u32,
    },
}

impl<'a> DepositTarget<'a> {
    /// Whole-parcel buffer deposit with `capacity` staging bytes (offset 0).
    pub fn buffer(destination: &'a Parcel, capacity: u64) -> Self {
        Self::Buffer {
            destination,
            dst_offset: 0,
            capacity,
        }
    }

    /// Buffer deposit sized for `count` structured elements of `T`.
    ///
    /// Capacity uses [`StructuredBufferElement::gpu_element_stride`], which is the
    /// packed Slang ABI stride for [`struct@crate::GpuType`] (not `size_of::<T>()`).
    pub fn buffer_elements<T: StructuredBufferElement>(destination: &'a Parcel, count: u64) -> Self {
        Self::buffer(destination, count.saturating_mul(T::gpu_element_stride() as u64))
    }

    /// Buffer deposit starting at `dst_offset` within `destination`.
    pub fn buffer_at(destination: &'a Parcel, dst_offset: u64, capacity: u64) -> Self {
        Self::Buffer {
            destination,
            dst_offset,
            capacity,
        }
    }

    /// Texture-region deposit. Prefer a non-zero `src_row_pitch` (device footprint pitch).
    #[allow(clippy::too_many_arguments)]
    pub fn texture(
        destination: &'a Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        capacity: u64,
        src_row_pitch: u32,
    ) -> Self {
        Self::Texture {
            destination,
            x,
            y,
            width,
            height,
            capacity,
            src_row_pitch,
        }
    }
}

/// Shared state for one recorded deposit relationship.
pub(crate) struct DepositBinding {
    pub(crate) scheme_id: u64,
    pub(crate) deposit_id: u32,
    pub(crate) capacity: u64,
    pub(crate) affinity: u64,
    pub(crate) pending: Mutex<Option<BufferHandle>>,
    pub(crate) ctx: Context,
    pub(crate) pool: Arc<DepositExchangePool>,
    pub(crate) scheme_alive: Arc<AtomicBool>,
}

impl DepositBinding {
    pub(crate) fn discard_pending(&self) {
        if let Some(handle) = self.pending.lock().unwrap_or_else(|e| e.into_inner()).take() {
            self.pool.return_handle(handle, 0);
        }
        self.scheme_alive.store(false, Ordering::Release);
    }
}

/// Stable deposit relationship recorded in one [`Scheme`].
///
/// Topology (destination copy) is recorded at bind time. Each submission tenders staging
/// bytes via [`Self::write`] or `(&deposit << &data)?`; [`Scheme::submit`] claims the
/// occurrence internally and graph execution consumes it at the copy dispatch.
///
/// `<<` applies to **this submission only**. [`MemoryExchange::bind_deposit`] is what
/// survives across submissions. Offset and partial fills stay on [`Self::write`].
///
/// ```ignore
/// (&deposit << &uniforms)?;      // StructuredBufferElement value
/// (&deposit << vertices.as_slice())?;
/// (&deposit << pixels.as_slice())?; // raw `[u8]`
/// ```
#[derive(Clone)]
pub struct DepositTransaction {
    pub(crate) inner: Arc<DepositBinding>,
}

impl DepositTransaction {
    /// Staging capacity declared for this deposit.
    pub fn capacity(&self) -> u64 {
        self.inner.capacity
    }

    /// Stable declaration index within the owning [`Scheme`].
    pub fn id(&self) -> u32 {
        self.inner.deposit_id
    }

    /// Write `data` into a settled (or newly allocated) physical staging backing.
    ///
    /// Never waits: if every prior backing is still in flight, allocates another.
    /// Must be called before [`Scheme::submit`] for every deposit referenced this frame.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), GoldyError> {
        if !self.inner.scheme_alive.load(Ordering::Acquire) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "DepositTransaction belongs to a dropped scheme"
            )));
        }
        if offset.saturating_add(data.len() as u64) > self.inner.capacity {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "deposit write: [{offset}..{}] exceeds declaration size {}",
                offset + data.len() as u64,
                self.inner.capacity
            )));
        }
        let mut pending = self.inner.pending.lock().unwrap_or_else(|e| e.into_inner());
        let handle = if let Some(handle) = *pending {
            handle
        } else {
            let handle = self
                .inner
                .pool
                .take_or_alloc(&self.inner.ctx, self.inner.capacity, self.inner.affinity)?;
            *pending = Some(handle);
            handle
        };
        self.inner.pool.write_handle(&self.inner.ctx, handle, offset, data)
    }

    /// Write typed elements, packed for [`struct@crate::GpuType`] the same way as
    /// [`crate::Runtime::acquire_buffer_with_data`].
    pub fn write_data<T: StructuredBufferElement>(&self, offset: u64, data: &[T]) -> Result<(), GoldyError> {
        let encoded = T::gpu_encode_slice(data);
        self.write(offset, encoded.as_ref())
    }

    /// Write `data` at offset 0.
    pub fn write_bytes(&self, data: &[u8]) -> Result<(), GoldyError> {
        self.write(0, data)
    }
}

impl std::fmt::Debug for DepositTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DepositTransaction")
            .field("scheme_id", &self.inner.scheme_id)
            .field("deposit_id", &self.inner.deposit_id)
            .field("capacity", &self.inner.capacity)
            .finish_non_exhaustive()
    }
}

impl PartialEq for DepositTransaction {
    fn eq(&self, other: &Self) -> bool {
        self.inner.scheme_id == other.inner.scheme_id && self.inner.deposit_id == other.inner.deposit_id
    }
}

impl Eq for DepositTransaction {}

impl std::hash::Hash for DepositTransaction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.scheme_id.hash(state);
        self.inner.deposit_id.hash(state);
    }
}

impl<T: StructuredBufferElement> Shl<&T> for &DepositTransaction {
    type Output = Result<(), GoldyError>;

    fn shl(self, data: &T) -> Self::Output {
        self.write_data(0, std::slice::from_ref(data))
    }
}

impl<T: StructuredBufferElement> Shl<&[T]> for &DepositTransaction {
    type Output = Result<(), GoldyError>;

    fn shl(self, data: &[T]) -> Self::Output {
        self.write_data(0, data)
    }
}

impl Shl<&[u8]> for &DepositTransaction {
    type Output = Result<(), GoldyError>;

    fn shl(self, data: &[u8]) -> Self::Output {
        self.write(0, data)
    }
}

/// Linear per-submission deposit claim. Claimed at submit, consumed at the copy dispatch.
pub(crate) struct DepositClaim {
    handle: BufferHandle,
    capacity: u64,
    pool: Arc<DepositExchangePool>,
    consumed: bool,
}

impl DepositClaim {
    pub(crate) fn new(handle: BufferHandle, capacity: u64, pool: Arc<DepositExchangePool>) -> Self {
        Self {
            handle,
            capacity,
            pool,
            consumed: false,
        }
    }

    pub(crate) fn resolved(&self) -> crate::task_graph::ResolvedDeposit {
        crate::task_graph::ResolvedDeposit {
            parent: self.handle,
            offset: 0,
            len: self.capacity,
        }
    }

    /// Settle the claim and park the backing until `ready_after`.
    pub(crate) fn consume(mut self, ready_after: TimelineValue) {
        self.consumed = true;
        self.pool.return_handle(self.handle, ready_after);
    }
}

impl Drop for DepositClaim {
    fn drop(&mut self) {
        if !self.consumed {
            self.pool.return_handle(self.handle, 0);
        }
    }
}

/// Consume each deposit claim in `ids` at timeline `tv`.
pub(crate) fn consume_deposit_claims(
    ids: impl IntoIterator<Item = u32>,
    claims: &mut std::collections::HashMap<u32, Option<DepositClaim>>,
    tv: TimelineValue,
) {
    for id in ids {
        if let Some(slot) = claims.get_mut(&id) {
            if let Some(claim) = slot.take() {
                claim.consume(tv);
            }
        }
    }
}

/// Park every still-unconsumed claim at `tv` (submit failure / partial enqueue).
pub(crate) fn park_unconsumed_deposit_claims(
    claims: &mut std::collections::HashMap<u32, Option<DepositClaim>>,
    tv: TimelineValue,
) {
    for slot in claims.values_mut() {
        if let Some(claim) = slot.take() {
            claim.consume(tv);
        }
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
        self.bind_deposit(scheme, DepositTarget::buffer(destination.whole(), capacity))
    }
}
