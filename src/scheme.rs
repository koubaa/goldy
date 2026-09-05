//! Retained scheme — goldy's primary submission unit.
//!
//! A [`Scheme`] is a set of dispatches and precedences, first-class, retained across
//! submissions. Schemes persist across frames. Structural mutation (new nodes, bindings)
//! drops retained command lists. Params-only mutation (pipeline, user slots, dispatch
//! dims) keeps other partitions' retained lists and re-records only partitions whose
//! baked payload changed. A clean scheme resubmits with zero recording cost.
//!
//! **Construction**: `Scheme::new(&ctx)` — bound to one context for its lifetime.
//! **Submission**: `scheme.submit()` — submits, and submits again, using the retained path
//! when clean.

#[cfg(feature = "graphics")]
use crate::backend::RenderCommand;
use crate::backend::{BufferHandle, GpuCommand};
use crate::buffer::{Allocation, BufferSource};
use crate::context::Context;
use crate::cpu_dispatch::{CpuBindingExec, CpuDispatchExec, CpuMain};
use crate::error::GoldyError;
use crate::handles::TextureHandle;
use crate::parcel::Parcel;
#[cfg(feature = "graphics")]
use crate::render_target::RenderTarget;
use crate::retained_pool::StampedParcel;
#[cfg(feature = "graphics")]
use crate::swapchain_pool::{AcquiredPresent, PresentLease, SwapchainPool};
use crate::task_graph::cross_submit::ResourceKey;
#[cfg(feature = "graphics")]
use crate::task_graph::cross_submit::ResourceKeyMap;
#[cfg(feature = "graphics")]
use crate::task_graph::DeferredPresentAcquire;
use crate::task_graph::IrSubmitState;
#[cfg(feature = "graphics")]
use crate::task_graph::ResolvedPresentSlot;
use crate::task_graph::ResourceId;
#[cfg(feature = "graphics")]
use crate::task_graph::ShaderResourceSlot;
#[cfg(feature = "graphics")]
use crate::task_graph::PRESENT_LEASE_SLOT_PLACEHOLDER;
use crate::task_graph::{DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding, TaskNode};
use crate::texture::TextureCopyFootprint;
use crate::timeline::TimelineValue;
#[cfg(feature = "graphics")]
use crate::timeline::{PromiseResolver, TimelinePromise};
use crate::types::{
    BufferFlags, DispatchShape, ResourceAccess, ResourceHandle, TextureFlags, TextureFormat, TextureKind,
};
#[cfg(feature = "graphics")]
use crate::types::{DepthFormat, IndexFormat};
use crate::validation_env;
#[cfg(feature = "graphics")]
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_SCHEME_ID: AtomicU64 = AtomicU64::new(1);

/// Per-withdraw staging buffer pool with scheme-lifetime ownership.
///
/// Returned staging buffers are stamped with the submission timeline that must retire
/// before reuse (`ready_after`). This matches transient-pool epoch gating: dropping a
/// [`Submission`] without consuming does not require a CPU wait, but in-flight
/// staging is not handed to a later submit until `gpu_progress` passes that stamp.
enum WithdrawStagingAllocSpec {
    Buffer { byte_size: u64 },
    Texture { layout: TextureCopyFootprint },
}

/// Staging buffer parked in a withdraw pool until its submission timeline retires.
struct StampedStagingBuffer {
    handle: BufferHandle,
    ready_after: TimelineValue,
}

pub(crate) struct WithdrawStagingPool {
    handles: Mutex<Vec<StampedStagingBuffer>>,
    alloc_spec: WithdrawStagingAllocSpec,
    ctx: Context,
    scheme_alive: AtomicBool,
}

impl WithdrawStagingPool {
    fn new_buffer(ctx: &Context, byte_size: u64) -> Arc<Self> {
        Arc::new(Self {
            handles: Mutex::new(Vec::new()),
            alloc_spec: WithdrawStagingAllocSpec::Buffer { byte_size },
            ctx: ctx.clone(),
            scheme_alive: AtomicBool::new(true),
        })
    }

    fn new_texture(ctx: &Context, layout: TextureCopyFootprint) -> Arc<Self> {
        Arc::new(Self {
            handles: Mutex::new(Vec::new()),
            alloc_spec: WithdrawStagingAllocSpec::Texture { layout },
            ctx: ctx.clone(),
            scheme_alive: AtomicBool::new(true),
        })
    }

    fn take_or_alloc(
        &self,
        backend: &mut dyn crate::backend::GpuBackend,
        device: crate::backend::DeviceHandle,
    ) -> Result<BufferHandle, GoldyError> {
        let ctx = self.ctx.backend_handle();
        let progress = backend.gpu_progress(ctx);
        let handle = {
            let mut pool = self.handles.lock().unwrap_or_else(|e| e.into_inner());
            pool.iter()
                .position(|entry| entry.ready_after <= progress)
                .map(|pos| pool.swap_remove(pos).handle)
        };
        match self.alloc_spec {
            WithdrawStagingAllocSpec::Buffer { byte_size } => {
                if let Some(handle) = handle {
                    if validation_env::scheme_validation_enabled() {
                        let cap = backend.buffer_size(handle);
                        if cap < byte_size {
                            return Err(GoldyError::Backend(anyhow::anyhow!(
                                "recycled withdraw staging buffer capacity {cap} is smaller than withdraw byte size {byte_size}"
                            )));
                        }
                    }
                    Ok(handle)
                } else {
                    backend
                        .alloc_readback_buffer(device, byte_size)
                        .map_err(|e| self.ctx.classify(e))
                }
            }
            WithdrawStagingAllocSpec::Texture { layout } => {
                if let Some(handle) = handle {
                    if validation_env::scheme_validation_enabled() {
                        let cap = backend.buffer_size(handle);
                        if cap < layout.staging_bytes {
                            return Err(GoldyError::Backend(anyhow::anyhow!(
                                "recycled texture withdraw staging capacity {cap} is smaller than required {}",
                                layout.staging_bytes
                            )));
                        }
                    }
                    Ok(handle)
                } else {
                    backend
                        .alloc_texture_readback_staging(device, layout)
                        .map_err(|e| self.ctx.classify(e))
                }
            }
        }
    }

    pub(crate) fn return_handle(&self, handle: BufferHandle, ready_after: TimelineValue) {
        if self.scheme_alive.load(Ordering::Acquire) {
            self.handles
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(StampedStagingBuffer { handle, ready_after });
        } else {
            let _ = self.ctx.wait_until(ready_after);
            let mut backend = self.ctx.device().inner.backend.lock().unwrap();
            backend.free_readback_buffer(handle);
        }
    }

    fn mark_scheme_dropped_and_drain(&self) {
        self.scheme_alive.store(false, Ordering::Release);
        let mut pool = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(max_ready) = pool.iter().map(|entry| entry.ready_after).max() {
            let _ = self.ctx.wait_until(max_ready);
        }
        let mut backend = self.ctx.device().inner.backend.lock().unwrap();
        for entry in pool.drain(..) {
            backend.free_readback_buffer(entry.handle);
        }
    }
}

impl fmt::Debug for WithdrawStagingPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WithdrawStagingPool")
            .field("scheme_alive", &self.scheme_alive.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

/// Cloneable GPU submission identity and timeline (no exchange claims).
#[derive(Debug, Clone)]
pub(crate) struct SubmissionHandle {
    core: Arc<SubmissionCore>,
}

#[derive(Debug)]
struct SubmissionCore {
    scheme_id: u64,
    timeline: TimelineValue,
}

impl SubmissionHandle {
    /// Timeline value for this submission (crate-internal clearing clock).
    pub fn timeline_value(&self) -> TimelineValue {
        self.core.timeline
    }

    /// Block until this submission's GPU work has completed.
    pub fn wait(&self, ctx: &Context) -> Result<(), GoldyError> {
        ctx.wait_until(self.core.timeline)?;
        Ok(())
    }

    pub(crate) fn scheme_id(&self) -> u64 {
        self.core.scheme_id
    }
}

impl From<SubmissionHandle> for TimelineValue {
    fn from(handle: SubmissionHandle) -> Self {
        handle.timeline_value()
    }
}

/// Dense index of an exchange claim slot on a [`Submission`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ClaimKey {
    #[cfg(feature = "graphics")]
    Present {
        present_idx: u32,
    },
    Withdraw {
        withdraw_idx: u32,
    },
}

/// Stable erased present relationship recorded in one [`Scheme`].
///
/// Reusable across submissions. Extract each submission's product with [`Self::claim`].
#[cfg(feature = "graphics")]
#[derive(Clone)]
pub struct Transaction {
    pub(crate) scheme_id: u64,
    pub(crate) key: ClaimKey,
    /// Scheme-unique present binding ([`ResourceId::PresentLease`]).
    pub(crate) binding_id: u32,
    /// Live exchange backing generation (shared with the owning pool).
    pub(crate) generation: Arc<std::sync::atomic::AtomicU64>,
}

#[cfg(feature = "graphics")]
impl fmt::Debug for Transaction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Transaction")
            .field("scheme_id", &self.scheme_id)
            .field("key", &self.key)
            .field("binding_id", &self.binding_id)
            .field(
                "generation",
                &self.generation.load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

/// Unique receipt returned by [`Scheme::submit`].
///
/// Owns untaken exchange claims (present and withdraw). Dropping this receipt
/// discards every claim that has not been taken.
///
/// GPU completion is observed via [`Self::is_settled`] / [`Self::wait_until_settled`]
/// — not via raw timeline values.
pub struct Submission {
    handle: SubmissionHandle,
    /// Context that submitted this work (for settlement waits).
    ctx: Context,
    /// Present claim slots (parallel to [`Self::claim_bindings`] / generations).
    #[cfg(feature = "graphics")]
    present_claims: Vec<Mutex<Option<Box<dyn crate::exchange::ClaimImpl>>>>,
    /// Scheme-unique present binding id for each present claim slot.
    #[cfg(feature = "graphics")]
    claim_bindings: Vec<u32>,
    /// Pool generation snapshotted when each present claim's drawable was acquired.
    #[cfg(feature = "graphics")]
    claim_generations: Vec<u64>,
    /// Withdraw claim slots; taken by [`crate::exchange::WithdrawTransaction::claim`].
    withdraw_claims: Vec<Mutex<Option<crate::exchange::WithdrawSlot>>>,
}

impl Drop for Submission {
    fn drop(&mut self) {
        let ready_after = self.handle.timeline_value();
        for claim_mutex in &self.withdraw_claims {
            if let Ok(mut slot) = claim_mutex.lock() {
                if let Some(withdraw) = slot.take() {
                    withdraw.pool.return_handle(withdraw.staging, ready_after);
                }
            }
        }
        #[cfg(feature = "graphics")]
        for claim_mutex in &self.present_claims {
            if let Ok(mut slot) = claim_mutex.lock() {
                if let Some(claim) = slot.take() {
                    claim.discard_best_effort();
                }
            }
        }
    }
}

impl fmt::Debug for Submission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("Submission");
        debug
            .field("scheme_id", &self.handle.scheme_id())
            .field("settled", &self.is_settled());
        #[cfg(feature = "graphics")]
        debug.field("present_claims", &self.present_claims.len());
        debug.field("withdraw_claims", &self.withdraw_claims.len()).finish()
    }
}

impl Submission {
    /// Cloneable timeline / scheme identity (does not share claim ownership).
    #[cfg(test)]
    pub(crate) fn handle(&self) -> SubmissionHandle {
        self.handle.clone()
    }

    /// Crate-internal clearing epoch for this submission.
    pub(crate) fn timeline_value(&self) -> TimelineValue {
        self.handle.timeline_value()
    }

    /// True when this submission's GPU work has retired.
    pub fn is_settled(&self) -> bool {
        self.ctx.gpu_progress() >= self.handle.timeline_value()
    }

    /// Block until this submission's GPU work has completed.
    pub fn wait_until_settled(&self) -> Result<(), GoldyError> {
        self.handle.wait(&self.ctx)
    }

    /// Like [`Self::wait_until_settled`] but returns `Ok(false)` on timeout.
    pub fn wait_until_settled_timeout(&self, timeout_ms: u32) -> Result<bool, GoldyError> {
        match self.ctx.wait_until_timeout(self.handle.timeline_value(), timeout_ms) {
            Ok(()) => Ok(true),
            Err(GoldyError::SubmitTimeout) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Block until this submission's GPU work has completed (caller-supplied context).
    #[cfg(test)]
    pub(crate) fn wait(&self, ctx: &Context) -> Result<(), GoldyError> {
        self.handle.wait(ctx)
    }

    #[cfg(feature = "graphics")]
    pub(crate) fn take_present_claim(
        &mut self,
        scheme_id: u64,
        key: ClaimKey,
        binding_id: u32,
        generation: u64,
    ) -> Result<crate::exchange::Claim, GoldyError> {
        if self.handle.scheme_id() != scheme_id {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "Transaction belongs to a different scheme than this submission"
            )));
        }
        let ClaimKey::Present { present_idx } = key else {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "present claim key required for surface transaction"
            )));
        };
        let idx = present_idx as usize;
        let expected_binding = self.claim_bindings.get(idx).copied().ok_or_else(|| {
            GoldyError::Backend(anyhow::anyhow!(
                "claim index {} out of range for submission ({} claims)",
                idx,
                self.present_claims.len()
            ))
        })?;
        if expected_binding != binding_id {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "transaction binding {} does not match claim slot binding {}",
                binding_id,
                expected_binding
            )));
        }
        let expected_generation = self.claim_generations.get(idx).copied().ok_or_else(|| {
            GoldyError::Backend(anyhow::anyhow!(
                "claim generation index {} out of range for submission",
                idx
            ))
        })?;
        if expected_generation != generation {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "transaction generation {generation} is stale for claim published at generation {expected_generation}"
            )));
        }
        let claim_mutex = self.present_claims.get(idx).ok_or_else(|| {
            GoldyError::Backend(anyhow::anyhow!(
                "claim index {} out of range for submission ({} claims)",
                idx,
                self.present_claims.len()
            ))
        })?;
        let mut slot = claim_mutex.lock().unwrap_or_else(|e| e.into_inner());
        let implementation = slot
            .take()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("claim already consumed for this submission")))?;
        Ok(crate::exchange::Claim::from_impl(implementation))
    }

    pub(crate) fn take_withdraw_claim(
        &mut self,
        scheme_id: u64,
        key: ClaimKey,
    ) -> Result<crate::exchange::WithdrawSlot, GoldyError> {
        if self.handle.scheme_id() != scheme_id {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "WithdrawTransaction belongs to a different scheme than this submission"
            )));
        }
        let withdraw_idx = match key {
            ClaimKey::Withdraw { withdraw_idx } => withdraw_idx,
            #[cfg(feature = "graphics")]
            ClaimKey::Present { .. } => {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "withdraw claim key required for memory withdrawal"
                )));
            }
        };
        let idx = withdraw_idx as usize;
        let claim_mutex = self.withdraw_claims.get(idx).ok_or_else(|| {
            GoldyError::Backend(anyhow::anyhow!(
                "withdraw index {} out of range for submission ({} withdrawals)",
                idx,
                self.withdraw_claims.len()
            ))
        })?;
        let mut slot = claim_mutex.lock().unwrap_or_else(|e| e.into_inner());
        slot.take()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("withdraw claim already consumed for this submission")))
    }

    /// Submit timeline stamped on the acquired present frame, if still held.
    #[cfg(all(test, feature = "graphics"))]
    pub(crate) fn present_frame_submit_timeline(&self, idx: usize) -> Option<TimelineValue> {
        let claim_mutex = self.present_claims.get(idx)?;
        let slot = claim_mutex.lock().unwrap_or_else(|e| e.into_inner());
        slot.as_ref()?.debug_submit_timeline()
    }
}

impl From<&Submission> for TimelineValue {
    fn from(submission: &Submission) -> Self {
        submission.timeline_value()
    }
}

#[cfg(feature = "graphics")]
struct PresentBinding {
    pool: Arc<crate::swapchain_pool::SwapchainPoolInner>,
    pool_lease_id: u32,
}

#[cfg(feature = "graphics")]
struct PresentTransactionInfo {
    /// Scheme-unique binding id used in IR as [`ResourceId::PresentLease`].
    binding_id: u32,
    pool: Arc<crate::swapchain_pool::SwapchainPoolInner>,
    /// Pool-local lease id for eager-acquire provenance checks.
    pool_lease_id: u32,
}

/// Validate registered present exchanges against real drawable IR accesses.
#[cfg(feature = "graphics")]
fn validate_present_exchange_bindings(
    ir: &GraphIR,
    present_transactions: &[PresentTransactionInfo],
) -> Result<(), GoldyError> {
    use std::collections::{HashMap, HashSet};

    let registered: HashSet<u32> = present_transactions.iter().map(|t| t.binding_id).collect();

    let mut first_access: HashMap<u32, NodeAccess> = HashMap::new();
    let mut has_write: HashSet<u32> = HashSet::new();
    let mut accessed: HashSet<u32> = HashSet::new();

    for node in &ir.nodes {
        for b in &node.bindings {
            let ResourceId::PresentLease(id) = b.resource else {
                continue;
            };
            accessed.insert(id);
            if b.access.writes() {
                has_write.insert(id);
            }
            first_access.entry(id).or_insert(b.access);
        }
    }

    for tx in present_transactions {
        if !accessed.contains(&tx.binding_id) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "present exchange binding {} registered but scheme never accesses its PresentLease",
                tx.binding_id
            )));
        }
        if !has_write.contains(&tx.binding_id) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "present exchange binding {} has no Write/Overwrite/ReadWrite access to its PresentLease",
                tx.binding_id
            )));
        }
        match first_access.get(&tx.binding_id) {
            Some(NodeAccess::Write | NodeAccess::Overwrite) => {}
            Some(other) => {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "present exchange binding {}: first PresentLease access must be Write or Overwrite, got {:?}",
                    tx.binding_id,
                    other
                )));
            }
            None => unreachable!("accessed set implies first_access entry"),
        }
    }

    for id in accessed {
        if !registered.contains(&id) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PresentLease binding {} accessed in IR but no exchange transaction registered",
                id
            )));
        }
    }

    Ok(())
}

/// Parcel stamps read by a present easement for `binding_id` (copy-to-present sources).
#[cfg(feature = "graphics")]
fn present_easement_source_stamps(
    ir: &GraphIR,
    binding_id: u32,
    resource_stamps: &ResourceKeyMap<Arc<crate::parcel::ParcelStamp>>,
) -> Vec<Arc<crate::parcel::ParcelStamp>> {
    let dst = ResourceId::PresentLease(binding_id);
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for node in &ir.nodes {
        let key = match &node.kind {
            NodeKind::CopyTexture { src, dst: d, .. } if *d == dst => Some(ResourceKey::Texture(*src)),
            NodeKind::CopyRenderTarget { src, dst: d, .. } if *d == dst => Some(ResourceKey::RenderTarget(*src)),
            _ => None,
        };
        let Some(key) = key else {
            continue;
        };
        if let Some(stamp) = resource_stamps.get(&key) {
            let ptr = Arc::as_ptr(stamp);
            if seen.insert(ptr) {
                out.push(Arc::clone(stamp));
            }
        } else {
            tracing::warn!(
                target: "goldy::scheme",
                binding_id,
                ?key,
                "present easement: copy source has no registered stamp; WAR hazard not tracked"
            );
        }
    }
    out
}

#[cfg(feature = "graphics")]
fn claim_present_easement_promises(
    ir: &GraphIR,
    present_transactions: &[PresentTransactionInfo],
    resource_stamps: &ResourceKeyMap<Arc<crate::parcel::ParcelStamp>>,
) -> Vec<Mutex<Option<PromiseResolver>>> {
    let mut resolvers = Vec::with_capacity(present_transactions.len());
    for tx in present_transactions {
        let (promise, resolver) = TimelinePromise::new();
        for stamp in present_easement_source_stamps(ir, tx.binding_id, resource_stamps) {
            stamp.push_pending(promise.clone());
        }
        resolvers.push(Mutex::new(Some(resolver)));
    }
    resolvers
}

enum WithdrawSource {
    Buffer {
        source: BufferHandle,
        src_offset: u64,
        #[allow(dead_code)]
        source_backing: Arc<Allocation>,
        byte_size: u64,
    },
    Texture {
        source: TextureHandle,
        #[allow(dead_code)]
        source_backing: crate::texture::TextureBacking,
        layout: TextureCopyFootprint,
    },
}

struct WithdrawInfo {
    source: WithdrawSource,
    staging_pool: Arc<WithdrawStagingPool>,
}

/// Stable index of a scheme-held lease declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseId(pub(crate) u32);

/// Stable identity of one recorded scheme node, returned when the node is finalized.
///
/// Nodes are only ever appended to a scheme, so an id stays valid — and keeps pointing at
/// the same dispatch site — for the life of the scheme it came from. That makes it usable
/// as a key for per-site history across frames (see
/// [Shader Specialization Prediction](https://koubaa.github.io/goldy/design/shader-specialization.html)),
/// not just as an argument to [`Scheme::set_node_pipeline`] and its siblings.
///
/// Ids carry the originating scheme's identity, so passing one to a different scheme is
/// rejected instead of silently addressing an unrelated node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId {
    scheme_id: u64,
    index: u32,
}

impl NodeId {
    /// Position of this node in its scheme's recording order.
    pub fn index(&self) -> usize {
        self.index as usize
    }
}

/// Marker type for texture leases acquired via [`Scheme::lease_texture`].
pub struct LeaseTexture;

/// Marker type for buffer leases acquired via [`Scheme::lease_buffer`].
pub struct LeaseBuffer;

/// Marker type for render-target leases acquired via [`Scheme::lease_render_target`].
#[cfg(feature = "graphics")]
pub struct LeaseRenderTarget;

/// Epoch-gated pool of physical staging parcels for one logical deposit.
struct DepositPool {
    size: u64,
    /// All physical parcels kept alive for retained CB variants.
    parcels: Vec<Parcel>,
    /// Parcel selected for the next submit (`None` until [`DepositTransaction::write`](crate::exchange::DepositTransaction::write)).
    pending: Option<usize>,
}

impl DepositPool {
    fn new(size: u64) -> Self {
        Self {
            size,
            parcels: Vec::new(),
            pending: None,
        }
    }

    fn select_or_alloc(&mut self, ctx: &Context) -> Result<usize, GoldyError> {
        if let Some(idx) = self.pending {
            return Ok(idx);
        }
        if let Some(idx) = self.parcels.iter().position(|p| p.is_settled_on(ctx)) {
            self.pending = Some(idx);
            return Ok(idx);
        }
        let parcel = ctx
            .with_transient_pool(|pool| {
                pool.acquire_buffer(
                    ctx,
                    self.size,
                    crate::types::BufferKind::Scattered,
                    BufferFlags::CPU_WRITABLE,
                    None,
                )
            })
            .map_err(|e| ctx.classify(e))?;
        self.parcels.push(parcel);
        let idx = self.parcels.len() - 1;
        self.pending = Some(idx);
        Ok(idx)
    }

    fn stage(&mut self, ctx: &Context, offset: u64, data: &[u8]) -> Result<(), GoldyError> {
        if offset.saturating_add(data.len() as u64) > self.size {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "deposit write: [{offset}..{}] exceeds declaration size {}",
                offset + data.len() as u64,
                self.size
            )));
        }
        let idx = self.select_or_alloc(ctx)?;
        self.parcels[idx]
            .write_bytes(offset, data)
            .map_err(|e| ctx.classify(e))?;
        Ok(())
    }

    fn resolve_pending(&self) -> Option<crate::task_graph::ResolvedDeposit> {
        let idx = self.pending?;
        let parcel = &self.parcels[idx];
        let parent = parcel.buffer_handle().expect("deposit parcels are whole buffers");
        Some(crate::task_graph::ResolvedDeposit {
            parent,
            offset: 0,
            len: parcel.byte_size(),
        })
    }

    fn stamp_pending(&mut self, ctx: crate::backend::ContextHandle, tv: TimelineValue) {
        if let Some(idx) = self.pending.take() {
            self.parcels[idx].mark_referenced(ctx, tv);
        }
    }

    fn return_all(self, ctx: &Context) {
        for mut parcel in self.parcels {
            let ready_after = parcel.last_referenced();
            parcel.release_bookkeeping();
            ctx.with_transient_pool(|pool| pool.return_buffer_parcel(parcel, ready_after));
        }
    }
}

/// One-submission tenancy of pool property held by a [`Scheme`].
///
/// Leases have no cross-scheme identity; the scheme owns the N=1 backing parcel
/// for the declaration's lifetime.
pub struct Lease<T> {
    pub(crate) id: LeaseId,
    _marker: PhantomData<T>,
}

/// Outcome counters for [`Scheme::submit`] (retention-recovery assertions and telemetry).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayStats {
    /// Submissions that skipped IR re-recording because the backend reused a retained graph
    /// (native CB replay on Vulkan/DX12, soft re-record on WebGPU). Absent when the `metal`
    /// feature is enabled so macOS CI can compile this struct without the field.
    #[cfg(not(feature = "metal"))]
    pub resubmit_hits: u64,
    /// Submissions that recorded (first submit, post-mutation submits, retention misses).
    pub records: u64,
    /// Re-records caused by a foreign scheme changing shared-parcel topology.
    pub topology_records: u64,
    /// Submissions that found the scheme clean: no structural, params, or topology dirtiness.
    ///
    /// Unlike [`Self::resubmit_hits`], this counts the scheme's own state rather than backend
    /// command-list retention, so it means the same thing on Metal and WebGPU (which re-encode
    /// every frame) as it does on Vulkan and DX12. That makes it the portable signal for
    /// per-site history across frames.
    pub clean_submits: u64,
}

/// How much of a retained scheme the next [`Scheme::submit`] must rebuild.
///
/// Ordered so [`SchemeDirty::Structure`] wins over [`SchemeDirty::Params`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SchemeDirty {
    /// Resubmit retained command lists; skip fingerprint hashing.
    Clean,
    /// Bindings unchanged. Recompute partition fingerprints and re-record only
    /// partitions whose baked payload (pipeline, user slots, dispatch dims) changed.
    Params,
    /// Nodes or bindings changed. Drop all retained CBs and rebuild the schedule.
    Structure,
}

/// Snapshot of dirty/replay state taken at the start of [`Scheme::submit`].
struct IrSubmitPrep {
    topo_dirty: bool,
    structurally_dirty: bool,
    params_dirty: bool,
    ir_clean: bool,
    had_replay: bool,
    deposit_resolutions: std::collections::HashMap<u32, crate::task_graph::ResolvedDeposit>,
}

/// A retained scheme: a set of dispatches held across submissions with COW dirty tracking.
///
/// Build the scheme's nodes once via [`Self::node`]; call [`Self::submit`] every frame.
/// While clean, `submit` pays neither recording nor fingerprint-hashing cost.
pub struct Scheme {
    ir: GraphIR,
    submit_state: IrSubmitState,
    /// Context this scheme submits on. Fixed at construction; many schemes per context,
    /// exactly one context per scheme.
    ctx: Context,
    /// N=1 backing parcels for [`Lease`] declarations, indexed by [`LeaseId`].
    leases: Vec<Parcel>,
    /// N=1 backing render targets for [`Lease<LeaseRenderTarget>`] declarations, indexed by [`LeaseId`].
    #[cfg(feature = "graphics")]
    rt_leases: Vec<RenderTarget>,
    /// Epoch-gated CPU-writable staging pools for deposit declarations.
    deposits: Vec<DepositPool>,
    /// Host functions and staging for [`NodeKind::CpuDispatch`] nodes, indexed by `cpu_id`.
    cpu_dispatches: Vec<CpuDispatchExec>,
    /// COW dirty level: structural vs params-only vs clean. Cleared by a successful submit.
    dirty: SchemeDirty,
    /// Set by foreign schemes when shared-parcel interaction topology changes.
    topology_dirty: Arc<AtomicBool>,
    /// Parcels this scheme registered on at the last record (for silent edge teardown).
    prev_topology_parcels: Vec<(ResourceKey, Arc<crate::parcel::ParcelStamp>)>,
    stats: ReplayStats,
    next_withdraw_id: u32,
    /// Process-unique identity for cross-scheme [`Submission`] / withdraw pairing.
    scheme_id: u64,
    /// Memory withdrawals: N-backed staging per submission.
    withdraws: Vec<WithdrawInfo>,
    /// Interned present bindings: index is [`ResourceId::PresentLease`] id.
    #[cfg(feature = "graphics")]
    present_bindings: Vec<PresentBinding>,
    /// Registered present exchanges: one claim slot per transaction, keyed by dense present_idx.
    #[cfg(feature = "graphics")]
    present_transactions: Vec<PresentTransactionInfo>,
    /// Record-time diagnostics flushed on [`Self::submit`].
    record_errors: Vec<String>,
    /// Accel handles already GPU-built on this object (`build_blas` / `build_tlas`).
    prior_built_accels: HashSet<u64>,
}

fn parcel_gpu_buffer(parcel: &Parcel) -> Result<(BufferHandle, u64), GoldyError> {
    match parcel.resource_id() {
        ResourceId::Buffer(h) => Ok((h, 0)),
        ResourceId::BufferRange { parent, offset, .. } => Ok((parent, offset)),
        _ => Err(GoldyError::Backend(anyhow::anyhow!(
            "acceleration-structure geometry requires a buffer parcel"
        ))),
    }
}

fn parcel_has_accel_input(parcel: &Parcel) -> bool {
    parcel
        .buffer_descriptor()
        .map(|(_, flags)| flags.contains(BufferFlags::ACCEL_INPUT))
        .unwrap_or(true)
}

impl Scheme {
    /// Create a scheme bound to `ctx`.
    pub fn new(ctx: &Context) -> Self {
        Self {
            ir: GraphIR::default(),
            submit_state: IrSubmitState::new(),
            ctx: ctx.clone(),
            leases: Vec::new(),
            #[cfg(feature = "graphics")]
            rt_leases: Vec::new(),
            deposits: Vec::new(),
            cpu_dispatches: Vec::new(),
            dirty: SchemeDirty::Structure,
            topology_dirty: Arc::new(AtomicBool::new(false)),
            prev_topology_parcels: Vec::new(),
            stats: ReplayStats::default(),
            next_withdraw_id: 0,
            scheme_id: NEXT_SCHEME_ID.fetch_add(1, Ordering::Relaxed),
            withdraws: Vec::new(),
            #[cfg(feature = "graphics")]
            present_bindings: Vec::new(),
            #[cfg(feature = "graphics")]
            present_transactions: Vec::new(),
            record_errors: Vec::new(),
            prior_built_accels: HashSet::new(),
        }
    }

    /// Context this scheme submits on.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// True when the next [`Self::submit`] must re-record at least one partition.
    pub fn is_dirty(&self) -> bool {
        self.dirty != SchemeDirty::Clean
    }

    fn mark_structure_dirty(&mut self) {
        self.dirty = SchemeDirty::Structure;
    }

    fn mark_params_dirty(&mut self) {
        if self.dirty < SchemeDirty::Params {
            self.dirty = SchemeDirty::Params;
        }
    }

    /// True when a foreign scheme changed shared-parcel topology since the last record.
    #[doc(hidden)]
    pub fn is_topology_dirty(&self) -> bool {
        self.topology_dirty.load(Ordering::Acquire)
    }

    /// Submission outcome counters.
    pub fn replay_stats(&self) -> ReplayStats {
        self.stats
    }

    /// Per-partition timeline values from the most recent successful submit (diagnostics/tests).
    #[doc(hidden)]
    pub fn partition_last_tvs(&self) -> &[Option<TimelineValue>] {
        self.submit_state.partition_last_tvs()
    }

    /// Register stamp targets collected during compute-node recording.
    pub(crate) fn apply_compute_stamps(&mut self, stamps: &[std::sync::Arc<crate::parcel::ParcelStamp>]) {
        for stamp in stamps {
            self.submit_state.register_stamp(stamp.clone());
        }
    }

    /// Append a GPU buffer-to-buffer copy between two buffer parcels (identity only; no bytes in IR).
    ///
    /// Record once while parcel identities are stable; refresh source bytes via [`crate::Buffer::write`]
    /// on a [`crate::types::BufferFlags::CPU_WRITABLE`] staging parcel before each [`Self::submit`].
    pub fn copy_buffer_parcel(
        &mut self,
        src: &Parcel,
        src_offset: u64,
        dst: &Parcel,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), GoldyError> {
        self.mark_structure_dirty();
        let src_resource = src.resource_id();
        let dst_resource = dst.resource_id();
        if !matches!(src_resource, ResourceId::Buffer(_) | ResourceId::BufferRange { .. })
            || !matches!(dst_resource, ResourceId::Buffer(_) | ResourceId::BufferRange { .. })
        {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_buffer_parcel requires buffer parcels"
            )));
        }
        self.submit_state.register_parcel_stamp(src);
        self.submit_state.register_parcel_stamp(dst);
        let dst_access = if dst_offset == 0 && size == dst.byte_size() {
            NodeAccess::Overwrite
        } else {
            NodeAccess::Write
        };
        self.ir.nodes.push(TaskNode {
            label: "copy_buffer_parcel",
            bindings: vec![
                ResourceBinding {
                    resource: src_resource,
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: dst_resource,
                    access: dst_access,
                },
            ],
            kind: NodeKind::CopyBuffer {
                src: src_resource,
                src_offset,
                dst: dst_resource,
                dst_offset,
                size,
            },
        });
        Ok(())
    }

    /// GPU-build a triangle BLAS from a vertex (and optional index) parcel.
    ///
    /// Vertex data is tightly packed `float3` (or a larger stride). Index buffer is 32-bit.
    /// Geometry parcels should be created with [`crate::types::BufferFlags::ACCEL_INPUT`].
    pub fn build_blas(
        &mut self,
        dest: &crate::AccelerationStructure,
        vertices: &Parcel,
        vertex_count: u32,
        vertex_stride: u32,
        indices: Option<(&Parcel, u32)>,
    ) -> Result<(), GoldyError> {
        self.mark_structure_dirty();
        if dest.kind != crate::accel::AccelKind::Blas {
            return Err(GoldyError::Validation(
                "build_blas requires a triangle BLAS, but the destination is a TLAS. \
                 hint: create with AccelerationStructure::blas_triangles, or call build_tlas \
                 for instance structures."
                    .into(),
            ));
        }
        if !parcel_has_accel_input(vertices) {
            return Err(GoldyError::Validation(
                "build_blas vertex parcel is missing BufferFlags::ACCEL_INPUT. \
                 hint: acquire the geometry buffer with BufferFlags::ACCEL_INPUT \
                 (Vulkan/DX12/Metal require AS-build usage on BLAS inputs)."
                    .into(),
            ));
        }
        if vertex_count == 0 || vertex_stride < 12 {
            return Err(GoldyError::Validation(
                "build_blas requires vertex_count > 0 and vertex_stride >= 12 (float3). \
                 hint: pass the vertex count and stride used to create the BLAS."
                    .into(),
            ));
        }
        let vertex_bytes = vertex_count as u64 * vertex_stride as u64;
        if vertex_bytes > vertices.byte_size() {
            return Err(GoldyError::Validation(format!(
                "build_blas vertex range ({vertex_count} vertices × {vertex_stride} bytes = {vertex_bytes}) \
                 exceeds parcel size {} bytes. \
                 hint: shrink vertex_count/stride or acquire a larger ACCEL_INPUT buffer.",
                vertices.byte_size()
            )));
        }
        let (vertex_buffer, vertex_offset) = parcel_gpu_buffer(vertices)?;
        self.submit_state.register_parcel_stamp(vertices);
        let mut bindings = vec![
            ResourceBinding {
                resource: vertices.resource_id(),
                access: NodeAccess::Read,
            },
            ResourceBinding {
                resource: dest.resource_id(),
                access: NodeAccess::Overwrite,
            },
        ];
        let (index_buffer, index_offset, index_count) = if let Some((idx, count)) = indices {
            self.submit_state.register_parcel_stamp(idx);
            bindings.push(ResourceBinding {
                resource: idx.resource_id(),
                access: NodeAccess::Read,
            });
            let (h, off) = parcel_gpu_buffer(idx)?;
            if !parcel_has_accel_input(idx) {
                return Err(GoldyError::Validation(
                    "build_blas index parcel is missing BufferFlags::ACCEL_INPUT. \
                     hint: acquire index buffers used as BLAS geometry with BufferFlags::ACCEL_INPUT."
                        .into(),
                ));
            }
            if count == 0 || count % 3 != 0 {
                return Err(GoldyError::Validation(
                    "build_blas index_count must be a positive multiple of 3 (triangle list). \
                     hint: pass 3 × triangle_count 32-bit indices."
                        .into(),
                ));
            }
            let index_bytes = count as u64 * 4;
            if index_bytes > idx.byte_size() {
                return Err(GoldyError::Validation(format!(
                    "build_blas index range ({count} × 4 = {index_bytes} bytes) exceeds parcel size {} bytes. \
                     hint: shrink index_count or acquire a larger ACCEL_INPUT index buffer.",
                    idx.byte_size()
                )));
            }
            (Some(h), off, count)
        } else {
            if !vertex_count.is_multiple_of(3) {
                return Err(GoldyError::Validation(
                    "build_blas without indices requires vertex_count to be a multiple of 3. \
                     hint: pass an index parcel, or provide 3 vertices per triangle."
                        .into(),
                ));
            }
            (None, 0, 0)
        };
        self.ir.nodes.push(TaskNode {
            label: "build_blas",
            bindings,
            kind: NodeKind::BuildAccelerationStructure(crate::backend::AccelBuildCommand::BlasTriangles {
                dest: dest.handle,
                vertex_buffer,
                vertex_offset,
                vertex_count,
                vertex_stride,
                index_buffer,
                index_offset,
                index_count,
            }),
        });
        dest.mark_gpu_built();
        Ok(())
    }

    /// GPU-build a TLAS from BLAS instances.
    pub fn build_tlas(
        &mut self,
        dest: &crate::AccelerationStructure,
        instances: &[crate::AccelInstance<'_>],
    ) -> Result<(), GoldyError> {
        self.mark_structure_dirty();
        if dest.kind != crate::accel::AccelKind::Tlas {
            return Err(GoldyError::Validation(
                "build_tlas requires a TLAS, but the destination is a BLAS. \
                 hint: create with AccelerationStructure::tlas, or call build_blas for triangle geometry."
                    .into(),
            ));
        }
        if instances.is_empty() {
            return Err(GoldyError::Validation(
                "build_tlas requires at least one instance. \
                 hint: pass a non-empty &[AccelInstance] of BLASes."
                    .into(),
            ));
        }
        let mut bindings = vec![ResourceBinding {
            resource: dest.resource_id(),
            access: NodeAccess::Overwrite,
        }];
        let mut rec = Vec::with_capacity(instances.len());
        for inst in instances {
            if inst.blas.kind != crate::accel::AccelKind::Blas {
                return Err(GoldyError::Validation(
                    "build_tlas instance must reference a BLAS, not a TLAS. \
                     hint: AccelInstance.blas must be AccelerationStructure::blas_triangles."
                        .into(),
                ));
            }
            bindings.push(ResourceBinding {
                resource: inst.blas.resource_id(),
                access: NodeAccess::Read,
            });
            rec.push(crate::backend::AccelInstanceRecord {
                blas: inst.blas.handle,
                transform: inst.transform,
                mask: inst.mask,
                custom_index: inst.custom_index,
            });
        }
        dest.retain_blases(instances);
        self.ir.nodes.push(TaskNode {
            label: "build_tlas",
            bindings,
            kind: NodeKind::BuildAccelerationStructure(crate::backend::AccelBuildCommand::Tlas {
                dest: dest.handle,
                instances: rec.into(),
            }),
        });
        dest.mark_gpu_built();
        Ok(())
    }
    ///
    /// `dst_offset` is relative to the start of `destination` (added to any buffer-range base).
    /// The recorded copy size is `capacity.min(destination.byte_size().saturating_sub(dst_offset))`.
    pub(crate) fn register_deposit_buffer(
        &mut self,
        destination: &Parcel,
        dst_offset: u64,
        capacity: u64,
    ) -> Result<crate::exchange::DepositTransaction, GoldyError> {
        if capacity == 0 {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "bind_deposit_buffer requires non-zero capacity"
            )));
        }
        self.mark_structure_dirty();
        let dst_resource = destination.resource_id();
        if !matches!(dst_resource, ResourceId::Buffer(_) | ResourceId::BufferRange { .. }) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "bind_deposit_buffer requires a buffer parcel destination"
            )));
        }
        let remaining = destination.byte_size().saturating_sub(dst_offset);
        if remaining == 0 {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "bind_deposit_buffer: dst_offset {dst_offset} exceeds destination size {}",
                destination.byte_size()
            )));
        }
        let copy_size = capacity.min(remaining);
        let deposit_id = u32::try_from(self.deposits.len()).expect("deposit id overflow");
        self.deposits.push(DepositPool::new(capacity));
        self.submit_state.register_parcel_stamp(destination);
        let src_resource = ResourceId::Deposit(deposit_id);
        let abs_dst_offset = destination.source_offset() + dst_offset;
        let dst_access = if dst_offset == 0 && copy_size == destination.byte_size() {
            NodeAccess::Overwrite
        } else {
            NodeAccess::Write
        };
        self.ir.nodes.push(TaskNode {
            label: "deposit_buffer",
            bindings: vec![
                ResourceBinding {
                    resource: src_resource,
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: dst_resource,
                    access: dst_access,
                },
            ],
            kind: NodeKind::CopyBuffer {
                src: src_resource,
                src_offset: 0,
                dst: dst_resource,
                dst_offset: abs_dst_offset,
                size: copy_size,
            },
        });
        Ok(crate::exchange::DepositTransaction {
            scheme_id: self.scheme_id,
            deposit_id,
            capacity,
        })
    }

    /// Register a destination-bound texture-region deposit (called by [`crate::MemoryExchange`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn register_deposit_texture(
        &mut self,
        destination: &crate::Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        capacity: u64,
        src_row_pitch: u32,
    ) -> Result<crate::exchange::DepositTransaction, GoldyError> {
        if capacity == 0 {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "bind_deposit_texture requires non-zero capacity"
            )));
        }
        self.mark_structure_dirty();
        let x_end = x
            .checked_add(width)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("bind_deposit_texture: x+width overflow")))?;
        let y_end = y
            .checked_add(height)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("bind_deposit_texture: y+height overflow")))?;
        if x_end > destination.width() || y_end > destination.height() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "bind_deposit_texture: {}x{} at ({},{}) exceeds {}x{} texture",
                width,
                height,
                x,
                y,
                destination.width(),
                destination.height()
            )));
        }
        let bpp = u64::from(destination.format().bytes_per_pixel());
        let min_bytes = if src_row_pitch == 0 {
            (width as u64) * (height as u64) * bpp
        } else {
            (src_row_pitch as u64) * (height as u64)
        };
        if min_bytes > capacity {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "bind_deposit_texture: copy range exceeds deposit capacity"
            )));
        }
        let deposit_id = u32::try_from(self.deposits.len()).expect("deposit id overflow");
        self.deposits.push(DepositPool::new(capacity));
        let th = destination.gpu_handle();
        let src_resource = ResourceId::Deposit(deposit_id);
        let dst_access = if x == 0 && y == 0 && width == destination.width() && height == destination.height() {
            NodeAccess::Overwrite
        } else {
            NodeAccess::Write
        };
        self.ir.nodes.push(TaskNode {
            label: "deposit_texture",
            bindings: vec![
                ResourceBinding {
                    resource: src_resource,
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::Texture(th),
                    access: dst_access,
                },
            ],
            kind: NodeKind::CopyBufferToTexture {
                src: src_resource,
                src_offset: 0,
                src_row_pitch,
                dst: th,
                x,
                y,
                width,
                height,
            },
        });
        Ok(crate::exchange::DepositTransaction {
            scheme_id: self.scheme_id,
            deposit_id,
            capacity,
        })
    }

    /// Stage bytes for a deposit transaction (called by [`crate::exchange::DepositTransaction::write`]).
    pub(crate) fn stage_deposit(
        &mut self,
        scheme_id: u64,
        deposit_id: u32,
        offset: u64,
        data: &[u8],
    ) -> Result<(), GoldyError> {
        if scheme_id != self.scheme_id {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "DepositTransaction belongs to a different scheme"
            )));
        }
        let pool = self
            .deposits
            .get_mut(deposit_id as usize)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("deposit write: unknown deposit {deposit_id}")))?;
        pool.stage(&self.ctx, offset, data)
    }

    /// Append a CPU-writable buffer → texture copy node (identity only; no bytes in IR).
    ///
    /// Record once while parcel identities are stable; refresh source bytes via [`crate::Buffer::write`]
    /// on a [`crate::types::BufferFlags::CPU_WRITABLE`] staging parcel before each [`Self::submit`].
    ///
    /// `src_row_pitch`: pass `0` when the source is tightly packed (`width * height * bpp`);
    /// the backend will repack into an intermediate footprint-aligned buffer at submit time.
    /// Pass the actual footprint row pitch (from [`crate::Device::texture_copy_footprint`]) when
    /// the source was allocated and written with that pitch — the backend will then copy directly,
    /// skipping the intermediate buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_buffer_to_texture_parcel(
        &mut self,
        src: &Parcel,
        src_offset: u64,
        src_row_pitch: u32,
        dst: &crate::Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), GoldyError> {
        self.mark_structure_dirty();
        let src_resource = src.resource_id();
        if !matches!(src_resource, ResourceId::Buffer(_) | ResourceId::BufferRange { .. }) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_buffer_to_texture_parcel requires a buffer parcel source"
            )));
        }
        let x_end = x
            .checked_add(width)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("copy_buffer_to_texture_parcel: x+width overflow")))?;
        let y_end = y
            .checked_add(height)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("copy_buffer_to_texture_parcel: y+height overflow")))?;
        if x_end > dst.width() || y_end > dst.height() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_buffer_to_texture_parcel: {}x{} at ({},{}) exceeds {}x{} texture",
                width,
                height,
                x,
                y,
                dst.width(),
                dst.height()
            )));
        }
        let th = dst.gpu_handle();
        self.submit_state.register_parcel_stamp(src);
        let dst_access = if x == 0 && y == 0 && width == dst.width() && height == dst.height() {
            NodeAccess::Overwrite
        } else {
            NodeAccess::Write
        };
        self.ir.nodes.push(TaskNode {
            label: "copy_buffer_to_texture",
            bindings: vec![
                ResourceBinding {
                    resource: src_resource,
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::Texture(th),
                    access: dst_access,
                },
            ],
            kind: NodeKind::CopyBufferToTexture {
                src: src_resource,
                src_offset,
                src_row_pitch,
                dst: th,
                x,
                y,
                width,
                height,
            },
        });
        Ok(())
    }

    /// Append a zero-fill node for `parcel[offset..offset+size]`.
    ///
    /// Zero-fill a buffer region via an upload micro-scheme.
    pub fn clear_parcel(&mut self, parcel: &Parcel, offset: u64, size: u64) -> Result<(), GoldyError> {
        self.mark_structure_dirty();
        let buffer = parcel
            .buffer_handle()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("clear_parcel: requires a buffer parcel")))?;
        let abs_offset = match parcel.resource_id() {
            ResourceId::BufferRange { offset: base, .. } => base + offset,
            _ => offset,
        };
        let clear_size = if size == 0 {
            parcel.byte_size().saturating_sub(offset)
        } else {
            size
        };
        self.submit_state.register_parcel_stamp(parcel);
        self.ir.nodes.push(TaskNode {
            label: "clear_parcel",
            bindings: vec![ResourceBinding {
                resource: parcel.resource_id(),
                access: NodeAccess::Overwrite,
            }],
            kind: NodeKind::ClearBuffer {
                buffer,
                offset: abs_offset,
                size: clear_size,
            },
        });
        Ok(())
    }

    /// Append a compute dispatch node to the scheme IR.
    pub(crate) fn commit_compute_dispatch(
        &mut self,
        label: &'static str,
        pipeline: crate::backend::ComputePipelineHandle,
        bindings: Vec<ResourceBinding>,
        resource_slots: Vec<u32>,
        user_slots: Vec<u32>,
        dispatch: DispatchDim,
    ) {
        self.mark_structure_dirty();
        self.ir.nodes.push(TaskNode {
            label,
            bindings,
            kind: NodeKind::Dispatch {
                pipeline,
                resource_slots,
                user_slots,
                dispatch,
            },
        });
    }

    /// Append a render pass node to the scheme IR.
    #[cfg(feature = "graphics")]
    pub(crate) fn commit_render_pass(
        &mut self,
        label: &'static str,
        target: crate::backend::RenderTargetHandle,
        color_load: crate::types::TargetLoad,
        bindings: Vec<ResourceBinding>,
        commands: Vec<RenderCommand>,
        stamp_targets: &[std::sync::Arc<crate::parcel::ParcelStamp>],
    ) {
        self.apply_compute_stamps(stamp_targets);
        self.mark_structure_dirty();
        self.ir.nodes.push(TaskNode {
            label,
            bindings,
            kind: NodeKind::RenderPass {
                target,
                color_load,
                commands,
            },
        });
    }

    /// Declare a transient texture lease backed by the context's transient pool (N=1).
    ///
    /// The backing parcel is held until the scheme is dropped. Structural mutation.
    pub fn lease_texture(
        &mut self,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<Lease<LeaseTexture>, GoldyError> {
        self.mark_structure_dirty();
        let texture = self
            .ctx
            .with_transient_pool(|pool| pool.acquire_texture(&self.ctx, width, height, format, access, flags))
            .map_err(|e| self.ctx.classify(e))?;
        let id = LeaseId(u32::try_from(self.leases.len()).expect("lease id overflow"));
        self.leases.push(texture.into_lease_parcel());
        Ok(Lease {
            id,
            _marker: PhantomData,
        })
    }

    /// Declare a transient buffer lease backed by the context's transient pool (N=1).
    ///
    /// The backing parcel is held until the scheme is dropped. Structural mutation.
    ///
    /// # Write-first invariant
    ///
    /// The pool may reissue a previously-used buffer parcel whose epoch has retired.
    /// The recycled bytes are **not** cleared. The first node that accesses this lease
    /// must declare [`NodeAccess::Write`], [`NodeAccess::Overwrite`], or `ReadWrite`, never pure `Read` — otherwise
    /// the shader observes the previous tenant's data.
    ///
    /// A full inaugural-write shape check (unique-minimal-write scheme validation per
    /// design §8) is not yet implemented; callers are responsible for this invariant today.
    pub fn lease_buffer(&mut self, size: u64) -> Result<Lease<LeaseBuffer>, GoldyError> {
        self.lease_buffer_with(
            size,
            crate::types::BufferKind::Scattered,
            crate::types::BufferFlags::empty(),
        )
    }

    /// Like [`Self::lease_buffer`] but with explicit kind and flags.
    ///
    /// Use this when the shader requires a buffer kind other than `Scattered` (e.g.
    /// `Broadcast` for uniform buffers). The pool bins buffers by `(size, kind, flags)`,
    /// so only identically-described buffers are ever reused across submissions.
    ///
    /// See [`Self::lease_buffer`] for the write-first invariant that applies to all buffer
    /// leases regardless of kind.
    pub fn lease_buffer_with(
        &mut self,
        size: u64,
        kind: crate::types::BufferKind,
        flags: crate::types::BufferFlags,
    ) -> Result<Lease<LeaseBuffer>, GoldyError> {
        self.mark_structure_dirty();
        let backing = self
            .ctx
            .with_transient_pool(|pool| pool.acquire_buffer(&self.ctx, size, kind, flags, None))
            .map_err(|e| self.ctx.classify(e))?;
        let id = LeaseId(u32::try_from(self.leases.len()).expect("lease id overflow"));
        self.leases.push(backing);
        Ok(Lease {
            id,
            _marker: PhantomData,
        })
    }

    /// Declare a render-target lease owned by this scheme (N=1).
    ///
    /// The backing render target is held until the scheme is dropped. Structural mutation.
    #[cfg(feature = "graphics")]
    pub fn lease_render_target(
        &mut self,
        width: u32,
        height: u32,
        format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<Lease<LeaseRenderTarget>, GoldyError> {
        self.mark_structure_dirty();
        let rt = RenderTarget::new_with_depth(self.ctx.device(), width, height, format, depth_format)
            .map_err(|e| self.ctx.classify(e))?;
        let handle = rt.backend_handle();
        let stamp = rt.stamp_handle();
        self.submit_state
            .register_stamp_parts(ResourceId::RenderTarget(handle), stamp);
        let id = LeaseId(u32::try_from(self.rt_leases.len()).expect("render target lease id overflow"));
        self.rt_leases.push(rt);
        Ok(Lease {
            id,
            _marker: PhantomData,
        })
    }

    /// Borrow the backing render target for a scheme-held lease.
    #[cfg(feature = "graphics")]
    pub(crate) fn rt(&self, lease: &Lease<LeaseRenderTarget>) -> &RenderTarget {
        &self.rt_leases[lease.id.0 as usize]
    }

    /// Typed resource descriptor handle for a scheme-held texture lease (advanced binding).
    pub fn lease_handle(&self, lease: &Lease<LeaseTexture>, access: ResourceAccess) -> Option<ResourceHandle> {
        self.leases[lease.id.0 as usize].handle(access)
    }

    /// Typed resource descriptor handle for a scheme-held buffer lease (advanced binding).
    pub fn lease_buffer_handle(&self, lease: &Lease<LeaseBuffer>, access: ResourceAccess) -> Option<ResourceHandle> {
        self.leases[lease.id.0 as usize].handle(access)
    }

    /// Declare a compute dispatch node, returning a builder for access declarations.
    ///
    /// Calling this marks the scheme dirty (structural mutation).
    pub fn node<'a>(
        &'a mut self,
        label: &'static str,
        pipeline: &crate::compute::ComputePipeline,
    ) -> SchemeNodeBuilder<'a> {
        self.mark_structure_dirty();
        SchemeNodeBuilder {
            scheme: self,
            label,
            pipeline: pipeline.handle,
            rt_pipeline: None,
            bindings: Vec::new(),
            resource_slots: Vec::new(),
            user_slots: Vec::new(),
            slot_access: pipeline.slot_access.clone(),
        }
    }

    /// Declare a ray-tracing dispatch (`TraceRays` / `DispatchRays`).
    ///
    /// Bind raygen resource parameters with [`SchemeNodeBuilder::with_parcel`], then
    /// [`SchemeNodeBuilder::dispatch`] with ray counts `(width, height, depth)` — not
    /// compute workgroups.
    pub fn trace_rays<'a>(
        &'a mut self,
        label: &'static str,
        pipeline: &crate::rt_pipeline::RayTracingPipeline,
    ) -> SchemeNodeBuilder<'a> {
        self.mark_structure_dirty();
        SchemeNodeBuilder {
            scheme: self,
            label,
            pipeline: 0,
            rt_pipeline: Some(pipeline.handle),
            bindings: Vec::new(),
            resource_slots: Vec::new(),
            user_slots: Vec::new(),
            slot_access: pipeline.slot_access.clone(),
        }
    }

    /// Identity of the node appended most recently. Call sites push first, then ask.
    fn last_node_id(&self) -> NodeId {
        debug_assert!(!self.ir.nodes.is_empty(), "last_node_id after a push");
        NodeId {
            scheme_id: self.scheme_id,
            index: (self.ir.nodes.len() - 1) as u32,
        }
    }

    /// Borrow a recorded compute dispatch node for in-place payload mutation.
    ///
    /// `api` names the caller for error text.
    fn dispatch_node_mut(&mut self, node: NodeId, api: &str) -> Result<&mut NodeKind, GoldyError> {
        if node.scheme_id != self.scheme_id {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "{api}: node id belongs to scheme {}, not scheme {}",
                node.scheme_id,
                self.scheme_id
            )));
        }
        let index = node.index();
        let kind = self
            .ir
            .nodes
            .get_mut(index)
            .map(|n| &mut n.kind)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("{api}: node {index} out of range")))?;
        if !matches!(kind, NodeKind::Dispatch { .. }) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "{api}: node {index} is not a compute dispatch"
            )));
        }
        Ok(kind)
    }

    /// Replace the compute pipeline on an existing dispatch node.
    ///
    /// Bindings stay as recorded. Marks the scheme params-dirty: the next submit
    /// recomputes partition fingerprints and re-records only partitions whose baked
    /// pipeline (or other payload) changed. Other retained partitions resubmit.
    pub fn set_node_pipeline(
        &mut self,
        node: NodeId,
        pipeline: &crate::compute::ComputePipeline,
    ) -> Result<(), GoldyError> {
        let changed = match self.dispatch_node_mut(node, "set_node_pipeline")? {
            NodeKind::Dispatch { pipeline: slot, .. } if *slot != pipeline.handle => {
                *slot = pipeline.handle;
                true
            }
            _ => false,
        };
        if changed {
            self.mark_params_dirty();
        }
        Ok(())
    }

    /// Replace the direct dispatch dimensions on an existing dispatch node.
    ///
    /// Marks the scheme params-dirty. Indirect dispatches become direct.
    pub fn set_node_dispatch(&mut self, node: NodeId, x: u32, y: u32, z: u32) -> Result<(), GoldyError> {
        let next = DispatchDim::Direct { x, y, z };
        let changed = match self.dispatch_node_mut(node, "set_node_dispatch")? {
            NodeKind::Dispatch { dispatch, .. } if *dispatch != next => {
                *dispatch = next;
                true
            }
            _ => false,
        };
        if changed {
            self.mark_params_dirty();
        }
        Ok(())
    }

    /// Replace one virtual-main scalar parameter (`with_param` slot) on a dispatch node.
    ///
    /// `param_index` is the nth recorded `with_param` value. Marks the scheme params-dirty.
    ///
    /// Scalar params are baked into the emitted command list, so a value that changes every
    /// frame re-records its partition every frame. Facts that flip often belong in a bound
    /// parcel instead.
    pub fn set_node_param(&mut self, node: NodeId, param_index: usize, value: u32) -> Result<(), GoldyError> {
        let changed = match self.dispatch_node_mut(node, "set_node_param")? {
            NodeKind::Dispatch { user_slots, .. } => {
                if param_index >= user_slots.len() {
                    return Err(GoldyError::Backend(anyhow::anyhow!(
                        "set_node_param: param_index {param_index} out of range (node has {} params)",
                        user_slots.len()
                    )));
                }
                let changed = user_slots[param_index] != value;
                user_slots[param_index] = value;
                changed
            }
            _ => false,
        };
        if changed {
            self.mark_params_dirty();
        }
        Ok(())
    }

    /// Begin recording a CPU dispatch: a serial host function over whole buffer parcels.
    ///
    /// Bind parcels with [`SchemeCpuNodeBuilder::with_parcel`] (or
    /// [`SchemeCpuNodeBuilder::with_lease`]) and scalars with
    /// [`SchemeCpuNodeBuilder::with_param`], then finish with
    /// [`SchemeCpuNodeBuilder::dispatch`]. The function's parameters are the node's
    /// "virtual main": one `&[T]` / `&mut [T]` per bound parcel in binding order,
    /// followed by one scalar per `with_param`. See [`crate::cpu_dispatch`] for the
    /// execution model and its cost.
    pub fn cpu_node<'a>(&'a mut self, label: &'static str) -> SchemeCpuNodeBuilder<'a> {
        self.mark_structure_dirty();
        SchemeCpuNodeBuilder {
            scheme: self,
            label,
            bindings: Vec::new(),
            params: Vec::new(),
        }
    }

    /// Number of CPU dispatch nodes recorded on this scheme.
    #[doc(hidden)]
    pub fn cpu_dispatch_count(&self) -> usize {
        self.cpu_dispatches.len()
    }

    /// Mark every CPU dispatch's staging as referenced by the submission at `tv`.
    fn stamp_cpu_dispatches(&self, tv: TimelineValue) {
        if self.cpu_dispatches.is_empty() {
            return;
        }
        let ctx_h = self.ctx.backend_handle();
        for exec in &self.cpu_dispatches {
            exec.stamp(ctx_h, tv);
        }
    }

    /// Record GPU-orderable buffer reuse dependencies enforced at submit-worker execute time.
    pub fn record_reuse_parcel(&mut self, parcel: &crate::Parcel) {
        self.submit_state.record_reuse_epochs(&parcel.last_referenced());
    }

    /// Record GPU-orderable reuse for all parcels in a buffer.
    pub fn record_reuse_buffer(&mut self, buffer: &crate::Buffer) {
        self.submit_state.record_reuse_epochs(&buffer.last_referenced());
    }

    /// Defer a host-visible buffer write until the submission worker after prior uses of
    /// `ready_after` retire on the CPU.
    ///
    /// Applied by the DX12, Vulkan, and Metal submission workers when
    /// [`crate::DeviceCapabilities::host_sidecar_on_submit_worker`] is true.
    pub fn defer_host_write(
        &mut self,
        ready_after: &crate::Buffer,
        buffer: &crate::Buffer,
        offset: u64,
        data: Box<[u8]>,
    ) {
        self.submit_state
            .defer_host_write(&ready_after.last_referenced(), buffer, offset, data);
    }

    fn prepare_ir_submit(&mut self) -> Result<IrSubmitPrep, GoldyError> {
        if !self.submit_state.all_stamps_alive() {
            // Dropping a retained-pool resource invalidates schemes that still bind it.
            self.submit_state.invalidate_retention();
            return Err(GoldyError::StaleResource);
        }

        let topo_dirty = self.topology_dirty.load(Ordering::Acquire);
        let structurally_dirty = self.dirty == SchemeDirty::Structure;
        let params_dirty = self.dirty == SchemeDirty::Params;
        {
            let _tz = crate::tracy_zone!("scheme.submit.dirty_check");
            if structurally_dirty || topo_dirty {
                self.submit_state.invalidate_retention();
            }
        }

        {
            let _tz = crate::tracy_zone!("scheme.submit.easement_gate");
            use crate::task_graph::cross_submit::net_access_per_resource;
            let net = net_access_per_resource(&self.ir);
            let ctx = self.ctx.backend_handle();
            for (key, access) in &net {
                if access.writes {
                    if let Some(stamp) = self.submit_state.resource_stamps().get(key) {
                        stamp.drain_pending_for_submit_gate(ctx);
                    }
                }
            }
        }

        Ok(IrSubmitPrep {
            topo_dirty,
            structurally_dirty,
            params_dirty,
            ir_clean: self.dirty == SchemeDirty::Clean && !topo_dirty,
            had_replay: self.submit_state.has_cb_replay(),
            deposit_resolutions: self.resolve_deposits_for_submit()?,
        })
    }

    fn teardown_replay_if_disabled(&mut self, had_replay: bool) {
        if had_replay && !self.submit_state.has_cb_replay() {
            use crate::task_graph::cross_submit::clear_scheme_topology_registration;
            clear_scheme_topology_registration(self.scheme_id, &self.prev_topology_parcels);
            self.prev_topology_parcels.clear();
            self.topology_dirty.store(false, Ordering::Release);
        }
    }

    fn stamp_deposits_and_cpu(&mut self, tv: TimelineValue) {
        let ctx_h = self.ctx.backend_handle();
        for pool in &mut self.deposits {
            pool.stamp_pending(ctx_h, tv);
        }
        self.stamp_cpu_dispatches(tv);
    }

    fn finish_ir_submit_bookkeeping(
        &mut self,
        tv: TimelineValue,
        part_result: &crate::task_graph::PartitionSubmitResult,
        prep: &IrSubmitPrep,
    ) {
        self.ctx.advance_high_water_timeline(tv);
        self.dirty = SchemeDirty::Clean;

        let structurally_dirty = prep.structurally_dirty;
        let params_dirty = prep.params_dirty;
        let topo_dirty = prep.topo_dirty;
        if prep.ir_clean {
            self.stats.clean_submits += 1;
        }
        // Standalone upload partitions never increment `PartitionSubmitResult.records`,
        // but the first submit after IR mutation still counts as a scheme record.
        // Params-only mutation also counts: at least one partition typically re-records,
        // and Metal (no CB retention) still needs a visible records bump.
        let retention_recorded = part_result.records > 0;
        let recorded = retention_recorded || structurally_dirty || params_dirty;
        // Topology edges exist to invalidate baked barriers in retained CBs — only
        // (re)register when CB replay is enabled and we actually recorded.
        let on_record_path =
            self.submit_state.has_cb_replay() && (structurally_dirty || topo_dirty || retention_recorded);

        if on_record_path {
            use crate::task_graph::cross_submit::{net_access_per_resource, reregister_scheme_topology};
            let net = net_access_per_resource(&self.ir);
            self.prev_topology_parcels = reregister_scheme_topology(
                &net,
                self.submit_state.resource_stamps(),
                &self.prev_topology_parcels,
                self.scheme_id,
                self.ctx.backend_handle(),
                &self.topology_dirty,
            );
        }

        if recorded {
            self.stats.records += 1;
            if topo_dirty && !structurally_dirty {
                self.stats.topology_records += 1;
            }
            if topo_dirty {
                self.topology_dirty.store(false, Ordering::Release);
            }
        }
        // Params-dirty submits often mix a re-record with retained-partition hits.
        // Count those hits even when `recorded` is true (`params_dirty` always is).
        // Require an actual partition-level cache hit. `all_from_cache()` is also
        // true when CB replay is disabled (fresh encodes leave `records == 0`),
        // and those must not count as retention hits.
        if part_result.resubmit_hits > 0 {
            #[cfg(not(feature = "metal"))]
            {
                self.stats.resubmit_hits += 1;
            }
        }

        if structurally_dirty || topo_dirty || params_dirty {
            tracing::debug!(
                target: "goldy::scheme",
                scheme_id = self.scheme_id,
                structurally_dirty,
                params_dirty,
                topo_dirty,
                partition_records = part_result.records,
                partition_resubmits = part_result.resubmit_hits,
                scheme_recorded = recorded,
                "submit dirty"
            );
        } else {
            tracing::debug!(
                target: "goldy::scheme",
                scheme_id = self.scheme_id,
                partition_records = part_result.records,
                partition_resubmits = part_result.resubmit_hits,
                scheme_recorded = recorded,
                all_partitions_from_cache = {
                    #[cfg(feature = "graphics")]
                    {
                        part_result.all_from_cache()
                    }
                    #[cfg(not(feature = "graphics"))]
                    {
                        part_result.records == 0
                    }
                },
                "submit clean (not dirty)"
            );
        }
    }

    /// Submit the scheme: resubmit the retained command list when clean, re-record when dirty.
    ///
    /// On a clean resubmit, bound parcels' reference tables are stamped with the new
    /// timeline value, keeping the context transient pool's reuse gates correct across
    /// retained submissions.
    ///
    /// When present exchange transactions are recorded and no early acquire was supplied,
    /// swapchain drawables are acquired lazily — after non-present partitions have been
    /// submitted — and stored on the returned [`Submission`] for [`Transaction::claim`].
    ///
    /// Per-partition command-buffer reuse legality (Vulkan `SIMULTANEOUS_USE`, DX12
    /// non-reset retained allocators) is enforced in the IR submit loop — no whole-scheme
    /// CPU wait here.
    #[cfg(feature = "graphics")]
    pub fn submit(&mut self) -> Result<Submission, GoldyError> {
        self.submit_with_acquired_presents(Vec::new())
    }

    #[cfg(not(feature = "graphics"))]
    pub fn submit(&mut self) -> Result<Submission, GoldyError> {
        self.submit_without_presents()
    }

    /// Like [`Self::submit`], but uses pre-acquired swapchain drawables.
    ///
    /// `acquired` must be empty (deferred acquire) or contain one entry per present
    /// grant, in grant order, with matching lease ids. Consumed claims are moved onto
    /// the [`Submission`]; leftovers are cancelled on drop.
    #[cfg(feature = "graphics")]
    pub fn submit_with_acquired_presents(
        &mut self,
        mut acquired: Vec<AcquiredPresent>,
    ) -> Result<Submission, GoldyError> {
        let prep = self.prepare_ir_submit()?;

        validate_present_exchange_bindings(&self.ir, &self.present_transactions)?;
        if let Some(msg) = self.record_errors.first() {
            return Err(GoldyError::Validation(msg.clone()));
        }
        crate::task_graph::validate::validate_graph_with_prior_built_accels(&self.ir, &self.prior_built_accels)?;

        let submit_result = {
            let grant_count = self.present_transactions.len();
            let mut present_slots = Vec::with_capacity(grant_count);
            // Fixed-size slots indexed by grant order so partial acquires leave holes
            // rather than shifting later bindings.
            let surface_frames: Vec<Mutex<Option<crate::surface::Frame>>> =
                (0..grant_count).map(|_| Mutex::new(None)).collect();
            // Parallel to surface_frames: generation snapshotted at acquire.
            let surface_generations: Vec<std::sync::Mutex<u64>> =
                (0..grant_count).map(|_| std::sync::Mutex::new(0)).collect();
            // Snapshot grant pools so the deferred-acquire closure does not borrow `self`
            // across the mutable `submit_state` call below.
            let present_grant_pools: Vec<(u32, Arc<crate::swapchain_pool::SwapchainPoolInner>, u32)> = self
                .present_transactions
                .iter()
                .map(|g| (g.binding_id, Arc::clone(&g.pool), g.pool_lease_id))
                .collect();
            let binding_to_idx: std::collections::HashMap<u32, usize> = present_grant_pools
                .iter()
                .enumerate()
                .map(|(i, (id, _, _))| (*id, i))
                .collect();
            let acquire_ctx = self.ctx.clone();

            if !acquired.is_empty() {
                if acquired.len() != present_grant_pools.len() {
                    return Err(GoldyError::Backend(anyhow::anyhow!(
                        "submit_with_acquired_presents: got {} acquired presents, scheme has {} present grants",
                        acquired.len(),
                        present_grant_pools.len()
                    )));
                }
                // Validate every claim before converting any to Frame. A mid-loop
                // Err after into_parts() would drop Frame and implicitly present.
                for ((binding_id, pool, pool_lease_id), claim) in present_grant_pools.iter().zip(acquired.iter()) {
                    if !std::sync::Arc::ptr_eq(claim.pool(), pool) || claim.lease_id() != *pool_lease_id {
                        return Err(GoldyError::Backend(anyhow::anyhow!(
                            "submit_with_acquired_presents: lease provenance mismatch \
                             (binding {}, expected pool-local {}, claim pool-local {})",
                            binding_id,
                            pool_lease_id,
                            claim.lease_id()
                        )));
                    }
                }
                for ((binding_id, _, _), claim) in present_grant_pools.iter().zip(acquired.drain(..)) {
                    let idx = binding_to_idx[binding_id];
                    let (_lease_id, _pool, slot_id, generation, handle, uav_index, surface_frame) = claim.into_parts();
                    present_slots.push(ResolvedPresentSlot {
                        binding_id: *binding_id,
                        generation,
                        slot_id,
                        handle,
                        uav_index,
                    });
                    *surface_frames[idx].lock().unwrap_or_else(|e| e.into_inner()) = Some(surface_frame);
                    *surface_generations[idx].lock().unwrap_or_else(|e| e.into_inner()) = generation;
                }
            }

            let _tz = crate::tracy_zone!("scheme.submit.pipelined");
            let mut partial = crate::task_graph::PartitionSubmitResult::default();
            let mut partial_tv = self.ctx.gpu_progress();
            let result = {
                // Deferred acquire: run Surface::begin (DXGI wait) only for bindings
                // needed by the upcoming present partition that are not yet resolved.
                let mut deferred_acquire = |needed: &[u32],
                                            slots: &mut Vec<ResolvedPresentSlot>|
                 -> anyhow::Result<()> {
                    for &binding_id in needed {
                        let idx = *binding_to_idx.get(&binding_id).ok_or_else(|| {
                            anyhow::anyhow!("deferred present acquire: unknown binding id {binding_id}")
                        })?;
                        let (_, pool, _) = &present_grant_pools[idx];
                        let (slot_id, generation, surface_frame, uav_index, handle) = SwapchainPool::acquire_slot(pool)
                            .map_err(|e| anyhow::anyhow!("{}", acquire_ctx.classify(e)))?;
                        slots.push(ResolvedPresentSlot {
                            binding_id,
                            generation,
                            slot_id,
                            handle,
                            uav_index,
                        });
                        *surface_frames[idx].lock().unwrap_or_else(|e| e.into_inner()) = Some(surface_frame);
                        *surface_generations[idx].lock().unwrap_or_else(|e| e.into_inner()) = generation;
                    }
                    Ok(())
                };
                let deferred: Option<&mut DeferredPresentAcquire<'_>> =
                    if present_grant_pools.is_empty() || !present_slots.is_empty() {
                        None
                    } else {
                        Some(&mut deferred_acquire)
                    };
                self.submit_state.submit_pipelined_and_retain_with_presents(
                    &self.ctx,
                    &self.ir,
                    &mut present_slots,
                    deferred,
                    &prep.deposit_resolutions,
                    prep.ir_clean,
                    &mut partial,
                    &mut partial_tv,
                    &self.cpu_dispatches,
                )
            };
            self.teardown_replay_if_disabled(prep.had_replay);
            match result {
                Ok(ok) => Ok((ok, surface_frames, surface_generations, partial)),
                Err(e) => Err((e, surface_frames, partial, partial_tv)),
            }
        };

        let ((tv, part_result), surface_frames, surface_generations, _partial) = match submit_result {
            Ok(ok) => ok,
            Err((e, surface_frames, partial, partial_tv)) => {
                self.stamp_cpu_dispatches(partial_tv);
                return Err(self.cleanup_failed_present_submit(e, surface_frames, &partial, partial_tv));
            }
        };

        self.stamp_deposits_and_cpu(tv);

        // Stamp each acquired frame with the timeline of the partition that wrote it.
        for (binding_id, binding_tv) in &part_result.present_binding_tvs {
            if let Some(idx) = self
                .present_transactions
                .iter()
                .position(|g| g.binding_id == *binding_id)
            {
                if let Ok(mut slot) = surface_frames[idx].lock() {
                    if let Some(frame) = slot.as_mut() {
                        frame.note_submit_timeline(*binding_tv);
                    }
                }
            }
        }

        self.finish_ir_submit_bookkeeping(tv, &part_result, &prep);

        let present_resolvers = claim_present_easement_promises(
            &self.ir,
            &self.present_transactions,
            self.submit_state.resource_stamps(),
        );
        let claim_generations: Vec<u64> = surface_generations
            .into_iter()
            .map(|m| *m.lock().unwrap_or_else(|e| e.into_inner()))
            .collect();
        let submission = self.finish_submit_frame(tv, surface_frames, claim_generations, present_resolvers)?;
        Ok(submission)
    }

    #[cfg(not(feature = "graphics"))]
    fn submit_without_presents(&mut self) -> Result<Submission, GoldyError> {
        let prep = self.prepare_ir_submit()?;
        if let Some(msg) = self.record_errors.first() {
            return Err(GoldyError::Validation(msg.clone()));
        }
        crate::task_graph::validate::validate_graph_with_prior_built_accels(&self.ir, &self.prior_built_accels)?;
        let mut present_slots = Vec::new();
        let mut partial = crate::task_graph::PartitionSubmitResult::default();
        let mut partial_tv = self.ctx.gpu_progress();
        let (tv, part_result) = self
            .submit_state
            .submit_pipelined_and_retain_with_presents(
                &self.ctx,
                &self.ir,
                &mut present_slots,
                None,
                &prep.deposit_resolutions,
                prep.ir_clean,
                &mut partial,
                &mut partial_tv,
                &self.cpu_dispatches,
            )
            .map_err(|e| {
                self.stamp_cpu_dispatches(partial_tv);
                self.ctx.advance_high_water_timeline(partial_tv);
                self.ctx.classify(e)
            })?;

        self.teardown_replay_if_disabled(prep.had_replay);
        self.stamp_deposits_and_cpu(tv);
        self.finish_ir_submit_bookkeeping(tv, &part_result, &prep);
        self.finish_submit(tv)
    }

    /// Settle high-water, source WAR, and present frames after a mid-submit failure.
    ///
    /// Bindings whose present partition already ran get an unpublished claim discarded
    /// at their copy timeline. Acquired but unsubmitted frames are cancelled. Never-
    /// acquired slots are left alone.
    #[cfg(feature = "graphics")]
    fn cleanup_failed_present_submit(
        &self,
        e: anyhow::Error,
        surface_frames: Vec<Mutex<Option<crate::surface::Frame>>>,
        partial: &crate::task_graph::PartitionSubmitResult,
        partial_tv: TimelineValue,
    ) -> GoldyError {
        self.ctx.advance_high_water_timeline(partial_tv);

        for (grant, frame_mutex) in self.present_transactions.iter().zip(surface_frames) {
            let submitted_tv = partial
                .present_binding_tvs
                .iter()
                .find(|(id, _)| *id == grant.binding_id)
                .map(|(_, tv)| *tv);
            let mut slot = frame_mutex.into_inner().unwrap_or_else(|e| e.into_inner());
            let Some(mut frame) = slot.take() else {
                continue;
            };
            if let Some(tv) = submitted_tv {
                frame.note_submit_timeline(tv);
                let (promise, resolver) = TimelinePromise::new();
                for stamp in
                    present_easement_source_stamps(&self.ir, grant.binding_id, self.submit_state.resource_stamps())
                {
                    stamp.push_pending(promise.clone());
                }
                resolver.resolve(tv);
                // Referenced drawable: discard without present (same as claim.discard).
                let claim = crate::exchange::SurfaceClaimImpl::new(frame);
                let _ = crate::exchange::ClaimImpl::discard(Box::new(claim));
            } else {
                // Acquired for a partition that never submitted — unsubmitted cancel.
                frame.cancel();
            }
        }

        self.ctx.classify(e)
    }

    #[cfg(feature = "graphics")]
    fn finish_submit_frame(
        &mut self,
        tv_dispatch: TimelineValue,
        present_frames: Vec<Mutex<Option<crate::surface::Frame>>>,
        claim_generations: Vec<u64>,
        present_resolvers: Vec<Mutex<Option<PromiseResolver>>>,
    ) -> Result<Submission, GoldyError> {
        // Resolve source WAR from the known copy/present-partition timeline immediately.
        // Claim consumption waits for presentation independently; it must not gate source reuse.
        let mut present_claims = Vec::with_capacity(present_frames.len());
        let claim_bindings: Vec<u32> = self.present_transactions.iter().map(|g| g.binding_id).collect();
        debug_assert_eq!(
            claim_bindings.len(),
            present_frames.len(),
            "present transaction count must match acquired frames"
        );
        debug_assert_eq!(
            claim_bindings.len(),
            claim_generations.len(),
            "present transaction count must match claim generations"
        );
        for (frame_mutex, resolver_mutex) in present_frames.into_iter().zip(present_resolvers) {
            let frame = frame_mutex.into_inner().unwrap_or_else(|e| e.into_inner());
            let resolver = resolver_mutex.into_inner().unwrap_or_else(|e| e.into_inner());
            if let Some(resolver) = resolver {
                match frame.as_ref().and_then(|f| f.submit_timeline()).filter(|&tv| tv != 0) {
                    Some(tv) => resolver.resolve(tv),
                    // No usable copy timeline: abandon so the next writer is not blocked forever.
                    None => drop(resolver),
                }
            }
            let claim = frame
                .map(|f| Box::new(crate::exchange::SurfaceClaimImpl::new(f)) as Box<dyn crate::exchange::ClaimImpl>);
            present_claims.push(Mutex::new(claim));
        }

        self.finish_submit(tv_dispatch, present_claims, claim_bindings, claim_generations)
    }

    fn finish_submit(
        &mut self,
        tv_dispatch: TimelineValue,
        #[cfg(feature = "graphics")] present_claims: Vec<Mutex<Option<Box<dyn crate::exchange::ClaimImpl>>>>,
        #[cfg(feature = "graphics")] claim_bindings: Vec<u32>,
        #[cfg(feature = "graphics")] claim_generations: Vec<u64>,
    ) -> Result<Submission, GoldyError> {
        if self.withdraws.is_empty() {
            return Ok(Submission {
                handle: SubmissionHandle {
                    core: Arc::new(SubmissionCore {
                        scheme_id: self.scheme_id,
                        timeline: tv_dispatch,
                    }),
                },
                ctx: self.ctx.clone(),
                #[cfg(feature = "graphics")]
                present_claims,
                #[cfg(feature = "graphics")]
                claim_bindings,
                #[cfg(feature = "graphics")]
                claim_generations,
                withdraw_claims: Vec::new(),
            });
        }

        let device = self.ctx.device().inner.handle;
        let mut copy_cmds = Vec::with_capacity(self.withdraws.len());
        let mut withdraw_claims = Vec::with_capacity(self.withdraws.len());
        let mut staging_handles = Vec::with_capacity(self.withdraws.len());

        {
            let mut backend = self.ctx.device().inner.backend.lock().unwrap();
            for withdraw in &self.withdraws {
                let staging = withdraw.staging_pool.take_or_alloc(&mut **backend, device)?;
                if validation_env::scheme_validation_enabled() {
                    if staging_handles.contains(&staging) {
                        return Err(GoldyError::Backend(anyhow::anyhow!(
                            "duplicate withdraw staging buffer handle in one submission"
                        )));
                    }
                    staging_handles.push(staging);
                }
                match &withdraw.source {
                    WithdrawSource::Buffer {
                        source,
                        src_offset,
                        byte_size,
                        ..
                    } => {
                        copy_cmds.push(GpuCommand::CopyBuffer {
                            src: *source,
                            src_offset: *src_offset,
                            dst: staging,
                            dst_offset: 0,
                            size: *byte_size,
                        });
                    }
                    WithdrawSource::Texture { source, layout, .. } => {
                        copy_cmds.push(GpuCommand::CopyTextureToReadback {
                            src: *source,
                            dst: staging,
                            layout: *layout,
                        });
                    }
                }
                withdraw_claims.push(Mutex::new(Some(crate::exchange::WithdrawSlot {
                    staging,
                    pool: Arc::clone(&withdraw.staging_pool),
                })));
            }
        }

        if validation_env::scheme_validation_enabled() {
            debug_assert_eq!(withdraw_claims.len(), self.withdraws.len());
        }

        let tv_copy = {
            let submit_result = {
                let mut backend = self.ctx.device().inner.backend.lock().unwrap();
                backend.submit_standalone(self.ctx.backend_handle(), &copy_cmds, None)
            };
            submit_result.map_err(|e| self.ctx.classify(e))?
        };
        self.ctx.advance_high_water_timeline(tv_copy);

        Ok(Submission {
            handle: SubmissionHandle {
                core: Arc::new(SubmissionCore {
                    scheme_id: self.scheme_id,
                    timeline: tv_copy,
                }),
            },
            ctx: self.ctx.clone(),
            #[cfg(feature = "graphics")]
            present_claims,
            #[cfg(feature = "graphics")]
            claim_bindings,
            #[cfg(feature = "graphics")]
            claim_generations,
            withdraw_claims,
        })
    }

    /// Map a pool-local [`PresentLease`] to a scheme-unique present binding id.
    ///
    /// Two leases from different pools that both use local id `0` receive distinct
    /// binding ids. Reusing the same lease returns the same binding.
    #[cfg(feature = "graphics")]
    fn intern_present_binding(&mut self, lease: &PresentLease) -> u32 {
        for (i, binding) in self.present_bindings.iter().enumerate() {
            if Arc::ptr_eq(&binding.pool, &lease.pool) && binding.pool_lease_id == lease.id {
                return i as u32;
            }
        }
        let id = self.present_bindings.len() as u32;
        self.present_bindings.push(PresentBinding {
            pool: Arc::clone(&lease.pool),
            pool_lease_id: lease.id,
        });
        id
    }

    /// True when this scheme already has a present exchange transaction for `lease`.
    #[cfg(feature = "graphics")]
    pub(crate) fn has_present_transaction_for(&self, lease: &PresentLease) -> bool {
        self.present_bindings.iter().enumerate().any(|(i, binding)| {
            Arc::ptr_eq(&binding.pool, &lease.pool)
                && binding.pool_lease_id == lease.id
                && self.present_transactions.iter().any(|t| t.binding_id == i as u32)
        })
    }

    /// Record a present exchange over a swapchain lease and return its transaction.
    ///
    /// Metadata-only: does not append IR nodes or touch the drawable. Real
    /// [`PresentLease`] bindings in dispatch/copy nodes determine deferred acquire,
    /// ordering, and settlement. Called by [`crate::SurfaceExchange`] bind helpers.
    /// Calling twice for the same lease reuses the existing transaction rather than
    /// creating a second claim slot.
    #[cfg(feature = "graphics")]
    pub(crate) fn register_present_exchange(&mut self, lease: &PresentLease) -> Transaction {
        let binding_id = self.intern_present_binding(lease);
        let generation = lease.generation_handle();
        let present_idx = if let Some((idx, _)) = self
            .present_transactions
            .iter()
            .enumerate()
            .find(|(_, t)| t.binding_id == binding_id)
        {
            idx as u32
        } else {
            self.mark_structure_dirty();
            let present_idx = self.present_transactions.len() as u32;
            self.present_transactions.push(PresentTransactionInfo {
                binding_id,
                pool: Arc::clone(&lease.pool),
                pool_lease_id: lease.id,
            });
            present_idx
        };
        Transaction {
            scheme_id: self.scheme_id,
            key: ClaimKey::Present { present_idx },
            binding_id,
            generation,
        }
    }

    /// Copy an offscreen render target into a present lease drawable.
    #[cfg(feature = "graphics")]
    pub fn copy_to_present(&mut self, src: &Lease<LeaseRenderTarget>, dst: &PresentLease) {
        self.mark_structure_dirty();
        let binding_id = self.intern_present_binding(dst);
        let handle = self.rt_leases[src.id.0 as usize].backend_handle();
        self.ir.nodes.push(TaskNode {
            label: "copy_to_present",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::RenderTarget(handle),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::PresentLease(binding_id),
                    access: NodeAccess::Overwrite,
                },
            ],
            kind: NodeKind::CopyRenderTarget {
                src: handle,
                dst: ResourceId::PresentLease(binding_id),
            },
        });
    }

    /// Copy a texture (UAV-writable parcel) into a present lease drawable.
    ///
    /// Copy a texture into a present lease (swapchain drawable).
    /// but targets a scheme [`PresentLease`] instead of the task-graph swapchain output.
    ///
    /// Record this after all compute nodes that write `src`. The present slot is
    /// resolved by [`Self::submit`] at acquire time — the same partition-slot-key
    /// mechanism used by [`Self::copy_to_present`].
    #[cfg(feature = "graphics")]
    pub fn copy_texture_to_present(&mut self, src: &crate::Texture, dst: &PresentLease) {
        self.mark_structure_dirty();
        let binding_id = self.intern_present_binding(dst);
        let src_h = src.gpu_handle();
        let stamp = src.whole().stamp_handle();
        self.submit_state
            .register_stamp_parts(ResourceId::Texture(src_h), stamp);
        self.ir.nodes.push(TaskNode {
            label: "copy_texture_to_present",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::Texture(src_h),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::PresentLease(binding_id),
                    access: NodeAccess::Overwrite,
                },
            ],
            kind: NodeKind::CopyTexture {
                src: src_h,
                dst: ResourceId::PresentLease(binding_id),
                dst_buffer_layout: None,
            },
        });
    }

    /// Full-texture GPU copy between two textures (same size and format).
    ///
    /// `src` needs [`TextureFlags::COPY_SRC`]; `dst` needs [`TextureFlags::COPY_DST`].
    /// Convenience wrapper around [`Self::copy_texture_region`].
    pub fn copy_texture(&mut self, src: &crate::Texture, dst: &crate::Texture) -> Result<(), GoldyError> {
        if src.width() != dst.width() || src.height() != dst.height() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_texture: size mismatch {}x{} → {}x{}",
                src.width(),
                src.height(),
                dst.width(),
                dst.height()
            )));
        }
        self.copy_texture_region(src, 0, 0, dst, 0, 0, src.width(), src.height())
    }

    /// Copy a rectangular texel region from `src` into `dst`.
    ///
    /// Formats must match. The region must be non-empty and in-bounds on both textures.
    /// Same-texture overlapping copies are rejected. `src` needs [`TextureFlags::COPY_SRC`];
    /// `dst` needs [`TextureFlags::COPY_DST`].
    #[allow(clippy::too_many_arguments)]
    pub fn copy_texture_region(
        &mut self,
        src: &crate::Texture,
        src_x: u32,
        src_y: u32,
        dst: &crate::Texture,
        dst_x: u32,
        dst_y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), GoldyError> {
        if width == 0 || height == 0 {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_texture_region: extent must be non-zero (got {width}x{height})"
            )));
        }
        if src.format() != dst.format() {
            return Err(GoldyError::Validation(format!(
                "copy_texture_region: format mismatch {:?} → {:?}. \
                 hint: copies require identical TextureFormat on src and dst; \
                 convert in a compute/render pass if you need a format change.",
                src.format(),
                dst.format()
            )));
        }
        if !src.flags().contains(TextureFlags::COPY_SRC) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_texture_region: source requires TextureFlags::COPY_SRC"
            )));
        }
        if !dst.flags().contains(TextureFlags::COPY_DST) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_texture_region: destination requires TextureFlags::COPY_DST"
            )));
        }
        let src_x_end = src_x
            .checked_add(width)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("copy_texture_region: src x+width overflow")))?;
        let src_y_end = src_y
            .checked_add(height)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("copy_texture_region: src y+height overflow")))?;
        let dst_x_end = dst_x
            .checked_add(width)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("copy_texture_region: dst x+width overflow")))?;
        let dst_y_end = dst_y
            .checked_add(height)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("copy_texture_region: dst y+height overflow")))?;
        if src_x_end > src.width() || src_y_end > src.height() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_texture_region: src region {}x{} at ({},{}) exceeds {}x{}",
                width,
                height,
                src_x,
                src_y,
                src.width(),
                src.height()
            )));
        }
        if dst_x_end > dst.width() || dst_y_end > dst.height() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_texture_region: dst region {}x{} at ({},{}) exceeds {}x{}",
                width,
                height,
                dst_x,
                dst_y,
                dst.width(),
                dst.height()
            )));
        }
        let src_h = src.gpu_handle();
        let dst_h = dst.gpu_handle();
        if src_h == dst_h {
            let src_rect = (src_x, src_y, src_x_end, src_y_end);
            let dst_rect = (dst_x, dst_y, dst_x_end, dst_y_end);
            let overlap = src_rect.0 < dst_rect.2
                && dst_rect.0 < src_rect.2
                && src_rect.1 < dst_rect.3
                && dst_rect.1 < src_rect.3;
            if overlap {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "copy_texture_region: overlapping same-texture copies are not supported"
                )));
            }
        }

        self.mark_structure_dirty();
        self.submit_state
            .register_stamp_parts(ResourceId::Texture(src_h), src.whole().stamp_handle());
        self.submit_state
            .register_stamp_parts(ResourceId::Texture(dst_h), dst.whole().stamp_handle());
        let full_dst = dst_x == 0 && dst_y == 0 && width == dst.width() && height == dst.height();
        let dst_access = if full_dst {
            NodeAccess::Overwrite
        } else {
            NodeAccess::Write
        };
        // Full-texture copies keep the compact `CopyTexture` node (and its present/readback
        // variants). Partial copies use `CopyTextureRegion`.
        if full_dst && src_x == 0 && src_y == 0 && width == src.width() && height == src.height() {
            self.ir.nodes.push(TaskNode {
                label: "copy_texture",
                bindings: vec![
                    ResourceBinding {
                        resource: ResourceId::Texture(src_h),
                        access: NodeAccess::Read,
                    },
                    ResourceBinding {
                        resource: ResourceId::Texture(dst_h),
                        access: NodeAccess::Overwrite,
                    },
                ],
                kind: NodeKind::CopyTexture {
                    src: src_h,
                    dst: ResourceId::Texture(dst_h),
                    dst_buffer_layout: None,
                },
            });
        } else {
            self.ir.nodes.push(TaskNode {
                label: "copy_texture_region",
                bindings: vec![
                    ResourceBinding {
                        resource: ResourceId::Texture(src_h),
                        access: NodeAccess::Read,
                    },
                    ResourceBinding {
                        resource: ResourceId::Texture(dst_h),
                        access: dst_access,
                    },
                ],
                kind: NodeKind::CopyTextureRegion {
                    src: src_h,
                    dst: dst_h,
                    src_x,
                    src_y,
                    dst_x,
                    dst_y,
                    width,
                    height,
                },
            });
        }
        Ok(())
    }

    /// Copy an offscreen render target into a texture deed parcel (for CPU readback via
    /// [`crate::MemoryExchange::bind_withdraw`]).
    ///
    /// The destination must be a texture parcel with [`TextureFlags::COPY_DST`], homed on
    /// this scheme's context, and matching the render target's width, height, and format.
    #[cfg(feature = "graphics")]
    pub fn copy_to_texture(&mut self, src: &Lease<LeaseRenderTarget>, dst: &Parcel) -> Result<(), GoldyError> {
        let src_rt = &self.rt_leases[src.id.0 as usize];
        if !dst.is_homed_on(&self.ctx) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "parcel home device does not match scheme context"
            )));
        }
        dst.texture_handle()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("copy_to_texture requires texture parcel")))?;
        let (width, height, format, _, flags) = dst
            .texture_descriptor()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("copy_to_texture requires texture parcel")))?;
        if !flags.contains(TextureFlags::COPY_DST) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_to_texture requires TextureFlags::COPY_DST"
            )));
        }
        if width == 0 || height == 0 {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_to_texture requires non-zero texture dimensions"
            )));
        }
        if width != src_rt.width() || height != src_rt.height() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_to_texture: texture {width}x{height} does not match render target {}x{}",
                src_rt.width(),
                src_rt.height()
            )));
        }
        if format != src_rt.format() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_to_texture: texture format {format:?} does not match render target {:?}",
                src_rt.format()
            )));
        }

        let src_handle = src_rt.backend_handle();
        self.mark_structure_dirty();
        self.submit_state.register_parcel_stamp(dst);
        let dst_resource = dst.resource_id();
        self.ir.nodes.push(TaskNode {
            label: "copy_to_texture",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::RenderTarget(src_handle),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: dst_resource,
                    access: NodeAccess::Overwrite,
                },
            ],
            kind: NodeKind::CopyRenderTarget {
                src: src_handle,
                dst: dst_resource,
            },
        });
        Ok(())
    }

    /// Begin recording an offscreen render pass on this scheme.
    #[cfg(feature = "graphics")]
    pub fn render_pass<'a>(
        &'a mut self,
        label: &'static str,
        rt: &Lease<LeaseRenderTarget>,
        color_load: crate::types::TargetLoad,
    ) -> SchemeRenderPassBuilder<'a> {
        self.mark_structure_dirty();
        let handle = self.rt_leases[rt.id.0 as usize].backend_handle();
        let access = if color_load.overwrites() {
            NodeAccess::Overwrite
        } else {
            NodeAccess::Write
        };
        SchemeRenderPassBuilder {
            scheme: self,
            label,
            target: handle,
            color_load,
            bindings: vec![ResourceBinding {
                resource: ResourceId::RenderTarget(handle),
                access,
            }],
            commands: Vec::new(),
            pending_push_constants: Vec::new(),
            pending_named: HashMap::new(),
        }
    }
}

impl Scheme {
    /// Number of IR nodes recorded in the scheme.
    ///
    /// Intended for tests and debug tooling only. Do **not** use for synchronisation.
    #[doc(hidden)]
    pub fn ir_node_count(&self) -> usize {
        self.ir.nodes.len()
    }

    /// Recorded task-graph nodes (tests / diagnostics only).
    #[doc(hidden)]
    pub fn ir_nodes(&self) -> &[crate::task_graph::TaskNode] {
        &self.ir.nodes
    }

    /// True when the IR contains a copy-to-present blit node.
    #[cfg(feature = "graphics")]
    #[doc(hidden)]
    pub fn test_has_copy_render_target_to_present(&self) -> bool {
        use crate::task_graph::{NodeKind, ResourceId};
        self.ir.nodes.iter().any(|node| {
            matches!(
                &node.kind,
                NodeKind::CopyRenderTarget { dst, .. }
                    if matches!(dst, ResourceId::PresentLease(_))
            )
        })
    }

    /// True when any dispatch node binds a present lease.
    #[cfg(feature = "graphics")]
    #[doc(hidden)]
    pub fn test_has_present_lease_dispatch_binding(&self) -> bool {
        use crate::task_graph::ResourceId;
        self.ir.nodes.iter().any(|node| {
            node.bindings
                .iter()
                .any(|b| matches!(b.resource, ResourceId::PresentLease(_)))
        })
    }

    /// Resolve pending deposit stages into concrete handles for this submit.
    fn resolve_deposits_for_submit(
        &self,
    ) -> Result<std::collections::HashMap<u32, crate::task_graph::ResolvedDeposit>, GoldyError> {
        let mut out = std::collections::HashMap::new();
        let mut referenced = std::collections::HashSet::new();
        for node in &self.ir.nodes {
            for b in &node.bindings {
                if let ResourceId::Deposit(id) = b.resource {
                    referenced.insert(id);
                }
            }
        }
        for id in referenced {
            let pool = self
                .deposits
                .get(id as usize)
                .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("submit: IR references unknown Deposit({id})")))?;
            let resolved = pool.resolve_pending().ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!("submit: Deposit({id}) was not written before submit"))
            })?;
            out.insert(id, resolved);
        }
        Ok(out)
    }

    /// Test/telemetry: number of physical parcels owned by deposit `id`.
    #[doc(hidden)]
    pub fn deposit_parcel_count(&self, deposit: &crate::exchange::DepositTransaction) -> usize {
        self.deposits
            .get(deposit.deposit_id as usize)
            .map(|p| p.parcels.len())
            .unwrap_or(0)
    }

    /// Test helper: mark the first physical deposit parcel as still in flight at `tv`.
    #[doc(hidden)]
    pub fn test_mark_deposit_inflight(&mut self, deposit: &crate::exchange::DepositTransaction, tv: TimelineValue) {
        let ctx = self.ctx.backend_handle();
        let pool = &mut self.deposits[deposit.deposit_id as usize];
        pool.pending = None;
        if let Some(parcel) = pool.parcels.first() {
            parcel.mark_referenced(ctx, tv);
        }
    }

    /// Test helper: number of retained CB slot variants across all partitions.
    #[doc(hidden)]
    pub fn test_retained_slot_variant_count(&self) -> usize {
        self.submit_state.retained_slot_variant_count()
    }

    /// Test helper: whether a CB replay ledger is attached.
    #[doc(hidden)]
    pub fn test_has_cb_replay(&self) -> bool {
        self.submit_state.has_cb_replay()
    }
}

impl Drop for Scheme {
    fn drop(&mut self) {
        for withdraw in &self.withdraws {
            withdraw.staging_pool.mark_scheme_dropped_and_drain();
        }

        use crate::task_graph::cross_submit::clear_scheme_topology_registration;
        clear_scheme_topology_registration(self.scheme_id, &self.prev_topology_parcels);

        let hw = self.ctx.high_water_timeline();
        let progress = self.ctx.gpu_progress();
        // Skip when already retired: wait_until → finish_timeline_wait used to flush the
        // submission worker even when progress >= hw, deadlocking multi-window teardown.
        if hw > 0 && progress < hw {
            let _ = self.ctx.wait_until(hw);
        }
        self.submit_state.release_backend_retained_graphs(&self.ctx);

        let ctx = self.ctx.clone();
        for pool in std::mem::take(&mut self.deposits) {
            pool.return_all(&ctx);
        }
        for exec in std::mem::take(&mut self.cpu_dispatches) {
            exec.release(&ctx);
        }
        for mut parcel in self.leases.drain(..) {
            let ready_after = parcel.last_referenced();
            parcel.release_bookkeeping();
            if parcel.texture_descriptor().is_some() {
                let home_device = parcel.home_device().clone();
                let texture = crate::Texture::from_returned_parcel(parcel, home_device);
                ctx.with_transient_pool(|pool| {
                    pool.adopt(StampedParcel {
                        hold: crate::retained_pool::RetainedHold::Texture(texture),
                        ready_after,
                    });
                });
            } else {
                ctx.with_transient_pool(|pool| pool.return_buffer_parcel(parcel, ready_after));
            }
        }
        #[cfg(feature = "graphics")]
        self.rt_leases.clear();
    }
}

impl Scheme {
    /// Register a memory withdrawal over a buffer or texture deed parcel.
    ///
    /// Called by [`crate::MemoryExchange::bind_withdraw`].
    pub(crate) fn register_withdraw(
        &mut self,
        parcel: &Parcel,
    ) -> Result<crate::exchange::WithdrawTransaction, GoldyError> {
        self.mark_structure_dirty();
        self.submit_state.register_parcel_stamp(parcel);
        if !parcel.is_homed_on(&self.ctx) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "parcel home device does not match scheme context"
            )));
        }

        let (source, byte_size, read_kind, staging_pool) = if parcel.buffer_handle().is_some() {
            let source_backing = parcel.grant_buffer_keepalive().map_err(|e| self.ctx.classify(e))?;
            let source = parcel.buffer_handle().ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!("bind_withdraw requires buffer or texture parcel"))
            })?;
            let byte_size = parcel.byte_size();
            if byte_size == 0 {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "bind_withdraw requires non-zero buffer byte size"
                )));
            }
            let staging_pool = WithdrawStagingPool::new_buffer(&self.ctx, byte_size);
            (
                WithdrawSource::Buffer {
                    source,
                    src_offset: parcel.source_offset(),
                    source_backing,
                    byte_size,
                },
                byte_size,
                crate::exchange::WithdrawReadKind::Buffer,
                staging_pool,
            )
        } else if parcel.texture_handle().is_some() {
            let source_backing = parcel.grant_texture_keepalive().map_err(|e| self.ctx.classify(e))?;
            let source = parcel.texture_handle().ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!("bind_withdraw requires buffer or texture parcel"))
            })?;
            let (width, height, format, access, flags) = parcel.texture_descriptor().ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!("bind_withdraw requires buffer or texture parcel"))
            })?;
            if !flags.contains(TextureFlags::COPY_SRC) {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "bind_withdraw texture requires TextureFlags::COPY_SRC"
                )));
            }
            if matches!(access, TextureKind::Interpolated) {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "bind_withdraw texture requires a storage-writable texture (TextureKind::Direct or DirectInterpolated); \
                     TextureKind::Interpolated is sampled-only and cannot be a compute output"
                )));
            }
            if width == 0 || height == 0 {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "bind_withdraw texture requires non-zero texture dimensions"
                )));
            }
            let layout = {
                let query_result = {
                    let backend = self.ctx.device().inner.backend.lock().unwrap();
                    backend.query_texture_copy_footprint(self.ctx.device().inner.handle, width, height, format)
                };
                query_result.map_err(|e| self.ctx.classify(e))?
            };
            let staging_pool = WithdrawStagingPool::new_texture(&self.ctx, layout);
            (
                WithdrawSource::Texture {
                    source,
                    source_backing,
                    layout,
                },
                layout.logical_bytes,
                crate::exchange::WithdrawReadKind::Texture(layout),
                staging_pool,
            )
        } else {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "bind_withdraw requires buffer or texture parcel"
            )));
        };

        let ir_withdraw_id = self.next_withdraw_id;
        self.next_withdraw_id += 1;
        let withdraw_idx = self.withdraws.len() as u32;
        self.withdraws.push(WithdrawInfo {
            source,
            staging_pool: Arc::clone(&staging_pool),
        });
        let resource = parcel.resource_id();
        self.ir.nodes.push(TaskNode {
            label: "withdraw",
            bindings: vec![ResourceBinding {
                resource,
                access: NodeAccess::Read,
            }],
            kind: NodeKind::WithdrawRead {
                withdraw_id: ir_withdraw_id,
            },
        });
        Ok(crate::exchange::WithdrawTransaction {
            scheme_id: self.scheme_id,
            key: ClaimKey::Withdraw { withdraw_idx },
            byte_size,
            read_kind,
            ctx: self.ctx.clone(),
        })
    }
}

pub(crate) fn node_access_to_resource_access(access: NodeAccess) -> ResourceAccess {
    match access {
        NodeAccess::Read => ResourceAccess::Read,
        NodeAccess::Write | NodeAccess::Overwrite => ResourceAccess::Write,
        NodeAccess::ReadWrite => ResourceAccess::ReadWrite,
    }
}

const DISPATCH_SHAPE_BYTE_SIZE: u64 = std::mem::size_of::<DispatchShape>() as u64;
const DISPATCH_SHAPE_STRIDE: u32 = DISPATCH_SHAPE_BYTE_SIZE as u32;

fn validate_dispatch_shape_parcel(parcel: &Parcel) -> Result<u64, GoldyError> {
    if parcel.buffer_handle().is_none() {
        return Err(GoldyError::Backend(anyhow::anyhow!(
            "dispatch(shape parcel): requires a buffer parcel holding a DispatchShape"
        )));
    }
    if parcel.byte_size() < DISPATCH_SHAPE_BYTE_SIZE {
        return Err(GoldyError::Backend(anyhow::anyhow!(
            "dispatch(shape parcel): parcel byte size {} is smaller than DispatchShape ({} bytes)",
            parcel.byte_size(),
            DISPATCH_SHAPE_BYTE_SIZE
        )));
    }
    match parcel.buffer_element_stride() {
        Some(stride) if stride == DISPATCH_SHAPE_STRIDE => {}
        Some(stride) => {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "dispatch(shape parcel): expected element stride {DISPATCH_SHAPE_STRIDE}, got {stride}"
            )));
        }
        None if parcel.byte_size() == DISPATCH_SHAPE_BYTE_SIZE => {}
        None => {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "dispatch(shape parcel): expected element stride {DISPATCH_SHAPE_STRIDE}"
            )));
        }
    }
    Ok(parcel.source_offset())
}

/// Binding surface for [`SchemeNodeBuilder::with_parcel`]: deeds, acquired buffers, scheme-held
/// leases, samplers, and textures.
///
/// Returns a `(resource_identity, bindless_slot_index)` pair where:
/// - `resource_identity` is `Some((ResourceId, Option<ParcelStamp>))` for resources that
///   participate in barrier generation. The stamp is `Some` for parcel-backed resources
///   (buffers, textures) that also participate in cross-scheme hazard tracking.
///   `resource_identity` is `None` for barrier-free resources such as samplers, which only need
///   a bindless slot.
/// - `bindless_slot_index` is the raw heap index to write into the push-constant layout.
type SchemeBindIdentity = Option<(ResourceId, Option<Arc<crate::parcel::ParcelStamp>>)>;
type SchemeBindResult = (SchemeBindIdentity, Option<u32>);

pub(crate) trait SchemeBindable {
    fn resolve(&self, scheme: &Scheme, access: ResourceAccess) -> SchemeBindResult;
    fn accel_kind(&self) -> Option<crate::accel::AccelKind> {
        None
    }

    fn prior_built_accel_handle(&self) -> Option<u64> {
        None
    }
}

impl SchemeBindable for Parcel {
    fn resolve(&self, _: &Scheme, access: ResourceAccess) -> SchemeBindResult {
        (
            Some((self.resource_id(), Some(self.stamp_handle()))),
            self.resource_index(access),
        )
    }
}

impl SchemeBindable for crate::Buffer {
    fn resolve(&self, _: &Scheme, access: ResourceAccess) -> SchemeBindResult {
        let parcel = self.whole();
        (
            Some((parcel.resource_id(), Some(parcel.stamp_handle()))),
            parcel.resource_index(access),
        )
    }
}

impl SchemeBindable for crate::buffer::Allocation {
    fn resolve(&self, _: &Scheme, access: ResourceAccess) -> SchemeBindResult {
        (
            Some((ResourceId::Buffer(self.handle), None)),
            self.resource_index(access),
        )
    }
}

impl<T> SchemeBindable for Lease<T> {
    fn resolve(&self, scheme: &Scheme, access: ResourceAccess) -> SchemeBindResult {
        let parcel = &scheme.leases[self.id.0 as usize];
        // TODO(inaugural-check): enforce that the first access to a buffer lease is Write,
        // Overwrite, or ReadWrite — never pure Read. The pool may recycle a buffer whose bytes
        // come from a previous submission; a Read-only first access would observe stale data.
        // This requires a per-scheme "has-been-written" bit per lease slot; deferred until
        // the unique-minimal-write shape-check lands (design §8).
        (
            Some((parcel.resource_id(), Some(parcel.stamp_handle()))),
            parcel.resource_index(access),
        )
    }
}

impl SchemeBindable for crate::Sampler {
    fn resolve(&self, _: &Scheme, access: ResourceAccess) -> SchemeBindResult {
        // Samplers carry no GPU-written data: no RAW/WAW hazard, no barrier, no stamp.
        // Only the bindless heap index is needed.
        (None, self.resource_index(access))
    }
}

impl SchemeBindable for crate::AccelerationStructure {
    fn resolve(&self, _: &Scheme, access: ResourceAccess) -> SchemeBindResult {
        let _ = access;
        (
            Some((self.resource_id(), None)),
            self.resource_index(crate::types::ResourceAccess::Read),
        )
    }

    fn accel_kind(&self) -> Option<crate::accel::AccelKind> {
        Some(self.kind)
    }

    fn prior_built_accel_handle(&self) -> Option<u64> {
        self.is_gpu_built().then_some(self.handle)
    }
}

impl SchemeBindable for crate::Texture {
    fn resolve(&self, _: &Scheme, access: ResourceAccess) -> SchemeBindResult {
        // `TextureKind::Direct` storage images have no SRV; when a shader slot is reflected
        // as read-only (ResourceAccess::Read) but the texture only has a UAV descriptor,
        // fall back to the UAV bindless index — matching historical submit behaviour in
        // `collect_bindless_indices_into`.
        let slot = self.resource_index(access).or_else(|| {
            if access == ResourceAccess::Read {
                self.resource_index(ResourceAccess::Write)
                    .or_else(|| self.resource_index(ResourceAccess::ReadWrite))
            } else {
                None
            }
        });
        let parcel = self.whole();
        (Some((parcel.resource_id(), Some(parcel.stamp_handle()))), slot)
    }
}

/// Builder for a single compute dispatch node within a [`Scheme`].
pub struct SchemeNodeBuilder<'a> {
    scheme: &'a mut Scheme,
    label: &'static str,
    pipeline: crate::backend::ComputePipelineHandle,
    rt_pipeline: Option<crate::backend::RayTracingPipelineHandle>,
    bindings: Vec<ResourceBinding>,
    resource_slots: Vec<u32>,
    user_slots: Vec<u32>,
    /// Per-slot descriptor access required by the shader signature (from pipeline
    /// reflection), in shader-signature order. Lets [`Self::with_parcel`] pick the
    /// correct SRV/UAV descriptor independent of the graph [`NodeAccess`].
    slot_access: Vec<Option<ResourceAccess>>,
}

impl<'a> SchemeNodeBuilder<'a> {
    /// Declare that this node accesses a bindable resource (retained deed or scheme-held lease).
    ///
    /// The resource's bindless index is appended to `resource_slots` in call order.
    /// The nth call corresponds to the nth resource-kind parameter in the shader signature.
    #[allow(private_bounds)]
    pub fn with_parcel(mut self, bindable: &impl SchemeBindable, access: NodeAccess) -> Self {
        // The graph `access` drives barriers; the *descriptor* (SRV vs UAV) is chosen
        // from the shader signature's reflected requirement for this slot, so a
        // `Scattered<T>` read still binds its UAV without the caller passing raw handles.
        // Slots with no reflected preference fall back to the graph access.
        let slot_idx = self.resource_slots.len();
        let descriptor_access = self
            .slot_access
            .get(slot_idx)
            .copied()
            .flatten()
            .unwrap_or_else(|| node_access_to_resource_access(access));
        let (resource_identity, slot) = bindable.resolve(self.scheme, descriptor_access);
        if bindable.accel_kind() == Some(crate::accel::AccelKind::Blas) {
            self.scheme.record_errors.push(
                "with_parcel bound a BLAS as a shader Accel parameter. \
                 hint: RayQuery / TraceRay take a TLAS. Build the BLAS, then Scheme::build_tlas, \
                 and bind the TLAS."
                    .into(),
            );
        }
        if let Some(h) = bindable.prior_built_accel_handle() {
            self.scheme.prior_built_accels.insert(h);
        }
        let slot = slot.unwrap_or_else(|| {
            panic!(
                "with_parcel: resource has no descriptor for {access:?} access; \
                 check BufferKind/TextureKind is compatible with NodeAccess"
            );
        });
        if let Some((resource, maybe_stamp)) = resource_identity {
            if let Some(stamp) = maybe_stamp {
                self.scheme.submit_state.register_stamp_parts(resource, stamp);
            }
            self.bindings.push(ResourceBinding { resource, access });
        }
        self.resource_slots.push(slot);
        self
    }

    /// Register dependency on all parcels of a buffer without emitting shader slots.
    pub fn with_buffer_dependency(mut self, buffer: &crate::Buffer, access: NodeAccess) -> Self {
        self.scheme.submit_state.register_buffer_stamps(buffer);
        for parcel in buffer.parcels() {
            self.bindings.push(ResourceBinding {
                resource: parcel.resource_id(),
                access,
            });
        }
        self
    }

    /// Append one scalar virtual-main parameter (region B).
    ///
    /// The nth call corresponds to the nth scalar-kind parameter in the shader signature.
    /// Values are u32 wire words (`f32` via `f32::to_bits()`, etc.).
    pub fn with_param(mut self, value: u32) -> Self {
        use crate::backend::shared::MAX_USER_SLOTS;
        assert!(
            self.user_slots.len() < MAX_USER_SLOTS,
            "with_param: at most {MAX_USER_SLOTS} scalar params per dispatch"
        );
        self.user_slots.push(value);
        self
    }

    /// Declare explicit resource view handles (internal: present-lease slot bookkeeping).
    ///
    /// Replaces resource slot indices while preserving trailing
    /// [`PRESENT_LEASE_SLOT_PLACEHOLDER`] entries appended by [`Self::with_present`].
    #[cfg(test)]
    pub(crate) fn with_views(mut self, handles: &[crate::types::ResourceHandle]) -> Self {
        let trailing_placeholders: Vec<u32> = self
            .resource_slots
            .iter()
            .rev()
            .take_while(|&&s| s == PRESENT_LEASE_SLOT_PLACEHOLDER)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        self.resource_slots = handles.iter().map(|h| h.index()).collect();
        self.resource_slots.extend_from_slice(&trailing_placeholders);
        self
    }

    /// Declare access to a present lease (swapchain drawable) at the current shader
    /// resource slot index.
    ///
    /// Inserts [`PRESENT_LEASE_SLOT_PLACEHOLDER`] at the current position in
    /// `resource_slots` so the resolver can patch it to the correct bindless index
    /// at submit time. Call order must match the shader's resource parameter order.
    #[cfg(feature = "graphics")]
    pub fn with_present_access(mut self, lease: &PresentLease, access: NodeAccess) -> Self {
        let binding_id = self.scheme.intern_present_binding(lease);
        self.bindings.push(ResourceBinding {
            resource: ResourceId::PresentLease(binding_id),
            access,
        });
        self.resource_slots.push(PRESENT_LEASE_SLOT_PLACEHOLDER);
        self
    }

    /// Declare a UAV write to a present lease (swapchain drawable).
    #[cfg(feature = "graphics")]
    pub fn with_present(self, lease: &PresentLease) -> Self {
        self.with_present_access(lease, NodeAccess::Write)
    }

    /// Finalize the node with fixed workgroup dimensions.
    ///
    /// The returned [`NodeId`] addresses this node in [`Scheme::set_node_pipeline`],
    /// [`Scheme::set_node_dispatch`], and [`Scheme::set_node_param`].
    pub fn dispatch(self, x: u32, y: u32, z: u32) -> NodeId {
        self.push_dispatch_node(DispatchDim::Direct { x, y, z })
    }

    /// Finalize the node with a device-resident [`DispatchShape`] parcel (indirect dispatch).
    ///
    /// The shape parcel's ordering dependency is registered automatically and is not a shader
    /// resource slot. Fixed workgroup counts use [`Self::dispatch`] instead.
    pub fn dispatch_shape_parcel(self, parcel: &Parcel) -> Result<NodeId, GoldyError> {
        if self.rt_pipeline.is_some() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "trace_rays does not support indirect DispatchShape parcels; use dispatch(width, height, depth)"
            )));
        }
        let offset = validate_dispatch_shape_parcel(parcel)?;
        let resource = parcel.resource_id();
        self.scheme
            .submit_state
            .register_stamp_parts(resource, parcel.stamp_handle());
        let mut bindings = self.bindings;
        bindings.push(ResourceBinding {
            resource,
            access: NodeAccess::Read,
        });
        let buffer = parcel
            .buffer_handle()
            .expect("validate_dispatch_shape_parcel ensures buffer parcel");
        self.scheme.ir.nodes.push(TaskNode {
            label: self.label,
            bindings,
            kind: NodeKind::Dispatch {
                pipeline: self.pipeline,
                resource_slots: self.resource_slots,
                user_slots: self.user_slots,
                dispatch: DispatchDim::Indirect { buffer, offset },
            },
        });
        Ok(self.scheme.last_node_id())
    }

    fn push_dispatch_node(self, dispatch: DispatchDim) -> NodeId {
        #[cfg(feature = "graphics")]
        {
            let present_bindings = self
                .bindings
                .iter()
                .filter(|b| matches!(b.resource, ResourceId::PresentLease(_)))
                .count();
            let present_slots = self
                .resource_slots
                .iter()
                .filter(|&&s| s == PRESENT_LEASE_SLOT_PLACEHOLDER)
                .count();
            debug_assert_eq!(
                present_bindings, present_slots,
                "present lease bindings must align with PRESENT_LEASE_SLOT_PLACEHOLDER entries (label={})",
                self.label
            );
        }
        // Do not assert resource_slots.len() >= bindings.len(): samplers add slots
        // without bindings, and with_buffer_dependency adds bindings without slots.
        // Present placeholders are resolved by declaration order, not binding index.
        self.scheme.ir.nodes.push(TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: if let Some(rt) = self.rt_pipeline {
                let (width, height, depth) = match dispatch {
                    DispatchDim::Direct { x, y, z } => (x, y, z),
                    DispatchDim::Indirect { .. } => {
                        panic!("trace_rays does not support indirect dispatch")
                    }
                };
                NodeKind::TraceRays {
                    pipeline: rt,
                    resource_slots: self.resource_slots,
                    user_slots: self.user_slots,
                    width,
                    height,
                    depth,
                }
            } else {
                NodeKind::Dispatch {
                    pipeline: self.pipeline,
                    resource_slots: self.resource_slots,
                    user_slots: self.user_slots,
                    dispatch,
                }
            },
        });
        self.scheme.last_node_id()
    }
}

/// Named or positional shader resource for a graphics draw.
///
/// Named bindings (`ShaderBinding::read("scene", &scene)`) are resolved against the
/// pipeline's merged virtual-main contract at [`SchemeRenderPassBuilder::set_pipeline`].
/// Extra names are allowed so one pass-level set can serve multiple pipeline switches.
#[cfg(feature = "graphics")]
pub struct ShaderBinding<'a> {
    name: &'a str,
    slot: ShaderResourceSlot<'a>,
}

#[cfg(feature = "graphics")]
impl<'a> ShaderBinding<'a> {
    /// Bind a read-only parcel to the pipeline resource named `name`.
    pub fn read(name: &'a str, parcel: &'a Parcel) -> Self {
        Self {
            name,
            slot: ShaderResourceSlot::Parcel {
                parcel,
                access: NodeAccess::Read,
            },
        }
    }

    /// Bind a writable parcel to the pipeline resource named `name`.
    pub fn write(name: &'a str, parcel: &'a Parcel) -> Self {
        Self {
            name,
            slot: ShaderResourceSlot::Parcel {
                parcel,
                access: NodeAccess::Write,
            },
        }
    }

    /// Bind a sampler to the pipeline resource named `name`.
    pub fn sampler(name: &'a str, sampler: &'a crate::Sampler) -> Self {
        Self {
            name,
            slot: ShaderResourceSlot::Sampler(sampler),
        }
    }
}

/// One parcel bound to a CPU dispatch before [`SchemeCpuNodeBuilder::dispatch`] runs.
struct PendingCpuBinding {
    resource: ResourceId,
    stamp: Arc<crate::parcel::ParcelStamp>,
    buffer: BufferHandle,
    offset: u64,
    byte_size: u64,
    keepalive: Arc<Allocation>,
    access: NodeAccess,
}

impl PendingCpuBinding {
    fn from_parcel(label: &str, parcel: &Parcel, access: NodeAccess) -> Result<Self, GoldyError> {
        let Some(buffer) = parcel.buffer_handle() else {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "cpu_node({label}): with_parcel requires a buffer parcel (textures are not supported)"
            )));
        };
        let keepalive = parcel.grant_buffer_keepalive().map_err(GoldyError::Backend)?;
        Ok(Self {
            resource: parcel.resource_id(),
            stamp: parcel.stamp_handle(),
            buffer,
            offset: parcel.source_offset(),
            byte_size: parcel.byte_size(),
            keepalive,
            access,
        })
    }
}

/// Builder for a CPU dispatch node within a [`Scheme`]; see [`Scheme::cpu_node`].
pub struct SchemeCpuNodeBuilder<'a> {
    scheme: &'a mut Scheme,
    label: &'static str,
    bindings: Vec<Result<PendingCpuBinding, GoldyError>>,
    params: Vec<u32>,
}

impl SchemeCpuNodeBuilder<'_> {
    /// Bind a whole buffer parcel as the next slice parameter of the virtual main.
    ///
    /// `access` drives both graph ordering and staging: `Read` downloads only,
    /// `Write` / `ReadWrite` download then upload, `Overwrite` uploads only (the slice
    /// arrives zeroed). The matching parameter must be `&[T]` for `Read` and `&mut [T]`
    /// otherwise, with the parcel byte size a multiple of `size_of::<T>()`.
    ///
    /// Errors are reported by [`Self::dispatch`].
    pub fn with_parcel(mut self, parcel: &Parcel, access: NodeAccess) -> Self {
        self.bindings
            .push(PendingCpuBinding::from_parcel(self.label, parcel, access));
        self
    }

    /// Bind a scheme-held buffer lease as the next slice parameter of the virtual main.
    pub fn with_lease(mut self, lease: &Lease<LeaseBuffer>, access: NodeAccess) -> Self {
        let parcel = &self.scheme.leases[lease.id.0 as usize];
        self.bindings
            .push(PendingCpuBinding::from_parcel(self.label, parcel, access));
        self
    }

    /// Append one scalar virtual-main parameter (`u32` wire word; `f32` via `to_bits()`).
    pub fn with_param(mut self, value: u32) -> Self {
        self.params.push(value);
        self
    }

    /// Finalize the node with its virtual main.
    ///
    /// Validates the function signature against the bindings (arity, mutability vs
    /// access, element size) and allocates the node's staging buffers. Fails without
    /// recording anything when validation or allocation fails.
    pub fn dispatch<M, F: CpuMain<M>>(self, main: F) -> Result<(), GoldyError> {
        let Self {
            scheme,
            label,
            bindings,
            params,
        } = self;
        let bindings = bindings.into_iter().collect::<Result<Vec<_>, _>>()?;
        let shapes: Vec<(NodeAccess, u64)> = bindings.iter().map(|b| (b.access, b.byte_size)).collect();
        crate::cpu_dispatch::validate_signature(label, &F::signature(), &shapes, params.len())?;

        let ctx = &scheme.ctx;
        let device = ctx.device();

        // Allocate all staging first so a failure part-way leaves nothing recorded.
        type Staging = (Option<BufferHandle>, Option<Parcel>);
        let release_staging = |staged: Vec<Staging>| {
            let mut backend = device.inner.backend.lock().unwrap();
            for (readback, _) in &staged {
                if let Some(h) = readback {
                    backend.free_readback_buffer(*h);
                }
            }
            drop(backend);
            for (_, upload) in staged {
                if let Some(mut p) = upload {
                    let ready_after = p.last_referenced();
                    p.release_bookkeeping();
                    ctx.with_transient_pool(|pool| pool.return_buffer_parcel(p, ready_after));
                }
            }
        };

        let mut staged: Vec<Staging> = Vec::with_capacity(bindings.len());
        for b in &bindings {
            // Every access except Overwrite lets the function observe the parcel's
            // current bytes, so it needs a device→host copy first.
            let readback = if b.access != NodeAccess::Overwrite && b.byte_size > 0 {
                let handle = {
                    let mut backend = device.inner.backend.lock().unwrap();
                    backend.alloc_readback_buffer(device.inner.handle, b.byte_size)
                };
                match handle {
                    Ok(h) => Some(h),
                    Err(e) => {
                        release_staging(staged);
                        return Err(ctx.classify(e));
                    }
                }
            } else {
                None
            };
            let upload = if b.access.writes() && b.byte_size > 0 {
                let acquired = ctx.with_transient_pool(|pool| {
                    pool.acquire_buffer(
                        ctx,
                        b.byte_size,
                        crate::types::BufferKind::Scattered,
                        BufferFlags::CPU_WRITABLE,
                        None,
                    )
                });
                match acquired {
                    Ok(p) => Some(p),
                    Err(e) => {
                        staged.push((readback, None));
                        release_staging(staged);
                        return Err(ctx.classify(e));
                    }
                }
            } else {
                None
            };
            staged.push((readback, upload));
        }

        let cpu_id = scheme.cpu_dispatches.len() as u32;
        let mut ir_bindings: Vec<ResourceBinding> = Vec::with_capacity(bindings.len());
        let mut execs: Vec<CpuBindingExec> = Vec::with_capacity(bindings.len());
        for (b, (readback, upload)) in bindings.into_iter().zip(staged) {
            scheme.submit_state.register_stamp_parts(b.resource, b.stamp);
            ir_bindings.push(ResourceBinding {
                resource: b.resource,
                access: b.access,
            });
            execs.push(CpuBindingExec {
                resource: b.resource,
                buffer: b.buffer,
                offset: b.offset,
                byte_size: b.byte_size,
                access: b.access,
                readback,
                upload,
                _keepalive: b.keepalive,
            });
        }
        scheme
            .cpu_dispatches
            .push(CpuDispatchExec::new(label, main, execs, params));
        scheme.mark_structure_dirty();
        scheme.ir.nodes.push(TaskNode {
            label,
            bindings: ir_bindings,
            kind: NodeKind::CpuDispatch { cpu_id },
        });
        Ok(())
    }
}

/// Deferred push-constant slot recorded before [`SchemeRenderPassBuilder::set_pipeline`].
///
/// Read and read-write handles are captured at record time; the descriptor actually
/// bound is chosen from pipeline reflection when the pipeline is set.
#[cfg(feature = "graphics")]
struct PendingPushConstant {
    graph_access: NodeAccess,
    read_handle: Option<ResourceHandle>,
    read_write_handle: Option<ResourceHandle>,
}

#[cfg(feature = "graphics")]
impl PendingPushConstant {
    fn from_parcel(parcel: &Parcel, access: NodeAccess) -> Self {
        Self {
            graph_access: access,
            read_handle: parcel.handle(ResourceAccess::Read),
            read_write_handle: parcel
                .handle(ResourceAccess::ReadWrite)
                .or_else(|| parcel.handle(ResourceAccess::Write)),
        }
    }

    fn from_sampler(sampler: &crate::Sampler) -> Self {
        Self {
            graph_access: NodeAccess::Read,
            read_handle: sampler.handle(ResourceAccess::Read),
            read_write_handle: None,
        }
    }

    fn resolve(&self, slot_access: &[Option<ResourceAccess>], slot_idx: usize) -> ResourceHandle {
        let descriptor_access = slot_access
            .get(slot_idx)
            .copied()
            .flatten()
            .unwrap_or_else(|| node_access_to_resource_access(self.graph_access));
        match descriptor_access {
            ResourceAccess::Read => self.read_handle.or(self.read_write_handle),
            ResourceAccess::Write | ResourceAccess::ReadWrite => self.read_write_handle.or(self.read_handle),
        }
        .unwrap_or_else(|| {
            panic!(
                "render pass resource slot {slot_idx}: no descriptor for {descriptor_access:?}; \
                 check BufferKind/TextureKind is compatible with the shader parameter"
            )
        })
    }
}

/// Builder for a render pass recorded on a [`Scheme`].
#[cfg(feature = "graphics")]
pub struct SchemeRenderPassBuilder<'a> {
    scheme: &'a mut Scheme,
    label: &'static str,
    target: crate::backend::RenderTargetHandle,
    color_load: crate::types::TargetLoad,
    bindings: Vec<ResourceBinding>,
    commands: Vec<RenderCommand>,
    pending_push_constants: Vec<PendingPushConstant>,
    pending_named: HashMap<String, PendingPushConstant>,
}

#[cfg(feature = "graphics")]
impl<'a> SchemeRenderPassBuilder<'a> {
    /// Declare a read or write dependency on a parcel deed.
    ///
    /// When [`Self::set_pipeline`] is called, parcels declared here are also registered
    /// for push-constant resource binding in call order.
    pub fn with_parcel(&mut self, parcel: &Parcel, access: NodeAccess) -> &mut Self {
        self.scheme.submit_state.register_parcel_stamp(parcel);
        self.bindings.push(ResourceBinding {
            resource: parcel.resource_id(),
            access,
        });
        self.pending_push_constants
            .push(PendingPushConstant::from_parcel(parcel, access));
        self
    }

    /// Register dependency on all parcels of a buffer without push-constant binding.
    pub fn with_buffer_dependency(&mut self, buffer: &crate::Buffer, access: NodeAccess) -> &mut Self {
        self.scheme.submit_state.register_buffer_stamps(buffer);
        for parcel in buffer.parcels() {
            self.bindings.push(ResourceBinding {
                resource: parcel.resource_id(),
                access,
            });
        }
        self
    }

    /// Declare push-constant slots in shader parameter order and register graph bindings.
    ///
    /// [`Self::set_pipeline`] emits typed bindless resource binds from these
    /// handles before each pipeline bind.
    pub fn with_shader_resources(&mut self, slots: &[ShaderResourceSlot<'_>]) -> &mut Self {
        for slot in slots {
            match slot {
                ShaderResourceSlot::Parcel { parcel, access } => {
                    self.scheme.submit_state.register_parcel_stamp(parcel);
                    self.bindings.push(ResourceBinding {
                        resource: parcel.resource_id(),
                        access: *access,
                    });
                    let pending = PendingPushConstant::from_parcel(parcel, *access);
                    if pending.read_handle.is_none() && pending.read_write_handle.is_none() {
                        panic!(
                            "ShaderResourceSlot::Parcel: mosaic parcels cannot be push-constant slots; \
                             use with_parcel for geometry bindings"
                        );
                    }
                    self.pending_push_constants.push(pending);
                }
                ShaderResourceSlot::Sampler(sampler) => {
                    self.pending_push_constants
                        .push(PendingPushConstant::from_sampler(sampler));
                }
            }
        }
        self
    }

    /// Bind resources by pipeline parameter name.
    ///
    /// Extra names are kept so the same set can serve several pipeline switches.
    /// Missing required names and wrong resource categories are rejected when
    /// [`Self::set_pipeline`] / [`Self::set_mesh_pipeline`] runs.
    pub fn with_shader_bindings(&mut self, bindings: &[ShaderBinding<'_>]) -> &mut Self {
        for binding in bindings {
            match binding.slot {
                ShaderResourceSlot::Parcel { parcel, access } => {
                    self.scheme.submit_state.register_parcel_stamp(parcel);
                    self.bindings.push(ResourceBinding {
                        resource: parcel.resource_id(),
                        access,
                    });
                    let pending = PendingPushConstant::from_parcel(parcel, access);
                    if pending.read_handle.is_none() && pending.read_write_handle.is_none() {
                        panic!(
                            "ShaderBinding `{}`: mosaic parcels cannot be push-constant slots",
                            binding.name
                        );
                    }
                    self.pending_named.insert(binding.name.to_string(), pending);
                }
                ShaderResourceSlot::Sampler(sampler) => {
                    self.pending_named
                        .insert(binding.name.to_string(), PendingPushConstant::from_sampler(sampler));
                }
            }
        }
        self
    }

    pub fn clear_depth(&mut self, depth: f32) -> &mut Self {
        self.commands.push(RenderCommand::ClearDepth(depth));
        self
    }

    pub fn set_pipeline(&mut self, pipeline: &crate::RenderPipeline) -> &mut Self {
        self.commands.push(RenderCommand::SetPipeline(pipeline.handle));
        if let Some(handles) =
            self.resolve_graphics_bindings(pipeline.resource_contract(), &pipeline.slot_access, "render pipeline")
        {
            self.commands.push(RenderCommand::BindResourcesTyped { handles });
        }
        self
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &impl BufferSource) -> &mut Self {
        self.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer: buffer.source_handle(),
            offset: buffer.source_offset(),
        });
        self
    }

    pub fn set_index_buffer(&mut self, buffer: &impl BufferSource, format: IndexFormat) -> &mut Self {
        self.commands.push(RenderCommand::SetIndexBuffer {
            buffer: buffer.source_handle(),
            offset: buffer.source_offset(),
            format,
        });
        self
    }

    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) -> &mut Self {
        self.commands.push(RenderCommand::Draw {
            vertex_count: vertices.end - vertices.start,
            instance_count: instances.end - instances.start,
            first_vertex: vertices.start,
            first_instance: instances.start,
        });
        self
    }

    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) -> &mut Self {
        self.commands.push(RenderCommand::DrawIndexed {
            index_count: indices.end - indices.start,
            instance_count: instances.end - instances.start,
            first_index: indices.start,
            base_vertex,
            first_instance: instances.start,
        });
        self
    }

    pub fn draw_fullscreen(&mut self) -> &mut Self {
        self.draw(0..3, 0..1)
    }

    /// Bind a [`crate::MeshPipeline`] and typed push-constant resources, like [`Self::set_pipeline`].
    pub fn set_mesh_pipeline(&mut self, pipeline: &crate::MeshPipeline) -> &mut Self {
        self.commands.push(RenderCommand::SetMeshPipeline(pipeline.handle));
        if let Some(handles) =
            self.resolve_graphics_bindings(pipeline.resource_contract(), &pipeline.slot_access, "mesh pipeline")
        {
            self.commands.push(RenderCommand::BindResourcesTyped { handles });
        }
        self
    }

    /// Dispatch mesh workgroups (`vkCmdDrawMeshTasksEXT` / `DispatchMesh`).
    pub fn dispatch_mesh(&mut self, x: u32, y: u32, z: u32) -> &mut Self {
        self.commands.push(RenderCommand::DispatchMesh { x, y, z });
        self
    }

    fn resolve_graphics_bindings(
        &self,
        contract: &crate::slang::graphics_link::PipelineResourceContract,
        slot_access: &[Option<ResourceAccess>],
        kind: &str,
    ) -> Option<Vec<ResourceHandle>> {
        if !self.pending_named.is_empty() {
            if contract.is_empty() {
                panic!(
                    "{kind}: named shader bindings were recorded but the pipeline has no virtual-main \
                     resource contract (non-[goldy_*] shaders still use positional with_shader_resources)"
                );
            }
            let provided: std::collections::HashSet<&str> = self.pending_named.keys().map(String::as_str).collect();
            let missing = crate::slang::graphics_link::missing_required_bindings(contract, &provided);
            if !missing.is_empty() {
                panic!(
                    "{kind}: missing required shader bindings [{}]. Bound names: [{}]",
                    missing.join(", "),
                    provided.into_iter().collect::<Vec<_>>().join(", ")
                );
            }
            let mut handles = Vec::with_capacity(contract.resources.len());
            for (i, res) in contract.resources.iter().enumerate() {
                let pending = self
                    .pending_named
                    .get(&res.name)
                    .unwrap_or_else(|| panic!("{kind}: missing binding `{}`", res.name));
                if let Some(handle) = pending.read_handle.or(pending.read_write_handle) {
                    if !handle.category().is_compatible_with(res.category) {
                        panic!(
                            "{kind}: binding `{}` expected {} but the resource is {}. \
                             Check BufferKind / texture / sampler against the shader parameter.",
                            res.name,
                            res.category.name(),
                            handle.category().name()
                        );
                    }
                }
                handles.push(pending.resolve(slot_access, i));
            }
            return Some(handles);
        }
        if self.pending_push_constants.is_empty() {
            return None;
        }
        if !contract.is_empty() && self.pending_push_constants.len() < contract.resources.len() {
            panic!(
                "{kind}: positional with_shader_resources has {} slots but the pipeline contract has {} \
                 ({}). Provide the merged order (fragment-first for raster, mesh-first for mesh), \
                 or use ShaderBinding names.",
                self.pending_push_constants.len(),
                contract.resources.len(),
                contract
                    .resources
                    .iter()
                    .map(|r| r.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Some(
            self.pending_push_constants
                .iter()
                .enumerate()
                .map(|(i, pending)| pending.resolve(slot_access, i))
                .collect(),
        )
    }

    pub fn finish(self) {
        let SchemeRenderPassBuilder {
            scheme,
            label,
            target,
            color_load,
            bindings,
            commands,
            pending_push_constants: _,
            pending_named: _,
        } = self;
        scheme.ir.nodes.push(TaskNode {
            label,
            bindings,
            kind: NodeKind::RenderPass {
                target,
                color_load,
                commands,
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::compute::ComputePipeline;
    use crate::device::Device;
    use crate::retained_pool::RetainedPool;
    use crate::shader::ShaderModule;
    use crate::task_graph::NodeAccess;
    use crate::task_graph::NodeKind;
    use crate::types::BufferFlags;
    use crate::types::ResourceAccess;
    use crate::BufferKind;
    use crate::MemoryExchange;
    use std::sync::Arc;

    fn mock_device() -> Arc<Device> {
        Arc::new(Device::from_backend(Box::new(MockBackend::new())).expect("mock device"))
    }

    fn mock_readback_counts(device: &Device) -> (usize, usize) {
        let backend = device.inner.backend.lock().unwrap();
        (backend.test_readback_alloc_count(), backend.test_readback_free_count())
    }

    fn mock_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(
            device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1,1,1)]
void cs_main(Scattered<uint> buf, ThreadId id) { buf[0] = 1; }
"#,
        )
        .expect("compile shader")
    }

    fn mock_texture_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(
            device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(DirectSpatial<float4> dst, ThreadId id) {
    if (id.x == 0 && id.y == 0) {
        dst[uint2(0, 0)] = float4(1.0, 0.0, 0.0, 1.0);
    }
}
"#,
        )
        .expect("compile texture shader")
    }

    fn mock_pipeline(device: &Device, shader: &ShaderModule) -> ComputePipeline {
        ComputePipeline::new(device, shader).expect("create pipeline")
    }

    #[cfg(feature = "graphics")]
    fn mock_render_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(device, "void main() {}").expect("compile render shader")
    }

    #[cfg(feature = "graphics")]
    fn mock_render_pipeline(device: &Device, shader: &ShaderModule) -> crate::RenderPipeline {
        crate::RenderPipeline::new(
            device,
            shader,
            shader,
            &crate::RenderPipelineDesc {
                target_format: crate::types::TextureFormat::Rgba8Unorm,
                ..Default::default()
            },
        )
        .expect("create render pipeline")
    }

    fn retained_buffer(pool: &mut RetainedPool) -> crate::Buffer {
        pool.acquire_buffer(
            32,
            crate::types::BufferKind::Scattered,
            None,
            crate::types::BufferFlags::empty(),
            None,
        )
        .expect("alloc buffer")
    }

    fn recording_scheme_with_parcel(
        device: &Arc<Device>,
        pool: &mut RetainedPool,
        ctx: &Context,
    ) -> (Scheme, crate::Buffer) {
        let shader = mock_shader(device);
        let pipeline = mock_pipeline(device, &shader);
        let buffer = retained_buffer(pool);

        let mut scheme = Scheme::new(ctx);
        scheme
            .node("a", &pipeline)
            .with_parcel(&*buffer, NodeAccess::Write)
            .dispatch(1, 1, 1);
        (scheme, buffer)
    }

    fn recording_scheme(device: &Arc<Device>, pool: &mut RetainedPool, ctx: &Context) -> (Scheme, crate::Buffer) {
        recording_scheme_with_parcel(device, pool, ctx)
    }

    fn clean_scheme(
        device: &Arc<Device>,
        pool: &mut RetainedPool,
    ) -> (Scheme, NodeId, crate::Buffer, crate::test_support::CbReuseOverride) {
        let cb = crate::test_support::CbReuseOverride::force_enabled();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(device);
        let pipeline = mock_pipeline(device, &shader);
        let parcel = retained_buffer(pool);

        let mut scheme = Scheme::new(&ctx);
        assert!(scheme.is_dirty(), "new scheme starts dirty");
        let node = scheme
            .node("a", &pipeline)
            .with_parcel(&parcel, NodeAccess::Write)
            .dispatch(1, 1, 1);

        scheme.submit().unwrap();
        assert!(!scheme.is_dirty(), "successful submit clears the dirty bit");
        assert_eq!(scheme.replay_stats().records, 1);
        #[cfg(not(feature = "metal"))]
        assert_eq!(scheme.replay_stats().resubmit_hits, 0);
        // Keep `parcel` alive: dropping a retained buffer bound to the scheme marks its
        // stamp dead and subsequent submits return `StaleResource`.
        (scheme, node, parcel, cb)
    }

    fn leased_texture_scheme(
        device: &Arc<Device>,
    ) -> (Scheme, Lease<LeaseTexture>, crate::test_support::CbReuseOverride) {
        let cb = crate::test_support::CbReuseOverride::force_enabled();
        let ctx = device.create_context().unwrap();
        let shader = mock_texture_shader(device);
        let pipeline = mock_pipeline(device, &shader);

        let mut scheme = Scheme::new(&ctx);
        let lease = scheme
            .lease_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::DirectInterpolated,
                TextureFlags::empty(),
            )
            .expect("lease texture");
        let _handle = scheme.leases[0].handle(ResourceAccess::Write).expect("lease handle");
        scheme
            .node("write_tex", &pipeline)
            .with_parcel(&lease, NodeAccess::Write)
            .dispatch(1, 1, 1);

        (scheme, lease, cb)
    }

    #[test]
    fn clear_and_full_deposit_buffer_bind_as_overwrite() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let buffer = retained_buffer(&mut pool);
        let parcel = &*buffer;
        let memory = MemoryExchange::new(&ctx);

        let mut clear_scheme = Scheme::new(&ctx);
        clear_scheme.clear_parcel(parcel, 0, parcel.byte_size()).expect("clear");
        assert_eq!(clear_scheme.ir.nodes[0].bindings[0].access, NodeAccess::Overwrite);

        let mut write_scheme = Scheme::new(&ctx);
        let deposit = memory
            .bind_deposit_buffer(&mut write_scheme, parcel, parcel.byte_size())
            .expect("bind full deposit");
        deposit
            .write(&mut write_scheme, 0, &vec![0u8; parcel.byte_size() as usize])
            .expect("full deposit write");
        assert_eq!(write_scheme.ir.nodes[0].bindings[1].access, NodeAccess::Overwrite);

        let mut partial_scheme = Scheme::new(&ctx);
        let partial_deposit = memory
            .bind_deposit_buffer_at(&mut partial_scheme, parcel, 4, 4)
            .expect("bind partial deposit");
        partial_deposit
            .write(&mut partial_scheme, 0, &[1, 2, 3, 4])
            .expect("partial deposit write");
        assert_eq!(partial_scheme.ir.nodes[0].bindings[1].access, NodeAccess::Write);
    }

    #[test]
    fn clean_submits_resubmit_without_rerecord() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let (mut scheme, _node0, _buf, _cb) = clean_scheme(&device, &mut pool);

        scheme.submit().unwrap();
        scheme.submit().unwrap();

        assert_eq!(scheme.replay_stats().records, 1, "only the first submit records");
        // Counted on every backend, including those without command-list retention.
        assert_eq!(
            scheme.replay_stats().clean_submits,
            2,
            "both post-record submits found the scheme clean"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            2,
            "subsequent clean submits resubmit"
        );
    }

    #[test]
    fn params_dirty_submit_is_not_a_clean_submit() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let (mut scheme, node0, _buf, _cb) = clean_scheme(&device, &mut pool);
        scheme.submit().unwrap();
        assert_eq!(scheme.replay_stats().clean_submits, 1);

        let shader = mock_shader(&device);
        let swapped = mock_pipeline(&device, &shader);
        scheme.set_node_pipeline(node0, &swapped).unwrap();
        scheme.submit().unwrap();
        assert_eq!(
            scheme.replay_stats().clean_submits,
            1,
            "a params-dirty submit breaks the clean streak"
        );

        scheme.submit().unwrap();
        assert_eq!(
            scheme.replay_stats().clean_submits,
            2,
            "the streak resumes once the swap is recorded"
        );
    }

    #[test]
    #[cfg(not(feature = "metal"))]
    fn clean_resubmit_performs_no_cpu_wait() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let (mut scheme, _node0, _buf, _cb) = clean_scheme(&device, &mut pool);

        scheme.submit().unwrap();
        scheme.submit().unwrap();

        device.with_mock_backend(|mock| {
            assert_eq!(
                mock.wait_until_count, 0,
                "clean scheme resubmits must not call wait_until on the submit path"
            );
        });
        assert!(
            !scheme.partition_last_tvs().is_empty(),
            "per-partition timelines are tracked after submit"
        );
    }

    #[test]
    fn mutation_marks_dirty_and_rerecords_once() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let (mut scheme, _node0, _buf, _cb) = clean_scheme(&device, &mut pool);
        scheme.submit().unwrap();

        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats(),
            ReplayStats {
                records: 1,
                resubmit_hits: 1,
                topology_records: 0,
                clean_submits: 1,
            }
        );
        #[cfg(feature = "metal")]
        assert_eq!(scheme.replay_stats().records, 1);

        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let parcel2 = retained_buffer(&mut pool);
        scheme
            .node("b", &pipeline)
            .with_parcel(&parcel2, NodeAccess::Write)
            .dispatch(1, 1, 1);

        assert!(scheme.is_dirty());
        scheme.submit().unwrap();
        scheme.submit().unwrap();

        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats(),
            ReplayStats {
                records: 2,
                resubmit_hits: 2,
                topology_records: 0,
                clean_submits: 2,
            }
        );
        #[cfg(feature = "metal")]
        assert_eq!(scheme.replay_stats().records, 2);
    }

    #[test]
    fn set_node_pipeline_rerecords_once_then_resubmits() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let (mut scheme, node0, _buf, _cb) = clean_scheme(&device, &mut pool);
        scheme.submit().unwrap();

        let shader = mock_shader(&device);
        let pipeline2 = mock_pipeline(&device, &shader);
        scheme.set_node_pipeline(node0, &pipeline2).unwrap();
        assert!(scheme.is_dirty());
        scheme.submit().unwrap();
        assert!(!scheme.is_dirty());
        scheme.submit().unwrap();

        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats(),
            ReplayStats {
                records: 2,
                resubmit_hits: 2,
                topology_records: 0,
                // Submits 2 and 4: the swap makes 3 params-dirty.
                clean_submits: 2,
            }
        );
        #[cfg(feature = "metal")]
        assert_eq!(scheme.replay_stats().records, 2);
    }

    #[test]
    fn node_id_from_another_scheme_is_rejected() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let (mut scheme_a, node_a, _buf_a, _cb) = clean_scheme(&device, &mut pool);
        let (scheme_b, _node_b, _buf_b, _cb_b) = clean_scheme(&device, &mut pool);

        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let err = scheme_a
            .set_node_pipeline(
                NodeId {
                    scheme_id: scheme_b.scheme_id,
                    index: node_a.index as u32,
                },
                &pipeline,
            )
            .expect_err("foreign node id must be rejected");
        assert!(err.to_string().contains("belongs to scheme"), "unexpected error: {err}");
        assert!(!scheme_a.is_dirty(), "a rejected id must not dirty the scheme");
    }

    #[test]
    fn set_node_pipeline_keeps_other_partition_retained() {
        let _cb = crate::test_support::CbReuseOverride::force_enabled();
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let p_a = mock_pipeline(&device, &shader);
        let p_a2 = mock_pipeline(&device, &shader);
        let p_b = mock_pipeline(&device, &shader);
        let p_c = mock_pipeline(&device, &shader);
        let buf0 = retained_buffer(&mut pool);
        let buf1 = retained_buffer(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let node_a = scheme
            .node("a", &p_a)
            .with_parcel(&buf0, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("b", &p_b)
            .with_parcel(&buf0, NodeAccess::Read)
            .with_parcel(&buf1, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("c", &p_c)
            .with_parcel(&buf1, NodeAccess::Read)
            .dispatch(1, 1, 1);

        scheme.submit().unwrap();
        scheme.submit().unwrap();
        let records_after_warmup = scheme.replay_stats().records;
        #[cfg(not(feature = "metal"))]
        let resubmits_after_warmup = scheme.replay_stats().resubmit_hits;

        scheme.set_node_pipeline(node_a, &p_a2).unwrap();
        scheme.submit().unwrap();

        assert_eq!(
            scheme.replay_stats().records,
            records_after_warmup + 1,
            "params-only pipeline swap must record once"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            resubmits_after_warmup + 1,
            "unchanged partition must resubmit instead of a full scheme re-record"
        );

        device.with_mock(|m| {
            let saw_new = m.retained_graphs.values().any(|cmds| {
                cmds.iter().any(|c| {
                    matches!(
                        c,
                        crate::backend::GraphCommand::Compute(crate::backend::GpuCommand::SetPipeline(p))
                            if *p == p_a2.handle
                    )
                })
            });
            assert!(saw_new, "re-recorded partition must bake the swapped pipeline");
        });
    }

    #[test]
    fn is_settled_true_before_first_reference() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let parcel = retained_buffer(&mut pool);
        assert!(parcel.is_settled(), "never-referenced parcel is settled");
    }

    #[test]
    fn frame_timeline_value_round_trip() {
        use crate::timeline::TimelineValue;

        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(device.clone());
        let (mut scheme, _buf) = recording_scheme(&device, &mut pool, &ctx);
        let frame = scheme.submit().unwrap();
        let tv = frame.timeline_value();
        assert!(tv > 0);
        assert_eq!(TimelineValue::from(frame.handle()), tv);
        assert_eq!(frame.timeline_value(), tv);
    }

    #[test]
    fn frame_wait_completes_submission() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(device.clone());
        let (mut scheme, _buf) = recording_scheme(&device, &mut pool, &ctx);
        let frame = scheme.submit().unwrap();
        frame.wait(&ctx).unwrap();
        assert!(ctx.gpu_progress() >= frame.timeline_value());
    }

    #[test]
    fn submit_returns_frame_without_calling_wait() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(device.clone());
        let (mut scheme, _buf) = recording_scheme(&device, &mut pool, &ctx);
        let frame = scheme.submit().unwrap();
        assert!(frame.timeline_value() > 0, "submit must return a frame token");
        // Non-blocking: a second submit must succeed without waiting on the first frame.
        let frame2 = scheme.submit().unwrap();
        assert!(frame2.timeline_value() >= frame.timeline_value());
        frame2.wait(&ctx).unwrap();
    }

    #[test]
    fn submit_stamps_parcel_references() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let parcel = retained_buffer(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("a", &pipeline)
            .with_parcel(&parcel, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let frame1 = scheme.submit().unwrap();
        assert_eq!(
            parcel.last_referenced_on(ctx.backend_handle()),
            Some(frame1.timeline_value())
        );

        let frame2 = scheme.submit().unwrap();
        assert!(
            frame2.timeline_value() >= frame1.timeline_value(),
            "timeline must be monotonic"
        );
        assert_eq!(
            parcel.last_referenced_on(ctx.backend_handle()),
            Some(frame2.timeline_value()),
            "resubmit path must also stamp parcel references"
        );
    }

    #[test]
    fn lease_texture_records_once_resubmits_clean() {
        let device = mock_device();
        let (mut scheme, _lease, _cb) = leased_texture_scheme(&device);

        scheme.submit().expect("first submit records");
        scheme.submit().expect("second submit resubmits");
        scheme.submit().expect("third submit resubmits");

        assert_eq!(scheme.replay_stats().records, 1, "exactly one record");
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            2,
            "remaining submits are retention hits"
        );
    }

    #[test]
    fn lease_backing_stamped_per_submit() {
        let device = mock_device();
        let (mut scheme, _lease, _cb) = leased_texture_scheme(&device);
        let ctx = scheme.ctx.clone();

        let frame1 = scheme.submit().unwrap();
        assert_eq!(
            scheme.leases[0].last_referenced_on(ctx.backend_handle()),
            Some(frame1.timeline_value())
        );

        let frame2 = scheme.submit().unwrap();
        assert!(frame2.timeline_value() >= frame1.timeline_value());
        assert_eq!(
            scheme.leases[0].last_referenced_on(ctx.backend_handle()),
            Some(frame2.timeline_value()),
            "lease backing must be stamped on resubmit"
        );
    }

    #[test]
    fn lease_backing_recycled_on_scheme_drop() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let outstanding_before = ctx.with_transient_pool(|pool| pool.outstanding_bytes().texture);

        {
            let mut scheme = Scheme::new(&ctx);
            let lease = scheme
                .lease_texture(
                    4,
                    4,
                    TextureFormat::Rgba8Unorm,
                    TextureKind::Interpolated,
                    TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                )
                .expect("lease");
            assert!(
                ctx.with_transient_pool(|pool| pool.outstanding_bytes().texture > outstanding_before),
                "leased backing counts as pool outstanding"
            );
            drop(lease);
            drop(scheme);
        }

        assert_eq!(
            ctx.with_transient_pool(|pool| pool.outstanding_bytes().texture),
            outstanding_before,
            "outstanding drops when scheme releases lease backings"
        );
        assert_eq!(
            ctx.with_transient_pool(|pool| pool.pending_count()),
            1,
            "dropped lease backing is parked in the pool"
        );
    }

    // ---- CPU dispatch nodes -------------------------------------------------

    fn u32_buffer(pool: &mut RetainedPool, data: &[u32]) -> crate::Buffer {
        pool.acquire_buffer_with_data(data, BufferKind::Scattered)
            .expect("alloc buffer")
    }

    fn read_u32(grant: &crate::WithdrawTransaction, frame: &mut Submission) -> Vec<u32> {
        let bytes = grant.claim(frame).expect("claim").consume().expect("consume");
        bytemuck::cast_slice(&bytes).to_vec()
    }

    #[test]
    fn cpu_node_appends_ir_node_and_allocates_staging() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let buf = u32_buffer(&mut pool, &[1, 2, 3, 4]);
        let (allocs_before, _) = mock_readback_counts(&device);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .cpu_node("double")
            .with_parcel(&buf, NodeAccess::ReadWrite)
            .dispatch(|data: &mut [u32]| {
                for d in data {
                    *d *= 2;
                }
            })
            .expect("record cpu node");

        assert_eq!(scheme.ir_node_count(), 1);
        assert_eq!(scheme.cpu_dispatch_count(), 1);
        assert!(scheme.is_dirty());
        match &scheme.ir.nodes[0].kind {
            NodeKind::CpuDispatch { cpu_id: 0 } => {}
            other => panic!("expected CpuDispatch node, got {other:?}"),
        }
        assert_eq!(scheme.ir.nodes[0].bindings.len(), 1);
        assert_eq!(scheme.ir.nodes[0].bindings[0].access, NodeAccess::ReadWrite);
        let (allocs_after, _) = mock_readback_counts(&device);
        assert_eq!(
            allocs_after - allocs_before,
            1,
            "ReadWrite binding allocates one readback staging"
        );
        let exec = &scheme.cpu_dispatches[0];
        assert!(exec.bindings[0].readback.is_some());
        assert!(exec.bindings[0].upload.is_some());

        let (_, frees_before) = mock_readback_counts(&device);
        drop(scheme);
        let (_, frees_after) = mock_readback_counts(&device);
        assert_eq!(frees_after - frees_before, 1, "scheme drop frees readback staging");
    }

    #[test]
    fn cpu_node_staging_follows_access() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let a = u32_buffer(&mut pool, &[1, 2]);
        let b = u32_buffer(&mut pool, &[3, 4]);
        let c = u32_buffer(&mut pool, &[5, 6]);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .cpu_node("mixed")
            .with_parcel(&a, NodeAccess::Read)
            .with_parcel(&b, NodeAccess::Overwrite)
            .with_parcel(&c, NodeAccess::Write)
            .dispatch(|_a: &[u32], _b: &mut [u32], _c: &mut [u32]| {})
            .expect("record");
        let b = &scheme.cpu_dispatches[0].bindings;
        assert!(b[0].readback.is_some() && b[0].upload.is_none(), "Read: download only");
        assert!(
            b[1].readback.is_none() && b[1].upload.is_some(),
            "Overwrite: upload only"
        );
        assert!(
            b[2].readback.is_some() && b[2].upload.is_some(),
            "Write: download then upload"
        );
    }

    #[test]
    fn cpu_node_rejects_mismatched_virtual_main() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let buf = u32_buffer(&mut pool, &[1, 2, 3, 4]);
        let (allocs_before, frees_before) = mock_readback_counts(&device);

        let mut scheme = Scheme::new(&ctx);
        // Two slice parameters, one binding.
        let err = scheme
            .cpu_node("bad_arity")
            .with_parcel(&buf, NodeAccess::Read)
            .dispatch(|_a: &[u32], _b: &[u32]| {})
            .unwrap_err();
        assert!(format!("{err}").contains("slice parameter"), "{err}");
        // Read access but `&mut [T]` parameter.
        let err = scheme
            .cpu_node("bad_mut")
            .with_parcel(&buf, NodeAccess::Read)
            .dispatch(|_a: &mut [u32]| {})
            .unwrap_err();
        assert!(format!("{err}").contains("Read"), "{err}");
        // Element size does not divide the parcel (16 bytes / 3).
        let err = scheme
            .cpu_node("bad_stride")
            .with_parcel(&buf, NodeAccess::Read)
            .dispatch(|_a: &[[u8; 3]]| {})
            .unwrap_err();
        assert!(format!("{err}").contains("multiple"), "{err}");
        // Scalar count mismatch.
        let err = scheme
            .cpu_node("bad_scalars")
            .with_parcel(&buf, NodeAccess::Read)
            .with_param(1)
            .dispatch(|_a: &[u32]| {})
            .unwrap_err();
        assert!(format!("{err}").contains("scalar"), "{err}");

        assert_eq!(scheme.ir_node_count(), 0, "failed records leave no node");
        assert_eq!(scheme.cpu_dispatch_count(), 0);
        let (allocs_after, frees_after) = mock_readback_counts(&device);
        assert_eq!(
            allocs_after - allocs_before,
            frees_after - frees_before,
            "no staging leaked"
        );
    }

    #[test]
    fn cpu_node_rejects_texture_parcel() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let tex = pool
            .acquire_texture(
                2,
                2,
                crate::types::TextureFormat::Rgba8Unorm,
                crate::types::TextureKind::Direct,
                crate::types::TextureFlags::empty(),
                None,
            )
            .expect("texture");
        let mut scheme = Scheme::new(&ctx);
        let err = scheme
            .cpu_node("tex")
            .with_parcel(&tex, NodeAccess::Read)
            .dispatch(|_a: &[u8]| {})
            .unwrap_err();
        assert!(format!("{err}").contains("buffer parcel"), "{err}");
    }

    #[test]
    fn cpu_node_is_isolated_in_its_own_wave_and_partition() {
        use crate::task_graph::analysis;

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let a = u32_buffer(&mut pool, &[0; 8]);
        let b = u32_buffer(&mut pool, &[0; 8]);
        let c = u32_buffer(&mut pool, &[0; 8]);

        let mut scheme = Scheme::new(&ctx);
        // wave 0: gpu writes A         | independent gpu writes C
        // wave 0': cpu reads A, writes B (same depth as C, but must be alone)
        // wave 1: gpu reads B
        scheme
            .node("produce_a", &pipeline)
            .with_parcel(&a, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .cpu_node("host")
            .with_parcel(&a, NodeAccess::Read)
            .with_parcel(&b, NodeAccess::Overwrite)
            .dispatch(|_a: &[u32], _b: &mut [u32]| {})
            .expect("record");
        scheme
            .node("independent_c", &pipeline)
            .with_parcel(&c, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("consume_b", &pipeline)
            .with_parcel(&b, NodeAccess::Read)
            .dispatch(1, 1, 1);

        let edges = analysis::build_edges(&scheme.ir);
        let schedule = analysis::schedule_waves(&scheme.ir, &edges);
        let waves: Vec<Vec<usize>> = schedule.waves.iter().map(|w| w.node_indices.clone()).collect();
        assert_eq!(
            waves,
            vec![vec![0, 2], vec![1], vec![3]],
            "cpu node (1) is peeled from its depth group into its own wave"
        );
        assert_eq!(analysis::wave_cpu_dispatch(&scheme.ir, &schedule.waves[1]), Some(0));
        assert!(
            !schedule.waves[1].barriers_before.is_empty(),
            "producer→cpu edge yields a barrier on A"
        );
        assert!(
            !schedule.waves[2].barriers_before.is_empty(),
            "cpu→consumer edge yields a barrier on B"
        );

        let parts = analysis::partition_wave_ranges(&scheme.ir, &schedule, true);
        assert_eq!(parts, vec![0..1, 1..2, 2..3], "cpu wave is its own partition");
    }

    #[test]
    fn cpu_node_runs_on_mock_and_resubmits() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let data = u32_buffer(&mut pool, &[1, 2, 3, 4]);
        let out = u32_buffer(&mut pool, &[0; 4]);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .cpu_node("scale_into")
            .with_parcel(&data, NodeAccess::ReadWrite)
            .with_parcel(&out, NodeAccess::Overwrite)
            .with_param(10)
            .with_param(0.5f32.to_bits())
            .dispatch(|data: &mut [u32], out: &mut [u32], add: u32, half: f32| {
                assert_eq!(half, 0.5);
                for (o, d) in out.iter_mut().zip(data.iter_mut()) {
                    *d += 1;
                    *o = *d + add;
                }
            })
            .expect("record");
        let grant_data = MemoryExchange::new(&ctx)
            .bind_withdraw(&mut scheme, &data)
            .expect("withdraw data");
        let grant_out = MemoryExchange::new(&ctx)
            .bind_withdraw(&mut scheme, &out)
            .expect("withdraw out");

        let mut frame = scheme.submit().expect("first submit");
        assert_eq!(read_u32(&grant_data, &mut frame), vec![2, 3, 4, 5]);
        assert_eq!(read_u32(&grant_out, &mut frame), vec![12, 13, 14, 15]);

        // Second submission observes the uploaded result of the first.
        let mut frame = scheme.submit().expect("second submit");
        assert_eq!(read_u32(&grant_data, &mut frame), vec![3, 4, 5, 6]);
        assert_eq!(read_u32(&grant_out, &mut frame), vec![13, 14, 15, 16]);
        assert!(!scheme.is_dirty());
    }

    #[test]
    fn cpu_node_download_and_upload_are_separate_transfer_submits() {
        use crate::backend::GpuCommand;

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let data = u32_buffer(&mut pool, &[1, 2, 3, 4]);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("gpu_write", &pipeline)
            .with_parcel(&data, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .cpu_node("host_rw")
            .with_parcel(&data, NodeAccess::ReadWrite)
            .dispatch(|_d: &mut [u32]| {})
            .expect("record");
        scheme
            .node("gpu_read", &pipeline)
            .with_parcel(&data, NodeAccess::Read)
            .dispatch(1, 1, 1);

        crate::test_support::mock_reset_tracking(&device);
        scheme.submit().expect("submit");

        let handle = data.buffer_handle().unwrap();
        let (downloads, uploads, barriers) = crate::test_support::with_mock(&device, |m| {
            let mut downloads = 0;
            let mut uploads = 0;
            let mut barriers = Vec::new();
            for batch in &m.recorded_compute_commands {
                for cmd in batch {
                    match cmd {
                        GpuCommand::CopyBuffer { src, dst, .. } => {
                            if *src == handle {
                                downloads += 1;
                            }
                            if *dst == handle {
                                uploads += 1;
                            }
                        }
                        GpuCommand::ResourceBarrier { buffers, .. } => {
                            barriers.extend(buffers.iter().filter(|(h, _)| *h == handle).map(|(_, u)| *u));
                        }
                        _ => {}
                    }
                }
            }
            (downloads, uploads, barriers)
        });
        assert_eq!(downloads, 1, "one device→staging copy");
        assert_eq!(uploads, 1, "one staging→device copy");
        assert!(
            barriers
                .iter()
                .any(|u| u.dst.kinds == crate::task_graph::UsageKindFlags::TRANSFER
                    && u.dst.access == crate::task_graph::NodeAccessUnion::ReadOnly
                    && u.src.kinds.contains(crate::task_graph::UsageKindFlags::COMPUTE)),
            "download barrier: compute write → transfer read; got {barriers:?}"
        );
        assert!(
            barriers
                .iter()
                .any(|u| u.dst.kinds == crate::task_graph::UsageKindFlags::TRANSFER
                    && u.dst.access == crate::task_graph::NodeAccessUnion::Write),
            "upload barrier: → transfer write; got {barriers:?}"
        );
    }

    #[test]
    fn withdraw_appends_ir_node() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        assert_eq!(scheme.ir_node_count(), 1);

        let _grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &parcel)
            .expect("withdraw");
        assert_eq!(scheme.ir_node_count(), 2);
        assert!(scheme.is_dirty(), "withdraw is structural");

        match &scheme.ir.nodes[1].kind {
            NodeKind::WithdrawRead { withdraw_id: 0 } => {}
            other => panic!("expected GrantRead node, got {other:?}"),
        }
    }

    #[test]
    fn withdraw_orders_after_writer() {
        use crate::task_graph::analysis;

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        let _grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &parcel)
            .expect("withdraw");

        let edges = analysis::build_edges(&scheme.ir);
        assert!(
            edges.contains(&(0, 1)),
            "dispatch (0) must precede grant_read (1); edges: {edges:?}"
        );
    }

    #[test]
    fn scheme_with_grant_retains() {
        let _cb = crate::test_support::CbReuseOverride::force_enabled();
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        let _grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &parcel)
            .expect("withdraw");

        scheme.submit().expect("first submit records");
        scheme.submit().expect("second submit resubmits");
        scheme.submit().expect("third submit resubmits");

        assert_eq!(scheme.replay_stats().records, 1, "exactly one record with grant node");
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            2,
            "remaining submits are retention hits"
        );
    }

    #[test]
    fn withdraw_survives_parcel_drop() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &parcel)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        drop(parcel);
        drop(pool);

        let loan = grant
            .claim(&mut frame)
            .expect("claim")
            .consume()
            .expect("read after parcel drop");
        assert_eq!(loan.len(), 32, "reads full logical buffer size");
    }

    #[test]
    fn withdraw_resubmit_after_parcel_drop() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &parcel)
            .expect("withdraw");
        let mut frame1 = scheme.submit().expect("submit 1");
        drop(parcel);
        drop(pool);
        // Retained ownership outranks the scheme: dropping the bound buffer kills its stamp.
        assert!(
            matches!(scheme.submit(), Err(GoldyError::StaleResource)),
            "resubmit after dropping a bound retained buffer must fail"
        );
        let loan1 = grant
            .claim(&mut frame1)
            .expect("claim")
            .consume()
            .expect("read frame1 after parcel drop");
        assert_eq!(loan1.len(), 32);
    }

    /// Returning a bound transient to the pool must invalidate the scheme — same as drop.
    ///
    /// Runs on the mock backend (all host platforms). GPU coverage for Metal/Vulkan/DX12
    /// is `scheme_return_transient_texture_invalidates_retained_scheme` in
    /// `tests/scheme_compute_integration.rs` (Vulkan/DX12 need a machine with those backends).
    #[test]
    fn return_transient_texture_invalidates_bound_scheme() {
        let device = mock_device();
        let _cb = crate::test_support::CbReuseOverride::force_enabled();
        let ctx = device.create_context().unwrap();
        let shader = mock_texture_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let tex = ctx
            .acquire_transient_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::DirectInterpolated,
                TextureFlags::empty(),
            )
            .expect("transient texture");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write_tex", &pipeline)
            .with_parcel(&tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme.submit().expect("first submit");

        ctx.return_transient_texture(tex);

        assert!(
            matches!(scheme.submit(), Err(GoldyError::StaleResource)),
            "resubmit after return_transient_texture must fail (stamp retired on pool return)"
        );
    }

    /// After stamp retirement, a re-acquired transient is a new deed and can bind a new scheme.
    #[test]
    fn return_transient_texture_reacquire_binds_fresh_scheme() {
        let device = mock_device();
        let _cb = crate::test_support::CbReuseOverride::force_enabled();
        let ctx = device.create_context().unwrap();
        let shader = mock_texture_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let tex = ctx
            .acquire_transient_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::DirectInterpolated,
                TextureFlags::empty(),
            )
            .expect("transient texture");
        let handle = tex.texture_handle();

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write_tex", &pipeline)
            .with_parcel(&tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme.submit().expect("first submit");
        ctx.return_transient_texture(tex);
        assert!(matches!(scheme.submit(), Err(GoldyError::StaleResource)));

        // Epoch-gate: mock may need progress for ready_after; wait on high water.
        let hw = ctx.high_water_timeline();
        if hw > 0 {
            let _ = ctx.wait_until(hw);
        }

        let tex2 = ctx
            .acquire_transient_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::DirectInterpolated,
                TextureFlags::empty(),
            )
            .expect("reacquire");
        assert_eq!(tex2.texture_handle(), handle, "pool should reuse GPU texture");

        let mut scheme2 = Scheme::new(&ctx);
        scheme2
            .node("write_tex", &pipeline)
            .with_parcel(&tex2, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme2
            .submit()
            .expect("fresh scheme with re-acquired texture must submit");
    }

    #[test]
    fn return_transient_buffer_invalidates_bound_scheme() {
        let device = mock_device();
        let _cb = crate::test_support::CbReuseOverride::force_enabled();
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let buf = ctx
            .acquire_transient_buffer(64, BufferKind::Scattered, BufferFlags::empty(), None)
            .expect("transient buffer");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("a", &pipeline)
            .with_parcel(&buf, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme.submit().expect("first submit");

        ctx.return_transient_buffer(buf);

        assert!(
            matches!(scheme.submit(), Err(GoldyError::StaleResource)),
            "resubmit after return_transient_buffer must fail"
        );
    }

    #[test]
    fn withdraw_concurrent_frames_succeed() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = pool
            .acquire_buffer_with_data(&[7u32; 8], BufferKind::Scattered)
            .expect("parcel");
        let mut scheme = Scheme::new(&ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &parcel)
            .expect("withdraw");
        let mut frame1 = scheme.submit().expect("first submit");
        let mut frame2 = scheme.submit().expect("second submit without waiting on frame1");

        let loan1 = grant.claim(&mut frame1).expect("claim").consume().expect("read frame1");
        let loan2 = grant.claim(&mut frame2).expect("claim").consume().expect("read frame2");
        assert_eq!(loan1.len(), 32);
        assert_eq!(loan2.len(), 32);
        for chunk in loan1.chunks_exact(4) {
            assert_eq!(u32::from_le_bytes(chunk.try_into().unwrap()), 7);
        }
        let (allocs, _) = mock_readback_counts(&device);
        assert_eq!(allocs, 2, "two live frames require two staging allocations");
    }

    #[test]
    fn withdraw_double_read_same_frame_errors() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &parcel)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let _loan = grant.claim(&mut frame).expect("claim").consume().expect("first read");
        let err = grant.claim(&mut frame).expect_err("second claim must fail");
        assert!(err.to_string().contains("already consumed"), "unexpected error: {err}");
    }

    #[test]
    fn grant_staging_pool_recycled_on_loan_drop() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = pool
            .acquire_buffer_with_data(&[3u32; 8], BufferKind::Scattered)
            .expect("parcel");
        let mut scheme = Scheme::new(&ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &parcel)
            .expect("withdraw");

        let mut frame1 = scheme.submit().expect("submit 1");
        {
            let loan = grant.claim(&mut frame1).expect("claim").consume().expect("read frame1");
            assert_eq!(loan.len(), 32);
        }
        let mut frame2 = scheme.submit().expect("submit 2 after loan drop");
        let loan2 = grant
            .claim(&mut frame2)
            .expect("claim")
            .consume()
            .expect("read frame2 after pool recycle");
        assert_eq!(loan2.len(), 32);
        let (allocs, _) = mock_readback_counts(&device);
        assert_eq!(allocs, 1, "pool recycles staging buffer on loan drop");
    }

    #[test]
    fn withdraw_rejects_foreign_device_parcel() {
        let device_a = mock_device();
        let device_b = mock_device();
        let mut pool = RetainedPool::new(device_a.clone());
        let ctx_a = device_a.create_context().unwrap();
        let ctx_b = device_b.create_context().unwrap();
        let parcel = retained_buffer(&mut pool);
        let mut scheme = Scheme::new(&ctx_b);
        let err = match MemoryExchange::new(scheme.context()).bind_withdraw(&mut scheme, &parcel) {
            Ok(_) => panic!("cross-device grant must fail"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("home device"), "unexpected error: {err}");
        drop(ctx_a);
    }

    #[test]
    fn withdraw_rejects_cross_scheme_frame() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = retained_buffer(&mut pool);

        let mut scheme_a = Scheme::new(&ctx);
        let grant_a = MemoryExchange::new(scheme_a.context())
            .bind_withdraw(&mut scheme_a, &parcel)
            .expect("grant_a");

        let mut scheme_b = Scheme::new(&ctx);
        let _grant_b = MemoryExchange::new(scheme_b.context())
            .bind_withdraw(&mut scheme_b, &parcel)
            .expect("grant_b");
        let mut frame_b = scheme_b.submit().expect("submit b");

        let err = grant_a.claim(&mut frame_b).expect_err("cross-scheme claim must fail");
        assert!(err.to_string().contains("different scheme"), "unexpected error: {err}");
    }

    #[test]
    fn withdraw_drop_scheme_with_outstanding_frame_frees_staging() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = pool
            .acquire_buffer_with_data(&[1u32; 8], BufferKind::Scattered)
            .expect("parcel");
        let mut scheme = Scheme::new(&ctx);
        let _grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &parcel)
            .expect("grant");
        let frame = scheme.submit().expect("submit");
        let (allocs_after_submit, frees_before) = mock_readback_counts(&device);
        assert_eq!(allocs_after_submit, 1, "submit allocates one staging buffer");
        drop(scheme);
        drop(frame);
        let (allocs, frees) = mock_readback_counts(&device);
        assert_eq!(frees, frees_before + 1, "outstanding frame frees staging on drop");
        assert_eq!(frees, allocs, "all staging buffers freed");
    }

    #[test]
    fn withdraw_rejects_zero_byte_buffer() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = pool.acquire_buffer(0, BufferKind::Scattered, None, crate::types::BufferFlags::empty(), None);
        if parcel.is_err() {
            // Pools/backends may reject zero-byte buffers; guard is still covered at grant_read.
            return;
        }
        let parcel = parcel.unwrap();
        let mut scheme = Scheme::new(&ctx);
        let err = match MemoryExchange::new(scheme.context()).bind_withdraw(&mut scheme, &parcel) {
            Ok(_) => panic!("zero-byte grant must fail"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("non-zero"), "unexpected error: {err}");
    }

    // ------------------------------------------------------------------
    // Texture grant tests
    // ------------------------------------------------------------------

    fn texture_parcel(pool: &mut RetainedPool) -> crate::Texture {
        pool.acquire_texture(
            4,
            4,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC,
            None,
        )
        .expect("texture parcel")
    }

    #[test]
    fn withdraw_texture_basic_succeeds() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let loan = grant
            .claim(&mut frame)
            .expect("claim")
            .consume()
            .expect("read texture grant");
        assert_eq!(loan.len(), 4 * 4 * 4, "Rgba8Unorm 4×4 = 64 bytes");
    }

    #[test]
    fn withdraw_texture_appends_ir_node() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let _grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");

        assert!(scheme.is_dirty(), "withdraw is structural");
        assert_eq!(scheme.ir_node_count(), 1);
        match &scheme.ir.nodes[0].kind {
            NodeKind::WithdrawRead { withdraw_id: 0 } => {}
            other => panic!("expected GrantRead node, got {other:?}"),
        }
    }

    #[test]
    fn withdraw_texture_staging_alloc_and_free() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let (allocs_before, frees_before) = mock_readback_counts(&device);
        assert_eq!(allocs_before, 1, "one staging alloc per submit");
        assert_eq!(frees_before, 0, "not freed yet");

        let loan = grant.claim(&mut frame).expect("claim").consume().expect("read");
        drop(loan);

        // After loan drop the handle returns to pool (scheme alive) — no free yet.
        let (_, frees_after_loan) = mock_readback_counts(&device);
        assert_eq!(frees_after_loan, 0, "pool recycles on loan drop");

        // Resubmit — pool recycles the same staging handle.
        let mut frame2 = scheme.submit().expect("resubmit");
        let (allocs_after_resubmit, _) = mock_readback_counts(&device);
        assert_eq!(allocs_after_resubmit, 1, "recycled: no new alloc");
        let _loan2 = grant.claim(&mut frame2).expect("claim").consume().expect("read frame2");

        // Drop scheme — pool drains and frees all handles.
        drop(_loan2);
        drop(frame2);
        drop(grant);
        drop(scheme);
        let (_, frees_final) = mock_readback_counts(&device);
        assert_eq!(frees_final, 1, "all staging freed on scheme drop");
    }

    #[test]
    fn withdraw_texture_double_read_same_frame_errors() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");

        let _loan = grant.claim(&mut frame).expect("claim").consume().expect("first read");
        let err = grant.claim(&mut frame).expect_err("second claim must fail");
        assert!(err.to_string().contains("already consumed"), "unexpected error: {err}");
    }

    #[test]
    fn withdraw_texture_concurrent_frames() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");
        let mut frame1 = scheme.submit().expect("first submit");
        let mut frame2 = scheme.submit().expect("second submit without waiting on frame1");

        let loan1 = grant.claim(&mut frame1).expect("claim").consume().expect("read frame1");
        let loan2 = grant.claim(&mut frame2).expect("claim").consume().expect("read frame2");
        assert_eq!(loan1.len(), loan2.len());

        let (allocs, _) = mock_readback_counts(&device);
        assert_eq!(allocs, 2, "two live frames require two staging allocations");
    }

    #[test]
    fn withdraw_texture_rejects_sampled_only_texture() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();

        let texture = pool
            .acquire_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::Interpolated,
                TextureFlags::COPY_SRC,
                None,
            )
            .expect("texture");
        let mut scheme = Scheme::new(&ctx);
        let err = match MemoryExchange::new(scheme.context()).bind_withdraw(&mut scheme, &texture) {
            Ok(_) => panic!("must reject Interpolated texture"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("sampled-only") || err.to_string().contains("storage-writable"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn withdraw_texture_rejects_missing_copy_src_flag() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();

        let texture = pool
            .acquire_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::empty(),
                None,
            )
            .expect("texture");
        let mut scheme = Scheme::new(&ctx);
        let err = match MemoryExchange::new(scheme.context()).bind_withdraw(&mut scheme, &texture) {
            Ok(_) => panic!("must reject missing COPY_SRC flag"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("COPY_SRC"), "unexpected error: {err}");
    }

    #[test]
    fn withdraw_texture_rejects_cross_scheme_frame() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx_a = device.create_context().unwrap();
        let ctx_b = device.create_context().unwrap();

        let texture = texture_parcel(&mut pool);

        let mut scheme_a = Scheme::new(&ctx_a);
        let grant_a = MemoryExchange::new(scheme_a.context())
            .bind_withdraw(&mut scheme_a, &texture)
            .expect("grant_a");
        let _frame_a = scheme_a.submit().expect("submit a");

        let mut scheme_b = Scheme::new(&ctx_b);
        let _grant_b = MemoryExchange::new(scheme_b.context())
            .bind_withdraw(&mut scheme_b, &texture)
            .expect("grant_b");
        let mut frame_b = scheme_b.submit().expect("submit b");

        let err = grant_a.claim(&mut frame_b).expect_err("cross-scheme claim must fail");
        assert!(err.to_string().contains("different scheme"), "unexpected error: {err}");
    }

    #[test]
    fn withdraw_texture_survives_parcel_drop() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, &texture)
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        drop(texture);
        drop(pool);

        let loan = grant
            .claim(&mut frame)
            .expect("claim")
            .consume()
            .expect("read after parcel drop");
        assert_eq!(loan.len(), 4 * 4 * 4);
    }

    // ------------------------------------------------------------------
    // Present-on-scheme tests
    // ------------------------------------------------------------------

    #[cfg(feature = "graphics")]
    struct MockWindow;

    #[cfg(feature = "graphics")]
    impl raw_window_handle::HasWindowHandle for MockWindow {
        fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            Ok(unsafe {
                raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Web(
                    raw_window_handle::WebWindowHandle::new(0),
                ))
            })
        }
    }

    #[cfg(feature = "graphics")]
    impl raw_window_handle::HasDisplayHandle for MockWindow {
        fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            Ok(unsafe {
                raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Web(
                    raw_window_handle::WebDisplayHandle::new(),
                ))
            })
        }
    }

    #[cfg(feature = "graphics")]
    fn mock_swapchain_pool(device: &Arc<Device>) -> (Context, crate::swapchain_pool::SwapchainPool) {
        let ctx = device.create_context().unwrap();
        let pool = crate::swapchain_pool::SwapchainPool::new(&ctx, &MockWindow, 2).expect("swapchain pool");
        (ctx, pool)
    }

    #[cfg(feature = "graphics")]
    fn mock_present_count(device: &Arc<Device>) -> usize {
        let backend = device.inner.backend.lock().unwrap();
        backend.test_surface_present_count()
    }

    #[cfg(feature = "graphics")]
    fn consume_present(tx: &Transaction, submission: &mut Submission) {
        tx.claim(submission).expect("claim").consume().expect("present");
    }

    #[cfg(feature = "graphics")]
    fn register_exchange_with_copy(scheme: &mut Scheme, lease: &crate::swapchain_pool::PresentLease) -> Transaction {
        let rt = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("render target");
        scheme.copy_to_present(&rt, lease);
        scheme.register_present_exchange(lease)
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn submit_rejects_dispatch_mesh_without_pipeline() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("rt");
        {
            let mut pass = scheme.render_pass("mesh", &rt, crate::types::TargetLoad::Discard);
            pass.dispatch_mesh(1, 1, 1);
            pass.finish();
        }
        let err = scheme.submit().expect_err("dispatch_mesh without pipeline");
        let s = err.to_string();
        assert!(s.contains("set_mesh_pipeline"), "unexpected error: {s}");
    }

    #[test]
    fn submit_rejects_blas_as_shader_accel() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let blas = crate::AccelerationStructure::blas_triangles(&device, 1, 3, 12).expect("BLAS");
        let pipeline = mock_pipeline(&device, &mock_shader(&device));
        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("trace", &pipeline)
            .with_parcel(&blas, NodeAccess::Read)
            .dispatch(1, 1, 1);
        let err = scheme.submit().expect_err("BLAS as Accel");
        let s = err.to_string();
        assert!(s.contains("BLAS"), "unexpected error: {s}");
    }

    #[test]
    fn build_blas_rejects_vertex_range_past_parcel() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let verts = pool
            .acquire_buffer(12, BufferKind::Scattered, None, BufferFlags::ACCEL_INPUT, None)
            .expect("verts");
        let blas = crate::AccelerationStructure::blas_triangles(&device, 2, 6, 12).expect("BLAS");
        let mut scheme = Scheme::new(&ctx);
        let err = scheme
            .build_blas(&blas, verts.whole(), 6, 12, None)
            .expect_err("vertex range");
        let s = err.to_string();
        assert!(s.contains("exceeds parcel size"), "unexpected error: {s}");
    }

    #[test]
    fn build_blas_rejects_index_count_not_multiple_of_three() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let verts = pool
            .acquire_buffer(36, BufferKind::Scattered, None, BufferFlags::ACCEL_INPUT, None)
            .expect("verts");
        let idx = pool
            .acquire_buffer(16, BufferKind::Scattered, None, BufferFlags::ACCEL_INPUT, None)
            .expect("idx");
        let blas = crate::AccelerationStructure::blas_triangles(&device, 2, 3, 12).expect("BLAS");
        let mut scheme = Scheme::new(&ctx);
        let err = scheme
            .build_blas(&blas, verts.whole(), 3, 12, Some((idx.whole(), 4)))
            .expect_err("index_count");
        let s = err.to_string();
        assert!(s.contains("multiple of 3"), "unexpected error: {s}");
    }

    #[test]
    fn tlas_retains_blas_after_caller_drop() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let blas = crate::AccelerationStructure::blas_triangles(&device, 1, 3, 12).expect("BLAS");
        let tlas = crate::AccelerationStructure::tlas(&device, 1).expect("TLAS");
        let blas_handle = blas.handle;
        let mut scheme = Scheme::new(&ctx);
        let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        scheme
            .build_tlas(
                &tlas,
                &[crate::AccelInstance {
                    blas: &blas,
                    transform: identity,
                    mask: 0xFF,
                    custom_index: 0,
                }],
            )
            .expect("build_tlas");
        drop(blas);
        device.with_mock(|m| {
            assert!(m.has_accel(blas_handle), "TLAS must keep the BLAS GPU object alive");
        });
        drop(tlas);
        device.with_mock(|m| {
            assert!(!m.has_accel(blas_handle), "BLAS should destroy after last TLAS drop");
        });
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn register_present_exchange_is_metadata_only() {
        let device = mock_device();
        let (ctx, pool) = mock_swapchain_pool(&device);
        let lease = pool.lease();

        let mut scheme = Scheme::new(&ctx);
        assert_eq!(scheme.ir_node_count(), 0);

        let tx = scheme.register_present_exchange(&lease);
        assert_eq!(scheme.ir_node_count(), 0, "registration must not append IR nodes");
        assert_eq!(
            match tx.key {
                ClaimKey::Present { present_idx } => present_idx,
                _ => panic!("expected present"),
            },
            0
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn register_present_exchange_marks_dirty() {
        let device = mock_device();
        let (ctx, pool) = mock_swapchain_pool(&device);
        let lease = pool.lease();

        let mut scheme = Scheme::new(&ctx);
        assert!(scheme.is_dirty(), "new scheme starts dirty");

        scheme.register_present_exchange(&lease);
        assert!(
            scheme.is_dirty(),
            "register_present_exchange must mark the scheme dirty"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn bind_before_write_leaves_coarse_in_non_present_partition() {
        use crate::task_graph::analysis;

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let parcel = retained_buffer(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        scheme.register_present_exchange(&lease);
        scheme
            .node("coarse", &pipeline)
            .with_parcel(&parcel, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("fine", &pipeline)
            .with_parcel(&parcel, NodeAccess::Read)
            .with_present(&lease)
            .dispatch(1, 1, 1);

        let partitions = analysis::describe_logical_partitions(
            &scheme.ir,
            &analysis::schedule_waves(&scheme.ir, &analysis::build_edges(&scheme.ir)),
        );
        assert!(
            partitions.iter().any(|p| p.is_pure_compute() && !p.has_present),
            "coarse compute must remain outside the present partition; got {partitions:?}"
        );
        assert!(
            partitions.last().is_some_and(|p| p.has_present),
            "present partition must be last; got {partitions:?}"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn submit_rejects_unused_present_transaction() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        scheme.register_present_exchange(&lease);

        let err = scheme.submit().expect_err("submit without drawable access");
        assert!(err.to_string().contains("never accesses"), "unexpected error: {err}");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn submit_rejects_unregistered_present_lease_access() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let mut pool = RetainedPool::new(device.clone());
        let parcel = retained_buffer(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write", &pipeline)
            .with_parcel(&parcel, NodeAccess::Write)
            .with_present(&lease)
            .dispatch(1, 1, 1);

        let err = scheme.submit().expect_err("submit without registered transaction");
        assert!(
            err.to_string().contains("no exchange transaction"),
            "unexpected error: {err}"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn submit_rejects_first_present_access_that_reads() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_texture_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut scheme = Scheme::new(&ctx);
        scheme.register_present_exchange(&lease);
        scheme
            .node("filter", &pipeline)
            .with_present_access(&lease, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);

        let err = scheme.submit().expect_err("first present touch must be Write");
        assert!(
            err.to_string()
                .contains("first PresentLease access must be Write or Overwrite"),
            "unexpected error: {err}"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn copy_to_present_appends_ir_node() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("render target");
        assert_eq!(scheme.ir_node_count(), 0);

        scheme.copy_to_present(&rt, &lease);
        assert_eq!(scheme.ir_node_count(), 1);

        match &scheme.ir.nodes[0].kind {
            NodeKind::CopyRenderTarget {
                dst: ResourceId::PresentLease(0),
                ..
            } => {}
            other => panic!("expected CopyRenderTarget{{dst:PresentLease(0)}}, got {other:?}"),
        }
        assert!(scheme.is_dirty(), "copy_to_present must mark the scheme dirty");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn copy_to_texture_appends_ir_node() {
        use crate::types::{TextureFlags, TextureFormat, TextureKind};

        let device = mock_device();
        let ctx = device.create_context().expect("context");
        let mut pool = crate::RetainedPool::new(device.clone());
        let tex = pool
            .acquire_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
                None,
            )
            .expect("texture");
        let tex_handle = tex.texture_handle().expect("texture handle");

        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(4, 4, TextureFormat::Rgba8Unorm, None)
            .expect("render target");
        assert_eq!(scheme.ir_node_count(), 0);

        scheme.copy_to_texture(&rt, &tex).expect("copy_to_texture");
        assert_eq!(scheme.ir_node_count(), 1);

        match &scheme.ir.nodes[0].kind {
            NodeKind::CopyRenderTarget {
                dst: ResourceId::Texture(h),
                ..
            } => assert_eq!(*h, tex_handle),
            other => panic!("expected CopyRenderTarget{{dst:Texture}}, got {other:?}"),
        }
        assert!(scheme.is_dirty(), "copy_to_texture must mark the scheme dirty");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn copy_to_texture_rejects_buffer_parcel() {
        use crate::types::{BufferKind, TextureFormat};

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().expect("context");
        let buffer = pool
            .acquire_buffer_sized::<u32>(4, BufferKind::Scattered, crate::types::BufferFlags::empty())
            .expect("buffer");

        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(4, 4, TextureFormat::Rgba8Unorm, None)
            .expect("render target");
        let err = scheme
            .copy_to_texture(&rt, &buffer)
            .expect_err("buffer parcel must fail");
        assert!(err.to_string().contains("texture parcel"), "unexpected error: {err}");
        assert_eq!(scheme.ir_node_count(), 0);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn copy_to_texture_rejects_missing_copy_dst_flag() {
        use crate::types::{TextureFlags, TextureFormat, TextureKind};

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().expect("context");
        let texture = pool
            .acquire_texture(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::COPY_SRC,
                None,
            )
            .expect("texture");

        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(4, 4, TextureFormat::Rgba8Unorm, None)
            .expect("render target");
        let err = scheme
            .copy_to_texture(&rt, &texture)
            .expect_err("missing COPY_DST must fail");
        assert!(err.to_string().contains("COPY_DST"), "unexpected error: {err}");
        assert_eq!(scheme.ir_node_count(), 0);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn copy_to_texture_rejects_dimension_mismatch() {
        use crate::types::{TextureFlags, TextureFormat, TextureKind};

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().expect("context");
        let texture = pool
            .acquire_texture(
                8,
                8,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::COPY_DST,
                None,
            )
            .expect("texture");

        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(4, 4, TextureFormat::Rgba8Unorm, None)
            .expect("render target");
        let err = scheme
            .copy_to_texture(&rt, &texture)
            .expect_err("dimension mismatch must fail");
        assert!(
            err.to_string().contains("does not match render target"),
            "unexpected error: {err}"
        );
        assert_eq!(scheme.ir_node_count(), 0);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn copy_to_texture_rejects_format_mismatch() {
        use crate::types::{TextureFlags, TextureFormat, TextureKind};

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().expect("context");
        let texture = pool
            .acquire_texture(
                4,
                4,
                TextureFormat::Bgra8Unorm,
                TextureKind::Direct,
                TextureFlags::COPY_DST,
                None,
            )
            .expect("texture");

        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(4, 4, TextureFormat::Rgba8Unorm, None)
            .expect("render target");
        let err = scheme
            .copy_to_texture(&rt, &texture)
            .expect_err("format mismatch must fail");
        assert!(
            err.to_string().contains("does not match render target"),
            "unexpected error: {err}"
        );
        assert_eq!(scheme.ir_node_count(), 0);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn with_present_placeholder_in_resource_slots_when_last() {
        // Correct ordering: with_views first, then with_present.
        // The placeholder must end up appended to the existing resource_slots.
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut scheme = Scheme::new(&ctx);

        // We can't directly inspect the builder — dispatch() consumes it and pushes into IR.
        // Instead, verify via the IR node's resource_slots.
        scheme
            .node("n", &pipeline)
            // Bind a fake slot (index 42) via raw slot push, simulated here by using
            // with_views with a ResourceHandle. We use a parcel handle for this.
            .with_present(&lease)
            .dispatch(1, 1, 1);

        // Node 0 must be a Dispatch whose resource_slots end with PRESENT_LEASE_SLOT_PLACEHOLDER.
        match &scheme.ir.nodes[0].kind {
            NodeKind::Dispatch { resource_slots, .. } => {
                assert!(
                    resource_slots.last() == Some(&crate::task_graph::PRESENT_LEASE_SLOT_PLACEHOLDER),
                    "last resource_slot must be PRESENT_LEASE_SLOT_PLACEHOLDER; got {resource_slots:?}"
                );
            }
            other => panic!("expected Dispatch node, got {other:?}"),
        }

        // Bindings must include PresentLease(0) as the last entry.
        assert!(
            scheme.ir.nodes[0]
                .bindings
                .iter()
                .any(|b| b.resource == ResourceId::PresentLease(0)),
            "bindings must contain PresentLease(0)"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn with_present_placeholder_preserved_when_with_views_follows() {
        // Regression test: with_views called AFTER with_present must preserve
        // the PRESENT_LEASE_SLOT_PLACEHOLDER that with_present appended.
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_texture_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let lease_tex = scheme_lease_texture_for_test(&device, &ctx);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n", &pipeline)
            .with_present(&lease) // pushes PLACEHOLDER
            .with_views(&[lease_tex]) // must preserve PLACEHOLDER
            .dispatch(1, 1, 1);

        match &scheme.ir.nodes[0].kind {
            NodeKind::Dispatch { resource_slots, .. } => {
                let has_placeholder = resource_slots
                    .iter()
                    .any(|s| *s == crate::task_graph::PRESENT_LEASE_SLOT_PLACEHOLDER);
                assert!(
                    has_placeholder,
                    "with_views must preserve PRESENT_LEASE_SLOT_PLACEHOLDER; \
                     resource_slots: {resource_slots:?}"
                );
                // The user handle must still be present at slot 0.
                assert_eq!(
                    resource_slots.len(),
                    2,
                    "expected [user_slot, PLACEHOLDER], got {resource_slots:?}"
                );
                assert_ne!(
                    resource_slots[0],
                    crate::task_graph::PRESENT_LEASE_SLOT_PLACEHOLDER,
                    "first slot must be the user handle, not a placeholder"
                );
            }
            other => panic!("expected Dispatch node, got {other:?}"),
        }
    }

    #[cfg(feature = "graphics")]
    fn mock_buf_then_present_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(
            device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> buf, DirectSpatial<float4> dst, ThreadId id) {
    buf[0] = 1u;
    if (id.x == 0 && id.y == 0) {
        dst[uint2(0, 0)] = float4(1.0, 0.0, 0.0, 1.0);
    }
}
"#,
        )
        .expect("compile buf+present shader")
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn with_present_placeholder_at_middle_shader_slot() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_buf_then_present_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let mut pool = RetainedPool::new(device.clone());
        let buf = pool
            .acquire_buffer(4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("buffer");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n", &pipeline)
            .with_parcel(&buf, NodeAccess::Read)
            .with_present(&lease)
            .dispatch(1, 1, 1);

        match &scheme.ir.nodes[0].kind {
            NodeKind::Dispatch { resource_slots, .. } => {
                assert_eq!(resource_slots.len(), 2);
                assert_ne!(resource_slots[0], PRESENT_LEASE_SLOT_PLACEHOLDER);
                assert_eq!(resource_slots[1], PRESENT_LEASE_SLOT_PLACEHOLDER);
            }
            other => panic!("expected Dispatch node, got {other:?}"),
        }
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn with_present_access_records_readwrite_binding() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_texture_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n", &pipeline)
            .with_present_access(&lease, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);

        let binding = scheme.ir.nodes[0]
            .bindings
            .iter()
            .find(|b| b.resource == ResourceId::PresentLease(0))
            .expect("present binding");
        assert_eq!(binding.access, NodeAccess::ReadWrite);
    }

    #[cfg(feature = "graphics")]
    fn mock_sampler_then_present_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(
            device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Filter samp, DirectSpatial<float4> dst, ThreadId id) {
    if (id.x == 0 && id.y == 0) {
        dst[uint2(0, 0)] = float4(1.0, 0.0, 0.0, 1.0);
    }
}
"#,
        )
        .expect("compile sampler+present shader")
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn with_present_after_sampler_submits_and_presents() {
        // Sampler contributes a resource slot without a hazard binding, so the
        // PresentLease binding index is 0 while the placeholder is at slot 1.
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_sampler_then_present_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let sampler = crate::Sampler::linear(&device).expect("sampler");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n", &pipeline)
            .with_parcel(&sampler, NodeAccess::Read)
            .with_present(&lease)
            .dispatch(1, 1, 1);
        let transaction = scheme.register_present_exchange(&lease);

        let dispatch = scheme
            .ir
            .nodes
            .iter()
            .find(|n| matches!(n.kind, NodeKind::Dispatch { .. }))
            .expect("dispatch node");
        match &dispatch.kind {
            NodeKind::Dispatch { resource_slots, .. } => {
                assert_eq!(resource_slots.len(), 2);
                assert_ne!(resource_slots[0], PRESENT_LEASE_SLOT_PLACEHOLDER);
                assert_eq!(resource_slots[1], PRESENT_LEASE_SLOT_PLACEHOLDER);
            }
            other => panic!("expected Dispatch node, got {other:?}"),
        }
        assert_eq!(
            dispatch.bindings.len(),
            1,
            "sampler must not emit a hazard binding; only PresentLease remains"
        );

        let before = mock_present_count(&device);
        let mut submission = scheme.submit().expect("submit with sampler-before-present");
        let claim = transaction.claim(&mut submission).expect("claim");
        claim.consume().expect("consume");
        assert_eq!(
            mock_present_count(&device),
            before + 1,
            "present placeholder after sampler must resolve through submit"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn with_present_after_buffer_dependency_submits_and_presents() {
        // Dependency-only bindings omit shader slots, so PresentLease is binding
        // index 1 while the placeholder is the only (index 0) resource slot.
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_texture_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let mut pool = RetainedPool::new(device.clone());
        let buf = pool
            .acquire_buffer(4, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .expect("buffer");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n", &pipeline)
            .with_buffer_dependency(&buf, NodeAccess::Read)
            .with_present(&lease)
            .dispatch(1, 1, 1);
        let transaction = scheme.register_present_exchange(&lease);

        let dispatch = scheme
            .ir
            .nodes
            .iter()
            .find(|n| matches!(n.kind, NodeKind::Dispatch { .. }))
            .expect("dispatch node");
        match &dispatch.kind {
            NodeKind::Dispatch { resource_slots, .. } => {
                assert_eq!(resource_slots, &[PRESENT_LEASE_SLOT_PLACEHOLDER]);
            }
            other => panic!("expected Dispatch node, got {other:?}"),
        }
        assert!(
            dispatch.bindings.len() >= 2,
            "dependency binding plus PresentLease expected; got {:?}",
            dispatch.bindings
        );

        let before = mock_present_count(&device);
        let mut submission = scheme.submit().expect("submit with dependency-before-present");
        let claim = transaction.claim(&mut submission).expect("claim");
        claim.consume().expect("consume");
        assert_eq!(
            mock_present_count(&device),
            before + 1,
            "present placeholder after dependency binding must resolve through submit"
        );
    }

    #[cfg(feature = "graphics")]
    fn scheme_lease_texture_for_test(device: &Arc<Device>, _ctx: &Context) -> crate::types::ResourceHandle {
        // Minimal helper: creates a retained texture parcel and returns a write handle.
        let mut pool = RetainedPool::new(device.clone());
        let parcel = pool
            .acquire_texture(
                4,
                4,
                crate::types::TextureFormat::Rgba8Unorm,
                crate::types::TextureKind::Direct,
                crate::types::TextureFlags::empty(),
                None,
            )
            .expect("texture parcel");
        parcel
            .handle(crate::types::ResourceAccess::Write)
            .expect("write handle")
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_exchange_submit_increments_present_count() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let present = register_exchange_with_copy(&mut scheme, &lease);

        let before = mock_present_count(&device);
        let mut submission = scheme.submit().expect("first submit");
        consume_present(&present, &mut submission);
        let after = mock_present_count(&device);
        assert_eq!(after, before + 1, "present must fire one swapchain present");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn transaction_claim_consume_presents_once() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let transaction = register_exchange_with_copy(&mut scheme, &lease);

        let before = mock_present_count(&device);
        let mut submission = scheme.submit().expect("submit");
        let handle = submission.handle();
        let claim = transaction.claim(&mut submission).expect("claim");
        claim.consume().expect("consume");
        assert_eq!(mock_present_count(&device), before + 1);
        // Timeline handle remains usable after the claim is consumed.
        let _ = handle.timeline_value();
        let err = transaction.claim(&mut submission).expect_err("second claim must fail");
        assert!(err.to_string().contains("already consumed"), "unexpected: {err}");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn dropping_submission_discards_untaken_claim() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let _present = register_exchange_with_copy(&mut scheme, &lease);
        let before = mock_present_count(&device);
        let submission = scheme.submit().expect("submit");
        drop(submission);
        assert_eq!(mock_present_count(&device), before, "discarded claim must not present");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn two_pools_receive_distinct_present_bindings() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let left_pool = crate::swapchain_pool::SwapchainPool::new(&ctx, &MockWindow, 2).expect("left pool");
        let right_pool = crate::swapchain_pool::SwapchainPool::new(&ctx, &MockWindow, 2).expect("right pool");
        let left = left_pool.lease();
        let right = right_pool.lease();
        assert_eq!(left.id, 0);
        assert_eq!(right.id, 0);

        let mut scheme = Scheme::new(&ctx);
        let rt_a = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("rt");
        let rt_b = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("rt");
        scheme.copy_to_present(&rt_a, &left);
        let left_grant = scheme.register_present_exchange(&left);
        scheme.copy_to_present(&rt_b, &right);
        let right_grant = scheme.register_present_exchange(&right);

        assert_ne!(
            left_grant.binding_id, right_grant.binding_id,
            "distinct pools must intern distinct scheme bindings"
        );
        assert_eq!(
            match left_grant.key {
                ClaimKey::Present { present_idx } => present_idx,
                _ => panic!("expected present"),
            },
            0
        );
        assert_eq!(
            match right_grant.key {
                ClaimKey::Present { present_idx } => present_idx,
                _ => panic!("expected present"),
            },
            1
        );

        let left_res = scheme.ir.nodes[0].bindings[1].resource;
        let right_res = scheme.ir.nodes[1].bindings[1].resource;
        assert_eq!(left_res, ResourceId::PresentLease(left_grant.binding_id));
        assert_eq!(right_res, ResourceId::PresentLease(right_grant.binding_id));
        assert_ne!(left_res, right_res);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn same_lease_reuses_present_binding_and_grant() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let first = scheme.register_present_exchange(&lease);
        let second = scheme.register_present_exchange(&lease);
        assert_eq!(
            match first.key {
                ClaimKey::Present { present_idx } => present_idx,
                _ => panic!("expected present"),
            },
            match second.key {
                ClaimKey::Present { present_idx } => present_idx,
                _ => panic!("expected present"),
            }
        );
        assert_eq!(first.binding_id, second.binding_id);
        assert_eq!(scheme.ir_node_count(), 0, "reuse must not append IR nodes");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn two_present_claims_consume_independently() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let left_pool = crate::swapchain_pool::SwapchainPool::new(&ctx, &MockWindow, 2).expect("left pool");
        let right_pool = crate::swapchain_pool::SwapchainPool::new(&ctx, &MockWindow, 2).expect("right pool");

        let mut scheme = Scheme::new(&ctx);
        let left_lease = left_pool.lease();
        let right_lease = right_pool.lease();
        let rt_a = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("rt");
        let rt_b = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("rt");
        scheme.copy_to_present(&rt_a, &left_lease);
        let left_tx = scheme.register_present_exchange(&left_lease);
        scheme.copy_to_present(&rt_b, &right_lease);
        let right_tx = scheme.register_present_exchange(&right_lease);

        let before = mock_present_count(&device);
        let mut submission = scheme.submit().expect("submit");
        let left_claim = left_tx.claim(&mut submission).expect("left claim");
        let right_claim = right_tx.claim(&mut submission).expect("right claim");

        left_claim.consume().expect("left present");
        assert_eq!(mock_present_count(&device), before + 1);
        drop(right_claim); // discard must not present
        assert_eq!(mock_present_count(&device), before + 1);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn eager_acquire_rejects_wrong_pool_with_matching_local_id() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let left_pool = crate::swapchain_pool::SwapchainPool::new(&ctx, &MockWindow, 2).expect("left pool");
        let right_pool = crate::swapchain_pool::SwapchainPool::new(&ctx, &MockWindow, 2).expect("right pool");
        let left = left_pool.lease();
        let right = right_pool.lease();
        assert_eq!(left.id, right.id);

        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("rt");
        scheme.copy_to_present(&rt, &left);
        let _grant = scheme.register_present_exchange(&left);

        let wrong = right_pool.acquire_present(&right).expect("acquire right");
        let err = scheme
            .submit_with_acquired_presents(vec![wrong])
            .expect_err("wrong pool must be rejected");
        assert!(err.to_string().contains("provenance"), "unexpected error: {err}");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn eager_acquire_mixed_provenance_cancels_without_presenting() {
        // First claim is valid; second is from the wrong pool. Validation must reject
        // before converting either AcquiredPresent into Frame, otherwise Drop would
        // implicitly present the already-converted frame.
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let left_pool = crate::swapchain_pool::SwapchainPool::new(&ctx, &MockWindow, 2).expect("left pool");
        let right_pool = crate::swapchain_pool::SwapchainPool::new(&ctx, &MockWindow, 2).expect("right pool");

        let mut scheme = Scheme::new(&ctx);
        let left = left_pool.lease();
        let right = right_pool.lease();
        let rt_a = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("rt");
        let rt_b = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("rt");
        scheme.copy_to_present(&rt_a, &left);
        let _left = scheme.register_present_exchange(&left);
        scheme.copy_to_present(&rt_b, &right);
        let _right = scheme.register_present_exchange(&right);

        let good = left_pool.acquire_present(&left_pool.lease()).expect("acquire left");
        let wrong = left_pool
            .acquire_present(&left_pool.lease())
            .expect("acquire left again for wrong slot");
        // Swap: grant order is left then right, but second claim is from left pool.
        let before = mock_present_count(&device);
        let err = scheme
            .submit_with_acquired_presents(vec![good, wrong])
            .expect_err("second claim must fail provenance for right grant");
        assert!(err.to_string().contains("provenance"), "unexpected: {err}");
        assert_eq!(
            mock_present_count(&device),
            before,
            "rejected eager submit must not present any converted frame"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn surface_exchange_bind_rejects_duplicate_for_same_lease() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let surface = crate::exchange::SurfaceExchange::new(&ctx, &MockWindow, crate::types::SurfaceConfig::default())
            .expect("surface exchange");
        let tex_a = mock_direct_texture(&device);
        let tex_b = mock_direct_texture(&device);

        let mut scheme = Scheme::new(&ctx);
        let first = surface.bind(&mut scheme, &tex_a).expect("first bind");
        let err = surface
            .bind(&mut scheme, &tex_b)
            .expect_err("second bind for same lease must fail");
        assert!(err.to_string().contains("already bound"), "unexpected: {err}");
        // First transaction still works; only one copy+grant recorded beyond the bind path.
        let copy_count = scheme
            .ir
            .nodes
            .iter()
            .filter(|n| n.label == "copy_texture_to_present")
            .count();
        assert_eq!(copy_count, 1, "rejected bind must not append a second copy");
        let mut submission = scheme.submit().expect("submit");
        first.claim(&mut submission).expect("claim").consume().expect("present");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn two_surfaces_bind_copy_resolve_and_claim_independently() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let left = crate::exchange::SurfaceExchange::new(&ctx, &MockWindow, crate::types::SurfaceConfig::default())
            .expect("left surface");
        let right = crate::exchange::SurfaceExchange::new(&ctx, &MockWindow, crate::types::SurfaceConfig::default())
            .expect("right surface");
        let left_tex = mock_direct_texture(&device);
        let right_tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write_left", &pipeline)
            .with_parcel(&left_tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("write_right", &pipeline)
            .with_parcel(&right_tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let left_tx = left.bind(&mut scheme, &left_tex).expect("bind left");
        let right_tx = right.bind(&mut scheme, &right_tex).expect("bind right");

        assert_ne!(left_tx.binding_id(), right_tx.binding_id());
        let copy_bindings: Vec<_> = scheme
            .ir
            .nodes
            .iter()
            .filter(|n| n.label == "copy_texture_to_present")
            .map(|n| match &n.kind {
                NodeKind::CopyTexture { dst, .. } => *dst,
                other => panic!("expected CopyTexture, got {other:?}"),
            })
            .collect();
        assert_eq!(copy_bindings.len(), 2);
        assert_eq!(copy_bindings[0], ResourceId::PresentLease(left_tx.binding_id()));
        assert_eq!(copy_bindings[1], ResourceId::PresentLease(right_tx.binding_id()));

        let before = mock_present_count(&device);
        let mut submission = scheme.submit().expect("submit");
        let left_stamp = submission
            .present_frame_submit_timeline(0)
            .expect("left claim must hold a stamped frame");
        let right_stamp = submission
            .present_frame_submit_timeline(1)
            .expect("right claim must hold a stamped frame");
        assert_eq!(left_stamp, submission.timeline_value());
        assert_eq!(right_stamp, submission.timeline_value());

        let left_claim = left_tx.claim(&mut submission).expect("left claim");
        let right_claim = right_tx.claim(&mut submission).expect("right claim");
        left_claim.consume().expect("left present");
        assert_eq!(mock_present_count(&device), before + 1);
        right_claim.discard().expect("right discard");
        assert_eq!(
            mock_present_count(&device),
            before + 1,
            "discard must not present the other surface"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn surface_claim_impl_drop_cancels_without_presenting() {
        // Raw SurfaceClaimImpl may be dropped on finish_submit_frame failure before
        // wrapping in Claim/Submission. Drop must cancel, not present via Frame::drop.
        let device = mock_device();
        let (_ctx, spool) = mock_swapchain_pool(&device);
        let acquired = spool.acquire_present(&spool.lease()).expect("acquire");
        let (_lease, _pool, _slot, _gen, _handle, _uav, frame) = acquired.into_parts();
        let before = mock_present_count(&device);
        drop(crate::exchange::SurfaceClaimImpl::new(frame));
        assert_eq!(
            mock_present_count(&device),
            before,
            "SurfaceClaimImpl Drop must cancel, not present"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn partial_acquire_failure_discards_submitted_binding_without_present() {
        use crate::task_graph::cross_submit::ResourceKey;
        use crate::timeline::PromiseState;

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let left = crate::exchange::SurfaceExchange::new(&ctx, &MockWindow, crate::types::SurfaceConfig::default())
            .expect("left");
        let right = crate::exchange::SurfaceExchange::new(&ctx, &MockWindow, crate::types::SurfaceConfig::default())
            .expect("right");
        let left_tex = mock_direct_texture(&device);
        let right_tex = mock_direct_texture(&device);
        let buf = retained_buffer(&mut pool);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        // Force distinct present partitions: left copy must precede a buf write that
        // the right path reads, so binding 1 is introduced in a later wave.
        scheme
            .node("write_left", &pipeline)
            .with_parcel(&left_tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let _left_tx = left.bind(&mut scheme, &left_tex).expect("bind left");
        scheme
            .node("bridge", &pipeline)
            .with_parcel(&left_tex, NodeAccess::Read)
            .with_parcel(&buf, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("write_right", &pipeline)
            .with_parcel(&right_tex, NodeAccess::Write)
            .with_parcel(&buf, NodeAccess::Read)
            .dispatch(1, 1, 1);
        let _right_tx = right.bind(&mut scheme, &right_tex).expect("bind right");

        right.fail_next_acquire();
        let before = mock_present_count(&device);
        let err = scheme
            .submit()
            .expect_err("right acquire must fail after left submitted");
        assert!(
            err.to_string().contains("test-injected acquire failure") || err.to_string().contains("acquire"),
            "unexpected: {err}"
        );
        assert_eq!(
            mock_present_count(&device),
            before,
            "partial failure must not present the submitted left frame"
        );

        let key = ResourceKey::Texture(left_tex.gpu_handle());
        let war = {
            let stamp = scheme.submit_state.resource_stamps().get(&key).expect("left tex stamp");
            let pending = stamp.pending.lock().unwrap();
            assert!(
                !pending.is_empty(),
                "left present partition must have submitted and registered WAR before right acquire failed"
            );
            pending[0].poll()
        };
        match war {
            PromiseState::Resolved(_) => {}
            other => panic!("left source WAR must settle on partial failure cleanup, got {other:?}"),
        }
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn resize_advances_generation_and_stales_prior_claim() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let surface = crate::exchange::SurfaceExchange::new(&ctx, &MockWindow, crate::types::SurfaceConfig::default())
            .expect("surface");
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write", &pipeline)
            .with_parcel(&tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let tx = surface.bind(&mut scheme, &tex).expect("bind");
        assert_eq!(tx.generation(), 0);
        let mut submission = scheme.submit().expect("submit");

        surface.resize(64, 64).expect("resize");
        assert_eq!(tx.generation(), 1);
        let err = tx.claim(&mut submission).expect_err("claim must be stale after resize");
        assert!(err.to_string().contains("stale"), "unexpected: {err}");

        let before = mock_present_count(&device);
        drop(submission);
        assert_eq!(
            mock_present_count(&device),
            before,
            "dropping stale submission must discard, not present"
        );

        // Next submit under the new generation still works.
        let mut submission2 = scheme.submit().expect("submit after resize");
        tx.claim(&mut submission2)
            .expect("fresh claim at new generation")
            .consume()
            .expect("present");
        assert_eq!(mock_present_count(&device), before + 1);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn resize_one_surface_does_not_stale_other_transaction() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let left = crate::exchange::SurfaceExchange::new(&ctx, &MockWindow, crate::types::SurfaceConfig::default())
            .expect("left");
        let right = crate::exchange::SurfaceExchange::new(&ctx, &MockWindow, crate::types::SurfaceConfig::default())
            .expect("right");
        let left_tex = mock_direct_texture(&device);
        let right_tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write_left", &pipeline)
            .with_parcel(&left_tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("write_right", &pipeline)
            .with_parcel(&right_tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let left_tx = left.bind(&mut scheme, &left_tex).expect("bind left");
        let right_tx = right.bind(&mut scheme, &right_tex).expect("bind right");
        let mut submission = scheme.submit().expect("submit");

        left.resize(80, 80).expect("resize left only");
        assert_eq!(left_tx.generation(), 1);
        assert_eq!(right_tx.generation(), 0);

        let err = left_tx.claim(&mut submission).expect_err("left claim stale");
        assert!(err.to_string().contains("stale"), "unexpected: {err}");
        right_tx
            .claim(&mut submission)
            .expect("right claim still current")
            .consume()
            .expect("right present");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_exchange_stamps_frame_with_present_partition_timeline() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let present = register_exchange_with_copy(&mut scheme, &lease);

        let submission = scheme.submit().expect("submit");
        // No read grants → finish_submit_frame keeps the present-partition tv as
        // the submission timeline; the acquired frame must carry the same stamp
        // so Present waits on that epoch rather than timeline_next-1.
        let stamped = submission
            .present_frame_submit_timeline(
                (match present.key {
                    ClaimKey::Present { present_idx } => present_idx,
                    _ => panic!("expected present"),
                }) as usize,
            )
            .expect("present frame must be stamped before consume");
        assert_eq!(
            stamped,
            submission.timeline_value(),
            "frame submit_tv must equal present-partition timeline"
        );
        let mut submission = submission;
        consume_present(&present, &mut submission);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_exchange_second_present_errors() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let present = register_exchange_with_copy(&mut scheme, &lease);

        let mut submission = scheme.submit().expect("submit");
        consume_present(&present, &mut submission);
        let err = present.claim(&mut submission).expect_err("second present must fail");
        assert!(err.to_string().contains("already consumed"), "unexpected error: {err}");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_grant_rejects_cross_scheme_submission() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme_a = Scheme::new(&ctx);
        let present_a = scheme_a.register_present_exchange(&lease);

        let mut scheme_b = Scheme::new(&ctx);
        register_exchange_with_copy(&mut scheme_b, &lease);
        let mut submission_b = scheme_b.submit().expect("submit b");

        let err = present_a
            .claim(&mut submission_b)
            .expect_err("cross-scheme present must fail");
        assert!(err.to_string().contains("different scheme"), "unexpected error: {err}");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_exchange_submit_twice_presents_independently() {
        // Each submit acquires a fresh swapchain frame; both must be presentable.
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let present = register_exchange_with_copy(&mut scheme, &lease);

        let mut submission1 = scheme.submit().expect("submit 1");
        let mut submission2 = scheme.submit().expect("submit 2");

        // Present in order; both must succeed.
        consume_present(&present, &mut submission1);
        consume_present(&present, &mut submission2);

        assert_eq!(mock_present_count(&device), 2, "two submits → two presents");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_exchange_scheme_records_once_per_slot() {
        // The present-aware retention path must record the first time a given
        // swapchain slot is seen and resubmit from cache on subsequent encounters.
        // Because the mock backend cycles through slots, the N-th submit may
        // record a new slot; we only assert that at least one resubmit occurs.
        let _cb = crate::test_support::CbReuseOverride::force_enabled();
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let present = register_exchange_with_copy(&mut scheme, &lease);

        // Submit many frames; the mock backend has a fixed pool of slot ids
        // so after depth frames we must see at least one cache hit.
        let depth = 6;
        for i in 0..depth {
            let mut submission = scheme.submit().expect(&format!("submit {i}"));
            consume_present(&present, &mut submission);
        }

        // At minimum, the scheme must have recorded (not resubmitted) at most
        // `depth` times — but with a 2-deep swapchain mock the 3rd submit
        // should have been a slot-keyed hit.
        #[cfg(not(feature = "metal"))]
        assert!(
            scheme.replay_stats().resubmit_hits > 0,
            "with {} submits over a 2-deep pool, expected retention hits; stats: {:?}",
            depth,
            scheme.replay_stats()
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn dropped_frame_without_present_cancels_swapchain() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        register_exchange_with_copy(&mut scheme, &lease);

        let before = mock_present_count(&device);
        {
            let _frame = scheme.submit().expect("submit");
            // Drop without calling present — must cancel, not implicitly present.
        }
        assert_eq!(
            mock_present_count(&device),
            before,
            "dropped Submission must not trigger swapchain present"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn copy_to_present_and_render_pass_partition_on_present_boundary() {
        use crate::task_graph::analysis;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_render_shader(&device);
        let pipeline = mock_render_pipeline(&device, &shader);

        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("rt");
        let mut pass = scheme.render_pass("render", &rt, crate::types::TargetLoad::Discard);
        pass.set_pipeline(&pipeline);
        pass.draw_fullscreen();
        pass.finish();
        scheme.copy_to_present(&rt, &lease);
        scheme.register_present_exchange(&lease);

        let partitions = analysis::describe_logical_partitions(
            &scheme.ir,
            &analysis::schedule_waves(&scheme.ir, &analysis::build_edges(&scheme.ir)),
        );
        assert!(
            partitions.len() >= 2,
            "render pass and present copy must land in separate logical partitions; got {partitions:?}"
        );
        assert!(
            !partitions[0].has_present,
            "first partition must not touch present lease"
        );
        assert!(
            partitions.iter().any(|p| p.has_present),
            "some partition must touch present lease"
        );
    }

    #[test]
    #[should_panic(expected = "with_parcel: resource has no descriptor")]
    fn with_parcel_panics_on_incompatible_access() {
        let device = mock_device();
        let ctx = device.create_context().expect("context");
        let mut pool = RetainedPool::new(device.clone());
        let texture = pool
            .acquire_texture(
                4,
                4,
                crate::types::TextureFormat::Rgba8Unorm,
                crate::types::TextureKind::Interpolated,
                crate::types::TextureFlags::empty(),
                None,
            )
            .expect("texture");
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("bad_bind", &pipeline)
            .with_parcel(&texture, NodeAccess::Write)
            .dispatch(1, 1, 1);
    }

    fn mock_direct_texture(device: &Arc<Device>) -> crate::Texture {
        let mut pool = RetainedPool::new(device.clone());
        pool.acquire_texture(
            4,
            4,
            crate::types::TextureFormat::Rgba8Unorm,
            crate::types::TextureKind::Direct,
            crate::types::TextureFlags::empty(),
            None,
        )
        .expect("direct texture")
    }

    #[cfg(feature = "graphics")]
    fn present_scheme_with_texture_copy(
        scheme: &mut Scheme,
        tex: &crate::Texture,
        lease: &PresentLease,
        pipeline: &ComputePipeline,
    ) -> Transaction {
        scheme
            .node("write_tex", pipeline)
            .with_parcel(tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme.copy_texture_to_present(tex, lease);
        scheme.register_present_exchange(lease)
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_easement_promise_resolved_at_submit_before_claim_consume() {
        use crate::task_graph::cross_submit::ResourceKey;
        use crate::timeline::PromiseState;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        present_scheme_with_texture_copy(&mut scheme, &tex, &lease, &pipeline);

        let key = ResourceKey::Texture(tex.gpu_handle());
        let submission = scheme.submit().expect("submit");
        let resolved_tv = {
            let stamp = scheme.submit_state.resource_stamps().get(&key).expect("texture stamp");
            assert_eq!(stamp.pending.lock().unwrap().len(), 1);
            // Source WAR resolves at submit from the known copy timeline. Dropping the
            // submission discards the drawable claim without abandoning the WAR promise.
            match stamp.pending.lock().unwrap()[0].poll() {
                PromiseState::Resolved(tv) => tv,
                other => panic!("expected Resolved after submit, got {other:?}"),
            }
        };
        assert_eq!(resolved_tv, submission.timeline_value());
        drop(submission);
        let after_drop = {
            let stamp = scheme.submit_state.resource_stamps().get(&key).expect("texture stamp");
            stamp.pending.lock().unwrap()[0].poll()
        };
        match after_drop {
            PromiseState::Resolved(_) => {}
            other => panic!("WAR must remain Resolved after claim discard, got {other:?}"),
        }
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_easement_tracks_render_target_copy_source_stamp() {
        use crate::task_graph::cross_submit::ResourceKey;
        use crate::timeline::PromiseState;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_render_shader(&device);
        let pipeline = mock_render_pipeline(&device, &shader);

        let mut scheme = Scheme::new(&ctx);
        let rt = scheme
            .lease_render_target(4, 4, crate::types::TextureFormat::Rgba8Unorm, None)
            .expect("rt");
        let rt_handle = scheme.rt(&rt).backend_handle();
        let mut pass = scheme.render_pass("render", &rt, crate::types::TargetLoad::Discard);
        pass.set_pipeline(&pipeline);
        pass.draw_fullscreen();
        pass.finish();
        scheme.copy_to_present(&rt, &lease);
        let _present = scheme.register_present_exchange(&lease);

        let key = ResourceKey::RenderTarget(rt_handle);
        assert!(
            scheme.submit_state.resource_stamps().contains_key(&key),
            "lease_render_target must register an RT stamp for present WAR"
        );

        let submission = scheme.submit().expect("submit");
        let resolved_tv = {
            let stamp = scheme.submit_state.resource_stamps().get(&key).expect("rt stamp");
            assert_eq!(
                stamp.pending.lock().unwrap().len(),
                1,
                "copy_to_present must attach a present-easement promise to the RT stamp"
            );
            match stamp.pending.lock().unwrap()[0].poll() {
                PromiseState::Resolved(tv) => tv,
                other => panic!("expected Resolved after submit, got {other:?}"),
            }
        };
        assert_eq!(resolved_tv, submission.timeline_value());
        drop(submission);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn present_consume_resolves_with_copy_timeline_not_display_present() {
        use crate::task_graph::cross_submit::ResourceKey;
        use crate::timeline::PromiseState;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        let present = present_scheme_with_texture_copy(&mut scheme, &tex, &lease, &pipeline);
        let mut submission = scheme.submit().expect("submit");
        let compute_tv = submission.timeline_value();
        let frame_submit_tv = submission
            .present_frame_submit_timeline(0)
            .expect("present frame must carry submit timeline");
        assert_eq!(
            frame_submit_tv, compute_tv,
            "mock present partition TV should match submission high-water when no grant staging follows"
        );

        consume_present(&present, &mut submission);

        let key = ResourceKey::Texture(tex.gpu_handle());
        let poll_state = {
            let stamp = scheme.submit_state.resource_stamps().get(&key).expect("texture stamp");
            stamp.pending.lock().unwrap()[0].poll()
        };
        match poll_state {
            PromiseState::Resolved(easement_tv) => {
                assert_eq!(
                    easement_tv, frame_submit_tv,
                    "present easement must resolve to the copy/present-partition timeline (easement={easement_tv}, copy={frame_submit_tv})"
                );
            }
            other => panic!("expected Resolved copy timeline, got {other:?}"),
        }
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn submit_gate_folds_resolved_present_promise_into_foreign_reads() {
        use crate::task_graph::cross_submit::ResourceKey;
        use crate::timeline::PromiseState;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));
        let ctx_handle = ctx.backend_handle();

        let mut scheme = Scheme::new(&ctx);
        let present = present_scheme_with_texture_copy(&mut scheme, &tex, &lease, &pipeline);
        let key = ResourceKey::Texture(tex.gpu_handle());

        let mut sub1 = scheme.submit().expect("submit 1");
        consume_present(&present, &mut sub1);

        let _sub2 = scheme.submit().expect("submit 2");
        let stamp = scheme.submit_state.resource_stamps().get(&key).expect("texture stamp");
        let sync = stamp.sync.lock().unwrap();
        assert!(
            sync.foreign_reads.get(ctx_handle).is_some(),
            "submit gate must fold resolved present promise into foreign_reads"
        );
        drop(sync);
        assert_eq!(
            stamp.pending.lock().unwrap().len(),
            1,
            "submit 2 claims a fresh present promise; frame 1's resolved promise must be pruned"
        );
        // Frame 2's promise is resolved at submit from the known copy timeline.
        assert!(
            matches!(stamp.pending.lock().unwrap()[0].poll(), PromiseState::Resolved(_)),
            "frame 2 present promise must be Resolved after submit"
        );
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn submit_gate_does_not_block_on_unconsumed_claim() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        let present = present_scheme_with_texture_copy(&mut scheme, &tex, &lease, &pipeline);
        let mut sub1 = scheme.submit().expect("submit 1");
        // Source WAR already resolved at submit; the next submit must not wait for consume.
        let _sub2 = scheme.submit().expect("submit 2 without consuming claim 1");
        consume_present(&present, &mut sub1);
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn texture_stamp_war_resolved_at_submit_survives_claim_discard() {
        use crate::task_graph::cross_submit::ResourceKey;
        use crate::timeline::PromiseState;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        present_scheme_with_texture_copy(&mut scheme, &tex, &lease, &pipeline);
        let key = ResourceKey::Texture(tex.gpu_handle());

        let submission = scheme.submit().expect("submit");
        let poll = {
            let stamp = scheme.submit_state.resource_stamps().get(&key).expect("texture stamp");
            stamp.pending.lock().unwrap()[0].poll()
        };
        assert!(
            matches!(poll, PromiseState::Resolved(_)),
            "WAR must resolve at submit, got {poll:?}"
        );
        drop(submission);
        let after = {
            let stamp = scheme.submit_state.resource_stamps().get(&key).expect("texture stamp");
            stamp.pending.lock().unwrap()[0].poll()
        };
        assert!(
            matches!(after, PromiseState::Resolved(_)),
            "discarding the claim must not abandon the resolved WAR, got {after:?}"
        );
    }

    /// shader and `copy_texture_to_present` on the same persistent `out_image` must resolve
    /// to one ledger cell (`ResourceSync`), and a present-path submit must record the copy
    /// read on that cell so cross-frame WAR enforcement can key off `foreign_reads`
    /// (present easement is an exercised-claim / foreign read, not a scheduled `last_reads`).
    #[cfg(feature = "graphics")]
    #[test]
    fn out_image_fine_write_and_present_copy_share_ledger_identity() {
        use crate::task_graph::cross_submit::{
            compute_cross_submit_sync, net_access_per_resource, ResourceKey, ResourceKeyMap,
        };
        use crate::task_graph::ResourceId;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_texture_shader(&device));
        let ctx_handle = ctx.backend_handle();
        let tex_handle = tex.gpu_handle();
        let key = ResourceKey::Texture(tex_handle);
        let expected_stamp = tex.whole().stamp_handle();

        let mut scheme = Scheme::new(&ctx);
        // Fine write then present copy on one Texture instance.
        scheme
            .node("fine_write", &pipeline)
            .with_parcel(&tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme.copy_texture_to_present(&tex, &lease);
        let present = scheme.register_present_exchange(&lease);

        let fine_bindings: Vec<_> = scheme
            .ir
            .nodes
            .iter()
            .filter(|n| n.label == "fine_write")
            .flat_map(|n| &n.bindings)
            .filter(|b| matches!(b.resource, ResourceId::Texture(h) if h == tex_handle))
            .collect();
        scheme
            .ir
            .nodes
            .iter()
            .find(|n| n.label == "copy_texture_to_present")
            .and_then(|n| {
                n.bindings.iter().find(|b| {
                    matches!(b.resource, ResourceId::Texture(h) if h == tex_handle) && b.access == NodeAccess::Read
                })
            })
            .expect("present copy must read out_image");
        assert_eq!(fine_bindings.len(), 1);
        assert_eq!(fine_bindings[0].access, NodeAccess::Write);

        let registered = scheme
            .submit_state
            .resource_stamps()
            .get(&key)
            .expect("out_image stamp registered before submit")
            .clone();
        // stamp_handle() returns a fresh Arc wrapper each call; the ledger cell is `sync`.
        assert!(
            Arc::ptr_eq(&registered.sync, &expected_stamp.sync),
            "fine write and present copy must share one ResourceSync ledger cell"
        );

        let net = net_access_per_resource(&scheme.ir);
        assert!(
            net[&key].reads && net[&key].writes,
            "scheme net access must include both sides"
        );

        let mut sub1 = scheme.submit().expect("submit frame 1");
        let frame1_tv = sub1.timeline_value();
        {
            let sync = registered.sync.lock().unwrap();
            let read_tv = sync
                .last_reads
                .get(ctx_handle)
                .expect("present-copy read must be on the ledger after submit");
            let write_tv = sync
                .last_write
                .get(ctx_handle)
                .expect("fine-write must be on the ledger after submit");
            assert!(
                read_tv <= frame1_tv && write_tv <= frame1_tv,
                "ledger epochs must not exceed submission high-water (read={read_tv}, write={write_tv}, submit={frame1_tv})"
            );
        }

        consume_present(&present, &mut sub1);

        // Frame 2 submit gate folds the resolved copy easement into foreign_reads.
        let _sub2 = scheme.submit().expect("submit frame 2");
        let present_read_tv = {
            let sync = registered.sync.lock().unwrap();
            sync.foreign_reads
                .get(ctx_handle)
                .expect("foreign_reads after present fold")
        };
        assert!(
            present_read_tv >= frame1_tv,
            "folded copy-read epoch must be at least the frame-1 submit tv"
        );

        // Loop-carried WAR (copy read frame N, fine write frame N+1) must plan a live wait.
        let mut write_only = ResourceKeyMap::default();
        write_only.insert(
            key,
            net_access_per_resource(&scheme.ir)[&key], // reads+writes in IR; override for next-frame write admission
        );
        write_only.get_mut(&key).unwrap().reads = false;
        let ledger = {
            let sync = registered.sync.lock().unwrap().clone();
            let mut ledger = crate::task_graph::cross_submit::LedgerSnapshot::default();
            ledger.insert(key, crate::task_graph::cross_submit::LedgerEntry { sync });
            ledger
        };
        let plan = compute_cross_submit_sync(&write_only, &ledger, ctx_handle);
        assert!(
            !plan.waits.is_empty(),
            "next-frame private write must serialize against prior present read via ledger wait"
        );
        assert_eq!(plan.waits[0].value, present_read_tv);
    }

    /// The actual submit path (not just `compute_cross_submit_sync` planning) must
    /// emit a live queue-wait on frame 2 against the prior frame's present-read
    /// epoch. Goldy `Scheme::submit` never touches `FrameOrchestrator`;
    /// this test isolates ledger enforcement from orchestrator slot stamping.
    #[cfg(feature = "graphics")]
    #[test]
    fn present_war_ledger_live_wait_on_second_submit_path() {
        use crate::task_graph::cross_submit::ResourceKey;
        use crate::timeline::{Epoch, PromiseState};

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_texture_shader(&device));
        let ctx_handle = ctx.backend_handle();
        let key = ResourceKey::Texture(tex.gpu_handle());

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("fine_write", &pipeline)
            .with_parcel(&tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme.copy_texture_to_present(&tex, &lease);
        let present = scheme.register_present_exchange(&lease);

        let mut sub1 = scheme.submit().expect("submit frame 1");
        let compute_tv = sub1.timeline_value();
        // Source WAR resolves at submit from the known copy timeline — before claim consume.
        let copy_tv = {
            let stamp = scheme
                .submit_state
                .resource_stamps()
                .get(&key)
                .expect("out_image stamp");
            match stamp.pending.lock().unwrap()[0].poll() {
                PromiseState::Resolved(tv) => tv,
                other => panic!("present promise must be resolved after submit, got {other:?}"),
            }
        };
        assert_eq!(
            copy_tv, compute_tv,
            "easement epoch must be the present-partition/copy timeline (copy={copy_tv}, compute={compute_tv})"
        );
        consume_present(&present, &mut sub1);

        let waits_before = device.with_mock_backend(|b| b.recorded_waits.len());
        let _sub2 = scheme.submit().expect("submit frame 2");
        let frame2_waits: Vec<Epoch> = device.with_mock_backend(|b| {
            b.recorded_waits[waits_before..]
                .iter()
                .flat_map(|w| w.iter().copied())
                .collect()
        });
        assert!(
            frame2_waits
                .iter()
                .any(|e| e.context == ctx_handle && e.value >= copy_tv),
            "frame 2 submit must live-wait on prior copy read via ledger (need wait>={copy_tv}, got {frame2_waits:?})"
        );
    }

    #[test]
    fn deposit_allocates_instead_of_waiting_while_in_flight() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let dst = pool
            .acquire_buffer(64, BufferKind::Scattered, Some(4), BufferFlags::empty(), None)
            .unwrap();

        let mut scheme = Scheme::new(&ctx);
        let upload = MemoryExchange::new(scheme.context())
            .bind_deposit_buffer(&mut scheme, dst.whole(), 64)
            .unwrap();

        let payload_a = vec![1u8; 64];
        upload.write(&mut scheme, 0, &payload_a).unwrap();
        assert_eq!(scheme.deposit_parcel_count(&upload), 1);
        let _sub1 = scheme.submit().unwrap();

        // Mock completes submits immediately; force the physical parcel back in-flight.
        scheme.test_mark_deposit_inflight(&upload, 1_000_000);

        let payload_b = vec![2u8; 64];
        upload.write(&mut scheme, 0, &payload_b).unwrap();
        assert_eq!(
            scheme.deposit_parcel_count(&upload),
            2,
            "in-flight staging must grow the pool instead of waiting"
        );
        let _sub2 = scheme.submit().unwrap();
    }

    #[test]
    fn deposit_reuses_settled_parcel() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let dst = pool
            .acquire_buffer(32, BufferKind::Scattered, Some(4), BufferFlags::empty(), None)
            .unwrap();
        let mut scheme = Scheme::new(&ctx);
        let upload = MemoryExchange::new(scheme.context())
            .bind_deposit_buffer(&mut scheme, dst.whole(), 32)
            .unwrap();

        upload.write(&mut scheme, 0, &[7u8; 32]).unwrap();
        let _ = scheme.submit().unwrap();
        upload.write(&mut scheme, 0, &[8u8; 32]).unwrap();
        assert_eq!(
            scheme.deposit_parcel_count(&upload),
            1,
            "settled staging parcel must be reused"
        );
        let _ = scheme.submit().unwrap();
    }

    #[test]
    fn deposit_rejects_submit_without_stage() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let dst = pool
            .acquire_buffer(16, BufferKind::Scattered, Some(4), BufferFlags::empty(), None)
            .unwrap();
        let mut scheme = Scheme::new(&ctx);
        let upload = MemoryExchange::new(scheme.context())
            .bind_deposit_buffer(&mut scheme, dst.whole(), 16)
            .unwrap();
        let err = scheme.submit().expect_err("must require stage before submit");
        let msg = format!("{err}");
        assert!(msg.contains("was not written"), "unexpected error: {msg}");
        let _ = upload; // binding kept for capacity/topology side effects
    }

    #[test]
    fn deposit_warms_slot_variants_per_physical_parcel() {
        let _cb = crate::test_support::CbReuseOverride::force_enabled();
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let dst = pool
            .acquire_buffer(32, BufferKind::Scattered, Some(4), BufferFlags::empty(), None)
            .unwrap();
        let mut scheme = Scheme::new(&ctx);
        let upload = MemoryExchange::new(scheme.context())
            .bind_deposit_buffer(&mut scheme, dst.whole(), 32)
            .unwrap();

        upload.write(&mut scheme, 0, &[1u8; 32]).unwrap();
        let _ = scheme.submit().unwrap();
        assert_eq!(scheme.replay_stats().records, 1);
        assert_eq!(scheme.test_retained_slot_variant_count(), 1);

        upload.write(&mut scheme, 0, &[2u8; 32]).unwrap();
        let _ = scheme.submit().unwrap();
        assert_eq!(scheme.test_retained_slot_variant_count(), 1);
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            1,
            "reusing the same physical parcel must hit the warmed variant"
        );

        scheme.test_mark_deposit_inflight(&upload, 1_000_000);
        upload.write(&mut scheme, 0, &[3u8; 32]).unwrap();
        assert_eq!(scheme.deposit_parcel_count(&upload), 2);
        let _ = scheme.submit().unwrap();
        assert_eq!(
            scheme.test_retained_slot_variant_count(),
            2,
            "a newly allocated physical parcel records a new slot variant"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(scheme.replay_stats().records, 2);
    }

    #[test]
    fn deposit_to_texture_resolves_at_submit() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let tex = pool
            .acquire_texture(
                1,
                1,
                crate::types::TextureFormat::Rgba8Unorm,
                crate::types::TextureKind::Direct,
                crate::types::TextureFlags::COPY_DST,
                None,
            )
            .unwrap();
        let mut scheme = Scheme::new(&ctx);
        let upload = MemoryExchange::new(scheme.context())
            .bind_deposit_texture(&mut scheme, &tex, 0, 0, 1, 1, 4, 0)
            .unwrap();
        upload.write(&mut scheme, 0, &[9, 8, 7, 6]).unwrap();
        let _ = scheme.submit().unwrap();
        assert_eq!(scheme.replay_stats().records, 1);
        assert!(scheme.deposit_parcel_count(&upload) >= 1);
    }

    #[test]
    fn deposit_scheme_drop_returns_parcels() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let dst = pool
            .acquire_buffer(16, BufferKind::Scattered, Some(4), BufferFlags::empty(), None)
            .unwrap();
        let mut scheme = Scheme::new(&ctx);
        let upload = MemoryExchange::new(scheme.context())
            .bind_deposit_buffer(&mut scheme, dst.whole(), 16)
            .unwrap();
        upload.write(&mut scheme, 0, &[4u8; 16]).unwrap();
        let _ = scheme.submit().unwrap();
        assert_eq!(scheme.deposit_parcel_count(&upload), 1);
        drop(scheme);
        // Drop must not panic and must release CB/parcel ownership back to the context.
        let _ = ctx.gpu_progress();
    }

    #[test]
    fn deposit_disable_cb_reuse_skips_replay_ledger() {
        let _cb = crate::test_support::CbReuseOverride::force_disabled();

        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(Arc::clone(&device));
        let dst = pool
            .acquire_buffer(16, BufferKind::Scattered, Some(4), BufferFlags::empty(), None)
            .unwrap();
        let mut scheme = Scheme::new(&ctx);
        let upload = MemoryExchange::new(scheme.context())
            .bind_deposit_buffer(&mut scheme, dst.whole(), 16)
            .unwrap();
        upload.write(&mut scheme, 0, &[1u8; 16]).unwrap();
        let _ = scheme.submit().unwrap();
        upload.write(&mut scheme, 0, &[2u8; 16]).unwrap();
        let _ = scheme.submit().unwrap();

        assert!(
            !scheme.test_has_cb_replay(),
            "CB-reuse disable override must tear down the replay ledger"
        );
        assert_eq!(
            scheme.test_retained_slot_variant_count(),
            0,
            "no slot variants are retained when replay is disabled"
        );
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            0,
            "fresh path must not count retention hits"
        );
    }
}
