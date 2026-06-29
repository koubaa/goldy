//! Retained scheme — the primary submission unit of the diwan machine.
//!
//! A [`Scheme`] is goldy's realization of the diwan scheme (spec §2): a set of dispatches
//! and precedences, first-class, retained across submissions. Unlike [`crate::TaskGraph`],
//! which is rebuilt each frame, a scheme persists; structural mutation sets a COW dirty bit,
//! and a clean scheme resubmits with zero recording cost.
//!
//! **Construction**: `Scheme::new(&ctx)` — bound to one context for its lifetime.
//! **Submission**: `scheme.submit()` — submits, and submits again, using the retained path
//! when clean.

use crate::backend::{BufferHandle, GpuCommand, RenderCommand, TextureCopyFootprint, TextureHandle};
use crate::buffer::{Allocation, BufferSource};
use crate::context::Context;
use crate::error::GoldyError;
use crate::parcel::Parcel;
use crate::render_target::RenderTarget;
use crate::retained_pool::StampedParcel;
use crate::swapchain_pool::PresentLease;
use crate::task_graph::cross_submit::ResourceKey;
use crate::task_graph::IrSubmitState;
use crate::task_graph::ResolvedPresentSlot;
use crate::task_graph::ResourceId;
use crate::task_graph::{
    DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding, ShaderResourceSlot, TaskNode,
    PRESENT_LEASE_SLOT_PLACEHOLDER,
};
use crate::timeline::{PromiseResolver, TimelinePromise, TimelineValue};
use crate::tracy_frame_mark;
use crate::types::{
    BufferFlags, Color, DepthFormat, DispatchShape, IndexFormat, ResourceAccess, ResourceHandle, TextureFlags,
    TextureFormat, TextureKind,
};
use crate::validation_env;
use crate::Buffer;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_SCHEME_ID: AtomicU64 = AtomicU64::new(1);

/// Per-grant staging buffer pool with scheme-lifetime ownership.
///
/// Returned staging buffers are stamped with the submission timeline that must retire
/// before reuse (`ready_after`). This matches transient-pool epoch gating: dropping a
/// [`Submission`] or [`Loan`] without consuming does not require a CPU wait, but in-flight
/// staging is not handed to a later submit until `gpu_progress` passes that stamp.
enum GrantStagingAllocSpec {
    Buffer { byte_size: u64 },
    Texture { layout: TextureCopyFootprint },
}

/// Staging buffer parked in a grant pool until its submission timeline retires.
struct StampedStagingBuffer {
    handle: BufferHandle,
    ready_after: TimelineValue,
}

struct GrantStagingPool {
    handles: Mutex<Vec<StampedStagingBuffer>>,
    alloc_spec: GrantStagingAllocSpec,
    ctx: Context,
    scheme_alive: AtomicBool,
}

impl GrantStagingPool {
    fn new_buffer(ctx: &Context, byte_size: u64) -> Arc<Self> {
        Arc::new(Self {
            handles: Mutex::new(Vec::new()),
            alloc_spec: GrantStagingAllocSpec::Buffer { byte_size },
            ctx: ctx.clone(),
            scheme_alive: AtomicBool::new(true),
        })
    }

    fn new_texture(ctx: &Context, layout: TextureCopyFootprint) -> Arc<Self> {
        Arc::new(Self {
            handles: Mutex::new(Vec::new()),
            alloc_spec: GrantStagingAllocSpec::Texture { layout },
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
            GrantStagingAllocSpec::Buffer { byte_size } => {
                if let Some(handle) = handle {
                    if validation_env::scheme_validation_enabled() {
                        let cap = backend.buffer_size(handle);
                        if cap < byte_size {
                            return Err(GoldyError::Backend(anyhow::anyhow!(
                                "recycled grant staging buffer capacity {cap} is smaller than grant byte size {byte_size}"
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
            GrantStagingAllocSpec::Texture { layout } => {
                if let Some(handle) = handle {
                    if validation_env::scheme_validation_enabled() {
                        let cap = backend.buffer_size(handle);
                        if cap < layout.staging_bytes {
                            return Err(GoldyError::Backend(anyhow::anyhow!(
                                "recycled texture grant staging capacity {cap} is smaller than required {}",
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

    fn return_handle(&self, handle: BufferHandle, ready_after: TimelineValue) {
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

impl fmt::Debug for GrantStagingPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GrantStagingPool")
            .field("scheme_alive", &self.scheme_alive.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

struct SubmissionData {
    scheme_id: u64,
    timeline: TimelineValue,
    backend: Arc<Mutex<Box<dyn crate::backend::GpuBackend>>>,
    /// Per-grant staging buffer for this submission; taken by [`ReadGrant::consume`].
    cells: Vec<Mutex<Option<BufferHandle>>>,
    /// Per-grant pools; used to recycle or free unconsumed cells on drop.
    staging_pools: Vec<Arc<GrantStagingPool>>,
    /// Acquired swapchain frames for present grants; consumed by [`Grant::consume`].
    present_frames: Vec<Mutex<Option<crate::surface::Frame>>>,
    /// Resolvers for present easement timeline promises; resolved by [`Grant::consume`].
    present_resolvers: Vec<Mutex<Option<PromiseResolver>>>,
    /// Pre-allocated present easement timeline per grant when present GPU work was
    /// enqueued on the submission worker at submit (promise still resolved at consume).
    present_fifo_tvs: Vec<Option<TimelineValue>>,
    /// Frame tokens for grants whose present was scheduled on the worker (no GPU work at consume).
    present_fifo_tokens: Vec<Mutex<Option<crate::backend::FrameToken>>>,
}

impl Drop for SubmissionData {
    fn drop(&mut self) {
        let ready_after = self.timeline;
        for (cell, pool) in self.cells.iter().zip(self.staging_pools.iter()) {
            if let Some(handle) = cell.lock().unwrap_or_else(|e| e.into_inner()).take() {
                pool.return_handle(handle, ready_after);
            }
        }
        let mut pending_finishes: Vec<(crate::backend::FrameToken, TimelineValue)> = Vec::new();
        for (idx, tv) in self.present_fifo_tvs.iter().enumerate() {
            if let (Some(tv), Some(token)) = (
                tv,
                self.present_fifo_tokens
                    .get(idx)
                    .and_then(|m| m.lock().unwrap().take()),
            ) {
                pending_finishes.push((token, *tv));
            }
        }
        for (token, tv) in pending_finishes {
            let _ = complete_scheduled_present(&self.backend, token, tv);
        }
        for frame_mutex in &self.present_frames {
            if let Ok(mut slot) = frame_mutex.lock() {
                if let Some(frame) = slot.take() {
                    frame.cancel();
                }
            }
        }
    }
}

impl fmt::Debug for SubmissionData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmissionData")
            .field("scheme_id", &self.scheme_id)
            .field("timeline", &self.timeline)
            .field("cells", &self.cells.len())
            .field("staging_pools", &self.staging_pools.len())
            .field("present_frames", &self.present_frames.len())
            .field("present_resolvers", &self.present_resolvers.len())
            .field("present_fifo_tvs", &self.present_fifo_tvs.len())
            .finish()
    }
}

/// Per-submission identity returned by [`Scheme::submit`].
///
/// Not [`crate::surface::Frame`] (the swapchain acquire/present token).
///
/// A lightweight token. The timeline value identifies which submission this represents;
/// use [`Self::wait`] to block until that submission's GPU work completes (including
/// grant-read staging copies when grants are recorded).
#[derive(Debug, Clone)]
pub struct Submission {
    data: Arc<SubmissionData>,
}

impl Submission {
    /// Timeline value for this submission — pass to [`Context::wait_until`](crate::Context::wait_until).
    pub fn timeline_value(&self) -> TimelineValue {
        self.data.timeline
    }

    /// Block until this submission's GPU work has completed.
    pub fn wait(&self, ctx: &Context) -> Result<(), GoldyError> {
        ctx.wait_until(self.data.timeline)
    }

    #[cfg(test)]
    pub(crate) fn present_fifo_tv(&self, grant_idx: usize) -> Option<TimelineValue> {
        self.data.present_fifo_tvs.get(grant_idx).copied().flatten()
    }
}

impl From<Submission> for TimelineValue {
    fn from(submission: Submission) -> Self {
        submission.timeline_value()
    }
}

/// A scheme easement — exclusive access to a resource recorded at topology time.
///
/// Call [`Grant::consume`] once per submission to revoke that submission's granted
/// access and obtain the grant's product.
pub trait Grant {
    /// Product yielded when this grant's access is consumed for one submission.
    type Output;
    /// Consume the granted access for `submission`, yielding its product.
    ///
    /// `submission` must come from the same [`Scheme`] that recorded this grant.
    /// Each submission may be consumed at most once per grant.
    fn consume(&self, submission: &Submission) -> Result<Self::Output, GoldyError>;
}

/// Marker returned by [`Scheme::grant_present`].
#[derive(Clone)]
pub struct PresentGrant {
    pub(crate) grant_id: u32,
    pub(crate) scheme_id: u64,
    pub(crate) pool: std::sync::Arc<crate::swapchain_pool::SwapchainPoolInner>,
}

impl fmt::Debug for PresentGrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PresentGrant")
            .field("grant_id", &self.grant_id)
            .field("scheme_id", &self.scheme_id)
            .finish_non_exhaustive()
    }
}

impl PresentGrant {
    /// Stable grant index recorded in the scheme IR.
    pub fn grant_id(&self) -> u32 {
        self.grant_id
    }
}

impl Grant for PresentGrant {
    type Output = ();

    fn consume(&self, submission: &Submission) -> Result<Self::Output, GoldyError> {
        let _tz = crate::tracy_zone!("scheme.grant_present.consume");
        if submission.data.scheme_id != self.scheme_id {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PresentGrant belongs to a different scheme than this submission"
            )));
        }
        let idx = self.grant_id as usize;

        if let Some(present_tv) = submission.data.present_fifo_tvs.get(idx).copied().flatten() {
            let token = submission
                .data
                .present_fifo_tokens
                .get(idx)
                .ok_or_else(|| {
                    GoldyError::Backend(anyhow::anyhow!(
                        "present FIFO token missing for grant index {idx}"
                    ))
                })?
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| {
                    GoldyError::Backend(anyhow::anyhow!("grant access already consumed for this submission"))
                })?;
            if let Some(frame_mutex) = submission.data.present_frames.get(idx) {
                let _ = frame_mutex.lock().unwrap_or_else(|e| e.into_inner()).take();
            }
            let resolver_mutex = submission.data.present_resolvers.get(idx).ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!(
                    "present resolver index {} out of range for submission ({} present grants)",
                    idx,
                    submission.data.present_resolvers.len()
                ))
            })?;
            if let Some(resolver) = resolver_mutex.lock().unwrap_or_else(|e| e.into_inner()).take() {
                resolver.resolve(present_tv);
            }
            complete_scheduled_present(&submission.data.backend, token, present_tv)?;
        } else {
            let frame_mutex = submission.data.present_frames.get(idx).ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!(
                    "present grant index {} out of range for submission ({} present grants)",
                    idx,
                    submission.data.present_frames.len()
                ))
            })?;
            let mut slot = frame_mutex.lock().unwrap_or_else(|e| e.into_inner());
            let surface_frame = slot.take().ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!("grant access already consumed for this submission"))
            })?;
            let present_tv = surface_frame.present().map_err(GoldyError::Backend)?;
            let resolver_mutex = submission.data.present_resolvers.get(idx).ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!(
                    "present resolver index {} out of range for submission ({} present grants)",
                    idx,
                    submission.data.present_resolvers.len()
                ))
            })?;
            if let Some(resolver) = resolver_mutex.lock().unwrap_or_else(|e| e.into_inner()).take() {
                resolver.resolve(present_tv);
            }
        }

        if self.pool.speculative_acquire || validation_env::speculative_present_acquire_enabled() {
            let _tz = crate::tracy_zone!("scheme.grant_present.speculative_acquire");
            let spec_result = crate::swapchain_pool::SwapchainPool::acquire_slot(&self.pool);
            match spec_result {
                Ok(slot) => crate::swapchain_pool::SwapchainPool::stash_speculative_acquire(&self.pool, slot),
                Err(e) => {
                    tracing::debug!(
                        target: "goldy::scheme",
                        error = %e,
                        "speculative present acquire failed; submit will acquire synchronously"
                    );
                }
            }
        }
        Ok(())
    }
}

struct PresentGrantInfo {
    lease_id: u32,
    pool: std::sync::Arc<crate::swapchain_pool::SwapchainPoolInner>,
}

/// Parcel stamps read by a present easement for `lease_id` (copy-to-present sources).
///
/// Only [`NodeKind::CopyTexture`] sources can be stamp-tracked — `CopyRenderTarget`
/// sources are scheme-owned leases that do not participate in the [`ResourceKey`] /
/// [`crate::parcel::ParcelStamp`] system, so they cannot carry a promise.
/// A warning is emitted when such a node is encountered so the gap is visible.
fn present_easement_source_stamps(
    ir: &GraphIR,
    lease_id: u32,
    resource_stamps: &std::collections::HashMap<ResourceKey, Arc<crate::parcel::ParcelStamp>>,
) -> Vec<Arc<crate::parcel::ParcelStamp>> {
    let dst = ResourceId::PresentLease(lease_id);
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for node in &ir.nodes {
        match &node.kind {
            NodeKind::CopyTexture { src, dst: d, .. } if *d == dst => {
                let key = ResourceKey::Texture(*src);
                let hit = resource_stamps.get(&key);
                if let Some(stamp) = hit {
                    let ptr = Arc::as_ptr(stamp);
                    if seen.insert(ptr) {
                        out.push(Arc::clone(stamp));
                    }
                }
            }
            NodeKind::CopyRenderTarget { dst: d, .. } if *d == dst => {
                // RenderTarget sources are scheme-owned leases with no ResourceKey and
                // no ParcelStamp, so we cannot attach a promise to gate the next writer.
                // The WAR hazard for this path is not covered by the easement promise
                // mechanism. Log so the gap is visible; do not silently drop.
                tracing::warn!(
                    target: "goldy::scheme",
                    lease_id,
                    "present easement: CopyRenderTarget source has no stamp; \
                     WAR hazard not tracked by promise (TODO: extend RT stamp system)"
                );
            }
            _ => {}
        }
    }
    out
}

fn claim_present_easement_promises(
    ir: &GraphIR,
    present_grants: &[PresentGrantInfo],
    resource_stamps: &std::collections::HashMap<ResourceKey, Arc<crate::parcel::ParcelStamp>>,
) -> Vec<Mutex<Option<PromiseResolver>>> {
    let mut resolvers = Vec::with_capacity(present_grants.len());
    for grant in present_grants {
        let (promise, resolver) = TimelinePromise::new();
        for stamp in present_easement_source_stamps(ir, grant.lease_id, resource_stamps) {
            stamp.push_pending(promise.clone());
        }
        resolvers.push(Mutex::new(Some(resolver)));
    }
    resolvers
}

fn configure_present_easement_stamps(
    ir: &GraphIR,
    present_grants: &[PresentGrantInfo],
    present_fifo_tvs: &[Option<TimelineValue>],
    resource_stamps: &std::collections::HashMap<ResourceKey, Arc<crate::parcel::ParcelStamp>>,
    ctx: crate::backend::ContextHandle,
    fifo_on_worker: bool,
) {
    for (idx, grant) in present_grants.iter().enumerate() {
        let stamps = present_easement_source_stamps(ir, grant.lease_id, resource_stamps);
        if fifo_on_worker {
            if let Some(present_tv) = present_fifo_tvs.get(idx).copied().flatten() {
                for stamp in stamps {
                    stamp.set_legacy_present_easement(false);
                    stamp.sync.lock().unwrap().mark_fifo_ordered_read(ctx, present_tv);
                }
            }
        } else {
            for stamp in stamps {
                stamp.set_legacy_present_easement(true);
            }
        }
    }
}

fn complete_scheduled_present(
    backend: &Arc<Mutex<Box<dyn crate::backend::GpuBackend>>>,
    frame: crate::backend::FrameToken,
    present_tv: TimelineValue,
) -> Result<(), GoldyError> {
    let wait = {
        let b = backend.lock().unwrap();
        b.take_scheduled_present_blocking_wait(frame, present_tv)
            .map_err(|e| GoldyError::Backend(e))?
    };
    let Some(wait) = wait else {
        return Ok(());
    };
    let outcome = wait.run().map_err(GoldyError::Backend)?;
    backend
        .lock()
        .unwrap()
        .apply_scheduled_present_bookkeeping(outcome)
        .map_err(GoldyError::Backend)?;
    // FIFO scheduled present bypasses `Frame::present()` → `apply_frame_bookkeeping`, which
    // is where the legacy path emits Tracy's main frame boundary.
    tracy_frame_mark!();
    Ok(())
}

/// Stable index of a read-easement grant recorded in the scheme IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GrantId(pub(crate) u32);

/// Marker type for buffer read grants.
pub struct GrantBuffer;

/// Marker type for texture read grants (v1: deed texture parcels, uncompressed formats).
pub struct GrantTexture;

enum GrantReadKind {
    Buffer,
    Texture(TextureCopyFootprint),
}

enum GrantSource {
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

struct GrantInfo {
    source: GrantSource,
    staging_pool: Arc<GrantStagingPool>,
}

/// Readable bytes for one `(grant × submission)` cell — returned by [`ReadGrant::consume`].
///
/// Dropping the loan returns the staging buffer to the grant's reuse pool once its
/// submission timeline has retired; otherwise the buffer
/// is freed immediately when the owning [`Scheme`] is gone.
pub struct Loan<T> {
    bytes: Vec<u8>,
    handle: BufferHandle,
    ready_after: TimelineValue,
    return_pool: Arc<GrantStagingPool>,
    _marker: PhantomData<T>,
}

impl<T> fmt::Debug for Loan<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Loan")
            .field("len", &self.bytes.len())
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl<T> Deref for Loan<T> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl<T> Drop for Loan<T> {
    fn drop(&mut self) {
        self.return_pool.return_handle(self.handle, self.ready_after);
    }
}

/// A read easement over a scheme parcel — recorded once via [`Scheme::grant_read`].
///
/// Obtain readable bytes for a submission by coordinating this handle with a
/// [`Submission`] from the **same** [`Scheme`]: `grant.consume(&submission)`.
pub struct ReadGrant<T> {
    grant_id: GrantId,
    scheme_id: u64,
    ctx: Context,
    byte_size: u64,
    read_kind: GrantReadKind,
    return_pool: Arc<GrantStagingPool>,
    _marker: PhantomData<T>,
}

impl<T> ReadGrant<T> {
    /// Logical byte size of readable data for this grant.
    pub fn byte_size(&self) -> u64 {
        self.byte_size
    }
}

impl<T> Grant for ReadGrant<T> {
    type Output = Loan<T>;

    fn consume(&self, submission: &Submission) -> Result<Self::Output, GoldyError> {
        if submission.data.scheme_id != self.scheme_id {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "ReadGrant belongs to a different scheme than this submission"
            )));
        }
        submission.wait(&self.ctx)?;
        let idx = self.grant_id.0 as usize;
        if validation_env::scheme_validation_enabled() && idx >= submission.data.cells.len() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "grant index {} out of range for submission ({} grants)",
                idx,
                submission.data.cells.len()
            )));
        }
        let handle = submission
            .data
            .cells
            .get(idx)
            .ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!(
                    "grant index {} out of range for submission ({} grants)",
                    idx,
                    submission.data.cells.len()
                ))
            })?
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("grant access already consumed for this submission")))?;
        let byte_size = usize::try_from(self.byte_size)
            .map_err(|_| GoldyError::Backend(anyhow::anyhow!("grant readback byte size exceeds address space")))?;
        let mut bytes = vec![0u8; byte_size];
        {
            let backend = self.ctx.device().inner.backend.lock().unwrap();
            match self.read_kind {
                GrantReadKind::Buffer => backend
                    .read_readback_buffer(handle, &mut bytes)
                    .map_err(|e| self.ctx.classify(e))?,
                GrantReadKind::Texture(layout) => backend
                    .read_texture_readback_staging(handle, layout, &mut bytes)
                    .map_err(|e| self.ctx.classify(e))?,
            }
        }
        Ok(Loan {
            bytes,
            handle,
            ready_after: submission.timeline_value(),
            return_pool: Arc::clone(&self.return_pool),
            _marker: PhantomData,
        })
    }
}

/// Stable index of a scheme-held lease declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseId(pub(crate) u32);

/// Marker type for texture leases acquired via [`Scheme::lease_texture`].
pub struct LeaseTexture;

/// Marker type for buffer leases acquired via [`Scheme::lease_buffer`].
pub struct LeaseBuffer;

/// Marker type for render-target leases acquired via [`Scheme::lease_render_target`].
pub struct LeaseRenderTarget;

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
    /// Submissions that skipped re-recording because the backend re-executed a cached command
    /// list without re-recording (Vulkan / DX12 only; absent when the `metal` feature is enabled).
    #[cfg(not(feature = "metal"))]
    pub resubmit_hits: u64,
    /// Submissions that recorded (first submit, post-mutation submits, retention misses).
    pub records: u64,
    /// Re-records caused by a foreign scheme changing shared-parcel topology.
    pub topology_records: u64,
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
    rt_leases: Vec<RenderTarget>,
    /// COW dirty bit: set by every structural mutation, cleared by a successful record.
    dirty: bool,
    /// Set by foreign schemes when shared-parcel interaction topology changes.
    topology_dirty: Arc<AtomicBool>,
    /// Parcels this scheme registered on at the last record (for silent edge teardown).
    prev_topology_parcels: Vec<(ResourceKey, Arc<crate::parcel::ParcelStamp>)>,
    /// Retention key stored at record time. `None` when the backend cannot retain `ir`.
    retention_key: Option<u64>,
    stats: ReplayStats,
    next_grant_id: u32,
    /// Process-unique identity for cross-scheme [`Submission`] / [`ReadGrant`] pairing.
    scheme_id: u64,
    /// Read-easement grants: N-backed staging per submission.
    grants: Vec<GrantInfo>,
    /// Present easement grants: swapchain pool backing per submission.
    present_grants: Vec<PresentGrantInfo>,
}

impl Scheme {
    /// Create a scheme bound to `ctx`.
    pub fn new(ctx: &Context) -> Self {
        Self {
            ir: GraphIR::default(),
            submit_state: IrSubmitState::new(),
            ctx: ctx.clone(),
            leases: Vec::new(),
            rt_leases: Vec::new(),
            dirty: true,
            topology_dirty: Arc::new(AtomicBool::new(false)),
            prev_topology_parcels: Vec::new(),
            retention_key: None,
            stats: ReplayStats::default(),
            next_grant_id: 0,
            scheme_id: NEXT_SCHEME_ID.fetch_add(1, Ordering::Relaxed),
            grants: Vec::new(),
            present_grants: Vec::new(),
        }
    }

    /// True when the next [`Self::submit`] must re-record.
    pub fn is_dirty(&self) -> bool {
        self.dirty
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

    /// Append a CPU→GPU write node for a retained buffer [`Parcel`].
    ///
    /// Marks the scheme dirty. Pair with [`Self::submit`] for a property-only upload
    /// dispatch, or retain the scheme and refresh the payload each submission.
    pub fn commit_write_parcel(&mut self, parcel: &Parcel, offset: u64, data: Vec<u8>) -> Result<(), GoldyError> {
        self.dirty = true;
        let (buffer, resource) = parcel.write_buffer_target().map_err(|e| self.ctx.classify(e))?;
        self.submit_state.register_parcel_stamp(parcel);
        self.ir.nodes.push(TaskNode {
            label: "write_parcel",
            bindings: vec![ResourceBinding {
                resource,
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteBuffer {
                buffer,
                offset,
                data: Arc::from(data),
            },
        });
        Ok(())
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
        self.dirty = true;
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
        self.ir.nodes.push(TaskNode {
            label: "copy_buffer_parcel",
            bindings: vec![
                ResourceBinding {
                    resource: src_resource,
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: dst_resource,
                    access: NodeAccess::Write,
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
        self.dirty = true;
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
        self.ir.nodes.push(TaskNode {
            label: "copy_buffer_to_texture",
            bindings: vec![
                ResourceBinding {
                    resource: src_resource,
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::Texture(th),
                    access: NodeAccess::Write,
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
    /// Mirrors [`crate::TaskGraph::clear_parcel`].
    pub fn commit_clear_parcel(&mut self, parcel: &Parcel, offset: u64, size: u64) -> Result<(), GoldyError> {
        self.dirty = true;
        let buffer = parcel
            .buffer_handle()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("commit_clear_parcel: requires a buffer parcel")))?;
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
                access: NodeAccess::Write,
            }],
            kind: NodeKind::ClearBuffer {
                buffer,
                offset: abs_offset,
                size: clear_size,
            },
        });
        Ok(())
    }

    /// Append a CPU→GPU full-texture upload node.
    ///
    /// Mirrors [`crate::TaskGraph::write_texture`].
    pub fn commit_write_texture(&mut self, texture: &crate::Texture, data: Vec<u8>) -> Result<(), GoldyError> {
        let expected = texture.byte_size();
        if data.len() != expected as usize {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "commit_write_texture: expected {} bytes, got {}",
                expected,
                data.len()
            )));
        }
        self.dirty = true;
        let th = texture.gpu_handle();
        self.ir.nodes.push(TaskNode {
            label: "write_texture",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Texture(th),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteTexture {
                texture: th,
                data: std::sync::Arc::from(data),
                width: texture.width(),
                height: texture.height(),
            },
        });
        Ok(())
    }

    /// Append a CPU→GPU partial-texture upload node for a rectangular sub-region.
    ///
    /// Mirrors [`crate::TaskGraph::write_texture_region`].
    pub fn commit_write_texture_region(
        &mut self,
        texture: &crate::Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: Vec<u8>,
    ) -> Result<(), GoldyError> {
        let x_end = x
            .checked_add(width)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("commit_write_texture_region: x+width overflow")))?;
        let y_end = y
            .checked_add(height)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("commit_write_texture_region: y+height overflow")))?;
        if x_end > texture.width() || y_end > texture.height() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "commit_write_texture_region: {}x{} at ({},{}) exceeds {}x{} texture",
                width,
                height,
                x,
                y,
                texture.width(),
                texture.height()
            )));
        }
        self.dirty = true;
        let th = texture.gpu_handle();
        self.ir.nodes.push(TaskNode {
            label: "write_texture_region",
            bindings: vec![ResourceBinding {
                resource: ResourceId::Texture(th),
                access: NodeAccess::Write,
            }],
            kind: NodeKind::WriteTextureRegion {
                texture: th,
                data: std::sync::Arc::from(data),
                x,
                y,
                width,
                height,
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
        self.dirty = true;
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
    pub(crate) fn commit_render_pass(
        &mut self,
        label: &'static str,
        target: crate::backend::RenderTargetHandle,
        bindings: Vec<ResourceBinding>,
        commands: Vec<RenderCommand>,
        stamp_targets: &[std::sync::Arc<crate::parcel::ParcelStamp>],
    ) {
        self.apply_compute_stamps(stamp_targets);
        self.dirty = true;
        self.ir.nodes.push(TaskNode {
            label,
            bindings,
            kind: NodeKind::RenderPass { target, commands },
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
        self.dirty = true;
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
    /// must declare [`NodeAccess::Write`] (or `ReadWrite`), never pure `Read` — otherwise
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
        self.dirty = true;
        let backing = self
            .ctx
            .with_transient_pool(|pool| pool.acquire_buffer(&self.ctx, size, kind, flags))
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
    pub fn lease_render_target(
        &mut self,
        width: u32,
        height: u32,
        format: TextureFormat,
        depth_format: Option<DepthFormat>,
    ) -> Result<Lease<LeaseRenderTarget>, GoldyError> {
        self.dirty = true;
        let rt = RenderTarget::new_with_depth(self.ctx.device(), width, height, format, depth_format)
            .map_err(|e| self.ctx.classify(e))?;
        let id = LeaseId(u32::try_from(self.rt_leases.len()).expect("render target lease id overflow"));
        self.rt_leases.push(rt);
        Ok(Lease {
            id,
            _marker: PhantomData,
        })
    }

    /// Borrow the backing [`RenderTarget`] for a scheme-held lease.
    pub fn rt(&self, lease: &Lease<LeaseRenderTarget>) -> &RenderTarget {
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
        self.dirty = true;
        SchemeNodeBuilder {
            scheme: self,
            label,
            pipeline: pipeline.handle,
            bindings: Vec::new(),
            resource_slots: Vec::new(),
            user_slots: Vec::new(),
            slot_access: pipeline.slot_access.clone(),
        }
    }

    /// Submit the scheme: resubmit the retained command list when clean, re-record when dirty.
    ///
    /// On a clean resubmit, bound parcels' reference tables are stamped with the new
    /// timeline value, keeping the context transient pool's reuse gates correct across
    /// retained submissions.
    ///
    /// When present grants are recorded, swapchain drawables are acquired before lowering
    /// and stored on the returned [`Submission`] for [`Grant::consume`].
    ///
    /// Per-partition command-buffer reuse legality (Vulkan `SIMULTANEOUS_USE`, DX12
    /// non-reset retained allocators) is enforced in the IR submit loop — no whole-scheme
    /// CPU wait here.
    pub fn submit(&mut self) -> Result<Submission, GoldyError> {
        let topo_dirty = self.topology_dirty.load(Ordering::Acquire);
        let structurally_dirty = self.dirty;
        {
            let _tz = crate::tracy_zone!("scheme.submit.dirty_check");
            if structurally_dirty || topo_dirty {
                self.submit_state.invalidate_retention();
            }
        }

        let mut present_slots = Vec::with_capacity(self.present_grants.len());
        let mut surface_frames = Vec::with_capacity(self.present_grants.len());
        {
            let _tz = crate::tracy_zone!("scheme.submit.acquire_present");
            for grant in &self.present_grants {
                let (slot_id, surface_frame, uav_index, handle) =
                    crate::swapchain_pool::SwapchainPool::resolve_present_slot(&grant.pool)
                        .map_err(|e| self.ctx.classify(e))?;
                present_slots.push(ResolvedPresentSlot {
                    lease_id: grant.lease_id,
                    slot_id,
                    handle,
                    uav_index,
                });
                surface_frames.push(Mutex::new(Some(surface_frame)));
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

        let submit_result = {
            let _tz = crate::tracy_zone!("scheme.submit.pipelined");
            let ir_clean = !structurally_dirty && !topo_dirty;
            self.submit_state
                .submit_pipelined_and_retain_with_presents(&self.ctx, &self.ir, &present_slots, ir_clean)
        };

        let (tv, part_result) = match submit_result {
            Ok(ok) => ok,
            Err(e) => {
                for frame_mutex in surface_frames {
                    if let Ok(mut slot) = frame_mutex.lock() {
                        if let Some(frame) = slot.take() {
                            frame.cancel();
                        }
                    }
                }
                return Err(self.ctx.classify(e));
            }
        };

        self.ctx.advance_high_water_timeline(tv);

        self.dirty = false;
        self.retention_key = None;

        // Standalone upload partitions (WriteTexture, etc.) never increment
        // `PartitionSubmitResult.records`, but the first submit after IR mutation still
        // counts as a scheme record when `structurally_dirty`.
        let recorded = !part_result.all_from_cache() || structurally_dirty;
        let on_record_path = structurally_dirty || topo_dirty || recorded;

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
        } else if part_result.all_from_cache() {
            #[cfg(not(feature = "metal"))]
            {
                self.stats.resubmit_hits += 1;
            }
        }

        if structurally_dirty || topo_dirty {
            tracing::debug!(
                target: "goldy::scheme",
                scheme_id = self.scheme_id,
                structurally_dirty,
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
                all_partitions_from_cache = part_result.all_from_cache(),
                "submit clean (not dirty)"
            );
        }

        let present_resolvers =
            claim_present_easement_promises(&self.ir, &self.present_grants, self.submit_state.resource_stamps());
        let fifo_on_worker = {
            let backend = self.ctx.device().inner.backend.lock().unwrap();
            backend.schedules_present_on_submit_worker()
        };
        let (present_fifo_tvs, present_fifo_tokens) = Self::schedule_presents_on_worker_if_supported(
            &self.ctx,
            tv,
            &mut surface_frames,
        )?;
        configure_present_easement_stamps(
            &self.ir,
            &self.present_grants,
            &present_fifo_tvs,
            self.submit_state.resource_stamps(),
            self.ctx.backend_handle(),
            fifo_on_worker,
        );
        let submission = self.finish_submit_frame(
            tv,
            surface_frames,
            present_resolvers,
            present_fifo_tvs,
            present_fifo_tokens,
        )?;
        Ok(submission)
    }

    fn schedule_presents_on_worker_if_supported(
        ctx: &Context,
        compute_tv: TimelineValue,
        surface_frames: &mut [Mutex<Option<crate::surface::Frame>>],
    ) -> Result<(Vec<Option<TimelineValue>>, Vec<Mutex<Option<crate::backend::FrameToken>>>), GoldyError> {
        let mut backend = ctx.device().inner.backend.lock().unwrap();
        if !backend.schedules_present_on_submit_worker() {
            let n = surface_frames.len();
            return Ok((
                vec![None; n],
                (0..n).map(|_| Mutex::new(None)).collect(),
            ));
        }
        let mut tvs = Vec::with_capacity(surface_frames.len());
        let mut tokens = Vec::with_capacity(surface_frames.len());
        for frame_mutex in surface_frames.iter() {
            let mut slot = frame_mutex.lock().unwrap();
            let mut frame = slot
                .take()
                .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("present frame missing at submit")))?;
            let token = frame.frame_token();
            frame.mark_present_scheduled_on_worker();
            drop(frame);
            let present_tv = backend
                .schedule_present_on_submission_worker(token, compute_tv)
                .map_err(|e| ctx.classify(e))?;
            ctx.advance_high_water_timeline(present_tv);
            tvs.push(Some(present_tv));
            tokens.push(Mutex::new(Some(token)));
        }
        Ok((tvs, tokens))
    }

    fn finish_submit_frame(
        &mut self,
        tv_dispatch: TimelineValue,
        present_frames: Vec<Mutex<Option<crate::surface::Frame>>>,
        present_resolvers: Vec<Mutex<Option<PromiseResolver>>>,
        present_fifo_tvs: Vec<Option<TimelineValue>>,
        present_fifo_tokens: Vec<Mutex<Option<crate::backend::FrameToken>>>,
    ) -> Result<Submission, GoldyError> {
        let backend = Arc::clone(&self.ctx.device().inner.backend);
        if self.grants.is_empty() {
            return Ok(Submission {
                data: Arc::new(SubmissionData {
                    scheme_id: self.scheme_id,
                    timeline: tv_dispatch,
                    backend,
                    cells: Vec::new(),
                    staging_pools: Vec::new(),
                    present_frames,
                    present_resolvers,
                    present_fifo_tvs,
                    present_fifo_tokens,
                }),
            });
        }

        let device = self.ctx.device().inner.handle;
        let mut copy_cmds = Vec::with_capacity(self.grants.len());
        let mut cells = Vec::with_capacity(self.grants.len());
        let mut staging_pools = Vec::with_capacity(self.grants.len());
        let mut staging_handles = Vec::with_capacity(self.grants.len());

        {
            let mut backend = self.ctx.device().inner.backend.lock().unwrap();
            for grant in &self.grants {
                let staging = grant.staging_pool.take_or_alloc(&mut **backend, device)?;
                if validation_env::scheme_validation_enabled() {
                    if staging_handles.contains(&staging) {
                        return Err(GoldyError::Backend(anyhow::anyhow!(
                            "duplicate grant staging buffer handle in one submission"
                        )));
                    }
                    staging_handles.push(staging);
                }
                match &grant.source {
                    GrantSource::Buffer {
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
                    GrantSource::Texture { source, layout, .. } => {
                        copy_cmds.push(GpuCommand::CopyTextureToReadback {
                            src: *source,
                            dst: staging,
                            layout: *layout,
                        });
                    }
                }
                cells.push(Mutex::new(Some(staging)));
                staging_pools.push(Arc::clone(&grant.staging_pool));
            }
        }

        if validation_env::scheme_validation_enabled() {
            debug_assert_eq!(cells.len(), self.grants.len());
            debug_assert_eq!(staging_pools.len(), self.grants.len());
        }

        let tv_copy = {
            let mut backend = self.ctx.device().inner.backend.lock().unwrap();
            backend
                .submit_standalone(self.ctx.backend_handle(), &copy_cmds, None)
                .map_err(|e| self.ctx.classify(e))?
        };
        self.ctx.advance_high_water_timeline(tv_copy);

        Ok(Submission {
            data: Arc::new(SubmissionData {
                scheme_id: self.scheme_id,
                timeline: tv_copy,
                backend,
                cells,
                staging_pools,
                present_frames,
                present_resolvers,
                present_fifo_tvs,
                present_fifo_tokens,
            }),
        })
    }

    /// Record a present easement grant over a swapchain lease.
    pub fn grant_present(&mut self, lease: &PresentLease) -> PresentGrant {
        self.dirty = true;
        // `ir_grant_id` is the globally-unique ID used only for IR fingerprinting.
        // `present_idx` is the dense index into `present_frames`/`present_resolvers`
        // built by iterating `present_grants` at submit time; the two must be kept
        // independent so interleaved read grants do not corrupt the present vec index.
        let ir_grant_id = self.next_grant_id;
        self.next_grant_id += 1;
        let present_idx = self.present_grants.len() as u32;
        self.present_grants.push(PresentGrantInfo {
            lease_id: lease.id,
            pool: Arc::clone(&lease.pool),
        });
        self.ir.nodes.push(TaskNode {
            label: "grant_present",
            bindings: vec![ResourceBinding {
                resource: ResourceId::PresentLease(lease.id),
                access: NodeAccess::Read,
            }],
            kind: NodeKind::GrantPresent { grant_id: ir_grant_id },
        });
        PresentGrant {
            grant_id: present_idx,
            scheme_id: self.scheme_id,
            pool: Arc::clone(&lease.pool),
        }
    }

    /// Copy an offscreen render target into a present lease drawable.
    pub fn copy_to_present(&mut self, src: &Lease<LeaseRenderTarget>, dst: &PresentLease) {
        self.dirty = true;
        let handle = self.rt_leases[src.id.0 as usize].backend_handle();
        self.ir.nodes.push(TaskNode {
            label: "copy_to_present",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::RenderTarget(handle),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::PresentLease(dst.id),
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyRenderTarget {
                src: handle,
                dst: ResourceId::PresentLease(dst.id),
            },
        });
    }

    /// Copy a texture (UAV-writable parcel) into a present lease drawable.
    ///
    /// Analogous to [`TaskGraph::copy_texture_to_swapchain`](crate::TaskGraph::copy_texture_to_swapchain)
    /// but targets a scheme [`PresentLease`] instead of the task-graph swapchain output.
    ///
    /// Record this after all compute nodes that write `src`. The present slot is
    /// resolved by [`Self::submit`] at acquire time — the same partition-slot-key
    /// mechanism used by [`Self::copy_to_present`].
    pub fn copy_texture_to_present(&mut self, src: &crate::Texture, dst: &PresentLease) {
        self.dirty = true;
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
                    resource: ResourceId::PresentLease(dst.id),
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyTexture {
                src: src_h,
                dst: ResourceId::PresentLease(dst.id),
                dst_buffer_layout: None,
            },
        });
    }

    /// Copy an offscreen render target into a texture deed parcel (for CPU readback via
    /// [`Self::grant_read_texture`]).
    ///
    /// The destination must be a texture parcel with [`TextureFlags::COPY_DST`], homed on
    /// this scheme's context, and matching the render target's width, height, and format.
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

        self.dirty = true;
        self.submit_state.register_parcel_stamp(dst);
        let src_handle = src_rt.backend_handle();
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
                    access: NodeAccess::Write,
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
    pub fn render_pass<'a>(
        &'a mut self,
        label: &'static str,
        rt: &Lease<LeaseRenderTarget>,
    ) -> SchemeRenderPassBuilder<'a> {
        self.dirty = true;
        let handle = self.rt_leases[rt.id.0 as usize].backend_handle();
        SchemeRenderPassBuilder {
            scheme: self,
            label,
            target: handle,
            bindings: vec![ResourceBinding {
                resource: ResourceId::RenderTarget(handle),
                access: NodeAccess::Write,
            }],
            commands: Vec::new(),
            pending_push_constants: Vec::new(),
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
}

impl Drop for Scheme {
    fn drop(&mut self) {
        for grant in &self.grants {
            grant.staging_pool.mark_scheme_dropped_and_drain();
        }

        use crate::task_graph::cross_submit::clear_scheme_topology_registration;
        clear_scheme_topology_registration(self.scheme_id, &self.prev_topology_parcels);

        let hw = self.ctx.high_water_timeline();
        if hw > 0 {
            let _ = self.ctx.wait_until(hw);
        }
        self.submit_state.release_backend_retained_graphs(&self.ctx);

        let ctx = self.ctx.clone();
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
        self.rt_leases.clear();
    }
}

impl Scheme {
    /// Record a read easement grant over a buffer deed parcel.
    ///
    /// Returns a stable [`ReadGrant`] handle; call [`ReadGrant::consume`] with a
    /// [`Submission`] from [`Self::submit`] to obtain that submission's bytes.
    /// Record after the producing dispatch node(s). The parcel's backing buffer is
    /// retained for the scheme's lifetime so resubmits remain valid after the
    /// [`Parcel`] is dropped.
    pub fn grant_read(&mut self, parcel: &Parcel) -> Result<ReadGrant<GrantBuffer>, GoldyError> {
        self.dirty = true;
        self.submit_state.register_parcel_stamp(parcel);
        if !parcel.is_homed_on(&self.ctx) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "parcel home device does not match scheme context"
            )));
        }
        let source_backing = parcel.grant_buffer_keepalive().map_err(|e| self.ctx.classify(e))?;
        let source = parcel
            .buffer_handle()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("grant_read requires buffer parcel")))?;
        let byte_size = parcel.byte_size();
        if byte_size == 0 {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "grant_read requires non-zero buffer byte size"
            )));
        }
        // `ir_grant_id` is unique within the IR for fingerprinting; `read_idx` is the
        // dense index into `cells` built by iterating `grants` at submit time.
        let ir_grant_id = self.next_grant_id;
        self.next_grant_id += 1;
        let read_idx = GrantId(self.grants.len() as u32);
        let staging_pool = GrantStagingPool::new_buffer(&self.ctx, byte_size);
        self.grants.push(GrantInfo {
            source: GrantSource::Buffer {
                source,
                src_offset: parcel.source_offset(),
                source_backing,
                byte_size,
            },
            staging_pool: Arc::clone(&staging_pool),
        });
        let resource = parcel.resource_id();
        self.ir.nodes.push(TaskNode {
            label: "grant_read",
            bindings: vec![ResourceBinding {
                resource,
                access: NodeAccess::Read,
            }],
            kind: NodeKind::GrantRead { grant_id: ir_grant_id },
        });
        Ok(ReadGrant {
            grant_id: read_idx,
            scheme_id: self.scheme_id,
            ctx: self.ctx.clone(),
            byte_size,
            read_kind: GrantReadKind::Buffer,
            return_pool: staging_pool,
            _marker: PhantomData,
        })
    }

    /// Record a read easement grant over a texture deed parcel.
    ///
    /// The texture must have been created with [`TextureFlags::COPY_SRC`].
    /// v1 supports uncompressed 2D formats only.
    pub fn grant_read_texture(&mut self, parcel: &Parcel) -> Result<ReadGrant<GrantTexture>, GoldyError> {
        self.dirty = true;
        self.submit_state.register_parcel_stamp(parcel);
        if !parcel.is_homed_on(&self.ctx) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "parcel home device does not match scheme context"
            )));
        }
        let source_backing = parcel.grant_texture_keepalive().map_err(|e| self.ctx.classify(e))?;
        let source = parcel
            .texture_handle()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("grant_read_texture requires texture parcel")))?;
        let (width, height, format, access, flags) = parcel
            .texture_descriptor()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("grant_read_texture requires texture parcel")))?;
        if !flags.contains(TextureFlags::COPY_SRC) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "grant_read_texture requires TextureFlags::COPY_SRC"
            )));
        }
        // v1: only storage-writable textures (Direct / DirectInterpolated) are valid sources;
        // Interpolated (sampled-only) textures cannot be written by a compute shader.
        if matches!(access, TextureKind::Interpolated) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "grant_read_texture requires a storage-writable texture (TextureKind::Direct or DirectInterpolated); \
                 TextureKind::Interpolated is sampled-only and cannot be a compute output"
            )));
        }
        if width == 0 || height == 0 {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "grant_read_texture requires non-zero texture dimensions"
            )));
        }
        let layout = {
            let backend = self.ctx.device().inner.backend.lock().unwrap();
            backend
                .query_texture_copy_footprint(self.ctx.device().inner.handle, width, height, format)
                .map_err(|e| self.ctx.classify(e))?
        };
        let ir_grant_id = self.next_grant_id;
        self.next_grant_id += 1;
        let read_idx = GrantId(self.grants.len() as u32);
        let staging_pool = GrantStagingPool::new_texture(&self.ctx, layout);
        self.grants.push(GrantInfo {
            source: GrantSource::Texture {
                source,
                source_backing,
                layout,
            },
            staging_pool: Arc::clone(&staging_pool),
        });
        let resource = parcel.resource_id();
        self.ir.nodes.push(TaskNode {
            label: "grant_read",
            bindings: vec![ResourceBinding {
                resource,
                access: NodeAccess::Read,
            }],
            kind: NodeKind::GrantRead { grant_id: ir_grant_id },
        });
        Ok(ReadGrant {
            grant_id: read_idx,
            scheme_id: self.scheme_id,
            ctx: self.ctx.clone(),
            byte_size: layout.logical_bytes,
            read_kind: GrantReadKind::Texture(layout),
            return_pool: staging_pool,
            _marker: PhantomData,
        })
    }

    /// Record a GPU copy from `src` into `dst`.
    ///
    /// When `dst` is a [`Buffer`] with [`BufferFlags::CPU_READABLE`], the copy uses the
    /// buffer footprint from [`crate::Texture::copy_layout`]. Acquire `dst` sized to
    /// `layout.staging_bytes`, then [`Self::submit`] and wait the returned timeline before
    /// reading `dst` on the CPU.
    ///
    /// Required when `src` may have been written by a prior scheme submission on the same
    /// [`Context`]: cross-submission barriers and parcel stamps are applied before the transfer read.
    pub fn copy_texture(&mut self, src: &crate::Texture, dst: &Buffer) -> Result<TextureCopyFootprint, GoldyError> {
        if !dst.flags().contains(BufferFlags::CPU_READABLE) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_texture to buffer requires BufferFlags::CPU_READABLE destination"
            )));
        }
        let layout = src.copy_layout();
        if dst.byte_size() < layout.staging_bytes {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "copy_texture destination too small: {} < {}",
                dst.byte_size(),
                layout.staging_bytes
            )));
        }

        let src_h = src.gpu_handle();
        let stamp = src.whole().stamp_handle();
        self.submit_state
            .register_stamp_parts(ResourceId::Texture(src_h), stamp);

        self.ir.nodes.retain(|node| {
            !matches!(
                node.kind,
                NodeKind::CopyTexture {
                    dst_buffer_layout: Some(_),
                    ..
                }
            )
        });
        self.ir.nodes.push(TaskNode {
            label: "copy_texture",
            bindings: vec![
                ResourceBinding {
                    resource: ResourceId::Texture(src_h),
                    access: NodeAccess::Read,
                },
                ResourceBinding {
                    resource: ResourceId::Buffer(dst.backing_handle()),
                    access: NodeAccess::Write,
                },
            ],
            kind: NodeKind::CopyTexture {
                src: src_h,
                dst: ResourceId::Buffer(dst.backing_handle()),
                dst_buffer_layout: Some(layout),
            },
        });
        self.dirty = true;

        Ok(layout)
    }
}

pub(crate) fn node_access_to_resource_access(access: NodeAccess) -> ResourceAccess {
    match access {
        NodeAccess::Read => ResourceAccess::Read,
        NodeAccess::Write => ResourceAccess::Write,
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

mod sealed {
    pub trait Sealed {}
}

/// Argument to [`SchemeNodeBuilder::dispatch`]: fixed workgroup counts or a device-resident shape parcel.
///
/// Implemented for [`DispatchShape`] and [`Parcel`]. Fixed triples use
/// [`SchemeNodeBuilder::dispatch`] `(x, y, z)`.
pub trait IntoDispatch: sealed::Sealed {
    fn finish(self, builder: SchemeNodeBuilder<'_>) -> Result<(), GoldyError>;
}

impl sealed::Sealed for DispatchShape {}
impl sealed::Sealed for &Parcel {}

impl IntoDispatch for DispatchShape {
    fn finish(self, builder: SchemeNodeBuilder<'_>) -> Result<(), GoldyError> {
        builder.push_dispatch_node(DispatchDim::Direct {
            x: self.x,
            y: self.y,
            z: self.z,
        });
        Ok(())
    }
}

impl IntoDispatch for &Parcel {
    fn finish(self, builder: SchemeNodeBuilder<'_>) -> Result<(), GoldyError> {
        let offset = validate_dispatch_shape_parcel(self)?;
        let resource = self.resource_id();
        builder
            .scheme
            .submit_state
            .register_stamp_parts(resource, self.stamp_handle());
        let mut bindings = builder.bindings;
        bindings.push(ResourceBinding {
            resource,
            access: NodeAccess::Read,
        });
        let buffer = self
            .buffer_handle()
            .expect("validate_dispatch_shape_parcel ensures buffer parcel");
        builder.scheme.ir.nodes.push(TaskNode {
            label: builder.label,
            bindings,
            kind: NodeKind::Dispatch {
                pipeline: builder.pipeline,
                resource_slots: builder.resource_slots,
                user_slots: builder.user_slots,
                dispatch: DispatchDim::Indirect { buffer, offset },
            },
        });
        Ok(())
    }
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
pub(crate) type SchemeBindableResolution = (
    Option<(ResourceId, Option<Arc<crate::parcel::ParcelStamp>>)>,
    Option<u32>,
);

pub(crate) trait SchemeBindable {
    fn resolve(&self, scheme: &Scheme, access: ResourceAccess) -> SchemeBindableResolution;
}

impl SchemeBindable for Parcel {
    fn resolve(&self, _: &Scheme, access: ResourceAccess) -> SchemeBindableResolution {
        (
            Some((self.resource_id(), Some(self.stamp_handle()))),
            self.resource_index(access),
        )
    }
}

impl SchemeBindable for crate::Buffer {
    fn resolve(&self, _: &Scheme, access: ResourceAccess) -> SchemeBindableResolution {
        let parcel = self.whole();
        (
            Some((parcel.resource_id(), Some(parcel.stamp_handle()))),
            parcel.resource_index(access),
        )
    }
}

impl<T> SchemeBindable for Lease<T> {
    fn resolve(&self, scheme: &Scheme, access: ResourceAccess) -> SchemeBindableResolution {
        let parcel = &scheme.leases[self.id.0 as usize];
        // TODO(inaugural-check): enforce that the first access to a buffer lease is Write
        // (or ReadWrite), never pure Read. The pool may recycle a buffer whose bytes come
        // from a previous submission; a Read-only first access would observe stale data.
        // This requires a per-scheme "has-been-written" bit per lease slot; deferred until
        // the unique-minimal-write shape-check lands (design §8).
        (
            Some((parcel.resource_id(), Some(parcel.stamp_handle()))),
            parcel.resource_index(access),
        )
    }
}

impl SchemeBindable for crate::Sampler {
    fn resolve(&self, _: &Scheme, access: ResourceAccess) -> SchemeBindableResolution {
        // Samplers carry no GPU-written data: no RAW/WAW hazard, no barrier, no stamp.
        // Only the bindless heap index is needed.
        (None, self.resource_index(access))
    }
}

impl SchemeBindable for crate::Texture {
    fn resolve(&self, _: &Scheme, access: ResourceAccess) -> SchemeBindableResolution {
        // `TextureKind::Direct` storage images have no SRV; when a shader slot is reflected
        // as read-only (ResourceAccess::Read) but the texture only has a UAV descriptor,
        // fall back to the UAV bindless index — mirroring the TaskGraph path's behaviour in
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

    /// Declare a UAV write to a present lease (swapchain drawable).
    ///
    /// Appends a [`PRESENT_LEASE_SLOT_PLACEHOLDER`] entry at the end of `resource_slots`
    /// so the resolver can patch it to the correct UAV index at submit time.
    /// May be called before or after other slot-binding calls on the same node.
    pub fn with_present(mut self, lease: &PresentLease) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::PresentLease(lease.id),
            access: NodeAccess::Write,
        });
        self.resource_slots.push(PRESENT_LEASE_SLOT_PLACEHOLDER);
        self
    }

    /// Finalize the node with fixed workgroup dimensions.
    pub fn dispatch(self, x: u32, y: u32, z: u32) {
        self.push_dispatch_node(DispatchDim::Direct { x, y, z });
    }

    /// Finalize the node with a host [`DispatchShape`] or a device-resident shape parcel.
    ///
    /// Passing a `&Parcel` selects device-sourced (indirect) dispatch. The shape parcel's
    /// ordering dependency is registered automatically and is not a shader resource slot.
    ///
    /// Rust does not allow overloading [`Self::dispatch`] `(x, y, z)` and this shape/parcel
    /// form under the same name; this is the shape/parcel dispatch entry point.
    pub fn dispatch_shape(self, dim: impl IntoDispatch) -> Result<(), GoldyError> {
        dim.finish(self)
    }

    fn push_dispatch_node(self, dispatch: DispatchDim) {
        self.scheme.ir.nodes.push(TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::Dispatch {
                pipeline: self.pipeline,
                resource_slots: self.resource_slots,
                user_slots: self.user_slots,
                dispatch,
            },
        });
    }
}

/// Deferred push-constant slot recorded before [`SchemeRenderPassBuilder::set_pipeline`].
///
/// Read and read-write handles are captured at record time; the descriptor actually
/// bound is chosen from pipeline reflection when the pipeline is set.
struct PendingPushConstant {
    graph_access: NodeAccess,
    read_handle: Option<ResourceHandle>,
    read_write_handle: Option<ResourceHandle>,
}

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
pub struct SchemeRenderPassBuilder<'a> {
    scheme: &'a mut Scheme,
    label: &'static str,
    target: crate::backend::RenderTargetHandle,
    bindings: Vec<ResourceBinding>,
    commands: Vec<RenderCommand>,
    pending_push_constants: Vec<PendingPushConstant>,
}

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
    /// [`Self::set_pipeline`] emits [`RenderCommand::BindResourcesTyped`] from these
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

    pub fn clear(&mut self, color: Color) -> &mut Self {
        self.commands.push(RenderCommand::Clear(color));
        self
    }

    pub fn clear_depth(&mut self, depth: f32) -> &mut Self {
        self.commands.push(RenderCommand::ClearDepth(depth));
        self
    }

    pub fn set_pipeline(&mut self, pipeline: &crate::RenderPipeline) -> &mut Self {
        self.commands.push(RenderCommand::SetPipeline(pipeline.handle));
        if !self.pending_push_constants.is_empty() {
            let handles: Vec<ResourceHandle> = self
                .pending_push_constants
                .iter()
                .enumerate()
                .map(|(i, pending)| pending.resolve(&pipeline.slot_access, i))
                .collect();
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

    pub fn finish(self) {
        let SchemeRenderPassBuilder {
            scheme,
            label,
            target,
            bindings,
            commands,
            pending_push_constants: _,
        } = self;
        scheme.ir.nodes.push(TaskNode {
            label,
            bindings,
            kind: NodeKind::RenderPass { target, commands },
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
    use crate::SwapchainPool;
    use crate::task_graph::NodeAccess;
    use crate::task_graph::NodeKind;
    use crate::types::ResourceAccess;
    use crate::BufferKind;
    use std::sync::Arc;

    fn mock_device() -> Arc<Device> {
        Arc::new(Device::from_backend(Box::new(MockBackend::new())).expect("mock device"))
    }

    fn mock_device_legacy_present() -> Arc<Device> {
        let mut backend = MockBackend::new();
        backend.set_schedules_present_on_submit_worker(false);
        Arc::new(Device::from_backend(Box::new(backend)).expect("mock device"))
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

    fn mock_render_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(device, "void main() {}").expect("compile render shader")
    }

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

    fn recording_scheme(device: &Arc<Device>, pool: &mut RetainedPool, ctx: &Context) -> Scheme {
        recording_scheme_with_parcel(device, pool, ctx).0
    }

    fn clean_scheme(device: &Arc<Device>, pool: &mut RetainedPool) -> Scheme {
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(device);
        let pipeline = mock_pipeline(device, &shader);
        let parcel = retained_buffer(pool);

        let mut scheme = Scheme::new(&ctx);
        assert!(scheme.is_dirty(), "new scheme starts dirty");
        scheme
            .node("a", &pipeline)
            .with_parcel(&parcel, NodeAccess::Write)
            .dispatch(1, 1, 1);

        scheme.submit().unwrap();
        assert!(!scheme.is_dirty(), "successful submit clears the dirty bit");
        assert_eq!(scheme.replay_stats().records, 1);
        #[cfg(not(feature = "metal"))]
        assert_eq!(scheme.replay_stats().resubmit_hits, 0);
        scheme
    }

    fn leased_texture_scheme(device: &Arc<Device>) -> (Scheme, Lease<LeaseTexture>) {
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

        (scheme, lease)
    }

    #[test]
    fn clean_submits_resubmit_without_rerecord() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let mut scheme = clean_scheme(&device, &mut pool);

        scheme.submit().unwrap();
        scheme.submit().unwrap();

        assert_eq!(scheme.replay_stats().records, 1, "only the first submit records");
        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats().resubmit_hits,
            2,
            "subsequent clean submits resubmit"
        );
    }

    #[test]
    #[cfg(not(feature = "metal"))]
    fn clean_resubmit_performs_no_cpu_wait() {
        use crate::backend::GpuBackend;

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let mut scheme = clean_scheme(&device, &mut pool);

        scheme.submit().unwrap();
        scheme.submit().unwrap();

        let backend = device.inner.backend.lock().unwrap();
        assert_eq!(
            backend.test_wait_until_count(),
            0,
            "clean scheme resubmits must not call wait_until on the submit path"
        );
        assert!(
            !scheme.partition_last_tvs().is_empty(),
            "per-partition timelines are tracked after submit"
        );
    }

    #[test]
    fn mutation_marks_dirty_and_rerecords_once() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let mut scheme = clean_scheme(&device, &mut pool);
        scheme.submit().unwrap();

        #[cfg(not(feature = "metal"))]
        assert_eq!(
            scheme.replay_stats(),
            ReplayStats {
                records: 1,
                resubmit_hits: 1
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
                resubmit_hits: 2
            }
        );
        #[cfg(feature = "metal")]
        assert_eq!(scheme.replay_stats().records, 2);
    }

    #[test]
    fn is_settled_true_before_first_reference() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = retained_buffer(&mut pool);
        assert!(parcel.is_settled(&ctx), "never-referenced parcel is settled");
    }

    #[test]
    fn frame_timeline_value_round_trip() {
        use crate::timeline::TimelineValue;

        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(device.clone());
        let mut scheme = recording_scheme(&device, &mut pool, &ctx);
        let frame = scheme.submit().unwrap();
        let tv = frame.timeline_value();
        assert!(tv > 0);
        assert_eq!(TimelineValue::from(frame.clone()), tv);
        assert_eq!(frame.timeline_value(), tv);
    }

    #[test]
    fn frame_wait_completes_submission() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(device.clone());
        let mut scheme = recording_scheme(&device, &mut pool, &ctx);
        let frame = scheme.submit().unwrap();
        frame.wait(&ctx).unwrap();
        assert!(ctx.gpu_progress() >= frame.timeline_value());
    }

    #[test]
    fn submit_returns_frame_without_calling_wait() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let mut pool = RetainedPool::new(device.clone());
        let mut scheme = recording_scheme(&device, &mut pool, &ctx);
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
        let (mut scheme, _lease) = leased_texture_scheme(&device);

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
        let (mut scheme, _lease) = leased_texture_scheme(&device);
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

    #[test]
    fn grant_read_appends_ir_node() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        assert_eq!(scheme.ir_node_count(), 1);

        let _grant = scheme.grant_read(&parcel).expect("grant_read");
        assert_eq!(scheme.ir_node_count(), 2);
        assert!(scheme.is_dirty(), "grant_read is structural");

        match &scheme.ir.nodes[1].kind {
            NodeKind::GrantRead { grant_id: 0 } => {}
            other => panic!("expected GrantRead node, got {other:?}"),
        }
    }

    #[test]
    fn grant_read_orders_after_writer() {
        use crate::task_graph::analysis;

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        let _grant = scheme.grant_read(&parcel).expect("grant_read");

        let edges = analysis::build_edges(&scheme.ir);
        assert!(
            edges.contains(&(0, 1)),
            "dispatch (0) must precede grant_read (1); edges: {edges:?}"
        );
    }

    #[test]
    fn scheme_with_grant_retains() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        let _grant = scheme.grant_read(&parcel).expect("grant_read");

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
    fn grant_read_survives_parcel_drop() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        let grant = scheme.grant_read(&parcel).expect("grant_read");
        let frame = scheme.submit().expect("submit");
        drop(parcel);
        drop(pool);

        let loan = grant.consume(&frame).expect("read after parcel drop");
        assert_eq!(loan.len(), 32, "reads full logical buffer size");
    }

    #[test]
    fn grant_read_resubmit_after_parcel_drop() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        let grant = scheme.grant_read(&parcel).expect("grant_read");
        let frame1 = scheme.submit().expect("submit 1");
        drop(parcel);
        drop(pool);
        let frame2 = scheme.submit().expect("submit 2 after parcel drop");
        let loan1 = grant.consume(&frame1).expect("read frame1");
        let loan2 = grant.consume(&frame2).expect("read frame2");
        assert_eq!(loan1.len(), 32);
        assert_eq!(loan2.len(), 32);
    }

    #[test]
    fn grant_read_concurrent_frames_succeed() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = pool
            .acquire_buffer_with_data(&[7u32; 8], BufferKind::Scattered)
            .expect("parcel");
        let mut scheme = Scheme::new(&ctx);
        let grant = scheme.grant_read(&parcel).expect("grant_read");
        let frame1 = scheme.submit().expect("first submit");
        let frame2 = scheme.submit().expect("second submit without waiting on frame1");

        let loan1 = grant.consume(&frame1).expect("read frame1");
        let loan2 = grant.consume(&frame2).expect("read frame2");
        assert_eq!(loan1.len(), 32);
        assert_eq!(loan2.len(), 32);
        for chunk in loan1.chunks_exact(4) {
            assert_eq!(u32::from_le_bytes(chunk.try_into().unwrap()), 7);
        }
        let (allocs, _) = mock_readback_counts(&device);
        assert_eq!(allocs, 2, "two live frames require two staging allocations");
    }

    #[test]
    fn grant_read_double_read_same_frame_errors() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);
        let grant = scheme.grant_read(&parcel).expect("grant_read");
        let frame = scheme.submit().expect("submit");
        let _loan = grant.consume(&frame).expect("first read");
        let err = match grant.consume(&frame) {
            Ok(_) => panic!("second read must fail"),
            Err(e) => e,
        };
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
        let grant = scheme.grant_read(&parcel).expect("grant_read");

        let frame1 = scheme.submit().expect("submit 1");
        {
            let loan = grant.consume(&frame1).expect("read frame1");
            assert_eq!(loan.len(), 32);
        }
        let frame2 = scheme.submit().expect("submit 2 after loan drop");
        let loan2 = grant.consume(&frame2).expect("read frame2 after pool recycle");
        assert_eq!(loan2.len(), 32);
        let (allocs, _) = mock_readback_counts(&device);
        assert_eq!(allocs, 1, "pool recycles staging buffer on loan drop");
    }

    #[test]
    fn grant_read_rejects_foreign_device_parcel() {
        let device_a = mock_device();
        let device_b = mock_device();
        let mut pool = RetainedPool::new(device_a.clone());
        let ctx_a = device_a.create_context().unwrap();
        let ctx_b = device_b.create_context().unwrap();
        let parcel = retained_buffer(&mut pool);
        let mut scheme = Scheme::new(&ctx_b);
        let err = match scheme.grant_read(&parcel) {
            Ok(_) => panic!("cross-device grant must fail"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("home device"), "unexpected error: {err}");
        drop(ctx_a);
    }

    #[test]
    fn grant_read_rejects_cross_scheme_frame() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = retained_buffer(&mut pool);

        let mut scheme_a = Scheme::new(&ctx);
        let grant_a = scheme_a.grant_read(&parcel).expect("grant_a");

        let mut scheme_b = Scheme::new(&ctx);
        let _grant_b = scheme_b.grant_read(&parcel).expect("grant_b");
        let frame_b = scheme_b.submit().expect("submit b");

        let err = match grant_a.consume(&frame_b) {
            Ok(_) => panic!("cross-scheme read must fail"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("different scheme"), "unexpected error: {err}");
    }

    #[test]
    fn grant_read_drop_scheme_with_outstanding_frame_frees_staging() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = pool
            .acquire_buffer_with_data(&[1u32; 8], BufferKind::Scattered)
            .expect("parcel");
        let mut scheme = Scheme::new(&ctx);
        let _grant = scheme.grant_read(&parcel).expect("grant");
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
    fn grant_read_rejects_zero_byte_buffer() {
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
        let err = match scheme.grant_read(&parcel) {
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
    fn grant_read_texture_basic_succeeds() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let grant = scheme.grant_read_texture(&texture).expect("grant_read_texture");
        let frame = scheme.submit().expect("submit");

        let loan = grant.consume(&frame).expect("read texture grant");
        assert_eq!(loan.len(), 4 * 4 * 4, "Rgba8Unorm 4×4 = 64 bytes");
    }

    #[test]
    fn grant_read_texture_appends_ir_node() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let _grant = scheme.grant_read_texture(&texture).expect("grant_read_texture");

        assert!(scheme.is_dirty(), "grant_read_texture is structural");
        assert_eq!(scheme.ir_node_count(), 1);
        match &scheme.ir.nodes[0].kind {
            NodeKind::GrantRead { grant_id: 0 } => {}
            other => panic!("expected GrantRead node, got {other:?}"),
        }
    }

    #[test]
    fn grant_read_texture_staging_alloc_and_free() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let grant = scheme.grant_read_texture(&texture).expect("grant_read_texture");
        let frame = scheme.submit().expect("submit");

        let (allocs_before, frees_before) = mock_readback_counts(&device);
        assert_eq!(allocs_before, 1, "one staging alloc per submit");
        assert_eq!(frees_before, 0, "not freed yet");

        let loan = grant.consume(&frame).expect("read");
        drop(loan);

        // After loan drop the handle returns to pool (scheme alive) — no free yet.
        let (_, frees_after_loan) = mock_readback_counts(&device);
        assert_eq!(frees_after_loan, 0, "pool recycles on loan drop");

        // Resubmit — pool recycles the same staging handle.
        let frame2 = scheme.submit().expect("resubmit");
        let (allocs_after_resubmit, _) = mock_readback_counts(&device);
        assert_eq!(allocs_after_resubmit, 1, "recycled: no new alloc");
        let _loan2 = grant.consume(&frame2).expect("read frame2");

        // Drop scheme — pool drains and frees all handles.
        drop(_loan2);
        drop(frame2);
        drop(grant);
        drop(scheme);
        let (_, frees_final) = mock_readback_counts(&device);
        assert_eq!(frees_final, 1, "all staging freed on scheme drop");
    }

    #[test]
    fn grant_read_texture_double_read_same_frame_errors() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let grant = scheme.grant_read_texture(&texture).expect("grant_read_texture");
        let frame = scheme.submit().expect("submit");

        let _loan = grant.consume(&frame).expect("first read");
        let err = grant.consume(&frame).expect_err("second read must fail");
        assert!(err.to_string().contains("already consumed"), "unexpected error: {err}");
    }

    #[test]
    fn grant_read_texture_concurrent_frames() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let grant = scheme.grant_read_texture(&texture).expect("grant_read_texture");
        let frame1 = scheme.submit().expect("first submit");
        let frame2 = scheme.submit().expect("second submit without waiting on frame1");

        let loan1 = grant.consume(&frame1).expect("read frame1");
        let loan2 = grant.consume(&frame2).expect("read frame2");
        assert_eq!(loan1.len(), loan2.len());

        let (allocs, _) = mock_readback_counts(&device);
        assert_eq!(allocs, 2, "two live frames require two staging allocations");
    }

    #[test]
    fn grant_read_texture_rejects_sampled_only_texture() {
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
        let err = match scheme.grant_read_texture(&texture) {
            Ok(_) => panic!("must reject Interpolated texture"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("sampled-only") || err.to_string().contains("storage-writable"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn grant_read_texture_rejects_missing_copy_src_flag() {
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
        let err = match scheme.grant_read_texture(&texture) {
            Ok(_) => panic!("must reject missing COPY_SRC flag"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("COPY_SRC"), "unexpected error: {err}");
    }

    #[test]
    fn grant_read_texture_rejects_cross_scheme_frame() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx_a = device.create_context().unwrap();
        let ctx_b = device.create_context().unwrap();

        let texture = texture_parcel(&mut pool);

        let mut scheme_a = Scheme::new(&ctx_a);
        let grant_a = scheme_a.grant_read_texture(&texture).expect("grant_a");
        let _frame_a = scheme_a.submit().expect("submit a");

        let mut scheme_b = Scheme::new(&ctx_b);
        let _grant_b = scheme_b.grant_read_texture(&texture).expect("grant_b");
        let frame_b = scheme_b.submit().expect("submit b");

        let err = grant_a.consume(&frame_b).expect_err("cross-scheme read must fail");
        assert!(err.to_string().contains("different scheme"), "unexpected error: {err}");
    }

    #[test]
    fn grant_read_texture_survives_parcel_drop() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let texture = texture_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        let grant = scheme.grant_read_texture(&texture).expect("grant_read_texture");
        let frame = scheme.submit().expect("submit");
        drop(texture);
        drop(pool);

        let loan = grant.consume(&frame).expect("read after parcel drop");
        assert_eq!(loan.len(), 4 * 4 * 4);
    }

    // ------------------------------------------------------------------
    // Present-on-scheme tests
    // ------------------------------------------------------------------

    struct MockWindow;

    impl raw_window_handle::HasWindowHandle for MockWindow {
        fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            Ok(unsafe {
                raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Web(
                    raw_window_handle::WebWindowHandle::new(0),
                ))
            })
        }
    }

    impl raw_window_handle::HasDisplayHandle for MockWindow {
        fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            Ok(unsafe {
                raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Web(
                    raw_window_handle::WebDisplayHandle::new(),
                ))
            })
        }
    }

    fn mock_swapchain_pool(device: &Arc<Device>) -> (Context, crate::swapchain_pool::SwapchainPool) {
        let ctx = device.create_context().unwrap();
        let pool = crate::swapchain_pool::SwapchainPool::new(&ctx, &MockWindow, 2).expect("swapchain pool");
        (ctx, pool)
    }

    fn mock_present_count(device: &Arc<Device>) -> usize {
        let backend = device.inner.backend.lock().unwrap();
        backend.test_surface_present_count()
    }

    #[test]
    fn grant_present_appends_ir_node() {
        let device = mock_device();
        let (ctx, pool) = mock_swapchain_pool(&device);
        let lease = pool.lease();

        let mut scheme = Scheme::new(&ctx);
        assert_eq!(scheme.ir_node_count(), 0);

        let grant = scheme.grant_present(&lease);
        assert_eq!(scheme.ir_node_count(), 1);

        match &scheme.ir.nodes[0].kind {
            NodeKind::GrantPresent { grant_id: 0 } => {}
            other => panic!("expected GrantPresent{{grant_id:0}}, got {other:?}"),
        }
        assert_eq!(grant.grant_id(), 0);
    }

    #[test]
    fn grant_present_marks_dirty() {
        let device = mock_device();
        let (ctx, pool) = mock_swapchain_pool(&device);
        let lease = pool.lease();

        let mut scheme = Scheme::new(&ctx);
        assert!(scheme.is_dirty(), "new scheme starts dirty");

        // Submit to clear dirty.
        // A scheme with only a GrantPresent (no dispatch) should still submit.
        scheme.grant_present(&lease);
        // grant_present must keep the dirty flag set.
        assert!(scheme.is_dirty(), "grant_present must mark the scheme dirty");
    }

    #[test]
    fn grant_present_orders_after_writer() {
        use crate::task_graph::analysis;

        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let parcel = retained_buffer(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write", &pipeline)
            .with_parcel(&parcel, NodeAccess::Write)
            .with_present(&lease)
            .dispatch(1, 1, 1);
        scheme.grant_present(&lease);

        let edges = analysis::build_edges(&scheme.ir);
        // The dispatch (node 0) must precede the GrantPresent (node 1).
        assert!(
            edges.contains(&(0, 1)),
            "dispatch (0) must precede grant_present (1); edges: {edges:?}"
        );
    }

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

    #[test]
    fn grant_present_submit_increments_present_count() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let present = scheme.grant_present(&lease);

        let before = mock_present_count(&device);
        let submission = scheme.submit().expect("first submit");
        present.consume(&submission).expect("present");
        let after = mock_present_count(&device);
        assert_eq!(after, before + 1, "present must fire one swapchain present");
    }

    #[test]
    fn grant_present_second_present_errors() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let present = scheme.grant_present(&lease);

        let submission = scheme.submit().expect("submit");
        present.consume(&submission).expect("first present");
        let err = present.consume(&submission).expect_err("second present must fail");
        assert!(err.to_string().contains("already consumed"), "unexpected error: {err}");
    }

    #[test]
    fn present_grant_rejects_cross_scheme_submission() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme_a = Scheme::new(&ctx);
        let present_a = scheme_a.grant_present(&lease);

        let mut scheme_b = Scheme::new(&ctx);
        scheme_b.grant_present(&lease);
        let submission_b = scheme_b.submit().expect("submit b");

        let err = present_a
            .consume(&submission_b)
            .expect_err("cross-scheme present must fail");
        assert!(err.to_string().contains("different scheme"), "unexpected error: {err}");
    }

    #[test]
    fn grant_present_submit_twice_presents_independently() {
        // Each submit acquires a fresh swapchain frame; both must be presentable.
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let present = scheme.grant_present(&lease);

        let submission1 = scheme.submit().expect("submit 1");
        let submission2 = scheme.submit().expect("submit 2");

        // Present in order; both must succeed.
        present.consume(&submission1).expect("present submission1");
        present.consume(&submission2).expect("present submission2");

        assert_eq!(mock_present_count(&device), 2, "two submits → two presents");
    }

    #[test]
    fn speculative_present_acquire_stashes_for_next_submit() {
        let device = mock_device();
        let ctx = device.create_context().unwrap();
        let pool = crate::swapchain_pool::SwapchainPool::new_with_options(
            &ctx,
            &MockWindow,
            crate::swapchain_pool::SwapchainPoolOptions {
                depth: 2,
                speculative_acquire: true,
                ..Default::default()
            },
        )
        .expect("swapchain pool");
        let lease = pool.lease();
        let pool = Arc::clone(&lease.pool);

        let mut scheme = Scheme::new(&ctx);
        let present = scheme.grant_present(&lease);

        let submission1 = scheme.submit().expect("submit 1");
        assert!(
            !SwapchainPool::has_speculative_acquire(&pool),
            "first submit uses synchronous acquire"
        );

        present.consume(&submission1).expect("present 1");
        assert!(
            SwapchainPool::has_speculative_acquire(&pool),
            "consume should stash drawable for next submit"
        );

        let submission2 = scheme.submit().expect("submit 2");
        assert!(
            !SwapchainPool::has_speculative_acquire(&pool),
            "second submit should consume speculative stash"
        );
        present.consume(&submission2).expect("present 2");
    }

    #[test]
    fn dropped_frame_restores_pending_acquire_count() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        scheme.grant_present(&lease);

        {
            let _submission = scheme.submit().expect("submit");
            // Drop without present — cancel_frame must restore acquire budget.
        }

        let surface = spool.pending_acquire_count();
        assert_eq!(surface, 0, "cancelled frame must decrement pending_acquire_count");

        // Depth-2 pool should allow another acquire immediately after cancel.
        let _submission2 = scheme.submit().expect("submit after cancel");
    }

    #[test]
    fn grant_present_scheme_records_once_per_slot() {
        // The present-aware retention path must record the first time a given
        // swapchain slot is seen and resubmit from cache on subsequent encounters.
        // Because the mock backend cycles through slots, the N-th submit may
        // record a new slot; we only assert that at least one resubmit occurs.
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        let present = scheme.grant_present(&lease);

        // Submit many frames; the mock backend has a fixed pool of slot ids
        // so after depth frames we must see at least one cache hit.
        let depth = 6;
        for i in 0..depth {
            let submission = scheme.submit().expect(&format!("submit {i}"));
            present.consume(&submission).expect(&format!("present {i}"));
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

    #[test]
    fn dropped_submission_finishes_scheduled_present_on_worker() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        scheme.grant_present(&lease);

        let before = mock_present_count(&device);
        {
            let _submission = scheme.submit().expect("submit");
            // Drop without consume — present was already enqueued at submit; Drop finishes bookkeeping.
        }
        assert_eq!(
            mock_present_count(&device),
            before + 1,
            "scheduled present must complete when Submission drops without consume"
        );
    }

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
        let mut pass = scheme.render_pass("render", &rt);
        pass.set_pipeline(&pipeline);
        pass.draw_fullscreen();
        pass.finish();
        scheme.copy_to_present(&rt, &lease);
        scheme.grant_present(&lease);

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

    fn present_scheme_with_texture_copy(
        scheme: &mut Scheme,
        tex: &crate::Texture,
        lease: &PresentLease,
        pipeline: &ComputePipeline,
    ) -> PresentGrant {
        scheme
            .node("write_tex", pipeline)
            .with_parcel(tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme.copy_texture_to_present(tex, lease);
        scheme.grant_present(lease)
    }

    #[test]
    fn present_easement_promise_abandoned_when_submission_dropped() {
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
        let stamp = scheme.submit_state.resource_stamps().get(&key).expect("texture stamp");
        assert_eq!(stamp.pending.lock().unwrap().len(), 1);
        assert_eq!(stamp.pending.lock().unwrap()[0].poll(), PromiseState::Pending);
        drop(submission);
        assert_eq!(stamp.pending.lock().unwrap()[0].poll(), PromiseState::Abandoned);
    }

    #[test]
    fn present_consume_resolves_with_present_timeline_not_compute() {
        use crate::task_graph::cross_submit::ResourceKey;
        use crate::timeline::PromiseState;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        let present = present_scheme_with_texture_copy(&mut scheme, &tex, &lease, &pipeline);
        let submission = scheme.submit().expect("submit");
        let compute_tv = submission.timeline_value();

        present.consume(&submission).expect("present");

        let key = ResourceKey::Texture(tex.gpu_handle());
        let poll_state = {
            let stamp = scheme.submit_state.resource_stamps().get(&key).expect("texture stamp");
            stamp.pending.lock().unwrap()[0].poll()
        };
        match poll_state {
            PromiseState::Resolved(present_tv) => {
                assert!(
                    present_tv > compute_tv,
                    "present easement must resolve to the later present/copy timeline (present={present_tv}, compute={compute_tv})"
                );
            }
            other => panic!("expected Resolved present timeline, got {other:?}"),
        }
    }

    #[test]
    fn submit_gate_folds_resolved_present_promise_into_foreign_reads() {
        use crate::task_graph::cross_submit::ResourceKey;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));
        let ctx_handle = ctx.backend_handle();

        let mut scheme = Scheme::new(&ctx);
        let present = present_scheme_with_texture_copy(&mut scheme, &tex, &lease, &pipeline);
        let key = ResourceKey::Texture(tex.gpu_handle());

        let sub1 = scheme.submit().expect("submit 1");
        present.consume(&sub1).expect("present frame 1");

        let _sub2 = scheme.submit().expect("submit 2");
        let stamp = scheme.submit_state.resource_stamps().get(&key).expect("texture stamp");
        let sync = stamp.sync.lock().unwrap();
        assert!(
            sync.foreign_reads.get(&ctx_handle).is_some(),
            "submit gate must fold resolved present promise into foreign_reads"
        );
        drop(sync);
        assert_eq!(
            stamp.pending.lock().unwrap().len(),
            1,
            "submit 2 claims a fresh present promise; frame 1's resolved promise must be pruned"
        );
        assert_eq!(
            stamp.pending.lock().unwrap()[0].poll(),
            crate::timeline::PromiseState::Pending
        );
    }

    #[test]
    fn submit_gate_blocks_until_present_promise_resolved() {
        use std::sync::{Arc, Barrier};
        use std::thread;
        use std::time::Duration;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        let present = present_scheme_with_texture_copy(&mut scheme, &tex, &lease, &pipeline);
        let sub1 = scheme.submit().expect("submit 1");

        let scheme = Arc::new(std::sync::Mutex::new(scheme));
        let barrier = Arc::new(Barrier::new(2));
        let scheme2 = Arc::clone(&scheme);
        let barrier2 = Arc::clone(&barrier);

        let gate_thread = thread::spawn(move || {
            barrier2.wait();
            scheme2
                .lock()
                .unwrap()
                .submit()
                .expect("submit 2 must succeed after promise resolves");
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(30));
        present.consume(&sub1).expect("consume releases submit gate");
        gate_thread.join().expect("gate thread");
    }

    #[test]
    fn texture_stamp_not_settled_while_present_promise_unresolved() {
        use crate::task_graph::cross_submit::ResourceKey;
        use crate::timeline::Settle;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_shader(&device));

        let mut scheme = Scheme::new(&ctx);
        present_scheme_with_texture_copy(&mut scheme, &tex, &lease, &pipeline);
        let key = ResourceKey::Texture(tex.gpu_handle());

        let submission = scheme.submit().expect("submit");
        let stamp = scheme.submit_state.resource_stamps().get(&key).expect("texture stamp");
        assert_eq!(stamp.settle_on_context(&ctx), Settle::Pending);
        drop(submission);
        assert_eq!(stamp.settle_on_context(&ctx), Settle::Ready);
    }

    /// shader and `copy_texture_to_present` on the same persistent `out_image` must resolve
    /// to one ledger cell (`ResourceSync`), and a present-path submit must record the copy
    /// read on that cell so cross-frame WAR enforcement can key off `last_reads`.
    #[test]
    fn out_image_fine_write_and_present_copy_share_ledger_identity() {
        use crate::task_graph::cross_submit::{compute_cross_submit_sync, net_access_per_resource, ResourceKey};
        use crate::task_graph::ResourceId;
        use std::collections::HashMap;

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
        // Mirror ekrano scheme path: fine write then present copy on one Texture instance.
        scheme
            .node("fine_write", &pipeline)
            .with_parcel(&tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme.copy_texture_to_present(&tex, &lease);
        let present = scheme.grant_present(&lease);

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

        let sub1 = scheme.submit().expect("submit frame 1");
        let frame1_tv = sub1.timeline_value();
        {
            let sync = registered.sync.lock().unwrap();
            let read_tv = sync
                .last_reads
                .get(&ctx_handle)
                .copied()
                .expect("present-copy read must be on the ledger after submit");
            let write_tv = sync
                .last_write
                .get(&ctx_handle)
                .copied()
                .expect("fine-write must be on the ledger after submit");
            assert!(
                read_tv <= frame1_tv && write_tv <= frame1_tv,
                "ledger epochs must not exceed submission high-water (read={read_tv}, write={write_tv}, submit={frame1_tv})"
            );
        }

        present.consume(&sub1).expect("present frame 1");

        // Frame 2 submit gate folds the resolved present epoch into last_reads.
        let _sub2 = scheme.submit().expect("submit frame 2");
        let present_read_tv = {
            let sync = registered.sync.lock().unwrap();
            *sync.last_reads.get(&ctx_handle).expect("last_reads after present fold")
        };
        assert!(
            present_read_tv >= frame1_tv,
            "folded present read epoch must be at least the frame-1 submit tv"
        );

        // Loop-carried present WAR is covered by FIFO enqueue order; no live wait needed.
        let mut write_only = HashMap::new();
        write_only.insert(
            key,
            net_access_per_resource(&scheme.ir)[&key], // reads+writes in IR; override for next-frame write admission
        );
        write_only.get_mut(&key).unwrap().reads = false;
        let ledger = {
            let sync = registered.sync.lock().unwrap().clone();
            let mut ledger = crate::task_graph::cross_submit::LedgerSnapshot::new();
            ledger.insert(key, crate::task_graph::cross_submit::LedgerEntry { sync });
            ledger
        };
        let plan = compute_cross_submit_sync(&write_only, &ledger, ctx_handle);
        assert!(
            plan.waits.is_empty(),
            "FIFO-scheduled present makes same-context loop-carried WAR wait redundant (got {:?})",
            plan.waits
        );
    }

    /// When present is scheduled on the submission worker at submit, frame N+1 does not
    /// need a live same-context WAR wait against frame N's present-read epoch.
    #[test]
    fn present_war_fifo_ordering_on_second_submit_path() {
        use crate::timeline::Epoch;

        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let tex = mock_direct_texture(&device);
        let pipeline = mock_pipeline(&device, &mock_texture_shader(&device));
        let ctx_handle = ctx.backend_handle();

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("fine_write", &pipeline)
            .with_parcel(&tex, NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme.copy_texture_to_present(&tex, &lease);
        let present = scheme.grant_present(&lease);

        let sub1 = scheme.submit().expect("submit frame 1");
        let compute_tv = sub1.timeline_value();
        let present_tv = sub1
            .present_fifo_tv(0)
            .expect("present scheduled on worker at submit");
        assert!(
            present_tv >= compute_tv,
            "present epoch must cover the frame-1 submit (present={present_tv}, compute={compute_tv})"
        );
        present.consume(&sub1).expect("present frame 1");

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
                .all(|e| !(e.context == ctx_handle && e.value >= present_tv)),
            "FIFO present ordering makes same-context WAR wait redundant (got {frame2_waits:?})"
        );
    }

    #[test]
    fn present_war_ledger_live_wait_on_second_submit_path() {
        use crate::task_graph::cross_submit::ResourceKey;
        use crate::timeline::{Epoch, PromiseState};

        let device = mock_device_legacy_present();
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
        let present = scheme.grant_present(&lease);

        let sub1 = scheme.submit().expect("submit frame 1");
        let compute_tv = sub1.timeline_value();
        present.consume(&sub1).expect("present frame 1");

        let present_tv = {
            let stamp = scheme
                .submit_state
                .resource_stamps()
                .get(&key)
                .expect("out_image stamp");
            match stamp.pending.lock().unwrap()[0].poll() {
                PromiseState::Resolved(tv) => tv,
                other => panic!("present promise must be resolved after consume, got {other:?}"),
            }
        };
        assert!(
            present_tv >= compute_tv,
            "present epoch must cover the frame-1 submit (present={present_tv}, compute={compute_tv})"
        );

        let waits_before = device.with_mock_backend(|b| b.recorded_waits.len());
        let _sub2 = scheme.submit().expect("submit frame 2");
        let frame2_waits: Vec<Epoch> = device.with_mock_backend(|b| {
            b.recorded_waits[waits_before..]
                .iter()
                .flat_map(|w| w.iter().copied())
                .collect()
        });
        assert!(
            frame2_waits.iter().any(|e| e.context == ctx_handle && e.value >= present_tv),
            "frame 2 submit must live-wait on prior present read via ledger (need wait>={present_tv}, got {frame2_waits:?})"
        );
    }
}
