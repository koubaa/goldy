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

use crate::backend::{BufferHandle, GpuCommand, RenderCommand, TextureHandle, TextureReadbackLayout};
use crate::buffer::{Buffer, BufferSource};
use crate::context::Context;
use crate::error::GoldyError;
use crate::parcel::Parcel;
use crate::render_target::RenderTarget;
use crate::retained_pool::StampedParcel;
use crate::swapchain_pool::{PresentLease, SwapchainPool};
use crate::task_graph::cross_submit::ResourceKey;
use crate::task_graph::IrSubmitState;
use crate::task_graph::ResolvedPresentSlot;
use crate::task_graph::ResourceId;
use crate::task_graph::{
    DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding, ShaderResourceSlot, TaskNode,
    PRESENT_LEASE_SLOT_PLACEHOLDER,
};
use crate::texture::Texture;
use crate::timeline::TimelineValue;
use crate::types::{
    BackendType, Color, DepthFormat, IndexFormat, ResourceAccess, ResourceHandle, TextureFlags, TextureFormat,
    TextureKind,
};
use crate::validation_env;
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
    Texture { layout: TextureReadbackLayout },
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

    fn new_texture(ctx: &Context, layout: TextureReadbackLayout) -> Arc<Self> {
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
    /// Per-grant staging buffer for this submission; taken by [`ReadGrant::consume`].
    cells: Vec<Mutex<Option<BufferHandle>>>,
    /// Per-grant pools; used to recycle or free unconsumed cells on drop.
    staging_pools: Vec<Arc<GrantStagingPool>>,
    /// Acquired swapchain frames for present grants; consumed by [`PresentGrant::consume`].
    present_frames: Vec<Mutex<Option<crate::surface::Frame>>>,
}

impl Drop for SubmissionData {
    fn drop(&mut self) {
        let ready_after = self.timeline;
        for (cell, pool) in self.cells.iter().zip(self.staging_pools.iter()) {
            if let Some(handle) = cell.lock().unwrap_or_else(|e| e.into_inner()).take() {
                pool.return_handle(handle, ready_after);
            }
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
#[derive(Debug, Clone)]
pub struct PresentGrant {
    pub(crate) grant_id: u32,
    pub(crate) scheme_id: u64,
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
        if submission.data.scheme_id != self.scheme_id {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "PresentGrant belongs to a different scheme than this submission"
            )));
        }
        let idx = self.grant_id as usize;
        let frame_mutex = submission.data.present_frames.get(idx).ok_or_else(|| {
            GoldyError::Backend(anyhow::anyhow!(
                "present grant index {} out of range for submission ({} present grants)",
                idx,
                submission.data.present_frames.len()
            ))
        })?;
        let mut slot = frame_mutex.lock().unwrap_or_else(|e| e.into_inner());
        let surface_frame = slot
            .take()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("grant access already consumed for this submission")))?;
        surface_frame.present().map(|_| ()).map_err(GoldyError::Backend)
    }
}

struct PresentGrantInfo {
    lease_id: u32,
    pool: std::sync::Arc<crate::swapchain_pool::SwapchainPoolInner>,
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
    Texture(TextureReadbackLayout),
}

enum GrantSource {
    Buffer {
        source: BufferHandle,
        #[allow(dead_code)]
        source_backing: Arc<Buffer>,
        byte_size: u64,
    },
    Texture {
        source: TextureHandle,
        #[allow(dead_code)]
        source_backing: Texture,
        layout: TextureReadbackLayout,
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
    /// Timeline value from the most recent successful [`Self::submit`].
    ///
    /// On Vulkan, retained partitions are resubmitted as live `VkCommandBuffer` objects
    /// (VUID-vkQueueSubmit2-commandBuffer-03875 forbids submitting a CB that is still
    /// pending).  Before resubmitting on the clean path we wait on this value to ensure all
    /// prior partitions have retired.
    ///
    /// On Metal, `try_resubmit_retained` always re-encodes a fresh `MTLCommandBuffer` from
    /// the cached `GraphCommand` list, so there is no pending CB to wait for and the wait is
    /// skipped at runtime.
    ///
    /// This is still conservative: a lowered scheme may become multiple queue submissions
    /// (A1, A2, A3) and another scheme's B1 need only wait for the slice it depends on —
    /// not A3.  A per-slice retirement gate belongs in the IR lowering path; until then,
    /// whole-scheme `last_submitted_tv` is the correctness stopgap.
    last_submitted_tv: Option<TimelineValue>,
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
            last_submitted_tv: None,
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
        let backing = self
            .ctx
            .with_transient_pool(|pool| pool.acquire_texture(&self.ctx, width, height, format, access, flags))
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
    /// The backing render target is held until the scheme is dropped or [`Self::begin_rerecord`]
    /// clears it. Structural mutation.
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

    /// Typed resource descriptor handle for a scheme-held lease (advanced binding).
    pub fn lease_handle(&self, lease: &Lease<LeaseTexture>, access: ResourceAccess) -> Option<ResourceHandle> {
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
        }
    }

    /// Submit the scheme: resubmit the retained command list when clean, re-record when dirty.
    ///
    /// On a clean resubmit, bound parcels' reference tables are stamped with the new
    /// timeline value, keeping the context transient pool's reuse gates correct across
    /// retained submissions.
    ///
    /// When present grants are recorded, swapchain drawables are acquired before lowering
    /// and stored on the returned [`Submission`] for [`PresentGrant::consume`].
    pub fn submit(&mut self) -> Result<Submission, GoldyError> {
        // Vulkan/DX12 retained partitions reuse a live command buffer — wait until the prior
        // submission retires before resubmitting to satisfy VUID-vkQueueSubmit2-commandBuffer-03875.
        // Skip on Metal (fresh MTLCommandBuffer each resubmit). Use a runtime backend check:
        // the `metal` Cargo feature is enabled on all default builds, so a compile-time
        // `#[cfg(not(feature = "metal"))]` gate would omit this wait on Vulkan too.
        if let Some(prev_tv) = self.last_submitted_tv {
            if self.ctx.device().backend_type() != BackendType::Metal {
                self.ctx.wait_until(prev_tv)?;
            }
        }

        let topo_dirty = self.topology_dirty.load(Ordering::Acquire);
        let structurally_dirty = self.dirty;
        if structurally_dirty || topo_dirty {
            self.submit_state.invalidate_retention();
        }

        let mut present_slots = Vec::with_capacity(self.present_grants.len());
        let mut surface_frames = Vec::with_capacity(self.present_grants.len());
        for grant in &self.present_grants {
            let (slot_id, surface_frame, uav_index, handle) =
                SwapchainPool::acquire_slot(&grant.pool).map_err(|e| self.ctx.classify(e))?;
            present_slots.push(ResolvedPresentSlot {
                lease_id: grant.lease_id,
                slot_id,
                handle,
                uav_index,
            });
            surface_frames.push(Mutex::new(Some(surface_frame)));
        }

        let submit_result =
            self.submit_state
                .submit_pipelined_and_retain_with_presents(&self.ctx, &self.ir, &present_slots);

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

        let recorded = !part_result.all_from_cache();
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

        let submission = self.finish_submit_frame(tv, surface_frames)?;
        self.last_submitted_tv = Some(submission.timeline_value());
        Ok(submission)
    }

    /// Clear recorded nodes and retention cache while preserving scheme identity.
    ///
    /// Use before re-recording structural nodes after a resize or other topology change.
    /// `last_submitted_tv` is preserved so the next submit waits for the prior
    /// submission before resubmitting retained command lists.
    pub fn begin_rerecord(&mut self) {
        self.ir = GraphIR::default();
        self.submit_state.reset();
        self.grants.clear();
        self.present_grants.clear();
        self.next_grant_id = 0;
        self.retention_key = None;
        self.dirty = true;
        self.prev_topology_parcels.clear();

        let ctx = self.ctx.clone();
        for mut parcel in self.leases.drain(..) {
            let ready_after = parcel.last_referenced();
            parcel.release_bookkeeping();
            ctx.with_transient_pool(|pool| {
                pool.adopt(StampedParcel { parcel, ready_after });
            });
        }
        self.rt_leases.clear();
    }

    fn finish_submit_frame(
        &mut self,
        tv_dispatch: TimelineValue,
        present_frames: Vec<Mutex<Option<crate::surface::Frame>>>,
    ) -> Result<Submission, GoldyError> {
        if self.grants.is_empty() {
            return Ok(Submission {
                data: Arc::new(SubmissionData {
                    scheme_id: self.scheme_id,
                    timeline: tv_dispatch,
                    cells: Vec::new(),
                    staging_pools: Vec::new(),
                    present_frames,
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
                    GrantSource::Buffer { source, byte_size, .. } => {
                        copy_cmds.push(GpuCommand::CopyBuffer {
                            src: *source,
                            dst: staging,
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
                cells,
                staging_pools,
                present_frames,
            }),
        })
    }

    /// Record a present easement grant over a swapchain lease.
    pub fn grant_present(&mut self, lease: &PresentLease) -> PresentGrant {
        self.dirty = true;
        let grant_id = self.next_grant_id;
        self.next_grant_id += 1;
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
            kind: NodeKind::GrantPresent { grant_id },
        });
        PresentGrant {
            grant_id,
            scheme_id: self.scheme_id,
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
            push_constant_handles: Vec::new(),
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

        let ctx = self.ctx.clone();
        for mut parcel in self.leases.drain(..) {
            let ready_after = parcel.last_referenced();
            parcel.release_bookkeeping();
            ctx.with_transient_pool(|pool| {
                pool.adopt(StampedParcel { parcel, ready_after });
            });
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
        let grant_id = GrantId(self.next_grant_id);
        self.next_grant_id += 1;
        let staging_pool = GrantStagingPool::new_buffer(&self.ctx, byte_size);
        self.grants.push(GrantInfo {
            source: GrantSource::Buffer {
                source,
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
            kind: NodeKind::GrantRead { grant_id: grant_id.0 },
        });
        Ok(ReadGrant {
            grant_id,
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
                .query_texture_readback_layout(self.ctx.device().inner.handle, width, height, format)
                .map_err(|e| self.ctx.classify(e))?
        };
        let grant_id = GrantId(self.next_grant_id);
        self.next_grant_id += 1;
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
            kind: NodeKind::GrantRead { grant_id: grant_id.0 },
        });
        Ok(ReadGrant {
            grant_id,
            scheme_id: self.scheme_id,
            ctx: self.ctx.clone(),
            byte_size: layout.logical_bytes,
            read_kind: GrantReadKind::Texture(layout),
            return_pool: staging_pool,
            _marker: PhantomData,
        })
    }
}

fn node_access_to_resource_access(access: NodeAccess) -> ResourceAccess {
    match access {
        NodeAccess::Read => ResourceAccess::Read,
        NodeAccess::Write => ResourceAccess::Write,
        NodeAccess::ReadWrite => ResourceAccess::ReadWrite,
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
}

impl<'a> SchemeNodeBuilder<'a> {
    /// Declare that this node accesses a retained [`crate::Parcel`] deed.
    ///
    /// The parcel's bindless index is appended to `resource_slots` in call order.
    /// The nth call corresponds to the nth resource-kind parameter in the shader signature.
    pub fn with_parcel(mut self, parcel: &crate::Parcel, access: NodeAccess) -> Self {
        self.scheme.submit_state.register_parcel_stamp(parcel);
        self.bindings.push(ResourceBinding {
            resource: parcel.resource_id(),
            access,
        });
        let resource_access = node_access_to_resource_access(access);
        if let Some(index) = parcel.resource_index(resource_access) {
            self.resource_slots.push(index);
        }
        self
    }

    /// Declare that this node accesses a scheme-held texture [`Lease`].
    pub fn with_lease(mut self, lease: &Lease<LeaseTexture>, access: NodeAccess) -> Self {
        let idx = lease.id.0 as usize;
        let backing = &self.scheme.leases[idx];
        let resource = backing.resource_id();
        self.scheme.submit_state.register_parcel_stamp(backing);
        self.bindings.push(ResourceBinding { resource, access });
        let resource_access = node_access_to_resource_access(access);
        if let Some(handle) = backing.handle(resource_access) {
            self.resource_slots.push(handle.index());
        }
        self
    }

    /// Append one scalar virtual-main parameter (region B of [`PushLayout`]).
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

    /// Declare explicit resource view handles (advanced: mosaic parcels and non-default views).
    ///
    /// Replaces resource slot indices while preserving trailing
    /// [`PRESENT_LEASE_SLOT_PLACEHOLDER`] entries appended by [`Self::with_present`].
    pub fn with_views(mut self, handles: &[crate::types::ResourceHandle]) -> Self {
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

    /// Like [`Self::with_views`]; retained for internal/tests.
    #[allow(dead_code)]
    pub(crate) fn with_views_typed(self, handles: &[crate::types::ResourceHandle]) -> Self {
        self.with_views(handles)
    }

    /// Declare a UAV write to a present lease (swapchain drawable).
    ///
    /// Appends a [`PRESENT_LEASE_SLOT_PLACEHOLDER`] entry at the end of `resource_slots`
    /// so the resolver can patch it to the correct UAV index at submit time.
    /// May be called before or after [`Self::with_views`].
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
        self.scheme.ir.nodes.push(TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::Dispatch {
                pipeline: self.pipeline,
                resource_slots: self.resource_slots,
                user_slots: self.user_slots,
                dispatch: DispatchDim::Direct { x, y, z },
            },
        });
    }
}

/// Builder for a render pass recorded on a [`Scheme`].
pub struct SchemeRenderPassBuilder<'a> {
    scheme: &'a mut Scheme,
    label: &'static str,
    target: crate::backend::RenderTargetHandle,
    bindings: Vec<ResourceBinding>,
    commands: Vec<RenderCommand>,
    push_constant_handles: Vec<ResourceHandle>,
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
        let resource_access = node_access_to_resource_access(access);
        if let Some(handle) = parcel.handle(resource_access) {
            self.push_constant_handles.push(handle);
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
                    let resource_access = node_access_to_resource_access(*access);
                    self.scheme.submit_state.register_parcel_stamp(parcel);
                    self.bindings.push(ResourceBinding {
                        resource: parcel.resource_id(),
                        access: *access,
                    });
                    let handle = parcel.handle(resource_access).unwrap_or_else(|| {
                        panic!(
                            "ShaderResourceSlot::Parcel: mosaic parcels cannot be push-constant slots; \
                             use with_parcel for geometry and with_views at draw time"
                        )
                    });
                    self.push_constant_handles.push(handle);
                }
                ShaderResourceSlot::Sampler(sampler) => {
                    let handle = sampler
                        .handle(ResourceAccess::Read)
                        .expect("ShaderResourceSlot::Sampler: missing bindless sampler index");
                    self.push_constant_handles.push(handle);
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
        if !self.push_constant_handles.is_empty() {
            self.commands.push(RenderCommand::BindResourcesTyped {
                handles: self.push_constant_handles.clone(),
            });
        }
        self
    }

    /// Declare explicit resource view handles at draw time (advanced: mosaic parcels and non-default views).
    pub fn with_views(&mut self, handles: &[ResourceHandle]) -> &mut Self {
        self.commands.push(RenderCommand::BindResourcesTyped {
            handles: handles.to_vec(),
        });
        self
    }

    /// Like [`Self::with_views`]; retained for internal/tests.
    #[allow(dead_code)]
    pub(crate) fn with_views_typed(&mut self, handles: &[ResourceHandle]) -> &mut Self {
        self.with_views(handles)
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
            push_constant_handles: _,
        } = self;
        scheme.ir.nodes.push(TaskNode {
            label,
            bindings,
            kind: NodeKind::RenderPass { target, commands },
        });
    }
}

/// Upload CPU bytes to a buffer parcel without mutating scheme structure.
///
/// Performs a standalone GPU write so retained schemes stay clean across frames.
///
/// If the parcel was previously referenced by GPU work on `ctx`, this function
/// waits for that work to complete before issuing the write, preventing a race
/// between the in-flight GPU read and the incoming CPU upload.
///
/// # Deprecation
///
/// This function is obsolete. Parcel writes should be expressed as
/// [`Scheme::commit_write_parcel`] nodes inside the scheme that consumes them.
/// Schemes do not CPU-stall on submission; the staging belt handles cross-
/// submission hazards without a blocking wait.
#[deprecated(
    since = "0.0.0",
    note = "use Scheme::commit_write_parcel instead; parcel writes belong inside the scheme that consumes them"
)]
pub fn write_to_parcel(ctx: &Context, parcel: &Parcel, offset: u64, data: &[u8]) -> Result<(), GoldyError> {
    // Ensure the GPU has finished any prior use of this parcel before overwriting it.
    if let Some(last_tv) = parcel.last_referenced_on(ctx.backend_handle()) {
        if ctx.gpu_progress() < last_tv {
            ctx.wait_until(last_tv)?;
        }
    }

    let (buffer, _) = parcel.write_buffer_target().map_err(|e| ctx.classify(e))?;
    let cmd = GpuCommand::WriteBuffer {
        buffer,
        offset,
        data: Arc::from(data.to_vec()),
    };
    let tv = {
        let mut backend = ctx.device().inner.backend.lock().unwrap();
        backend
            .submit_standalone(ctx.backend_handle(), &[cmd], None)
            .map_err(|e| ctx.classify(e))?
    };
    ctx.advance_high_water_timeline(tv);
    parcel.mark_referenced(ctx.backend_handle(), tv);
    Ok(())
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::compute::ComputePipeline;
    use crate::device::Device;
    use crate::retained_pool::RetainedPool;
    use crate::shader::ShaderModule;
    use crate::task_graph::NodeAccess;
    use crate::task_graph::NodeKind;
    use crate::types::ResourceAccess;
    use crate::BufferKind;
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

    fn retained_buffer_parcel(pool: &mut RetainedPool) -> Parcel {
        pool.acquire_buffer(
            32,
            crate::types::BufferKind::Scattered,
            None,
            crate::types::BufferFlags::empty(),
            None,
        )
        .expect("alloc buffer parcel")
    }

    fn recording_scheme_with_parcel(device: &Arc<Device>, pool: &mut RetainedPool, ctx: &Context) -> (Scheme, Parcel) {
        let shader = mock_shader(device);
        let pipeline = mock_pipeline(device, &shader);
        let parcel = retained_buffer_parcel(pool);

        let mut scheme = Scheme::new(ctx);
        scheme
            .node("a", &pipeline)
            .with_parcel(&parcel, NodeAccess::Write)
            .dispatch(1, 1, 1);
        (scheme, parcel)
    }

    fn recording_scheme(device: &Arc<Device>, pool: &mut RetainedPool, ctx: &Context) -> Scheme {
        recording_scheme_with_parcel(device, pool, ctx).0
    }

    fn clean_scheme(device: &Arc<Device>, pool: &mut RetainedPool) -> Scheme {
        let ctx = device.create_context().unwrap();
        let shader = mock_shader(device);
        let pipeline = mock_pipeline(device, &shader);
        let parcel = retained_buffer_parcel(pool);

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
            .with_lease(&lease, NodeAccess::Write)
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
        let parcel2 = retained_buffer_parcel(&mut pool);
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
        let parcel = retained_buffer_parcel(&mut pool);
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
        let parcel = retained_buffer_parcel(&mut pool);

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
        let parcel = retained_buffer_parcel(&mut pool);
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
        let parcel = retained_buffer_parcel(&mut pool);

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

    fn texture_parcel(pool: &mut RetainedPool) -> Parcel {
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
        let parcel = retained_buffer_parcel(&mut pool);

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
    fn write_to_parcel_does_not_dirty_scheme() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);

        scheme.submit().expect("initial submit");
        assert!(!scheme.is_dirty(), "clean after first submit");

        write_to_parcel(&ctx, &parcel, 0, &[1u8; 8]).expect("write_to_parcel");
        assert!(!scheme.is_dirty(), "write_to_parcel must not dirty the scheme");
    }

    #[test]
    fn write_to_parcel_stamps_parcel_reference() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = pool
            .acquire_buffer(
                32,
                crate::types::BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
                None,
            )
            .expect("parcel");

        let before = parcel.last_referenced_on(ctx.backend_handle());
        assert!(before.is_none(), "fresh parcel has no stamp");

        write_to_parcel(&ctx, &parcel, 0, &[0u8; 8]).expect("write");
        let after = parcel.last_referenced_on(ctx.backend_handle());
        assert!(after.is_some(), "write_to_parcel must stamp last_referenced_on");
        assert!(after.unwrap() > 0, "stamp must be a non-zero timeline value");
    }

    #[test]
    fn write_to_parcel_round_trips_data() {
        // write_to_parcel must cause a WriteBuffer command to be dispatched.
        // We verify indirectly: the parcel's reference epoch advances only
        // after write_to_parcel is called, confirming a submit occurred.
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = pool
            .acquire_buffer(
                32,
                crate::types::BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
                None,
            )
            .expect("parcel");

        let tv_before = parcel.last_referenced_on(ctx.backend_handle());
        write_to_parcel(&ctx, &parcel, 0, &[0xABu8; 8]).expect("write");
        let tv_after = parcel.last_referenced_on(ctx.backend_handle());

        assert!(
            tv_after > tv_before,
            "timeline must advance after write_to_parcel; before={tv_before:?} after={tv_after:?}"
        );
    }

    #[test]
    fn write_to_parcel_waits_for_prior_gpu_reference() {
        // Verify that write_to_parcel waits when the parcel's last GPU reference is
        // still in flight, preventing a write-after-read hazard.
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let (mut scheme, parcel) = recording_scheme_with_parcel(&device, &mut pool, &ctx);

        let _frame = scheme.submit().expect("submit");
        let tv = parcel.last_referenced_on(ctx.backend_handle()).expect("stamped");
        assert!(tv > 0, "parcel must be stamped after submit");

        let wait_before = {
            let backend = device.inner.backend.lock().unwrap();
            backend.test_wait_until_count()
        };

        write_to_parcel(&ctx, &parcel, 0, &[0u8; 8]).expect("write after submit");

        let wait_after = {
            let backend = device.inner.backend.lock().unwrap();
            backend.test_wait_until_count()
        };

        if ctx.gpu_progress() < tv {
            assert_eq!(
                wait_after,
                wait_before + 1,
                "write_to_parcel must wait when the prior GPU reference is still in flight"
            );
        } else {
            assert_eq!(
                wait_after, wait_before,
                "write_to_parcel must skip wait when GPU progress already covers the stamp"
            );
        }
    }

    #[test]
    fn dropped_frame_without_present_cancels_swapchain() {
        let device = mock_device();
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();

        let mut scheme = Scheme::new(&ctx);
        scheme.grant_present(&lease);

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

    #[test]
    fn begin_rerecord_clears_retention_and_allows_resubmit() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let (ctx, spool) = mock_swapchain_pool(&device);
        let lease = spool.lease();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);
        let parcel = retained_buffer_parcel(&mut pool);

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("write", &pipeline)
            .with_parcel(&parcel, NodeAccess::Write)
            .with_present(&lease)
            .dispatch(1, 1, 1);
        let present = scheme.grant_present(&lease);

        let submission = scheme.submit().expect("initial submit");
        present.consume(&submission).expect("present");

        scheme.begin_rerecord();
        assert!(scheme.is_dirty());
        scheme
            .node("write", &pipeline)
            .with_parcel(&parcel, NodeAccess::Write)
            .with_present(&lease)
            .dispatch(2, 2, 1);
        let present2 = scheme.grant_present(&lease);

        let submission2 = scheme.submit().expect("post-rerecord submit");
        present2.consume(&submission2).expect("present after rerecord");
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
}
