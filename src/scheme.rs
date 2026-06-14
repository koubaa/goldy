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

use crate::backend::{BufferHandle, GpuCommand};
use crate::buffer::Buffer;
use crate::context::Context;
use crate::error::GoldyError;
use crate::parcel::Parcel;
use crate::retained_pool::StampedParcel;
use crate::task_graph::IrSubmitState;
use crate::task_graph::{DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding, TaskNode};
use crate::timeline::TimelineValue;
use crate::types::{ResourceAccess, ResourceHandle, TextureFlags, TextureFormat, TextureKind};
use crate::validation_env;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_SCHEME_ID: AtomicU64 = AtomicU64::new(1);

/// Per-grant staging buffer pool with scheme-lifetime ownership.
/// TODO: this is an implementation detail of schemes for now, but eventually
/// it makes more sense to have device-scoped, size-bucketed pools for this.
/// That would back single schemes and multiple submissions and the reverse.
struct GrantStagingPool {
    handles: Mutex<Vec<BufferHandle>>,
    ctx: Context,
    scheme_alive: AtomicBool,
}

impl GrantStagingPool {
    fn new(ctx: &Context) -> Arc<Self> {
        Arc::new(Self {
            handles: Mutex::new(Vec::new()),
            ctx: ctx.clone(),
            scheme_alive: AtomicBool::new(true),
        })
    }

    fn take_or_alloc(
        &self,
        backend: &mut dyn crate::backend::GpuBackend,
        device: crate::backend::DeviceHandle,
        byte_size: u64,
    ) -> Result<BufferHandle, GoldyError> {
        let handle = {
            let mut pool = self.handles.lock().unwrap_or_else(|e| e.into_inner());
            pool.pop()
        };
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

    fn return_handle(&self, handle: BufferHandle) {
        if self.scheme_alive.load(Ordering::Acquire) {
            self.handles.lock().unwrap_or_else(|e| e.into_inner()).push(handle);
        } else {
            let mut backend = self.ctx.device().inner.backend.lock().unwrap();
            backend.free_readback_buffer(handle);
        }
    }

    fn mark_scheme_dropped_and_drain(&self) {
        self.scheme_alive.store(false, Ordering::Release);
        let mut backend = self.ctx.device().inner.backend.lock().unwrap();
        let mut pool = self.handles.lock().unwrap_or_else(|e| e.into_inner());
        for handle in pool.drain(..) {
            backend.free_readback_buffer(handle);
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

#[derive(Debug)]
struct FrameData {
    scheme_id: u64,
    timeline: TimelineValue,
    /// Per-grant staging buffer for this submission; taken by [`ReadGrant::read`].
    cells: Vec<Mutex<Option<BufferHandle>>>,
    /// Per-grant pools; used to recycle or free unconsumed cells on drop.
    staging_pools: Vec<Arc<GrantStagingPool>>,
}

impl Drop for FrameData {
    fn drop(&mut self) {
        for (cell, pool) in self.cells.iter().zip(self.staging_pools.iter()) {
            if let Some(handle) = cell.lock().unwrap_or_else(|e| e.into_inner()).take() {
                pool.return_handle(handle);
            }
        }
    }
}

/// Per-submission identity returned by [`Scheme::submit`].
///
/// Submission identity for a retained scheme — not [`crate::surface::Frame`]
/// (the present/acquire token). Re-exported at the crate root as [`SchemeFrame`](crate::SchemeFrame).
///
/// A lightweight token. The timeline value identifies which submission this frame
/// represents; use [`Self::wait`] to block until that submission's GPU work completes
/// (including grant-read staging copies when grants are recorded).
#[derive(Debug, Clone)]
pub struct Frame {
    data: Arc<FrameData>,
}

impl Frame {
    /// Timeline value for this submission — pass to [`Context::wait_until`](crate::Context::wait_until).
    pub fn timeline_value(&self) -> TimelineValue {
        self.data.timeline
    }

    /// Block until this submission's GPU work has completed.
    pub fn wait(&self, ctx: &Context) -> Result<(), GoldyError> {
        ctx.wait_until(self.data.timeline)
    }
}

impl From<Frame> for TimelineValue {
    fn from(frame: Frame) -> Self {
        frame.timeline_value()
    }
}

/// Stable index of a read-easement grant recorded in the scheme IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GrantId(pub(crate) u32);

/// Marker type for buffer read grants (v1: deed buffer parcels only).
pub struct GrantBuffer;

struct GrantInfo {
    source: BufferHandle,
    /// Keeps the deed buffer alive after the parcel is dropped.
    #[allow(dead_code)]
    source_backing: Arc<Buffer>,
    byte_size: u64,
    staging_pool: Arc<GrantStagingPool>,
}

/// Readable bytes for one `(grant × frame)` cell — returned by [`ReadGrant::read`].
///
/// Dropping the loan returns the staging buffer to the grant's reuse pool while the
/// owning [`Scheme`] is alive; otherwise the buffer is freed immediately.
pub struct Loan<T> {
    bytes: Vec<u8>,
    handle: BufferHandle,
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
        self.return_pool.return_handle(self.handle);
    }
}

/// A read easement over a scheme parcel — recorded once via [`Scheme::grant_read`].
///
/// Obtain readable bytes for a submission by coordinating this handle with a
/// [`Frame`] from the **same** [`Scheme`]: `grant.read(&frame)`.
pub struct ReadGrant<T> {
    grant_id: GrantId,
    scheme_id: u64,
    ctx: Context,
    byte_size: u64,
    return_pool: Arc<GrantStagingPool>,
    _marker: PhantomData<T>,
}

impl<T> ReadGrant<T> {
    /// Logical byte size of readable data for this grant.
    pub fn byte_size(&self) -> u64 {
        self.byte_size
    }

    /// Readable bytes for `frame`'s submission (full logical buffer size).
    ///
    /// `frame` must come from the same [`Scheme`] that created this grant.
    /// Blocks until this frame's GPU work (dispatch + grant staging copy) completes,
    /// then reads from that submission's dedicated staging buffer. Each frame may be
    /// read at most once per grant. Drop any unconsumed [`Frame`] tokens before dropping
    /// the scheme if you rely on staging-buffer reuse rather than immediate free.
    pub fn read(&self, frame: &Frame) -> Result<Loan<T>, GoldyError> {
        if frame.data.scheme_id != self.scheme_id {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "ReadGrant belongs to a different scheme than this SchemeFrame"
            )));
        }
        frame.wait(&self.ctx)?;
        let idx = self.grant_id.0 as usize;
        if validation_env::scheme_validation_enabled() && idx >= frame.data.cells.len() {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "grant index {} out of range for frame ({} grants)",
                idx,
                frame.data.cells.len()
            )));
        }
        let handle = frame
            .data
            .cells
            .get(idx)
            .ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!(
                    "grant index {} out of range for frame ({} grants)",
                    idx,
                    frame.data.cells.len()
                ))
            })?
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("grant already consumed for this SchemeFrame")))?;
        let byte_size = usize::try_from(self.byte_size)
            .map_err(|_| GoldyError::Backend(anyhow::anyhow!("grant readback byte size exceeds address space")))?;
        let mut bytes = vec![0u8; byte_size];
        {
            let backend = self.ctx.device().inner.backend.lock().unwrap();
            backend
                .read_readback_buffer(handle, &mut bytes)
                .map_err(|e| self.ctx.classify(e))?;
        }
        Ok(Loan {
            bytes,
            handle,
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
    /// COW dirty bit: set by every structural mutation, cleared by a successful record.
    dirty: bool,
    /// Retention key stored at record time. `None` when the backend cannot retain `ir`.
    retention_key: Option<u64>,
    /// Timeline value from the most recent successful [`Self::submit`].
    ///
    /// Before resubmitting a retained command list, we [`Context::wait_until`] this value so
    /// the backend CB is no longer pending (Vulkan VUID-vkQueueSubmit2-commandBuffer-03875).
    /// This is conservative: a lowered scheme may become multiple queue submissions (A1, A2,
    /// A3), and another scheme's B1 need only wait for the slice it depends on — not A3. A
    /// per-slice retirement gate belongs in the IR lowering path; until then, whole-scheme
    /// `last_submitted_tv` is the correctness stopgap.
    last_submitted_tv: Option<TimelineValue>,
    stats: ReplayStats,
    next_grant_id: u32,
    /// Process-unique identity for cross-scheme [`Frame`] / [`ReadGrant`] pairing.
    scheme_id: u64,
    /// Read-easement grants: N-backed staging per submission.
    grants: Vec<GrantInfo>,
}

impl Scheme {
    /// Create a scheme bound to `ctx`.
    pub fn new(ctx: &Context) -> Self {
        Self {
            ir: GraphIR::default(),
            submit_state: IrSubmitState::new(),
            ctx: ctx.clone(),
            leases: Vec::new(),
            dirty: true,
            retention_key: None,
            last_submitted_tv: None,
            stats: ReplayStats::default(),
            next_grant_id: 0,
            scheme_id: NEXT_SCHEME_ID.fetch_add(1, Ordering::Relaxed),
            grants: Vec::new(),
        }
    }

    /// True when the next [`Self::submit`] must re-record.
    pub fn is_dirty(&self) -> bool {
        self.dirty
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

    /// Typed resource descriptor handle for a scheme-held lease, for use in `bind_resources_typed`.
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
    pub fn submit(&mut self) -> Result<Frame, GoldyError> {
        let tv_dispatch = if !self.dirty {
            if let Some(key) = self.retention_key {
                if let Some(prev_tv) = self.last_submitted_tv {
                    self.ctx.wait_until(prev_tv)?;
                }
                if let Some(tv) = self.ctx.try_resubmit_retained(key)? {
                    self.submit_state
                        .apply_reference_stamps(self.ctx.backend_handle(), &self.ctx.device().inner, tv);
                    self.last_submitted_tv = Some(tv);
                    #[cfg(not(feature = "metal"))]
                    {
                        self.stats.resubmit_hits += 1;
                    }
                    tv
                } else {
                    self.record_and_retain()?
                }
            } else {
                self.record_and_retain()?
            }
        } else {
            self.record_and_retain()?
        };

        self.finish_submit_frame(tv_dispatch)
    }

    fn record_and_retain(&mut self) -> Result<TimelineValue, GoldyError> {
        let tv = self
            .submit_state
            .submit_pipelined_and_retain(&self.ctx, &self.ir)
            .map_err(|e| self.ctx.classify(e))?;
        self.submit_state
            .apply_reference_stamps(self.ctx.backend_handle(), &self.ctx.device().inner, tv);
        self.ctx.advance_high_water_timeline(tv);

        self.retention_key = if IrSubmitState::ir_can_retain(&self.ir) {
            Some(IrSubmitState::retention_fingerprint(&self.ir))
        } else {
            None
        };
        self.dirty = false;
        self.last_submitted_tv = Some(tv);
        self.stats.records += 1;
        Ok(tv)
    }

    fn finish_submit_frame(&mut self, tv_dispatch: TimelineValue) -> Result<Frame, GoldyError> {
        if self.grants.is_empty() {
            return Ok(Frame {
                data: Arc::new(FrameData {
                    scheme_id: self.scheme_id,
                    timeline: tv_dispatch,
                    cells: Vec::new(),
                    staging_pools: Vec::new(),
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
                let staging = grant
                    .staging_pool
                    .take_or_alloc(&mut **backend, device, grant.byte_size)?;
                if validation_env::scheme_validation_enabled() {
                    if staging_handles.contains(&staging) {
                        return Err(GoldyError::Backend(anyhow::anyhow!(
                            "duplicate grant staging buffer handle in one submission"
                        )));
                    }
                    staging_handles.push(staging);
                }
                copy_cmds.push(GpuCommand::CopyBuffer {
                    src: grant.source,
                    dst: staging,
                    size: grant.byte_size,
                });
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
                .submit_standalone(self.ctx.backend_handle(), &copy_cmds)
                .map_err(|e| self.ctx.classify(e))?
        };
        self.ctx.advance_high_water_timeline(tv_copy);

        Ok(Frame {
            data: Arc::new(FrameData {
                scheme_id: self.scheme_id,
                timeline: tv_copy,
                cells,
                staging_pools,
            }),
        })
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
    }
}

impl Scheme {
    /// Record a read easement grant over a buffer deed parcel.
    ///
    /// Returns a stable [`ReadGrant`] handle; call [`ReadGrant::read`] with a
    /// [`Frame`] from [`Self::submit`] to obtain that submission's bytes.
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
        let staging_pool = GrantStagingPool::new(&self.ctx);
        self.grants.push(GrantInfo {
            source,
            source_backing,
            byte_size,
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
            return_pool: staging_pool,
            _marker: PhantomData,
        })
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
    pub fn bind_parcel(mut self, parcel: &crate::Parcel, access: NodeAccess) -> Self {
        self.scheme.submit_state.register_parcel_stamp(parcel);
        self.bindings.push(ResourceBinding {
            resource: parcel.resource_id(),
            access,
        });
        self
    }

    /// Declare that this node reads a scheme-held [`Lease`].
    pub fn reads_lease(self, lease: &Lease<LeaseTexture>) -> Self {
        self.bind_lease(lease, NodeAccess::Read)
    }

    /// Declare that this node writes a scheme-held [`Lease`].
    pub fn writes_lease(self, lease: &Lease<LeaseTexture>) -> Self {
        self.bind_lease(lease, NodeAccess::Write)
    }

    fn bind_lease(mut self, lease: &Lease<LeaseTexture>, access: NodeAccess) -> Self {
        let idx = lease.id.0 as usize;
        let backing = &self.scheme.leases[idx];
        let resource = backing.resource_id();
        let stamp = backing.stamp_handle();
        self.scheme.submit_state.register_stamp(stamp);
        self.bindings.push(ResourceBinding { resource, access });
        self
    }

    /// Bind resource slots from typed [`crate::types::ResourceHandle`]s (region A indices only).
    pub fn bind_resources_typed(mut self, handles: &[crate::types::ResourceHandle]) -> Self {
        self.resource_slots = handles.iter().map(|h| h.index()).collect();
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
            .bind_parcel(&parcel, NodeAccess::Write)
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
            .bind_parcel(&parcel, NodeAccess::Write)
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
        let handle = scheme.leases[0].handle(ResourceAccess::Write).expect("lease handle");
        scheme
            .node("write_tex", &pipeline)
            .writes_lease(&lease)
            .bind_resources_typed(&[handle])
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
            .bind_parcel(&parcel2, NodeAccess::Write)
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
            .bind_parcel(&parcel, NodeAccess::Write)
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

        let loan = grant.read(&frame).expect("read after parcel drop");
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
        let loan1 = grant.read(&frame1).expect("read frame1");
        let loan2 = grant.read(&frame2).expect("read frame2");
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

        let loan1 = grant.read(&frame1).expect("read frame1");
        let loan2 = grant.read(&frame2).expect("read frame2");
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
        let _loan = grant.read(&frame).expect("first read");
        let err = match grant.read(&frame) {
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
            let loan = grant.read(&frame1).expect("read frame1");
            assert_eq!(loan.len(), 32);
        }
        let frame2 = scheme.submit().expect("submit 2 after loan drop");
        let loan2 = grant.read(&frame2).expect("read frame2 after pool recycle");
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

        let err = match grant_a.read(&frame_b) {
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
}
