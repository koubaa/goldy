//! Retained GPU property: [`Buffer`] (acquired aggregate) and [`Parcel`] (bindable unit).
//!
//! Acquire a [`Buffer`] from [`crate::retained_pool::RetainedPool`]; bind a [`Parcel`] to
//! dispatches and render passes. Each parcel is independently dependency-tracked.

use crate::backend::{BufferHandle, ContextHandle};
use crate::buffer::{Allocation, BufferSource, BufferView, StructuredBufferElement};
use crate::context::Context;
use crate::device::DeviceInner;
use crate::task_graph::ResourceId;
use crate::texture::TextureBacking;
use crate::timeline::{
    PromiseState, ReferenceTable, ResourceSync, Settle, TimelinePromise, TimelineValue, WRITE_KINDS_TRANSFER,
};
use crate::types::{BufferFlags, BufferKind, ResourceAccess, ResourceHandle, TextureFlags};
use crate::vram_allocator::ParcelType;
use std::borrow::Cow;
use std::ops::{Deref, Index};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// How a scheme interacts with a shared parcel for cross-scheme topology tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InteractionRole {
    Reads,
    Writes,
}

/// One scheme's registered interaction with a parcel.
#[derive(Debug, Clone)]
pub(crate) struct InteractionEdge {
    pub scheme_id: u64,
    pub role: InteractionRole,
    pub kind_bits: u8,
    pub ctx: ContextHandle,
    pub dirty_flag: Weak<AtomicBool>,
}

pub(crate) type InteractionSet = Vec<InteractionEdge>;

/// Shared stamp cell and home-device identity for scheme submit stamping.
pub(crate) struct ParcelStamp {
    pub(crate) sync: Arc<Mutex<ResourceSync>>,
    pub(crate) interaction_set: Arc<Mutex<InteractionSet>>,
    pub(crate) pending: Arc<Mutex<Vec<TimelinePromise>>>,
    pub(crate) home_device: Weak<DeviceInner>,
}

impl ParcelStamp {
    pub(crate) fn new(home_device: Weak<DeviceInner>) -> Self {
        Self {
            sync: Arc::new(Mutex::new(ResourceSync::default())),
            interaction_set: Arc::new(Mutex::new(Vec::new())),
            pending: Arc::new(Mutex::new(Vec::new())),
            home_device,
        }
    }

    pub(crate) fn clone_shared_cells(&self) -> Self {
        Self {
            sync: Arc::clone(&self.sync),
            interaction_set: Arc::clone(&self.interaction_set),
            pending: Arc::clone(&self.pending),
            home_device: self.home_device.clone(),
        }
    }

    pub(crate) fn merged_references(&self) -> ReferenceTable {
        self.sync.lock().unwrap().merged()
    }

    pub(crate) fn push_pending(&self, promise: TimelinePromise) {
        self.pending.lock().unwrap().push(promise);
    }

    /// Submission-order gate: block until every pending promise is resolved or abandoned,
    /// folding resolved values into [`ResourceSync::foreign_reads`]. This is not a GPU-completion wait.
    ///
    /// # Threading contract (strict — do not violate)
    ///
    /// This function **blocks the calling thread** until all pending easement promises
    /// on this stamp are settled. It is therefore a hard requirement that the thread
    /// resolving those promises (`PromiseResolver::resolve`, typically TID_PRESENT) is
    /// **never** the same thread that calls `submit()` (which calls this function).
    /// Violating this invariant causes an unrecoverable deadlock: submit waits for
    /// present-consume, while present-consume is stuck behind submit completing.
    ///
    /// The expected call topology is:
    ///   - Render/submit thread: `scheme.submit()` → `drain_pending_for_submit_gate`
    ///   - TID_PRESENT: `grant.consume()` → `resolver.resolve(copy_tv)`
    ///     where `copy_tv` is the present-partition submit timeline (last read of
    ///     copy-to-present sources), not the later display-present timeline.
    ///
    /// Do not fold the two roles onto one thread, even in fallback or teardown paths.
    pub(crate) fn drain_pending_for_submit_gate(&self, ctx: ContextHandle) {
        loop {
            let snapshot: Vec<TimelinePromise> = self.pending.lock().unwrap().clone();
            if snapshot.is_empty() {
                return;
            }
            for promise in &snapshot {
                if matches!(promise.poll(), PromiseState::Pending) {
                    promise.block();
                }
            }
            let mut pending = self.pending.lock().unwrap();
            let mut sync = self.sync.lock().unwrap();
            pending.retain(|promise| match promise.poll() {
                PromiseState::Pending => true,
                PromiseState::Resolved(tv) => {
                    sync.record_foreign_read(ctx, tv);
                    false
                }
                PromiseState::Abandoned => false,
            });
            if pending.is_empty() {
                return;
            }
        }
    }

    fn lazy_gc_pending(&self, ctx: ContextHandle) -> bool {
        let mut pending = self.pending.lock().unwrap();
        if pending.is_empty() {
            return false;
        }
        let mut sync = self.sync.lock().unwrap();
        let mut any_pending = false;
        pending.retain(|promise| match promise.poll() {
            PromiseState::Pending => {
                any_pending = true;
                true
            }
            PromiseState::Resolved(tv) => {
                sync.record_foreign_read(ctx, tv);
                false
            }
            PromiseState::Abandoned => false,
        });
        any_pending
    }

    /// Reuse-gate state for this stamp on `ctx`, including outstanding timeline promises.
    pub(crate) fn settle_on_context(&self, ctx: &Context) -> Settle {
        let ctx_handle = ctx.backend_handle();
        if self.lazy_gc_pending(ctx_handle) {
            return Settle::Pending;
        }
        let merged = self.sync.lock().unwrap().merged();
        if merged.is_empty() {
            return Settle::Ready;
        }
        let device = ctx.device();
        let mut waiting = None;
        for (c, tv) in merged.iter() {
            let progress = device
                .context_gpu_progress(c)
                .unwrap_or(crate::timeline::CONTEXT_DESTROYED_PROGRESS);
            if progress < tv {
                waiting = Some(waiting.map_or(tv, |w: TimelineValue| w.max(tv)));
            }
        }
        match waiting {
            Some(tv) => Settle::Waiting(tv),
            None => Settle::Ready,
        }
    }
}

/// Backing storage for a bindable [`Parcel`].
enum ParcelBacking {
    WholeBuffer(Arc<Allocation>),

    /// BufferRange is a sub-region of a partitioned buffer.
    ///
    /// This is an internal Goldy type. The public API is [`RetainedPool::acquire_record`]
    /// with [`ordinal`] / [`field`] descriptors; the resulting [`Buffer`] yields
    /// `BufferRange`-backed parcels via [`Buffer::unit`] / [`Buffer::field`].
    /// [`Parcel::from_buffer_range`] is intentionally `pub(crate)`.
    BufferRange {
        view: BufferView,
        parent: BufferHandle,
        parent_backing: Arc<Allocation>,
        offset: u64,
        len: u64,
    },
    Texture(TextureBacking),
}

/// A bindable unit of GPU property: whole buffer, buffer range, or texture.
///
/// Bind parcels to dispatches and render passes. Acquire [`Buffer`] from the retained pool
/// and index into it to obtain range parcels, or dereference a single-unit buffer.
pub struct Parcel {
    backing: ParcelBacking,
    stamp: ParcelStamp,
    /// Present only for texture parcels acquired directly (not via [`Buffer`]).
    bookkeeping: Option<BookkeepingGuard>,
}

impl Clone for Parcel {
    fn clone(&self) -> Self {
        Self {
            backing: match &self.backing {
                ParcelBacking::WholeBuffer(b) => ParcelBacking::WholeBuffer(Arc::clone(b)),
                ParcelBacking::BufferRange {
                    view,
                    parent,
                    parent_backing,
                    offset,
                    len,
                } => ParcelBacking::BufferRange {
                    view: view.clone(),
                    parent: *parent,
                    parent_backing: Arc::clone(parent_backing),
                    offset: *offset,
                    len: *len,
                },
                ParcelBacking::Texture(t) => ParcelBacking::Texture(t.clone()),
            },
            stamp: self.stamp.clone_shared_cells(),
            bookkeeping: None,
        }
    }
}

impl Parcel {
    pub(crate) fn from_whole_buffer(allocation: Arc<Allocation>, home_device: Weak<DeviceInner>) -> Self {
        Self {
            backing: ParcelBacking::WholeBuffer(allocation),
            stamp: ParcelStamp::new(home_device),
            bookkeeping: None,
        }
    }

    pub(crate) fn from_buffer_range(
        view: BufferView,
        parent: BufferHandle,
        parent_backing: Arc<Allocation>,
        offset: u64,
        len: u64,
        home_device: Weak<DeviceInner>,
    ) -> Self {
        Self {
            backing: ParcelBacking::BufferRange {
                view,
                parent,
                parent_backing,
                offset,
                len,
            },
            stamp: ParcelStamp::new(home_device),
            bookkeeping: None,
        }
    }

    pub(crate) fn from_texture(tex: TextureBacking, home_device: Weak<DeviceInner>) -> Self {
        Self {
            backing: ParcelBacking::Texture(tex),
            stamp: ParcelStamp::new(home_device),
            bookkeeping: None,
        }
    }

    /// Clone this parcel's stamp cells onto a new texture backing (non-owning views).
    pub(crate) fn clone_stamp_with_texture(&self, tex: TextureBacking) -> Self {
        Self {
            backing: ParcelBacking::Texture(tex),
            stamp: self.stamp.clone_shared_cells(),
            bookkeeping: None,
        }
    }

    /// Zoning / telemetry label (buffer vs texture).
    pub fn kind(&self) -> ParcelType {
        match &self.backing {
            ParcelBacking::WholeBuffer(_) | ParcelBacking::BufferRange { .. } => ParcelType::Buffer,
            ParcelBacking::Texture(_) => ParcelType::Texture,
        }
    }

    /// Approximate committed byte size for this bindable unit.
    pub fn byte_size(&self) -> u64 {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => b.byte_size(),
            ParcelBacking::BufferRange { len, .. } => *len,
            ParcelBacking::Texture(t) => t.byte_size() as u64,
        }
    }

    /// Resource descriptor index for how this parcel will be accessed in the current dispatch.
    pub fn resource_index(&self, access: ResourceAccess) -> Option<u32> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => b.resource_index(access),
            ParcelBacking::BufferRange { view, .. } => view.resource_index(access),
            ParcelBacking::Texture(t) => t.resource_index(access),
        }
    }

    /// Typed resource descriptor handle for validation and dispatch wiring.
    pub fn handle(&self, access: ResourceAccess) -> Option<ResourceHandle> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => b.handle(access),
            ParcelBacking::BufferRange { view, .. } => view.handle(access),
            ParcelBacking::Texture(t) => t.handle(access),
        }
    }

    /// Record the timeline of the most recent GPU work that referenced this parcel on `ctx`.
    pub fn mark_referenced(&self, ctx: ContextHandle, epoch: TimelineValue) {
        self.stamp
            .sync
            .lock()
            .unwrap()
            .record_write(ctx, epoch, WRITE_KINDS_TRANSFER);
    }

    /// Context-qualified last-referencing timelines for this parcel.
    pub fn last_referenced(&self) -> ReferenceTable {
        self.stamp.merged_references()
    }

    /// Last referencing timeline for a single context, if any.
    pub fn last_referenced_on(&self, ctx: ContextHandle) -> Option<TimelineValue> {
        self.stamp.merged_references().get(ctx)
    }

    /// Reuse-gate state for this parcel on `ctx`, including outstanding timeline promises.
    pub fn settle(&self, ctx: &Context) -> Settle {
        self.stamp.settle_on_context(ctx)
    }

    /// True when no in-flight GPU work on `ctx` still references this parcel.
    pub fn is_settled(&self, ctx: &Context) -> bool {
        matches!(self.settle(ctx), Settle::Ready)
    }

    /// Shared stamp cell updated by [`crate::Scheme`] at submit.
    pub(crate) fn stamp_handle(&self) -> Arc<ParcelStamp> {
        Arc::new(self.stamp.clone_shared_cells())
    }

    pub(crate) fn home_device(&self) -> &Weak<DeviceInner> {
        &self.stamp.home_device
    }

    /// Backend buffer handle and graph resource id for [`crate::Scheme::write_parcel`].
    pub(crate) fn write_buffer_target(&self) -> anyhow::Result<(BufferHandle, ResourceId)> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => {
                let h = b.gpu_buffer_handle();
                Ok((h, ResourceId::Buffer(h)))
            }
            ParcelBacking::BufferRange { .. } | ParcelBacking::Texture(_) => {
                anyhow::bail!("write_parcel is only valid for whole-buffer parcels")
            }
        }
    }

    /// Task-graph resource identity (runtime only).
    pub(crate) fn resource_id(&self) -> ResourceId {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => ResourceId::Buffer(b.gpu_buffer_handle()),
            ParcelBacking::BufferRange {
                parent, offset, len, ..
            } => ResourceId::BufferRange {
                parent: *parent,
                offset: *offset,
                len: *len,
            },
            ParcelBacking::Texture(t) => ResourceId::Texture(t.gpu_handle()),
        }
    }

    pub(crate) fn texture_handle(&self) -> Option<crate::backend::TextureHandle> {
        match &self.backing {
            ParcelBacking::Texture(t) => Some(t.gpu_handle()),
            _ => None,
        }
    }

    /// Whether this texture parcel owns its GPU resource (see [`TextureBacking::is_owned`]).
    ///
    /// Must not clone the backing: an owning clone's drop would destroy the live texture.
    pub(crate) fn texture_is_owned(&self) -> bool {
        match &self.backing {
            ParcelBacking::Texture(t) => t.is_owned(),
            _ => false,
        }
    }

    pub(crate) fn buffer_handle(&self) -> Option<BufferHandle> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => Some(b.gpu_buffer_handle()),
            ParcelBacking::BufferRange { parent, .. } => Some(*parent),
            _ => None,
        }
    }

    /// Host write into a whole-buffer parcel (used by scheme upload staging).
    pub(crate) fn write_bytes(&self, offset: u64, data: &[u8]) -> anyhow::Result<()> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => b.write(offset, data),
            ParcelBacking::BufferRange { .. } => {
                anyhow::bail!("write_bytes requires a whole-buffer parcel")
            }
            ParcelBacking::Texture(_) => anyhow::bail!("write_bytes requires a buffer parcel"),
        }
    }

    /// Structured-buffer element stride for whole-buffer parcels, if set at allocation.
    pub(crate) fn buffer_element_stride(&self) -> Option<u32> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => b.element_stride(),
            ParcelBacking::BufferRange { .. } => None,
            ParcelBacking::Texture(_) => None,
        }
    }

    pub(crate) fn grant_buffer_keepalive(&self) -> Result<Arc<Allocation>, anyhow::Error> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => Ok(Arc::clone(b)),
            ParcelBacking::BufferRange { parent_backing, .. } => Ok(Arc::clone(parent_backing)),
            ParcelBacking::Texture(_) => anyhow::bail!("grant_read requires buffer parcel"),
        }
    }

    pub(crate) fn grant_texture_keepalive(&self) -> Result<TextureBacking, anyhow::Error> {
        match &self.backing {
            // Never clone an owning backing: Drop on the clone would destroy the live GPU texture.
            ParcelBacking::Texture(t) => Ok(t.borrow()),
            _ => anyhow::bail!("grant_read_texture requires texture parcel"),
        }
    }

    pub(crate) fn release_bookkeeping(&mut self) {
        self.bookkeeping = None;
    }

    pub(crate) fn attach_bookkeeping(&mut self, guard: BookkeepingGuard) {
        self.bookkeeping = Some(guard);
    }

    /// Kind and flags for whole-buffer parcels; `None` for views and textures.
    ///
    /// Used by [`crate::transient_pool::TransientPool`] to key the buffer recycle bin on
    /// the full descriptor (not just size) so that buffers with different `BufferKind` or
    /// `BufferFlags` never alias.
    pub(crate) fn buffer_descriptor(&self) -> Option<(BufferKind, BufferFlags)> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => Some((b.access(), b.flags())),
            _ => None,
        }
    }

    pub(crate) fn texture_descriptor(
        &self,
    ) -> Option<(
        u32,
        u32,
        crate::types::TextureFormat,
        crate::types::TextureKind,
        crate::types::TextureFlags,
    )> {
        match &self.backing {
            ParcelBacking::Texture(t) => Some((t.width(), t.height(), t.format(), t.access(), t.flags())),
            _ => None,
        }
    }

    pub(crate) fn is_homed_on(&self, ctx: &Context) -> bool {
        self.stamp
            .home_device
            .upgrade()
            .is_some_and(|home| Arc::ptr_eq(&home, &ctx.device().inner))
    }

    /// Read this parcel's GPU bytes back to CPU memory (testing / diagnostics).
    pub fn read_to_cpu(&self, device: &crate::Device, output: &mut [u8]) -> anyhow::Result<()> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => b.read_to_cpu(device, output),
            ParcelBacking::BufferRange { view, .. } => view.read_to_cpu(device, output),
            ParcelBacking::Texture(_) => anyhow::bail!(
                "texture parcels are not host-readable; copy to a CPU_READABLE buffer parcel via a scheme first"
            ),
        }
    }
}

impl BufferSource for Parcel {
    fn source_handle(&self) -> BufferHandle {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => b.gpu_buffer_handle(),
            ParcelBacking::BufferRange { parent, .. } => *parent,
            ParcelBacking::Texture(_) => panic!("BufferSource is not implemented for texture parcels"),
        }
    }

    fn source_offset(&self) -> u64 {
        match &self.backing {
            ParcelBacking::WholeBuffer(_) => 0,
            ParcelBacking::BufferRange { offset, .. } => *offset,
            ParcelBacking::Texture(_) => 0,
        }
    }
}

/// Storage backing for an acquired [`Buffer`].
enum BufferStorage {
    Single(Arc<Allocation>),
    Partitioned {
        parent: Arc<Allocation>,
        field_names: Vec<Option<String>>,
    },
}

/// An acquired GPU buffer — possibly partitioned into independently bindable parcels.
///
/// Release by dropping or [`crate::retained_pool::RetainedPool::release_buffer`] /
/// [`crate::retained_pool::RetainedPool::release_texture`].
pub struct Buffer {
    storage: BufferStorage,
    units: Vec<Parcel>,
    bookkeeping: Option<BookkeepingGuard>,
    home_device: Weak<DeviceInner>,
}

impl Buffer {
    pub(crate) fn from_single(
        allocation: Allocation,
        bookkeeping: BookkeepingGuard,
        home_device: Weak<DeviceInner>,
    ) -> Self {
        let arc = Arc::new(allocation);
        let parcel = Parcel::from_whole_buffer(Arc::clone(&arc), home_device.clone());
        Self {
            storage: BufferStorage::Single(arc),
            units: vec![parcel],
            bookkeeping: Some(bookkeeping),
            home_device,
        }
    }

    pub(crate) fn from_partitioned(
        parent: Arc<Allocation>,
        views: Vec<BufferView>,
        field_names: Vec<Option<String>>,
        bookkeeping: BookkeepingGuard,
        home_device: Weak<DeviceInner>,
    ) -> Self {
        let backing = parent.gpu_buffer_handle();
        let units = views
            .into_iter()
            .map(|view| {
                let offset = view.offset();
                let len = view.size();
                Parcel::from_buffer_range(view, backing, Arc::clone(&parent), offset, len, home_device.clone())
            })
            .collect();
        Self {
            storage: BufferStorage::Partitioned { parent, field_names },
            units,
            bookkeeping: Some(bookkeeping),
            home_device,
        }
    }

    /// Number of bindable parcels in this buffer.
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// True when this buffer has more than one bindable parcel.
    pub fn is_partitioned(&self) -> bool {
        self.units.len() > 1
    }

    /// Obtain a bindable parcel by ordinal index.
    pub fn unit(&self, index: usize) -> &Parcel {
        &self.units[index]
    }

    /// Obtain a bindable parcel by field name.
    pub fn field(&self, name: &str) -> &Parcel {
        let idx = self
            .field_index(name)
            .unwrap_or_else(|| panic!("Buffer::field: unknown field {name:?}"));
        &self.units[idx]
    }

    fn field_index(&self, name: &str) -> Option<usize> {
        match &self.storage {
            BufferStorage::Single(_) => None,
            BufferStorage::Partitioned { field_names, .. } => {
                field_names.iter().position(|n| n.as_deref() == Some(name))
            }
        }
    }

    /// The whole-buffer parcel. Panics on partitioned buffers (bind individual units instead).
    pub fn whole(&self) -> &Parcel {
        assert!(
            !self.is_partitioned(),
            "Buffer::whole: cannot bind a partitioned buffer as one descriptor; bind individual parcels via indexing"
        );
        &self.units[0]
    }

    /// Zoning / telemetry label.
    pub fn kind(&self) -> ParcelType {
        ParcelType::Buffer
    }

    /// Total committed byte size (backing allocation).
    pub fn byte_size(&self) -> u64 {
        match &self.storage {
            BufferStorage::Single(b) => b.byte_size(),
            BufferStorage::Partitioned { parent, .. } => parent.byte_size(),
        }
    }

    /// Context-qualified last-referencing timelines merged across all parcels.
    pub fn last_referenced(&self) -> ReferenceTable {
        let mut merged = ReferenceTable::new();
        for unit in &self.units {
            for (ctx, tv) in unit.last_referenced().iter() {
                crate::timeline::mark_reference(&mut merged, ctx, tv);
            }
        }
        merged
    }

    pub fn is_settled(&self, ctx: &Context) -> bool {
        self.units.iter().all(|u| u.is_settled(ctx))
    }

    /// CPU write into a single-unit buffer.
    ///
    /// For [`crate::types::BufferFlags::CPU_WRITABLE`] buffers, this is a host-mapped
    /// memcpy (Metal/Vulkan) or a write into the paired UPLOAD mapping (DX12). It is
    /// **not** serialized behind in-flight GPU work: the caller must only write when the
    /// buffer is **settled** ([`Self::is_settled`] / host-observed progress past last use)
    /// or **fresh** (never GPU-referenced). Writing while the GPU still reads the buffer
    /// is a data race on Metal/Vulkan; on DX12 the staged bytes apply at the next
    /// `CopyBuffer` instead.
    ///
    /// Prefer [`crate::Scheme::stage_upload_buffer`] / epoch-gated staging pools, which
    /// select settled or newly allocated parcels. GPU visibility of a staging write is
    /// covered by same-frame scheme copy tests (e.g. `scheme_cpu_writable_staging_write_then_copy`),
    /// not by CPU→CPU `read_to_cpu` roundtrips alone.
    ///
    /// For other flags, backends may use a queue-ordered path (e.g. Metal blit+wait for
    /// non-`CPU_WRITABLE` Shared buffers).
    pub fn write(&self, offset: u64, data: &[u8]) -> anyhow::Result<()> {
        match &self.storage {
            BufferStorage::Single(b) => b.write(offset, data),
            BufferStorage::Partitioned { .. } => {
                anyhow::bail!("Buffer::write requires a single-unit buffer; write to a specific parcel instead")
            }
        }
    }

    /// Read the whole-buffer parcel back to CPU memory.
    pub fn read_to_cpu(&self, device: &crate::Device, output: &mut [u8]) -> anyhow::Result<()> {
        self.whole().read_to_cpu(device, output)
    }

    /// GPU clear on a single-unit buffer (see [`Self::clear`]).
    pub fn clear(&self, device: &crate::Device, offset: u64, size: u64) -> anyhow::Result<()> {
        match &self.storage {
            BufferStorage::Single(b) => b.clear(device, offset, size),
            BufferStorage::Partitioned { .. } => {
                anyhow::bail!("Buffer::clear requires a single-unit buffer; clear a specific parcel instead")
            }
        }
    }

    /// Record GPU reference on the whole-buffer parcel (single-unit buffers only).
    pub fn mark_referenced(&self, ctx: ContextHandle, epoch: TimelineValue) {
        self.whole().mark_referenced(ctx, epoch);
    }

    pub(crate) fn home_device(&self) -> &Weak<DeviceInner> {
        &self.home_device
    }

    pub(crate) fn release_bookkeeping(&mut self) {
        self.bookkeeping = None;
    }

    /// Peel a single-unit buffer into a transient-pool parcel (no bookkeeping).
    pub(crate) fn into_transient_parcel(mut self) -> anyhow::Result<Parcel> {
        if self.is_partitioned() {
            anyhow::bail!("into_transient_parcel requires a single-unit buffer");
        }
        self.release_bookkeeping();
        match self.storage {
            BufferStorage::Single(arc) => Ok(Parcel::from_whole_buffer(arc, self.home_device)),
            BufferStorage::Partitioned { .. } => unreachable!(),
        }
    }

    /// Iterate all bindable parcels for dependency registration.
    pub fn parcels(&self) -> impl Iterator<Item = &Parcel> {
        self.units.iter()
    }

    /// Extract the backing allocation from a single-unit buffer.
    #[allow(dead_code)] // retained-pool migration tests
    pub(crate) fn detach_allocation(mut self) -> anyhow::Result<Allocation> {
        self.release_bookkeeping();
        match self.storage {
            BufferStorage::Single(b) => {
                // Drop the whole-buffer parcel before unwrapping; it holds a second Arc ref.
                self.units.clear();
                Arc::try_unwrap(b)
                    .map_err(|_| anyhow::anyhow!("detach_allocation requires sole ownership of the backing allocation"))
            }
            BufferStorage::Partitioned { .. } => {
                anyhow::bail!("detach_allocation requires a single-unit buffer")
            }
        }
    }

    /// Backing allocation handle for whole-buffer operations (clear, write_parcel).
    pub(crate) fn backing_handle(&self) -> BufferHandle {
        match &self.storage {
            BufferStorage::Single(b) => b.gpu_buffer_handle(),
            BufferStorage::Partitioned { parent, .. } => parent.gpu_buffer_handle(),
        }
    }

    /// Creation flags for the backing allocation (e.g. [`crate::types::BufferFlags::CPU_READABLE`]).
    pub fn flags(&self) -> BufferFlags {
        match &self.storage {
            BufferStorage::Single(b) => b.flags(),
            BufferStorage::Partitioned { parent, .. } => parent.flags(),
        }
    }
}

/// An acquired GPU texture — one bindable parcel.
///
/// Release by dropping or [`crate::retained_pool::RetainedPool::release_texture`].
pub struct Texture {
    parcel: Parcel,
    bookkeeping: Option<BookkeepingGuard>,
    home_device: Weak<DeviceInner>,
}

impl Clone for Texture {
    fn clone(&self) -> Self {
        Self {
            parcel: self.parcel.clone(),
            bookkeeping: None,
            home_device: self.home_device.clone(),
        }
    }
}

impl Texture {
    pub(crate) fn from_backing(
        backing: TextureBacking,
        bookkeeping: BookkeepingGuard,
        home_device: Weak<DeviceInner>,
    ) -> Self {
        Self {
            parcel: Parcel::from_texture(backing, home_device.clone()),
            bookkeeping: Some(bookkeeping),
            home_device,
        }
    }

    pub(crate) fn from_parcel(parcel: Parcel, bookkeeping: BookkeepingGuard, home_device: Weak<DeviceInner>) -> Self {
        Self {
            parcel,
            bookkeeping: Some(bookkeeping),
            home_device,
        }
    }

    pub(crate) fn from_returned_parcel(parcel: Parcel, home_device: Weak<DeviceInner>) -> Self {
        Self {
            parcel,
            bookkeeping: None,
            home_device,
        }
    }

    pub(crate) fn from_borrowed_backing(backing: TextureBacking, home_device: Weak<DeviceInner>) -> Self {
        Self {
            parcel: Parcel::from_texture(backing, home_device.clone()),
            bookkeeping: None,
            home_device,
        }
    }

    /// The bindable parcel (same as `&*self`).
    pub fn whole(&self) -> &Parcel {
        &self.parcel
    }

    pub fn width(&self) -> u32 {
        self.parcel.texture_descriptor().expect("texture parcel").0
    }

    pub fn height(&self) -> u32 {
        self.parcel.texture_descriptor().expect("texture parcel").1
    }

    pub fn format(&self) -> crate::types::TextureFormat {
        self.parcel.texture_descriptor().expect("texture parcel").2
    }

    pub fn access(&self) -> crate::types::TextureKind {
        self.parcel.texture_descriptor().expect("texture parcel").3
    }

    pub fn flags(&self) -> crate::types::TextureFlags {
        self.parcel.texture_descriptor().expect("texture parcel").4
    }

    /// Buffer footprint for copying this texture into a destination buffer parcel.
    ///
    /// Requires [`TextureFlags::COPY_SRC`]. Swapchain drawables and other non-copyable
    /// textures do not satisfy that requirement and this method panics.
    pub fn copy_layout(&self) -> crate::backend::TextureCopyFootprint {
        if !self.flags().contains(TextureFlags::COPY_SRC) {
            panic!(
                "Texture::copy_layout requires TextureFlags::COPY_SRC; \
                 swapchain drawables and other non-copyable textures cannot be copied from"
            );
        }
        let home = self
            .home_device
            .upgrade()
            .expect("Texture::copy_layout: home device dropped");
        let backend = home.backend.lock().unwrap();
        backend
            .query_texture_copy_footprint(home.handle, self.width(), self.height(), self.format())
            .unwrap_or_else(|e| panic!("Texture::copy_layout backend query failed: {e}"))
    }

    pub fn gpu_handle(&self) -> crate::backend::TextureHandle {
        self.parcel.texture_handle().expect("texture parcel")
    }

    pub fn resource_index(&self, access: ResourceAccess) -> Option<u32> {
        self.parcel.resource_index(access)
    }

    pub fn handle(&self, access: ResourceAccess) -> Option<ResourceHandle> {
        self.parcel.handle(access)
    }

    pub fn byte_size(&self) -> u64 {
        self.parcel.byte_size()
    }

    pub fn is_owned(&self) -> bool {
        self.parcel.texture_is_owned()
    }

    pub fn kind(&self) -> ParcelType {
        ParcelType::Texture
    }

    pub fn last_referenced(&self) -> ReferenceTable {
        self.parcel.last_referenced()
    }

    pub fn is_settled(&self, ctx: &Context) -> bool {
        self.parcel.is_settled(ctx)
    }

    pub fn mark_referenced(&self, ctx: ContextHandle, epoch: TimelineValue) {
        self.parcel.mark_referenced(ctx, epoch);
    }

    pub fn set_debug_name(&self, name: &str) {
        self.parcel
            .grant_texture_keepalive()
            .expect("texture parcel")
            .set_debug_name(name);
    }

    /// Non-owning view sharing this texture's parcel stamp.
    pub fn borrow(&self) -> Self {
        let backing = self.parcel.grant_texture_keepalive().expect("texture parcel").borrow();
        Self {
            parcel: self.parcel.clone_stamp_with_texture(backing),
            bookkeeping: None,
            home_device: self.home_device.clone(),
        }
    }

    /// Wrap an externally-owned GPU texture (e.g. swapchain drawable).
    pub(crate) fn borrowed(
        device: &crate::device::Device,
        backend: Arc<Mutex<Box<dyn crate::backend::GpuBackend>>>,
        handle: crate::backend::TextureHandle,
        width: u32,
        height: u32,
        format: crate::types::TextureFormat,
    ) -> Self {
        Self::from_borrowed_backing(
            TextureBacking::borrowed(backend, handle, width, height, format),
            Arc::downgrade(&device.inner),
        )
    }

    #[deprecated(
        since = "0.1.0",
        note = "Use Scheme::write_texture_region() for batched, non-blocking uploads. \
                This method submits synchronously and stalls the GPU."
    )]
    #[allow(deprecated)]
    pub fn write_region(&self, x: u32, y: u32, width: u32, height: u32, data: &[u8]) -> anyhow::Result<()> {
        self.parcel
            .grant_texture_keepalive()?
            .write_region(x, y, width, height, data)
    }

    #[deprecated(
        since = "0.1.0",
        note = "Use Scheme::write_texture() for batched, non-blocking uploads. \
                This method submits synchronously and stalls the GPU."
    )]
    #[allow(deprecated)]
    pub fn write(&self, data: &[u8]) -> anyhow::Result<()> {
        self.parcel.grant_texture_keepalive()?.write(data)
    }

    #[deprecated(
        note = "Copy to a CPU_READABLE buffer parcel via a scheme, submit, wait the timeline, then read the buffer"
    )]
    #[allow(deprecated)]
    pub fn read_to_cpu(&self, output: &mut [u8]) -> anyhow::Result<()> {
        self.parcel.grant_texture_keepalive()?.read_to_cpu(output)
    }

    pub(crate) fn release_bookkeeping(&mut self) {
        self.bookkeeping = None;
        self.parcel.release_bookkeeping();
    }

    pub(crate) fn home_device(&self) -> &Weak<DeviceInner> {
        &self.home_device
    }

    pub(crate) fn into_lease_parcel(mut self) -> Parcel {
        if let Some(guard) = self.bookkeeping.take() {
            self.parcel.attach_bookkeeping(guard);
        }
        self.parcel
    }

    pub(crate) fn into_parcel(mut self) -> Parcel {
        self.release_bookkeeping();
        self.parcel
    }
}

impl Deref for Texture {
    type Target = Parcel;

    fn deref(&self) -> &Self::Target {
        &self.parcel
    }
}

impl Deref for Buffer {
    type Target = Parcel;

    fn deref(&self) -> &Self::Target {
        self.whole()
    }
}

impl Index<usize> for Buffer {
    type Output = Parcel;

    fn index(&self, index: usize) -> &Self::Output {
        &self.units[index]
    }
}

impl Index<&str> for Buffer {
    type Output = Parcel;

    fn index(&self, name: &str) -> &Self::Output {
        self.field(name)
    }
}

/// Initial contents for one field of an [`Buffer`] record.
pub enum Init {
    Data { bytes: Vec<u8>, count: u64, stride: u32 },
    Reserve { count: u64, stride: u32 },
}

impl Init {
    /// Upload a typed slice at acquisition (copied; not aliased).
    pub fn data<T: StructuredBufferElement>(data: &[T]) -> Self {
        Self::Data {
            bytes: bytemuck::cast_slice(data).to_vec(),
            count: data.len() as u64,
            stride: std::mem::size_of::<T>() as u32,
        }
    }

    /// Reserve uninitialized space for `count` elements of type `T`.
    pub fn reserve<T: StructuredBufferElement>(count: u64) -> Self {
        Self::Reserve {
            count,
            stride: std::mem::size_of::<T>() as u32,
        }
    }

    /// Reserve zero-initialized space for `count` elements of type `T`.
    pub fn zeros<T: StructuredBufferElement>(count: u64) -> Self {
        let stride = std::mem::size_of::<T>() as u32;
        let bytes_len = count.saturating_mul(stride as u64);
        let bytes = vec![0u8; bytes_len as usize];
        Self::Data { bytes, count, stride }
    }
}

/// One field specification for [`crate::RetainedPool::acquire_record`].
pub struct RecordField {
    pub name: Option<Cow<'static, str>>,
    pub init: Init,
}

/// Define a named field for record acquisition.
pub fn field(name: impl Into<Cow<'static, str>>, init: Init) -> RecordField {
    RecordField {
        name: Some(name.into()),
        init,
    }
}

/// Define an anonymous ordinal field for record acquisition.
pub fn ordinal(init: Init) -> RecordField {
    RecordField { name: None, init }
}

impl BufferSource for Buffer {
    fn source_handle(&self) -> BufferHandle {
        self.whole().source_handle()
    }

    fn source_offset(&self) -> u64 {
        self.whole().source_offset()
    }
}

/// Per-kind byte totals for resources currently held through a [`crate::retained_pool::RetainedPool`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BytesByKind {
    pub buffer: u64,
    pub texture: u64,
}

/// Internal counters shared with [`BookkeepingGuard`].
pub(crate) struct PoolBookkeeping {
    buffer_bytes: AtomicU64,
    texture_bytes: AtomicU64,
}

impl PoolBookkeeping {
    pub(crate) fn new() -> Self {
        Self {
            buffer_bytes: AtomicU64::new(0),
            texture_bytes: AtomicU64::new(0),
        }
    }

    pub(crate) fn add(&self, kind: ParcelType, bytes: u64) {
        match kind {
            ParcelType::Buffer => {
                self.buffer_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
            ParcelType::Texture => {
                self.texture_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
        }
    }

    pub(crate) fn subtract(&self, kind: ParcelType, bytes: u64) {
        match kind {
            ParcelType::Buffer => {
                self.buffer_bytes.fetch_sub(bytes, Ordering::Relaxed);
            }
            ParcelType::Texture => {
                self.texture_bytes.fetch_sub(bytes, Ordering::Relaxed);
            }
        }
    }

    pub(crate) fn snapshot(&self) -> BytesByKind {
        BytesByKind {
            buffer: self.buffer_bytes.load(Ordering::Relaxed),
            texture: self.texture_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Decrements retained-pool byte counters when the resource is dropped without `transfer_out`.
#[derive(Debug)]
pub(crate) struct BookkeepingGuard {
    pool: Weak<PoolBookkeeping>,
    kind: ParcelType,
    bytes: u64,
}

impl BookkeepingGuard {
    pub(crate) fn new(pool: Weak<PoolBookkeeping>, kind: ParcelType, bytes: u64) -> Self {
        Self { pool, kind, bytes }
    }
}

impl Drop for BookkeepingGuard {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            pool.subtract(self.kind, self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::device::Device;
    use crate::retained_pool::RetainedPool;
    use crate::timeline::{PromiseState, Settle, TimelinePromise};
    use std::sync::Arc;

    fn mock_device() -> Arc<Device> {
        Arc::new(Device::from_backend(Box::new(MockBackend::new())).expect("mock device"))
    }

    fn mock_parcel(_device: &Arc<Device>, pool: &mut RetainedPool) -> Parcel {
        let buffer = pool
            .acquire_record([field("x", Init::zeros::<u32>(4))])
            .expect("buffer");
        buffer.whole().clone()
    }

    #[test]
    fn settle_ready_when_never_referenced() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = mock_parcel(&device, &mut pool);
        assert_eq!(parcel.settle(&ctx), Settle::Ready);
        assert!(parcel.is_settled(&ctx));
    }

    #[test]
    fn settle_pending_with_unresolved_promise() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = mock_parcel(&device, &mut pool);
        let (promise, _resolver) = TimelinePromise::new();
        parcel.stamp_handle().push_pending(promise);
        assert_eq!(parcel.settle(&ctx), Settle::Pending);
        assert!(!parcel.is_settled(&ctx));
    }

    #[test]
    fn settle_waiting_when_epoch_unreached() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = mock_parcel(&device, &mut pool);
        parcel.mark_referenced(ctx.backend_handle(), 50);
        assert_eq!(parcel.settle(&ctx), Settle::Waiting(50));
        assert!(!parcel.is_settled(&ctx));
    }

    #[test]
    fn settle_lazy_gc_folds_resolved_promise_into_foreign_reads() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let ctx_handle = ctx.backend_handle();
        let parcel = mock_parcel(&device, &mut pool);
        let stamp = parcel.stamp_handle();
        let (promise, resolver) = TimelinePromise::new();
        stamp.push_pending(promise);
        resolver.resolve(25);
        assert_eq!(parcel.settle(&ctx), Settle::Waiting(25));
        assert!(stamp.pending.lock().unwrap().is_empty());
        let sync = stamp.sync.lock().unwrap();
        assert_eq!(sync.foreign_reads.get(ctx_handle), Some(25));
        assert!(sync.last_reads.is_empty());
    }

    #[test]
    fn settle_lazy_gc_drops_abandoned_promise() {
        let device = mock_device();
        let mut pool = RetainedPool::new(device.clone());
        let ctx = device.create_context().unwrap();
        let parcel = mock_parcel(&device, &mut pool);
        let stamp = parcel.stamp_handle();
        let (promise, resolver) = TimelinePromise::new();
        stamp.push_pending(promise);
        drop(resolver);
        assert_eq!(promise_state_after_abandon(&stamp), PromiseState::Abandoned);
        assert_eq!(parcel.settle(&ctx), Settle::Ready);
        assert!(stamp.pending.lock().unwrap().is_empty());
    }

    fn promise_state_after_abandon(stamp: &Arc<ParcelStamp>) -> PromiseState {
        stamp.pending.lock().unwrap()[0].poll()
    }
}
