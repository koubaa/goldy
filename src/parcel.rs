//! Retained GPU property: [`Buffer`] (acquired aggregate) and [`Parcel`] (bindable unit).
//!
//! Acquire a [`Buffer`] from [`crate::retained_pool::RetainedPool`]; bind a [`Parcel`] to
//! dispatches and render passes. Each parcel is independently dependency-tracked.

use crate::backend::{BufferHandle, ContextHandle};
use crate::buffer::{Allocation, BufferPool, BufferSource, BufferView, StructuredBufferElement};
use crate::context::Context;
use crate::device::DeviceInner;
use crate::task_graph::ResourceId;
use crate::texture::Texture;
use crate::timeline::{ReferenceTable, ResourceSync, TimelineValue, WRITE_KINDS_TRANSFER};
use crate::types::{ResourceAccess, ResourceHandle};
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

/// Shared stamp cell and home-device identity for [`TaskGraph`] submit stamping.
pub(crate) struct ParcelStamp {
    pub(crate) sync: Arc<Mutex<ResourceSync>>,
    pub(crate) interaction_set: Arc<Mutex<InteractionSet>>,
    pub(crate) home_device: Weak<DeviceInner>,
}

impl ParcelStamp {
    pub(crate) fn new(home_device: Weak<DeviceInner>) -> Self {
        Self {
            sync: Arc::new(Mutex::new(ResourceSync::default())),
            interaction_set: Arc::new(Mutex::new(Vec::new())),
            home_device,
        }
    }

    pub(crate) fn merged_references(&self) -> ReferenceTable {
        self.sync.lock().unwrap().merged()
    }
}

/// Backing storage for a bindable [`Parcel`].
enum ParcelBacking {
    WholeBuffer(Arc<Allocation>),
    BufferRange {
        view: BufferView,
        parent: BufferHandle,
        offset: u64,
        len: u64,
    },
    Texture(Texture),
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
                    offset,
                    len,
                } => ParcelBacking::BufferRange {
                    view: view.clone(),
                    parent: *parent,
                    offset: *offset,
                    len: *len,
                },
                ParcelBacking::Texture(t) => ParcelBacking::Texture(t.clone()),
            },
            stamp: ParcelStamp {
                sync: Arc::clone(&self.stamp.sync),
                interaction_set: Arc::clone(&self.stamp.interaction_set),
                home_device: self.stamp.home_device.clone(),
            },
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
        offset: u64,
        len: u64,
        home_device: Weak<DeviceInner>,
    ) -> Self {
        Self {
            backing: ParcelBacking::BufferRange {
                view,
                parent,
                offset,
                len,
            },
            stamp: ParcelStamp::new(home_device),
            bookkeeping: None,
        }
    }

    pub(crate) fn from_texture(tex: Texture, bookkeeping: BookkeepingGuard, home_device: Weak<DeviceInner>) -> Self {
        Self {
            backing: ParcelBacking::Texture(tex),
            stamp: ParcelStamp::new(home_device),
            bookkeeping: Some(bookkeeping),
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
        self.stamp.merged_references().get(&ctx).copied()
    }

    /// True when no in-flight GPU work on `ctx` still references this parcel.
    pub fn is_settled(&self, ctx: &Context) -> bool {
        ctx.parcel_ready(&self.last_referenced())
    }

    /// Shared stamp cell updated by [`crate::TaskGraph`] at submit.
    pub(crate) fn stamp_handle(&self) -> Arc<ParcelStamp> {
        Arc::new(ParcelStamp {
            sync: Arc::clone(&self.stamp.sync),
            interaction_set: Arc::clone(&self.stamp.interaction_set),
            home_device: self.stamp.home_device.clone(),
        })
    }

    pub(crate) fn home_device(&self) -> &Weak<DeviceInner> {
        &self.stamp.home_device
    }

    /// Backend buffer handle and graph resource id for [`crate::TaskGraph::write_parcel`].
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

    pub(crate) fn buffer_handle(&self) -> Option<BufferHandle> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => Some(b.gpu_buffer_handle()),
            ParcelBacking::BufferRange { parent, .. } => Some(*parent),
            _ => None,
        }
    }

    pub(crate) fn grant_buffer_keepalive(&self) -> Result<Arc<Allocation>, anyhow::Error> {
        match &self.backing {
            ParcelBacking::WholeBuffer(b) => Ok(Arc::clone(b)),
            ParcelBacking::BufferRange { .. } => anyhow::bail!("grant_read requires whole-buffer parcel"),
            ParcelBacking::Texture(_) => anyhow::bail!("grant_read requires buffer parcel"),
        }
    }

    pub(crate) fn grant_texture_keepalive(&self) -> Result<Texture, anyhow::Error> {
        match &self.backing {
            ParcelBacking::Texture(t) => Ok(t.clone()),
            _ => anyhow::bail!("grant_read_texture requires texture parcel"),
        }
    }

    pub(crate) fn release_bookkeeping(&mut self) {
        self.bookkeeping = None;
    }

    pub(crate) fn attach_bookkeeping(&mut self, guard: BookkeepingGuard) {
        self.bookkeeping = Some(guard);
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
            ParcelBacking::Texture(t) => t.read_to_cpu(output),
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
        pool: BufferPool,
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
        pool: BufferPool,
        views: Vec<BufferView>,
        field_names: Vec<Option<String>>,
        bookkeeping: BookkeepingGuard,
        home_device: Weak<DeviceInner>,
    ) -> Self {
        let backing = pool.backing_buffer().gpu_buffer_handle();
        let units = views
            .into_iter()
            .map(|view| {
                let offset = view.offset();
                let len = view.size();
                Parcel::from_buffer_range(view, backing, offset, len, home_device.clone())
            })
            .collect();
        Self {
            storage: BufferStorage::Partitioned { pool, field_names },
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
            BufferStorage::Partitioned { pool, .. } => pool.capacity(),
        }
    }

    /// Context-qualified last-referencing timelines merged across all parcels.
    pub fn last_referenced(&self) -> ReferenceTable {
        let mut merged = ReferenceTable::new();
        for unit in &self.units {
            for (ctx, tv) in unit.last_referenced() {
                merged
                    .entry(ctx)
                    .and_modify(|e| {
                        if tv > *e {
                            *e = tv;
                        }
                    })
                    .or_insert(tv);
            }
        }
        merged
    }

    pub fn is_settled(&self, ctx: &Context) -> bool {
        self.units.iter().all(|u| u.is_settled(ctx))
    }

    /// CPU write into a single-unit buffer (host-visible when [`crate::types::BufferFlags::CPU_READABLE`]).
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
            BufferStorage::Partitioned { pool, .. } => pool.backing_buffer().gpu_buffer_handle(),
        }
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
        let bytes = vec![0u8; (count * stride as u64) as usize];
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
