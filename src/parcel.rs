//! Opaque retained parcel — the shared unit moved between retained and transient containers.
//!
//! The client holds a [`Parcel`] by value and never sees backend resource handles.
//! Resource indices for shader binding are exposed via [`Parcel::resource_index`];
//! the runtime uses internal resource IDs when wiring [`crate::TaskGraph`] nodes.

use crate::backend::{BufferHandle, ContextHandle};
use crate::buffer::{Buffer, BufferPool, BufferSource, BufferView};
use crate::context::Context;
use crate::device::DeviceInner;
use crate::task_graph::ResourceId;
use crate::texture::Texture;
use crate::timeline::{mark_reference, ReferenceTable, TimelineValue};
use crate::types::{ResourceAccess, ResourceHandle};
use crate::vram_allocator::ParcelType;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Index into a [`Parcel`] mosaic's sub-ranges (returned by [`crate::retained_pool::MosaicBuilder`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MosaicSlot(pub u32);

/// Shared stamp cell and home-device identity for [`TaskGraph`] submit stamping.
pub(crate) struct ParcelStamp {
    pub(crate) references: Arc<Mutex<ReferenceTable>>,
    pub(crate) home_device: Weak<DeviceInner>,
}

impl ParcelStamp {
    pub(crate) fn new(home_device: Weak<DeviceInner>) -> Self {
        Self {
            references: Arc::new(Mutex::new(ReferenceTable::new())),
            home_device,
        }
    }
}

/// One backing buffer subdivided into multiple bindless sub-ranges (lifetime-homogeneous).
struct Mosaic {
    pool: BufferPool,
    views: Vec<BufferView>,
}

/// Storage variant for a retained parcel (not exposed to clients).
enum ParcelStorage {
    Buffer(Arc<Buffer>),
    Texture(Texture),
    Mosaic(Mosaic),
}

/// Per-kind byte totals for parcels currently held through a [`crate::retained_pool::RetainedPool`].
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

/// Decrements retained-pool byte counters when the parcel is dropped without `transfer_out`.
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

/// A deed-held GPU parcel (buffer or texture). Gate-free while retained.
///
/// Release by [`crate::retained_pool::RetainedPool::release`] (runtime reclaims when safe) or by dropping the parcel.
pub struct Parcel {
    storage: ParcelStorage,
    stamp: ParcelStamp,
    bookkeeping: Option<BookkeepingGuard>,
}

impl Parcel {
    pub(crate) fn from_buffer(buf: Buffer, bookkeeping: BookkeepingGuard, home_device: Weak<DeviceInner>) -> Self {
        Self {
            storage: ParcelStorage::Buffer(Arc::new(buf)),
            stamp: ParcelStamp::new(home_device),
            bookkeeping: Some(bookkeeping),
        }
    }

    pub(crate) fn from_texture(tex: Texture, bookkeeping: BookkeepingGuard, home_device: Weak<DeviceInner>) -> Self {
        Self {
            storage: ParcelStorage::Texture(tex),
            stamp: ParcelStamp::new(home_device),
            bookkeeping: Some(bookkeeping),
        }
    }

    pub(crate) fn from_mosaic(
        pool: BufferPool,
        views: Vec<BufferView>,
        bookkeeping: BookkeepingGuard,
        home_device: Weak<DeviceInner>,
    ) -> Self {
        Self {
            storage: ParcelStorage::Mosaic(Mosaic { pool, views }),
            stamp: ParcelStamp::new(home_device),
            bookkeeping: Some(bookkeeping),
        }
    }

    /// Sub-range of a mosaic parcel for vertex/index binding (`BufferSource`).
    ///
    /// Panics if `slot` is out of range or the parcel is not a mosaic.
    pub fn view(&self, slot: MosaicSlot) -> &BufferView {
        match &self.storage {
            ParcelStorage::Mosaic(m) => &m.views[slot.0 as usize],
            _ => panic!("Parcel::view called on non-mosaic parcel"),
        }
    }

    /// Zoning / telemetry label (buffer vs texture).
    pub fn kind(&self) -> ParcelType {
        match &self.storage {
            ParcelStorage::Buffer(_) | ParcelStorage::Mosaic(_) => ParcelType::Buffer,
            ParcelStorage::Texture(_) => ParcelType::Texture,
        }
    }

    /// Approximate committed byte size for accounting.
    pub fn byte_size(&self) -> u64 {
        match &self.storage {
            ParcelStorage::Buffer(b) => b.byte_size(),
            ParcelStorage::Texture(t) => t.byte_size() as u64,
            ParcelStorage::Mosaic(m) => m.pool.capacity(),
        }
    }

    /// Resource descriptor index for how this parcel will be accessed in the current dispatch.
    ///
    /// Mosaic parcels always return `None` — bind per-view via [`Parcel::view`].
    pub fn resource_index(&self, access: ResourceAccess) -> Option<u32> {
        match &self.storage {
            ParcelStorage::Buffer(b) => b.resource_index(access),
            ParcelStorage::Texture(t) => t.resource_index(access),
            ParcelStorage::Mosaic(_) => None,
        }
    }

    /// Typed resource descriptor handle for validation and dispatch wiring.
    pub fn handle(&self, access: ResourceAccess) -> Option<ResourceHandle> {
        match &self.storage {
            ParcelStorage::Buffer(b) => b.handle(access),
            ParcelStorage::Texture(t) => t.handle(access),
            ParcelStorage::Mosaic(_) => None,
        }
    }

    /// Record the timeline of the most recent GPU work that referenced this parcel on `ctx`.
    ///
    /// Monotonic per context: only increases; a smaller epoch is ignored.
    pub fn mark_referenced(&self, ctx: ContextHandle, epoch: TimelineValue) {
        let mut table = self.stamp.references.lock().unwrap();
        mark_reference(&mut table, ctx, epoch);
    }

    /// Context-qualified last-referencing timelines.
    pub fn last_referenced(&self) -> ReferenceTable {
        self.stamp.references.lock().unwrap().clone()
    }

    /// Last referencing timeline for a single context, if any.
    pub fn last_referenced_on(&self, ctx: ContextHandle) -> Option<TimelineValue> {
        self.stamp.references.lock().unwrap().get(&ctx).copied()
    }

    /// True when no in-flight GPU work on `ctx` still references this parcel.
    ///
    /// A settled parcel is immediately reusable. No epoch or timeline value is exposed;
    /// prefer this over [`Self::last_referenced`] for currency checks. See design §4.1.
    pub fn is_settled(&self, ctx: &Context) -> bool {
        ctx.parcel_ready(&self.last_referenced())
    }

    /// Shared stamp cell updated by [`crate::TaskGraph`] at submit.
    pub(crate) fn stamp_handle(&self) -> Arc<ParcelStamp> {
        Arc::new(ParcelStamp {
            references: Arc::clone(&self.stamp.references),
            home_device: self.stamp.home_device.clone(),
        })
    }

    pub(crate) fn home_device(&self) -> &Weak<DeviceInner> {
        &self.stamp.home_device
    }

    /// Backend buffer handle and graph resource id for [`crate::TaskGraph::write_parcel`].
    ///
    /// Valid only for non-mosaic buffer parcels acquired via
    /// [`crate::RetainedPool::acquire_buffer`].
    pub(crate) fn write_buffer_target(&self) -> anyhow::Result<(BufferHandle, ResourceId)> {
        match &self.storage {
            ParcelStorage::Buffer(b) => {
                let h = b.gpu_buffer_handle();
                Ok((h, ResourceId::Buffer(h)))
            }
            ParcelStorage::Texture(_) | ParcelStorage::Mosaic(_) => {
                anyhow::bail!("write_parcel is only valid for non-mosaic buffer parcels")
            }
        }
    }

    /// Task-graph resource identity (runtime only).
    pub(crate) fn resource_id(&self) -> ResourceId {
        match &self.storage {
            ParcelStorage::Buffer(b) => ResourceId::Buffer(b.gpu_buffer_handle()),
            ParcelStorage::Texture(t) => ResourceId::Texture(t.gpu_handle()),
            ParcelStorage::Mosaic(m) => ResourceId::Buffer(m.pool.backing_buffer().gpu_buffer_handle()),
        }
    }

    /// Backend texture handle (runtime only).
    pub(crate) fn texture_handle(&self) -> Option<crate::backend::TextureHandle> {
        match &self.storage {
            ParcelStorage::Texture(t) => Some(t.gpu_handle()),
            _ => None,
        }
    }

    /// Backend buffer handle for grant-read staging copies.
    pub(crate) fn buffer_handle(&self) -> Option<crate::backend::BufferHandle> {
        match &self.storage {
            ParcelStorage::Buffer(b) => Some(b.gpu_buffer_handle()),
            ParcelStorage::Mosaic(m) => Some(m.pool.backing_buffer().gpu_buffer_handle()),
            _ => None,
        }
    }

    /// Clone the backing [`Buffer`] keepalive for a grant-read source parcel.
    pub(crate) fn grant_buffer_keepalive(&self) -> Result<std::sync::Arc<crate::Buffer>, anyhow::Error> {
        match &self.storage {
            ParcelStorage::Buffer(b) => Ok(std::sync::Arc::clone(b)),
            ParcelStorage::Texture(_) => anyhow::bail!("grant_read requires buffer parcel"),
            ParcelStorage::Mosaic(_) => anyhow::bail!("grant_read requires non-mosaic buffer parcel"),
        }
    }

    /// Clone the backing [`Texture`] keepalive for a grant-read source parcel.
    pub(crate) fn grant_texture_keepalive(&self) -> Result<crate::Texture, anyhow::Error> {
        match &self.storage {
            ParcelStorage::Texture(t) => Ok(t.clone()),
            ParcelStorage::Buffer(_) => anyhow::bail!("grant_read_texture requires texture parcel"),
            ParcelStorage::Mosaic(_) => anyhow::bail!("grant_read_texture requires texture parcel"),
        }
    }

    /// Release pool bookkeeping so [`Drop`] does not double-decrement after [`RetainedPool::release`].
    pub(crate) fn release_bookkeeping(&mut self) {
        self.bookkeeping = None;
    }

    /// Attach fresh pool bookkeeping (transient-pool reuse path).
    ///
    /// Replaces any existing guard; used when a recycled parcel is re-issued and
    /// must track bytes against a fresh outstanding counter.
    pub(crate) fn attach_bookkeeping(&mut self, guard: BookkeepingGuard) {
        self.bookkeeping = Some(guard);
    }

    /// Texture allocation descriptor, if this parcel holds a texture.
    ///
    /// Used by [`crate::transient_pool::TransientPool`] to key recycle bins.
    pub(crate) fn texture_descriptor(
        &self,
    ) -> Option<(
        u32,
        u32,
        crate::types::TextureFormat,
        crate::types::TextureKind,
        crate::types::TextureFlags,
    )> {
        match &self.storage {
            ParcelStorage::Texture(t) => Some((t.width(), t.height(), t.format(), t.access(), t.flags())),
            _ => None,
        }
    }

    /// Extract the backing buffer from a non-mosaic buffer parcel.
    ///
    /// Consumes the parcel and releases retained-pool bookkeeping. The returned
    /// [`Buffer`] is independently owned (ekrano scratch pools and similar escape hatches).
    pub fn detach_buffer(mut self) -> anyhow::Result<Buffer> {
        self.release_bookkeeping();
        match self.storage {
            ParcelStorage::Buffer(b) => Arc::try_unwrap(b).map_err(|_| {
                anyhow::anyhow!("detach_buffer requires sole ownership of the backing buffer (outstanding read grant?)")
            }),
            ParcelStorage::Texture(_) | ParcelStorage::Mosaic(_) => {
                anyhow::bail!("detach_buffer requires a non-mosaic buffer parcel")
            }
        }
    }

    /// True when `ctx` was created on the same [`crate::Device`] as this parcel.
    pub(crate) fn is_homed_on(&self, ctx: &Context) -> bool {
        self.stamp
            .home_device
            .upgrade()
            .is_some_and(|home| Arc::ptr_eq(&home, &ctx.device().inner))
    }

    /// Extract the backing texture from a texture parcel.
    ///
    /// Consumes the parcel and releases retained-pool bookkeeping.
    pub fn detach_texture(mut self) -> anyhow::Result<Texture> {
        self.release_bookkeeping();
        match self.storage {
            ParcelStorage::Texture(t) => Ok(t),
            ParcelStorage::Buffer(_) | ParcelStorage::Mosaic(_) => {
                anyhow::bail!("detach_texture requires a texture parcel")
            }
        }
    }

    /// Bindless resource index for one mosaic sub-view.
    pub fn mosaic_view_resource_index(&self, slot: MosaicSlot, access: ResourceAccess) -> Option<u32> {
        self.view(slot).resource_index(access)
    }

    /// Read one mosaic sub-view back to CPU memory.
    pub fn mosaic_view_read_to_cpu(
        &self,
        device: &crate::Device,
        slot: MosaicSlot,
        output: &mut [u8],
    ) -> anyhow::Result<()> {
        self.view(slot).read_to_cpu(device, output)
    }
}

impl BufferSource for Parcel {
    fn source_handle(&self) -> BufferHandle {
        match &self.storage {
            ParcelStorage::Buffer(b) => b.gpu_buffer_handle(),
            ParcelStorage::Mosaic(_) => {
                panic!("use Parcel::view for mosaic vertex/index binding")
            }
            ParcelStorage::Texture(_) => {
                panic!("BufferSource is not implemented for texture parcels")
            }
        }
    }

    fn source_offset(&self) -> u64 {
        0
    }
}
