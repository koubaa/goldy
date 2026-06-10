//! Opaque retained parcel — the shared unit moved between retained and transient containers.
//!
//! The client holds a [`Parcel`] by value and never sees backend resource handles.
//! Resource indices for shader binding are exposed via [`Parcel::resource_index`];
//! the runtime uses internal resource IDs when wiring [`crate::TaskGraph`] nodes.

use crate::backend::{BufferHandle, ContextHandle};
use crate::buffer::{Buffer, BufferPool, BufferSource, BufferView};
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
    Buffer(Buffer),
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
            storage: ParcelStorage::Buffer(buf),
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

    /// Read buffer parcel contents back to CPU memory.
    ///
    /// Valid only for non-mosaic buffer parcels acquired via [`crate::RetainedPool::acquire_buffer`].
    pub fn read_to_cpu(&self, device: &crate::Device, output: &mut [u8]) -> anyhow::Result<()> {
        match &self.storage {
            ParcelStorage::Buffer(b) => b.read_to_cpu(device, output),
            ParcelStorage::Texture(_) | ParcelStorage::Mosaic(_) => {
                anyhow::bail!("read_to_cpu is only valid for non-mosaic buffer parcels")
            }
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

    /// Backend texture handle (runtime only; for tests comparing identity across transfer).
    #[cfg(test)]
    pub(crate) fn texture_handle(&self) -> Option<crate::backend::TextureHandle> {
        match &self.storage {
            ParcelStorage::Texture(t) => Some(t.gpu_handle()),
            _ => None,
        }
    }

    /// Backend buffer handle (runtime only; for tests and transfer identity checks).
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn buffer_handle(&self) -> Option<crate::backend::BufferHandle> {
        match &self.storage {
            ParcelStorage::Buffer(b) => Some(b.gpu_buffer_handle()),
            ParcelStorage::Mosaic(m) => Some(m.pool.backing_buffer().gpu_buffer_handle()),
            _ => None,
        }
    }

    /// Release pool bookkeeping so [`Drop`] does not double-decrement after [`RetainedPool::release`].
    pub(crate) fn release_bookkeeping(&mut self) {
        self.bookkeeping = None;
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
