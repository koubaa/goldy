//! Opaque retained parcel — the shared unit moved between retained and transient containers.
//!
//! The client holds a [`Parcel`] by value and never sees backend resource handles.
//! Resource indices for shader binding are exposed via [`Parcel::resource_index`];
//! the runtime uses internal resource IDs when wiring [`crate::TaskGraph`] nodes.

use crate::buffer::{Buffer, BufferPool, BufferView};
use crate::task_graph::ResourceId;
use crate::texture::Texture;
use crate::timeline::TimelineValue;
use crate::types::{ResourceAccess, ResourceHandle};
use crate::vram_allocator::ParcelType;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

/// Index into a [`Parcel`] mosaic's sub-ranges (returned by [`crate::retained_pool::MosaicBuilder`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MosaicSlot(pub u32);

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
/// Relinquish by [`crate::retained_pool::RetainedPool::transfer_out`] (stamped for the
/// transient pool) or by dropping the value (implicit relinquish).
pub struct Parcel {
    storage: ParcelStorage,
    /// Last referencing timeline; `0` means never referenced by GPU work.
    last_referenced: Arc<AtomicU64>,
    bookkeeping: Option<BookkeepingGuard>,
}

impl Parcel {
    pub(crate) fn from_buffer(buf: Buffer, bookkeeping: BookkeepingGuard) -> Self {
        Self {
            storage: ParcelStorage::Buffer(buf),
            last_referenced: Arc::new(AtomicU64::new(0)),
            bookkeeping: Some(bookkeeping),
        }
    }

    pub(crate) fn from_texture(tex: Texture, bookkeeping: BookkeepingGuard) -> Self {
        Self {
            storage: ParcelStorage::Texture(tex),
            last_referenced: Arc::new(AtomicU64::new(0)),
            bookkeeping: Some(bookkeeping),
        }
    }

    pub(crate) fn from_mosaic(pool: BufferPool, views: Vec<BufferView>, bookkeeping: BookkeepingGuard) -> Self {
        Self {
            storage: ParcelStorage::Mosaic(Mosaic { pool, views }),
            last_referenced: Arc::new(AtomicU64::new(0)),
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

    /// Overwrite a buffer parcel's contents in place.
    ///
    /// Observationally async: the backend may service this as a staged GPU copy (storage) or
    /// a direct mapped write (uniform). We never expose write completion and do not support
    /// readback on this path, so the synchronous case is an invisible special case of the
    /// async contract. A future depth>1 consumer that needs ordering should add a GPU-side
    /// queue wait (threading the device internally via the owned [`Buffer`]), not a
    /// client-facing CPU gate.
    ///
    /// Valid only for non-mosaic buffer parcels acquired via [`crate::RetainedPool::acquire_buffer`].
    pub fn copy_into<T: bytemuck::Pod>(&self, data: &[T]) -> anyhow::Result<()> {
        match &self.storage {
            ParcelStorage::Buffer(b) => b.write_data(0, data),
            ParcelStorage::Texture(_) | ParcelStorage::Mosaic(_) => {
                anyhow::bail!("Parcel::copy_into is only valid for non-mosaic buffer parcels")
            }
        }
    }

    /// Record the timeline of the most recent GPU work that referenced this parcel.
    ///
    /// Monotonic: only increases; a smaller epoch is ignored.
    pub fn mark_referenced(&self, epoch: TimelineValue) {
        self.last_referenced.fetch_max(epoch, Ordering::Relaxed);
    }

    /// Last recorded referencing timeline, if any.
    pub fn last_referenced(&self) -> Option<TimelineValue> {
        let v = self.last_referenced.load(Ordering::Relaxed);
        if v == 0 {
            None
        } else {
            Some(v)
        }
    }

    /// Shared stamp cell updated by [`crate::TaskGraph`] at submit.
    pub(crate) fn stamp_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.last_referenced)
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

    /// Release pool bookkeeping so [`Drop`] does not double-decrement after `transfer_out`.
    pub(crate) fn release_bookkeeping(&mut self) {
        self.bookkeeping = None;
    }
}
