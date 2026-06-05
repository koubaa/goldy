//! Opaque retained parcel — the shared unit moved between retained and transient containers.
//!
//! The client holds a [`Parcel`] by value and never sees backend resource handles.
//! Bindless indices for shader binding are exposed via [`Parcel::bindless_index`];
//! the runtime uses [`Parcel::resource_id`] when wiring [`TaskGraph`] nodes.

use crate::buffer::{Buffer, BufferPool, BufferView};
use crate::task_graph::ResourceId;
use crate::texture::Texture;
use crate::timeline::TimelineValue;
use crate::types::BindlessHandle;
use crate::vram_allocator::ParcelKind;
use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Weak;

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

    pub(crate) fn add(&self, kind: ParcelKind, bytes: u64) {
        match kind {
            ParcelKind::Buffer => {
                self.buffer_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
            ParcelKind::Texture => {
                self.texture_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
        }
    }

    pub(crate) fn subtract(&self, kind: ParcelKind, bytes: u64) {
        match kind {
            ParcelKind::Buffer => {
                self.buffer_bytes.fetch_sub(bytes, Ordering::Relaxed);
            }
            ParcelKind::Texture => {
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
    kind: ParcelKind,
    bytes: u64,
}

impl BookkeepingGuard {
    pub(crate) fn new(pool: Weak<PoolBookkeeping>, kind: ParcelKind, bytes: u64) -> Self {
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
    last_referenced: Option<TimelineValue>,
    bookkeeping: Option<BookkeepingGuard>,
}

impl Parcel {
    pub(crate) fn from_buffer(buf: Buffer, bookkeeping: BookkeepingGuard) -> Self {
        Self {
            storage: ParcelStorage::Buffer(buf),
            last_referenced: None,
            bookkeeping: Some(bookkeeping),
        }
    }

    pub(crate) fn from_texture(tex: Texture, bookkeeping: BookkeepingGuard) -> Self {
        Self {
            storage: ParcelStorage::Texture(tex),
            last_referenced: None,
            bookkeeping: Some(bookkeeping),
        }
    }

    pub(crate) fn from_mosaic(pool: BufferPool, views: Vec<BufferView>, bookkeeping: BookkeepingGuard) -> Self {
        Self {
            storage: ParcelStorage::Mosaic(Mosaic { pool, views }),
            last_referenced: None,
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
    pub fn kind(&self) -> ParcelKind {
        match &self.storage {
            ParcelStorage::Buffer(_) | ParcelStorage::Mosaic(_) => ParcelKind::Buffer,
            ParcelStorage::Texture(_) => ParcelKind::Texture,
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

    /// Bindless descriptor index for push constants / resource slots (not the backend handle).
    pub fn bindless_index(&self) -> Option<u32> {
        match &self.storage {
            ParcelStorage::Buffer(b) => b.bindless_index(),
            ParcelStorage::Texture(t) => t.bindless_index(),
            ParcelStorage::Mosaic(_) => None,
        }
    }

    /// Typed bindless handle for validation and dispatch wiring.
    pub fn bindless_handle(&self) -> Option<BindlessHandle> {
        match &self.storage {
            ParcelStorage::Buffer(b) => b.bindless_handle(),
            ParcelStorage::Texture(t) => t.bindless_handle(),
            ParcelStorage::Mosaic(_) => None,
        }
    }

    /// Record the timeline of the most recent GPU work that referenced this parcel.
    ///
    /// Monotonic: only increases; a smaller epoch is ignored.
    pub fn mark_referenced(&mut self, epoch: TimelineValue) {
        match self.last_referenced {
            Some(prev) if epoch <= prev => {}
            _ => self.last_referenced = Some(epoch),
        }
    }

    /// Last recorded referencing timeline, if any.
    pub fn last_referenced(&self) -> Option<TimelineValue> {
        self.last_referenced
    }

    /// Task-graph resource identity (runtime only).
    pub(crate) fn resource_id(&self) -> ResourceId {
        match &self.storage {
            ParcelStorage::Buffer(b) => ResourceId::Buffer(b.gpu_buffer_handle()),
            ParcelStorage::Texture(t) => ResourceId::Texture(t.handle()),
            ParcelStorage::Mosaic(m) => {
                ResourceId::Buffer(m.pool.backing_buffer().gpu_buffer_handle())
            }
        }
    }

    /// Backend texture handle (runtime only; for tests comparing identity across transfer).
    #[cfg(test)]
    pub(crate) fn texture_handle(&self) -> Option<crate::backend::TextureHandle> {
        match &self.storage {
            ParcelStorage::Texture(t) => Some(t.handle()),
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

    // TODO(Unit 2+): copy_into(&self, data) for host-visible parcels.
    // Strict contract: errors unless the parcel was acquired host-visible.
    // Not needed in Unit 1 — level textures take their bytes via construction-time `init`.
    #[allow(dead_code)]
    fn copy_into_stub(&self, _data: &[u8]) -> Result<()> {
        anyhow::bail!(
            "copy_into is not implemented yet; use construction-time init on acquire_*"
        )
    }
}
