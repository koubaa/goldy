//! Epoch-gated transient parcel pool — recycled GPU memory without a client-visible clock.
//!
//! There is one transient pool per context; this type is the engine.
//!
//! Relinquished resources enter as stamped parcels and are handed out again by lease
//! realization **only once every stamped epoch has retired**. Clients never compare
//! timeline values; the pool consumes `ready_after` internally through a progress
//! snapshot ([`Context::snapshot_gpu_progress`] / [`Context::parcel_ready`]).

use crate::context::Context;
use crate::parcel::{BookkeepingGuard, BytesByKind, Parcel, PoolBookkeeping, Texture};
use crate::retained_pool::{RetainedHold, StampedParcel};
use crate::timeline::{is_ready, ReferenceTable, TimelineValue};
use crate::types::{BufferFlags, BufferKind, TextureFlags, TextureFormat, TextureKind};
use crate::vram_allocator::ParcelType;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

/// Maximum number of ready (epoch-retired) entries to keep per buffer bin.
///
/// At N=1, depth=1 (the only proven config), one warm spare is enough to avoid
/// re-allocating on immediate reuse. Excess ready entries are dropped on
/// each frame-boundary [`Context::boundary_crossed`] → [`Self::drain_ready`] call.
const MAX_BUFFER_BIN_READY_SPARES: usize = 1;

/// Maximum number of ready (epoch-retired) entries to keep per texture bin.
///
/// Same policy as [`MAX_BUFFER_BIN_READY_SPARES`]. Dropping *all* ready textures on
/// every `drain_ready` (the old behavior) forced a fresh `alloc_texture` on the next
/// frame whenever the GPU had already retired the prior scratch — common on Metal
/// for small filter frames — and broke filter-scratch plateau tests.
const MAX_TEXTURE_BIN_READY_SPARES: usize = 1;

/// Recycle-bin key: parcels are interchangeable iff their allocation descriptors match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TextureKey {
    width: u32,
    height: u32,
    format: TextureFormat,
    access: TextureKind,
    flags: TextureFlags,
}

/// A parked texture resource awaiting epoch retirement.
///
/// Ready entries are kept as warm spares up to [`MAX_TEXTURE_BIN_READY_SPARES`] and
/// reissued by [`TransientPool::acquire_texture`]; excess ready entries are trimmed on
/// [`Self::drain_ready`].
struct TexturePendingEntry {
    parcel: Parcel,
    ready_after: ReferenceTable,
}

/// Recycle-bin key for buffer parcels: interchangeable iff size, kind, flags, and stride match.
///
/// Keying on size alone would allow an adopted non-Scattered buffer (from
/// [`crate::retained_pool::RetainedPool::release_buffer`]) to be handed out to a
/// [`TransientPool::acquire_buffer`] caller that expects a specific kind — which would produce
/// wrong descriptor categories or silent garbage in the shader. Stride is included so
/// scratch buffers with different structured strides never alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferKey {
    size: u64,
    kind: BufferKind,
    flags: BufferFlags,
    element_stride: Option<u32>,
}

impl BufferKey {
    fn from_parcel(parcel: &Parcel) -> Self {
        let (kind, flags) = parcel
            .buffer_descriptor()
            .expect("BufferKey::from_parcel requires a whole-buffer parcel");
        Self {
            size: parcel.byte_size(),
            kind,
            flags,
            element_stride: parcel.buffer_element_stride(),
        }
    }
}

/// A parked buffer parcel awaiting epoch retirement; reissued by [`TransientPool::acquire_buffer`].
struct BufferBinEntry {
    parcel: Parcel,
    ready_after: ReferenceTable,
}

/// Pending (not yet retired) + ready (warm spare) lists for one recycle-bin key.
///
/// Already-ready entries never re-query GPU progress; pending entries are promoted
/// with a caller-supplied progress snapshot (once per acquire/drain, not per parcel).
struct ResourceBin<E> {
    pending: Vec<E>,
    ready: Vec<E>,
}

impl<E> ResourceBin<E> {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            ready: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.ready.is_empty()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.pending.len() + self.ready.len()
    }
}

impl<E> Default for ResourceBin<E> {
    fn default() -> Self {
        Self::new()
    }
}

/// Epoch-gated recycling pool for transient parcels.
pub(crate) struct TransientPool {
    /// Bytes parked in recycle bins.
    pending: Arc<PoolBookkeeping>,
    /// Bytes held by clients through this pool (guard-decremented on drop).
    outstanding: Arc<PoolBookkeeping>,
    texture_bins: HashMap<TextureKey, ResourceBin<TexturePendingEntry>>,
    /// Buffer parcels keyed by `(size, kind, flags, stride)`; excess ready entries are trimmed by
    /// [`Self::drain_ready`] (see [`MAX_BUFFER_BIN_READY_SPARES`]).
    buffer_bins: HashMap<BufferKey, ResourceBin<BufferBinEntry>>,
    /// Monotonic count of fresh `alloc_buffer` calls made by [`Self::acquire_buffer`].
    ///
    /// Does **not** increment when a retired bin entry is reused. Exposed via
    /// [`crate::Context::transient_buffer_alloc_count`] for tests that verify the recycling
    /// path fires (alloc count stays flat across a reuse cycle).
    buffer_alloc_count: usize,
    /// Monotonic count of fresh `alloc_texture` calls made by [`Self::acquire_texture`].
    texture_alloc_count: usize,
}

impl TransientPool {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(PoolBookkeeping::new()),
            outstanding: Arc::new(PoolBookkeeping::new()),
            texture_bins: HashMap::new(),
            buffer_bins: HashMap::new(),
            buffer_alloc_count: 0,
            texture_alloc_count: 0,
        }
    }

    pub fn acquire_texture(
        &mut self,
        ctx: &Context,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<Texture> {
        let home_device = Arc::downgrade(&ctx.device().inner);
        let key = TextureKey {
            width,
            height,
            format,
            access,
            flags,
        };
        if let Some(bin) = self.texture_bins.get_mut(&key) {
            if let Some(entry) = take_ready_or_promote(
                bin,
                |e| &e.ready_after,
                |tables| ctx.snapshot_gpu_progress_for_tables(tables),
            ) {
                let bytes = entry.parcel.byte_size();
                self.pending.subtract(ParcelType::Texture, bytes);
                let guard = BookkeepingGuard::new(Arc::downgrade(&self.outstanding), ParcelType::Texture, bytes);
                self.outstanding.add(ParcelType::Texture, bytes);
                return Ok(Texture::from_parcel(entry.parcel, guard, home_device));
            }
        }

        let tex = ctx
            .device()
            .alloc_texture(width, height, format, access, flags)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.texture_alloc_count += 1;
        let bytes = tex.byte_size() as u64;
        self.outstanding.add(ParcelType::Texture, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.outstanding), ParcelType::Texture, bytes);
        Ok(Texture::from_backing(tex, guard, home_device))
    }

    /// Acquire a one-submission buffer lease backing parcel, reusing a retired bin entry when possible.
    ///
    /// `kind`, `flags`, and `element_stride` must match the values used to originally allocate any
    /// reused entry; they are also forwarded to the backend when a fresh allocation is needed.
    pub fn acquire_buffer(
        &mut self,
        ctx: &Context,
        size: u64,
        kind: BufferKind,
        flags: BufferFlags,
        element_stride: Option<u32>,
    ) -> Result<Parcel> {
        let key = BufferKey {
            size,
            kind,
            flags,
            element_stride,
        };
        if let Some(bin) = self.buffer_bins.get_mut(&key) {
            if let Some(entry) = take_ready_or_promote(
                bin,
                |e| &e.ready_after,
                |tables| ctx.snapshot_gpu_progress_for_tables(tables),
            ) {
                let bytes = entry.parcel.byte_size();
                self.pending.subtract(ParcelType::Buffer, bytes);
                let mut parcel = entry.parcel;
                parcel.attach_bookkeeping(BookkeepingGuard::new(
                    Arc::downgrade(&self.outstanding),
                    ParcelType::Buffer,
                    bytes,
                ));
                self.outstanding.add(ParcelType::Buffer, bytes);
                return Ok(parcel);
            }
        }

        let alloc = ctx
            .device()
            .alloc_buffer(size, kind, element_stride, flags)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.buffer_alloc_count += 1;
        let bytes = alloc.byte_size();
        self.outstanding.add(ParcelType::Buffer, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.outstanding), ParcelType::Buffer, bytes);
        let mut parcel = Parcel::from_whole_buffer(Arc::new(alloc), Arc::downgrade(&ctx.device().inner));
        parcel.attach_bookkeeping(guard);
        Ok(parcel)
    }

    /// Acquire a whole-buffer [`Buffer`] wrapper for clients that bind via [`Buffer`] (not raw [`Parcel`]).
    pub fn acquire_whole_buffer(
        &mut self,
        ctx: &Context,
        size: u64,
        kind: BufferKind,
        flags: BufferFlags,
        element_stride: Option<u32>,
    ) -> Result<crate::parcel::Buffer> {
        let home_device = Arc::downgrade(&ctx.device().inner);
        let parcel = self.acquire_buffer(ctx, size, kind, flags, element_stride)?;
        crate::parcel::Buffer::from_transient_parcel(parcel, home_device)
    }

    /// Return a scheme-held buffer lease parcel to the pool after its epoch retires.
    ///
    /// Called from [`crate::Scheme::drop`] for each buffer-backed lease. The parcel's
    /// bookkeeping must already be released by the caller.
    ///
    /// Retires the parcel stamp before parking: schemes that still bind the returned
    /// deed fail subsequent submit with [`crate::GoldyError::StaleResource`]. The parked
    /// entry gets a fresh stamp for the next acquire.
    pub(crate) fn return_buffer_parcel(&mut self, mut parcel: Parcel, ready_after: ReferenceTable) {
        parcel.retire_stamp_for_pool_return();
        let bytes = parcel.byte_size();
        let key = BufferKey::from_parcel(&parcel);
        self.pending.add(ParcelType::Buffer, bytes);
        let entry = BufferBinEntry { parcel, ready_after };
        let bin = self.buffer_bins.entry(key).or_default();
        park_entry(bin, entry, |e| &e.ready_after);
    }

    /// Return a scheme-held texture to the pool after its epoch retires.
    ///
    /// Called when filter-scratch (or other one-shot) textures are done for the frame.
    /// The texture's bookkeeping must still be attached; this method releases it.
    ///
    /// Retires the texture stamp before parking (same contract as
    /// [`Self::return_buffer_parcel`]): returning a bound transient must invalidate
    /// schemes that still reference it, on every backend.
    pub(crate) fn return_texture(&mut self, mut texture: Texture, ready_after: ReferenceTable) {
        texture.release_bookkeeping();
        let mut parcel = texture.into_parcel();
        parcel.retire_stamp_for_pool_return();
        let bytes = parcel.byte_size();
        let (width, height, format, access, flags) = parcel.texture_descriptor().expect("texture descriptor");
        let key = TextureKey {
            width,
            height,
            format,
            access,
            flags,
        };
        self.pending.add(ParcelType::Texture, bytes);
        let entry = TexturePendingEntry { parcel, ready_after };
        let bin = self.texture_bins.entry(key).or_default();
        park_entry(bin, entry, |e| &e.ready_after);
    }

    pub(crate) fn adopt(&mut self, stamped: StampedParcel) {
        let StampedParcel { hold, ready_after } = stamped;
        match hold {
            RetainedHold::Texture(texture) => {
                self.park_texture(texture, ready_after);
            }
            RetainedHold::Buffer(buffer) => {
                // Partitioned buffers (from `RetainedPool::acquire_record`) cannot be
                // reissued from the bin since the pool keys on single-parcel descriptors.
                // Drop them directly; the backend's deferred deletion queue provides the
                // same epoch-gated reclamation the bin would otherwise give.
                //
                // NOTE: as of this writing, no caller actually routes a partitioned
                // buffer through `park_buffer` → `adopt`. The composite
                // `cached_scheme_indirect` buffer is evicted via `drop(buf)` in
                // `alloc_or_reuse_scheme_indirect` and never touches this path. This
                // branch is defensive for future callers.
                let byte_size = buffer.byte_size();
                match buffer.into_transient_parcel() {
                    Ok(parcel) => {
                        // Stamp retirement happens inside return_buffer_parcel.
                        self.return_buffer_parcel(parcel, ready_after);
                    }
                    Err(_) => {
                        // Partitioned buffer — not binneable for reuse; drop it.
                        // `Buffer::drop` marks stamps dead and queues deferred destruction.
                        tracing::trace!(
                            byte_size,
                            "transient_pool: dropping partitioned buffer — \
                             epoch-gated reclamation via backend deletion queue",
                        );
                    }
                }
            }
        }
    }

    fn park_texture(&mut self, texture: Texture, ready_after: ReferenceTable) {
        self.return_texture(texture, ready_after);
    }

    /// Drop all parked textures (including not-yet-ready). Used when a Metal resize
    /// must free overflow-heap-backed textures immediately after waiting for GPU work.
    pub(crate) fn clear_textures(&mut self) {
        for bin in self.texture_bins.values() {
            for entry in bin.pending.iter().chain(bin.ready.iter()) {
                self.pending.subtract(ParcelType::Texture, entry.parcel.byte_size());
            }
        }
        self.texture_bins.clear();
    }

    /// Promote retired pending entries and drop excess ready spares.
    ///
    /// Snapshots GPU progress **once** for every context referenced by pending entries,
    /// then promotes/trims without further progress queries. Already-ready warm spares
    /// are never re-checked.
    pub fn drain_ready(&mut self, ctx: &Context) -> usize {
        let progress = ctx.snapshot_gpu_progress_for_tables(
            self.texture_bins
                .values()
                .flat_map(|bin| bin.pending.iter().map(|e| &e.ready_after))
                .chain(
                    self.buffer_bins
                        .values()
                        .flat_map(|bin| bin.pending.iter().map(|e| &e.ready_after)),
                ),
        );
        let mut released = self.trim_texture_bins(&progress);
        released += self.trim_buffer_bins(&progress);
        released
    }

    /// Drop excess epoch-retired texture bin entries beyond [`MAX_TEXTURE_BIN_READY_SPARES`].
    ///
    /// In-flight (not-ready) entries are never dropped — only ready spares above the cap.
    fn trim_texture_bins(&mut self, progress: &HashMap<crate::backend::ContextHandle, TimelineValue>) -> usize {
        let bookkeeping = Arc::clone(&self.pending);
        let mut trimmed = 0;
        for bin in self.texture_bins.values_mut() {
            promote_pending(bin, progress, |e| &e.ready_after);
            trimmed += trim_ready_spares(&mut bin.ready, MAX_TEXTURE_BIN_READY_SPARES, |entry| {
                bookkeeping.subtract(ParcelType::Texture, entry.parcel.byte_size());
            });
        }
        self.texture_bins.retain(|_, bin| !bin.is_empty());
        trimmed
    }

    /// Drop excess epoch-retired buffer bin entries beyond [`MAX_BUFFER_BIN_READY_SPARES`].
    ///
    /// In-flight (not-ready) entries are never dropped — only ready spares above the cap.
    /// Returns the number of entries dropped.
    fn trim_buffer_bins(&mut self, progress: &HashMap<crate::backend::ContextHandle, TimelineValue>) -> usize {
        let bookkeeping = Arc::clone(&self.pending);
        let mut trimmed = 0;
        for bin in self.buffer_bins.values_mut() {
            promote_pending(bin, progress, |e| &e.ready_after);
            trimmed += trim_ready_spares(&mut bin.ready, MAX_BUFFER_BIN_READY_SPARES, |entry| {
                bookkeeping.subtract(ParcelType::Buffer, entry.parcel.byte_size());
            });
        }
        self.buffer_bins.retain(|_, bin| !bin.is_empty());
        trimmed
    }

    #[cfg(test)]
    pub(crate) fn pending_bytes(&self) -> BytesByKind {
        self.pending.snapshot()
    }

    pub(crate) fn outstanding_bytes(&self) -> BytesByKind {
        self.outstanding.snapshot()
    }

    #[cfg(test)]
    pub(crate) fn pending_count(&self) -> usize {
        self.texture_bins.values().map(ResourceBin::len).sum::<usize>()
            + self.buffer_bins.values().map(ResourceBin::len).sum::<usize>()
    }

    /// Total number of fresh `alloc_buffer` calls made by [`Self::acquire_buffer`] since
    /// construction. Does not count bin reuses. Monotonically increasing.
    pub fn buffer_alloc_count(&self) -> usize {
        self.buffer_alloc_count
    }

    /// Total number of fresh `alloc_texture` calls made by [`Self::acquire_texture`] since
    /// construction. Does not count bin reuses. Monotonically increasing.
    pub fn texture_alloc_count(&self) -> usize {
        self.texture_alloc_count
    }
}

/// Park into `ready` when the epoch table is empty (immediately reusable); otherwise `pending`.
fn park_entry<E>(bin: &mut ResourceBin<E>, entry: E, ready_after: impl FnOnce(&E) -> &ReferenceTable) {
    if ready_after(&entry).is_empty() {
        bin.ready.push(entry);
    } else {
        bin.pending.push(entry);
    }
}

/// Take a warm spare, or promote at most one pending entry using a single progress snapshot.
fn take_ready_or_promote<E>(
    bin: &mut ResourceBin<E>,
    ready_after: impl Fn(&E) -> &ReferenceTable,
    snapshot: impl FnOnce(Vec<&ReferenceTable>) -> HashMap<crate::backend::ContextHandle, TimelineValue>,
) -> Option<E> {
    if !bin.ready.is_empty() {
        return Some(bin.ready.swap_remove(0));
    }
    if bin.pending.is_empty() {
        return None;
    }
    let progress = snapshot(bin.pending.iter().map(&ready_after).collect());
    if let Some(pos) = bin
        .pending
        .iter()
        .position(|e| is_ready(ready_after(e), &progress))
    {
        return Some(bin.pending.swap_remove(pos));
    }
    None
}

fn promote_pending<E>(
    bin: &mut ResourceBin<E>,
    progress: &HashMap<crate::backend::ContextHandle, TimelineValue>,
    ready_after: impl Fn(&E) -> &ReferenceTable,
) {
    let mut i = 0;
    while i < bin.pending.len() {
        if is_ready(ready_after(&bin.pending[i]), progress) {
            let entry = bin.pending.swap_remove(i);
            bin.ready.push(entry);
        } else {
            i += 1;
        }
    }
}

/// Keep the oldest `max_spares` ready entries; drop newest excess. Returns drop count.
fn trim_ready_spares<E>(ready: &mut Vec<E>, max_spares: usize, mut on_drop: impl FnMut(&E)) -> usize {
    let excess = ready.len().saturating_sub(max_spares);
    for _ in 0..excess {
        // Newest ready entries were pushed last; pop preserves the oldest warm spare(s).
        if let Some(entry) = ready.pop() {
            on_drop(&entry);
        }
    }
    excess
}

impl Default for TransientPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::device::Device;
    use crate::retained_pool::RetainedPool;

    fn test_device() -> Arc<Device> {
        Arc::new(Device::from_backend(Box::new(MockBackend::new())).expect("mock device"))
    }

    fn rgba_interpolated() -> (TextureFormat, TextureKind, TextureFlags) {
        (
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
        )
    }

    const TEST_BUFFER_SIZE: u64 = 64;
    const SCATTERED_EMPTY: (BufferKind, BufferFlags) = (BufferKind::Scattered, BufferFlags::empty());

    fn park_ready_buffer(ctx: &Context) {
        let alloc = ctx
            .device()
            .alloc_buffer(TEST_BUFFER_SIZE, SCATTERED_EMPTY.0, None, SCATTERED_EMPTY.1)
            .expect("alloc");
        let p = Parcel::from_whole_buffer(Arc::new(alloc), Arc::downgrade(&ctx.device().inner));
        ctx.with_transient_pool(|pool| pool.return_buffer_parcel(p, ReferenceTable::new()));
    }

    fn park_not_ready_buffer(ctx: &Context) {
        let mut ready_after = ReferenceTable::new();
        crate::timeline::mark_reference(&mut ready_after, ctx.test_backend_handle(), u64::MAX);
        let alloc = ctx
            .device()
            .alloc_buffer(TEST_BUFFER_SIZE, SCATTERED_EMPTY.0, None, SCATTERED_EMPTY.1)
            .expect("alloc");
        let p = Parcel::from_whole_buffer(Arc::new(alloc), Arc::downgrade(&ctx.device().inner));
        ctx.with_transient_pool(|pool| pool.return_buffer_parcel(p, ready_after));
    }

    fn park_ready_texture(ctx: &Context) {
        let (fmt, acc, flags) = rgba_interpolated();
        let tex = ctx.device().alloc_texture(8, 8, fmt, acc, flags).expect("alloc");
        let home = Arc::downgrade(&ctx.device().inner);
        let mut parcel = Parcel::from_texture(tex, home);
        parcel.retire_stamp_for_pool_return();
        ctx.with_transient_pool(|pool| {
            let bytes = parcel.byte_size();
            pool.pending.add(ParcelType::Texture, bytes);
            let key = TextureKey {
                width: 8,
                height: 8,
                format: fmt,
                access: acc,
                flags,
            };
            pool.texture_bins.entry(key).or_default().ready.push(TexturePendingEntry {
                parcel,
                ready_after: ReferenceTable::new(),
            });
        });
    }

    fn park_not_ready_texture(ctx: &Context) {
        let (fmt, acc, flags) = rgba_interpolated();
        let tex = ctx.device().alloc_texture(8, 8, fmt, acc, flags).expect("alloc");
        let home = Arc::downgrade(&ctx.device().inner);
        let mut parcel = Parcel::from_texture(tex, home);
        let mut ready_after = ReferenceTable::new();
        crate::timeline::mark_reference(&mut ready_after, ctx.test_backend_handle(), u64::MAX);
        parcel.retire_stamp_for_pool_return();
        ctx.with_transient_pool(|pool| {
            let bytes = parcel.byte_size();
            pool.pending.add(ParcelType::Texture, bytes);
            let key = TextureKey {
                width: 8,
                height: 8,
                format: fmt,
                access: acc,
                flags,
            };
            pool.texture_bins
                .entry(key)
                .or_default()
                .pending
                .push(TexturePendingEntry { parcel, ready_after });
        });
    }

    #[test]
    fn adopt_from_retained_pool_and_reuse() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut retained = RetainedPool::new(device.clone());
        let (fmt, acc, flags) = rgba_interpolated();
        let p = retained.acquire_texture(8, 8, fmt, acc, flags, None).unwrap();
        let handle_before = p.texture_handle().unwrap();

        retained.release_texture(&ctx, p);
        assert_eq!(retained.bytes_by_kind().texture, 0);
        assert_eq!(ctx.with_transient_pool(|t| t.pending_count()), 1);

        let p2 = ctx
            .with_transient_pool(|transient| transient.acquire_texture(&ctx, 8, 8, fmt, acc, flags))
            .unwrap();
        assert_eq!(p2.texture_handle(), Some(handle_before), "adopted parcel is reusable");
    }

    #[test]
    fn adopted_buffer_bins_and_reissues_via_acquire_buffer() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut retained = RetainedPool::new(device.clone());
        let b = retained
            .acquire_buffer(
                64,
                crate::types::BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
                None,
            )
            .unwrap();
        let handle_before = b.whole().buffer_handle().unwrap();
        retained.release_buffer(&ctx, b);
        assert_eq!(ctx.with_transient_pool(|t| t.pending_count()), 1);
        assert!(ctx.with_transient_pool(|t| t.pending_bytes().buffer >= 64));

        let released = ctx.with_transient_pool(|t| t.drain_ready(&ctx));
        assert_eq!(
            released, 0,
            "1 ready entry is within the cap (<= MAX_BUFFER_BIN_READY_SPARES); nothing trimmed"
        );

        let p = ctx
            .with_transient_pool(|pool| {
                pool.acquire_buffer(
                    &ctx,
                    64,
                    crate::types::BufferKind::Scattered,
                    crate::types::BufferFlags::empty(),
                    None,
                )
            })
            .expect("reuse binned buffer");
        assert_eq!(
            p.buffer_handle(),
            Some(handle_before),
            "adopted buffer parcel is reusable from buffer_bins"
        );
    }

    /// Tests the [`super::TransientPool::return_buffer_parcel`] → [`super::TransientPool::acquire_buffer`]
    /// round-trip: this is the path taken by [`crate::Scheme::drop`] when returning a buffer lease.
    #[test]
    fn return_buffer_parcel_reissues_on_acquire() {
        let device = test_device();
        let ctx = device.create_context().unwrap();

        // Acquire a fresh parcel directly from the pool (simulates lease_buffer allocation).
        let mut p = ctx
            .with_transient_pool(|pool| {
                pool.acquire_buffer(
                    &ctx,
                    64,
                    crate::types::BufferKind::Scattered,
                    crate::types::BufferFlags::empty(),
                    None,
                )
            })
            .expect("initial acquire");
        let handle_before = p.buffer_handle().expect("buffer handle");

        // Simulate Scheme::drop: release bookkeeping then return the parcel.
        let ready_after = p.last_referenced();
        p.release_bookkeeping();
        ctx.with_transient_pool(|pool| pool.return_buffer_parcel(p, ready_after));

        assert_eq!(ctx.with_transient_pool(|t| t.pending_count()), 1, "parcel is pending");

        // Re-acquire — the epoch table is empty so parcel is immediately ready.
        let p2 = ctx
            .with_transient_pool(|pool| {
                pool.acquire_buffer(
                    &ctx,
                    64,
                    crate::types::BufferKind::Scattered,
                    crate::types::BufferFlags::empty(),
                    None,
                )
            })
            .expect("reuse after return");
        assert_eq!(
            p2.buffer_handle(),
            Some(handle_before),
            "return_buffer_parcel → acquire_buffer must reuse the same GPU buffer"
        );
        assert_eq!(
            ctx.with_transient_pool(|t| t.pending_count()),
            0,
            "bin emptied after reuse"
        );
    }

    #[test]
    fn context_acquire_return_transient_buffer_reuses() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let alloc_before = ctx.transient_buffer_alloc_count();
        let buf = ctx
            .acquire_transient_buffer(128, BufferKind::Scattered, BufferFlags::empty(), Some(16))
            .expect("acquire");
        let handle = buf.whole().buffer_handle().expect("handle");
        assert_eq!(ctx.transient_buffer_alloc_count(), alloc_before + 1);
        ctx.return_transient_buffer(buf);
        let buf2 = ctx
            .acquire_transient_buffer(128, BufferKind::Scattered, BufferFlags::empty(), Some(16))
            .expect("reacquire");
        assert_eq!(buf2.whole().buffer_handle(), Some(handle));
        assert_eq!(
            ctx.transient_buffer_alloc_count(),
            alloc_before + 1,
            "reuse must not allocate again"
        );
    }

    #[test]
    fn buffer_bin_trim_drops_excess_ready_entries() {
        let device = test_device();
        let ctx = device.create_context().unwrap();

        for _ in 0..=MAX_BUFFER_BIN_READY_SPARES {
            park_ready_buffer(&ctx);
        }
        assert_eq!(
            ctx.with_transient_pool(|t| t.pending_count()),
            MAX_BUFFER_BIN_READY_SPARES + 1
        );

        ctx.with_transient_pool(|pool| pool.drain_ready(&ctx));

        assert_eq!(
            ctx.with_transient_pool(|t| t.pending_count()),
            MAX_BUFFER_BIN_READY_SPARES,
            "one excess ready entry must be trimmed"
        );
        assert_eq!(
            ctx.with_transient_pool(|t| t.pending_bytes().buffer),
            (MAX_BUFFER_BIN_READY_SPARES as u64) * TEST_BUFFER_SIZE
        );
    }

    #[test]
    fn buffer_bin_trim_preserves_not_ready_entries() {
        let device = test_device();
        let ctx = device.create_context().unwrap();

        park_not_ready_buffer(&ctx);
        park_not_ready_buffer(&ctx);
        park_ready_buffer(&ctx);

        ctx.with_transient_pool(|pool| pool.drain_ready(&ctx));

        assert_eq!(
            ctx.with_transient_pool(|t| t.pending_count()),
            3,
            "in-flight entries and the single ready spare within cap must survive trim"
        );
    }

    #[test]
    fn buffer_bin_trim_drops_excess_but_preserves_not_ready() {
        let device = test_device();
        let ctx = device.create_context().unwrap();

        park_not_ready_buffer(&ctx);
        park_not_ready_buffer(&ctx);
        for _ in 0..3 {
            park_ready_buffer(&ctx);
        }

        ctx.with_transient_pool(|pool| pool.drain_ready(&ctx));

        assert_eq!(
            ctx.with_transient_pool(|t| t.pending_count()),
            3,
            "2 not-ready + 1 ready spare; 2 excess ready entries dropped"
        );
    }

    #[test]
    fn texture_bin_trim_keeps_one_ready_spare() {
        let device = test_device();
        let ctx = device.create_context().unwrap();

        for _ in 0..=MAX_TEXTURE_BIN_READY_SPARES {
            park_ready_texture(&ctx);
        }
        assert_eq!(
            ctx.with_transient_pool(|t| t.pending_count()),
            MAX_TEXTURE_BIN_READY_SPARES + 1
        );

        ctx.with_transient_pool(|pool| pool.drain_ready(&ctx));

        assert_eq!(
            ctx.with_transient_pool(|t| t.pending_count()),
            MAX_TEXTURE_BIN_READY_SPARES,
            "ready texture spares must survive drain_ready (same policy as buffers)"
        );
    }

    #[test]
    fn texture_bin_trim_preserves_not_ready_entries() {
        let device = test_device();
        let ctx = device.create_context().unwrap();

        park_not_ready_texture(&ctx);
        park_not_ready_texture(&ctx);
        park_ready_texture(&ctx);

        ctx.with_transient_pool(|pool| pool.drain_ready(&ctx));

        assert_eq!(
            ctx.with_transient_pool(|t| t.pending_count()),
            3,
            "in-flight texture entries and one ready spare must survive trim"
        );
    }

    #[test]
    fn return_transient_texture_survives_flush_and_reissues() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let (fmt, acc, flags) = rgba_interpolated();

        let tex = ctx.acquire_transient_texture(8, 8, fmt, acc, flags).expect("texture");
        let handle = tex.texture_handle();
        let allocs_after_first = ctx.transient_texture_alloc_count();
        ctx.return_transient_texture(tex);

        // Simulate end-of-frame reclamation that used to drop all ready textures.
        ctx.flush_deferred_deletions();

        let tex2 = ctx.acquire_transient_texture(8, 8, fmt, acc, flags).expect("reacquire");
        assert_eq!(tex2.texture_handle(), handle, "warm spare must reissue after flush");
        assert_eq!(
            ctx.transient_texture_alloc_count(),
            allocs_after_first,
            "reissue must not fresh-alloc"
        );
    }
}
