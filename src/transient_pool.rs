//! Epoch-gated transient parcel pool — recycled GPU memory without a client-visible clock.
//!
//! There is one transient pool per context; this type is the engine.
//!
//! Relinquished resources enter as [`StampedParcel`]s and are handed out again by lease
//! realization **only once every stamped epoch has retired**. Clients never compare
//! timeline values; the pool consumes `ready_after` internally through
//! `Context::parcel_ready`.

use crate::context::Context;
use crate::parcel::{BookkeepingGuard, BytesByKind, Parcel, PoolBookkeeping};
use crate::retained_pool::{RetainedHold, StampedParcel};
use crate::timeline::ReferenceTable;
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
/// Reuse and drain currently happen as soon as [`Context::parcel_ready`] is true.
/// A future optimization could keep entries warm for a heuristic number of frames
/// past readiness before reissuing or dropping them, reducing allocation churn on
/// intermittent resize or ping-pong flip patterns.
struct TexturePendingEntry {
    parcel: Parcel,
    ready_after: ReferenceTable,
}

/// Recycle-bin key for buffer parcels: buffers are interchangeable iff size, kind, and flags match.
///
/// Keying on size alone would allow an adopted non-Scattered buffer (from
/// [`crate::retained_pool::RetainedPool::release_buffer`]) to be handed out to a
/// [`TransientPool::acquire_buffer`] caller that expects a specific kind — which would produce
/// wrong descriptor categories or silent garbage in the shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferKey {
    size: u64,
    kind: BufferKind,
    flags: BufferFlags,
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
        }
    }
}

/// A parked buffer parcel awaiting epoch retirement; reissued by [`TransientPool::acquire_buffer`].
struct BufferBinEntry {
    parcel: Parcel,
    ready_after: ReferenceTable,
}

/// Epoch-gated recycling pool for transient parcels.
pub struct TransientPool {
    /// Bytes parked in recycle bins.
    pending: Arc<PoolBookkeeping>,
    /// Bytes held by clients through this pool (guard-decremented on drop).
    outstanding: Arc<PoolBookkeeping>,
    texture_bins: HashMap<TextureKey, Vec<TexturePendingEntry>>,
    /// Buffer parcels keyed by `(size, kind, flags)`; excess ready entries are trimmed by
    /// [`Self::drain_ready`] (see [`MAX_BUFFER_BIN_READY_SPARES`]).
    buffer_bins: HashMap<BufferKey, Vec<BufferBinEntry>>,
    /// Monotonic count of fresh `alloc_buffer` calls made by [`Self::acquire_buffer`].
    ///
    /// Does **not** increment when a retired bin entry is reused. Exposed via
    /// [`crate::Context::transient_buffer_alloc_count`] for tests that verify the recycling
    /// path fires (alloc count stays flat across a reuse cycle).
    buffer_alloc_count: usize,
}

impl TransientPool {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(PoolBookkeeping::new()),
            outstanding: Arc::new(PoolBookkeeping::new()),
            texture_bins: HashMap::new(),
            buffer_bins: HashMap::new(),
            buffer_alloc_count: 0,
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
    ) -> Result<Parcel> {
        let key = TextureKey {
            width,
            height,
            format,
            access,
            flags,
        };
        if let Some(bin) = self.texture_bins.get_mut(&key) {
            if let Some(pos) = bin.iter().position(|e| ctx.parcel_ready(&e.ready_after)) {
                let entry = bin.swap_remove(pos);
                let bytes = entry.parcel.byte_size();
                self.pending.subtract(ParcelType::Texture, bytes);
                let mut parcel = entry.parcel;
                parcel.attach_bookkeeping(BookkeepingGuard::new(
                    Arc::downgrade(&self.outstanding),
                    ParcelType::Texture,
                    bytes,
                ));
                self.outstanding.add(ParcelType::Texture, bytes);
                return Ok(parcel);
            }
        }

        let tex = ctx
            .device()
            .alloc_texture(width, height, format, access, flags)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let bytes = tex.byte_size() as u64;
        self.outstanding.add(ParcelType::Texture, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.outstanding), ParcelType::Texture, bytes);
        Ok(Parcel::from_texture(tex, guard, Arc::downgrade(&ctx.device().inner)))
    }

    /// Acquire a one-submission buffer lease backing parcel, reusing a retired bin entry when possible.
    ///
    /// `kind` and `flags` must match the values used to originally allocate any reused entry;
    /// they are also forwarded to the backend when a fresh allocation is needed.
    pub fn acquire_buffer(&mut self, ctx: &Context, size: u64, kind: BufferKind, flags: BufferFlags) -> Result<Parcel> {
        let key = BufferKey { size, kind, flags };
        if let Some(bin) = self.buffer_bins.get_mut(&key) {
            if let Some(pos) = bin.iter().position(|e| ctx.parcel_ready(&e.ready_after)) {
                let entry = bin.swap_remove(pos);
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
            .alloc_buffer(size, kind, None, flags)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.buffer_alloc_count += 1;
        let bytes = alloc.byte_size();
        self.outstanding.add(ParcelType::Buffer, bytes);
        let guard = BookkeepingGuard::new(Arc::downgrade(&self.outstanding), ParcelType::Buffer, bytes);
        let mut parcel = Parcel::from_whole_buffer(Arc::new(alloc), Arc::downgrade(&ctx.device().inner));
        parcel.attach_bookkeeping(guard);
        Ok(parcel)
    }

    /// Return a scheme-held buffer lease parcel to the pool after its epoch retires.
    ///
    /// Called from [`crate::Scheme::drop`] for each buffer-backed lease. The parcel's
    /// bookkeeping must already be released by the caller.
    pub(crate) fn return_buffer_parcel(&mut self, parcel: Parcel, ready_after: ReferenceTable) {
        let bytes = parcel.byte_size();
        let key = BufferKey::from_parcel(&parcel);
        self.pending.add(ParcelType::Buffer, bytes);
        self.buffer_bins
            .entry(key)
            .or_default()
            .push(BufferBinEntry { parcel, ready_after });
    }

    pub(crate) fn adopt(&mut self, stamped: StampedParcel) {
        let StampedParcel { hold, ready_after } = stamped;
        match hold {
            RetainedHold::Texture(parcel) => {
                self.park_texture(parcel, ready_after);
            }
            RetainedHold::Buffer(buffer) => {
                let parcel = buffer
                    .into_transient_parcel()
                    .expect("buffer bin intake requires single-unit buffer");
                let bytes = parcel.byte_size();
                let key = BufferKey::from_parcel(&parcel);
                self.pending.add(ParcelType::Buffer, bytes);
                self.buffer_bins
                    .entry(key)
                    .or_default()
                    .push(BufferBinEntry { parcel, ready_after });
            }
        }
    }

    fn park_texture(&mut self, parcel: Parcel, ready_after: ReferenceTable) {
        let bytes = parcel.byte_size();
        let (width, height, format, access, flags) = parcel.texture_descriptor().expect("texture hold has descriptor");
        let key = TextureKey {
            width,
            height,
            format,
            access,
            flags,
        };
        self.pending.add(ParcelType::Texture, bytes);
        self.texture_bins
            .entry(key)
            .or_default()
            .push(TexturePendingEntry { parcel, ready_after });
    }

    pub fn drain_ready(&mut self, ctx: &Context) -> usize {
        let pending = &self.pending;
        let mut released = 0;
        for bin in self.texture_bins.values_mut() {
            bin.retain(|e| {
                if ctx.parcel_ready(&e.ready_after) {
                    pending.subtract(ParcelType::Texture, e.parcel.byte_size());
                    released += 1;
                    false
                } else {
                    true
                }
            });
        }
        self.texture_bins.retain(|_, bin| !bin.is_empty());
        released += self.trim_buffer_bins(ctx);
        released
    }

    /// Drop excess epoch-retired buffer bin entries beyond [`MAX_BUFFER_BIN_READY_SPARES`].
    ///
    /// In-flight (not-ready) entries are never dropped — only ready spares above the cap.
    /// Returns the number of entries dropped.
    fn trim_buffer_bins(&mut self, ctx: &Context) -> usize {
        let mut trimmed = 0;
        for bin in self.buffer_bins.values_mut() {
            let mut ready_indices: Vec<usize> = bin
                .iter()
                .enumerate()
                .filter(|(_, e)| ctx.parcel_ready(&e.ready_after))
                .map(|(i, _)| i)
                .collect();
            ready_indices.sort_unstable();

            let excess = ready_indices.len().saturating_sub(MAX_BUFFER_BIN_READY_SPARES);
            // Consume the highest-indexed ready entries first (descending). Removing by
            // descending index means each swap_remove only disturbs elements at higher
            // positions, so lower indices remain stable for subsequent iterations.
            let to_drop = ready_indices.split_off(ready_indices.len().saturating_sub(excess));
            for idx in to_drop.into_iter().rev() {
                let entry = bin.swap_remove(idx);
                self.pending.subtract(ParcelType::Buffer, entry.parcel.byte_size());
                trimmed += 1;
            }
        }
        self.buffer_bins.retain(|_, bin| !bin.is_empty());
        trimmed
    }

    pub fn pending_bytes(&self) -> BytesByKind {
        self.pending.snapshot()
    }

    pub fn outstanding_bytes(&self) -> BytesByKind {
        self.outstanding.snapshot()
    }

    pub fn pending_count(&self) -> usize {
        self.texture_bins.values().map(Vec::len).sum::<usize>() + self.buffer_bins.values().map(Vec::len).sum::<usize>()
    }

    /// Total number of fresh `alloc_buffer` calls made by [`Self::acquire_buffer`] since
    /// construction. Does not count bin reuses. Monotonically increasing.
    pub fn buffer_alloc_count(&self) -> usize {
        self.buffer_alloc_count
    }
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
    const SCATTERED_EMPTY: (BufferKind, BufferFlags) =
        (BufferKind::Scattered, BufferFlags::empty());

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
}
