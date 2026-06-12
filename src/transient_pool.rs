//! Epoch-gated transient parcel pool — recycled GPU memory without a client-visible clock.
//!
//! There is one transient pool per context (design §4); this type is the engine. Phase 1
//! exposes it as a standalone object for unit tests; phase 1.6 internalizes it on
//! [`crate::Context`] so programs never name the pool — leases and [`crate::RetainedPool::release`]
//! route into it automatically.
//!
//! Relinquished parcels enter as [`StampedParcel`]s and are handed out again by lease
//! realization **only once every stamped epoch has retired**. Clients never compare
//! timeline values; the pool consumes `ready_after` internally through
//! [`crate::Context::parcel_ready`].

use crate::context::Context;
use crate::parcel::{BookkeepingGuard, BytesByKind, Parcel, PoolBookkeeping};
use crate::retained_pool::StampedParcel;
use crate::timeline::ReferenceTable;
use crate::types::{TextureFlags, TextureFormat, TextureKind};
use crate::vram_allocator::ParcelType;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

/// Recycle-bin key: parcels are interchangeable iff their allocation descriptors match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TextureKey {
    width: u32,
    height: u32,
    format: TextureFormat,
    access: TextureKind,
    flags: TextureFlags,
}

/// A parked parcel awaiting epoch retirement.
struct PendingEntry {
    parcel: Parcel,
    ready_after: ReferenceTable,
}

/// Epoch-gated recycling pool for transient parcels.
///
/// Two byte populations are tracked separately:
///
/// - **pending** — parked in the pool, awaiting reuse ([`Self::pending_bytes`]);
/// - **outstanding** — handed out to clients and not yet returned
///   ([`Self::outstanding_bytes`]).
pub struct TransientPool {
    /// Bytes parked in recycle bins.
    pending: Arc<PoolBookkeeping>,
    /// Bytes held by clients through this pool (guard-decremented on drop).
    outstanding: Arc<PoolBookkeeping>,
    texture_bins: HashMap<TextureKey, Vec<PendingEntry>>,
    /// Non-reusable intake (buffer parcels): held until ready, then dropped.
    holding: Vec<PendingEntry>,
}

impl TransientPool {
    /// Create an empty transient pool.
    pub fn new() -> Self {
        Self {
            pending: Arc::new(PoolBookkeeping::new()),
            outstanding: Arc::new(PoolBookkeeping::new()),
            texture_bins: HashMap::new(),
            holding: Vec::new(),
        }
    }

    /// Acquire a texture parcel, recycling a parked one when its epochs have retired.
    ///
    /// Reuse requires an exact descriptor match `(width, height, format, access, flags)`
    /// **and** `ctx.parcel_ready(ready_after)`. Otherwise a fresh texture is allocated.
    /// 
    /// In the future, this could be made async to give the runtime an oppotunity to
    /// wait for a texture that is nearly ready (perhaps by passing in a desird epoch
    /// that can be compared against the epochs with parked textures)
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
                let mut parcel = entry.parcel;
                let bytes = parcel.byte_size();
                self.pending.subtract(ParcelType::Texture, bytes);
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

    /// Return a parcel acquired from this pool (crate-internal; superseded by leases).
    #[allow(dead_code)] // unit tests; lease retirement will call this internally
    pub(crate) fn recycle(&mut self, mut parcel: Parcel) {
        let ready_after = parcel.last_referenced();
        parcel.release_bookkeeping();
        self.park(StampedParcel { parcel, ready_after });
    }

    /// Intake a parcel relinquished from the retained pool or lease retirement path.
    pub(crate) fn adopt(&mut self, stamped: StampedParcel) {
        self.park(stamped);
    }

    /// Classify an incoming [`StampedParcel`] into a recycle bin or the non-reusable holding list.
    ///
    /// Textures keyed by allocation descriptor land in `texture_bins` for epoch-gated reuse.
    /// Buffer parcels (phase 1: never re-issued) go to `holding` until `ready_after` retires,
    /// then drop on drain.
    fn park(&mut self, stamped: StampedParcel) {
        let StampedParcel { parcel, ready_after } = stamped;
        let bytes = parcel.byte_size();
        match parcel.texture_descriptor() {
            Some((width, height, format, access, flags)) => {
                self.pending.add(ParcelType::Texture, bytes);
                let key = TextureKey {
                    width,
                    height,
                    format,
                    access,
                    flags,
                };
                self.texture_bins
                    .entry(key)
                    .or_default()
                    .push(PendingEntry { parcel, ready_after });
            }
            None => {
                // Phase 1: buffer parcels are not re-issued; hold until ready, then drop.
                self.pending.add(ParcelType::Buffer, bytes);
                self.holding.push(PendingEntry { parcel, ready_after });
            }
        }
    }

    /// Drop every parked parcel whose epochs have retired, freeing GPU memory.
    ///
    /// Returns the number of parcels released. This is the pool's
    /// memory-pressure relief valve; in steady state it should not be needed.
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
        self.holding.retain(|e| {
            if ctx.parcel_ready(&e.ready_after) {
                pending.subtract(ParcelType::Buffer, e.parcel.byte_size());
                released += 1;
                false
            } else {
                true
            }
        });
        released
    }

    /// Bytes parked in the pool awaiting reuse, by parcel kind.
    pub fn pending_bytes(&self) -> BytesByKind {
        self.pending.snapshot()
    }

    /// Bytes handed out through this pool and not yet returned, by parcel kind.
    pub fn outstanding_bytes(&self) -> BytesByKind {
        self.outstanding.snapshot()
    }

    /// Number of parked parcels (all bins plus non-reusable holding).
    pub fn pending_count(&self) -> usize {
        self.texture_bins.values().map(Vec::len).sum::<usize>() + self.holding.len()
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

    #[test]
    fn acquire_allocates_fresh_when_empty() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut pool = TransientPool::new();
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool.acquire_texture(&ctx, 32, 32, fmt, acc, flags).unwrap();
        assert_eq!(p.kind(), ParcelType::Texture);
        assert!(pool.outstanding_bytes().texture > 0);
        assert_eq!(pool.pending_bytes().texture, 0);
        assert_eq!(pool.pending_count(), 0);
    }

    #[test]
    fn recycle_then_acquire_reuses_same_texture_when_ready() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut pool = TransientPool::new();
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool.acquire_texture(&ctx, 16, 16, fmt, acc, flags).unwrap();
        let handle_before = p.texture_handle().unwrap();

        // Unreferenced parcel: ready immediately.
        pool.recycle(p);
        assert_eq!(pool.pending_count(), 1);
        assert!(pool.pending_bytes().texture > 0);
        assert_eq!(pool.outstanding_bytes().texture, 0);

        let p2 = pool.acquire_texture(&ctx, 16, 16, fmt, acc, flags).unwrap();
        assert_eq!(
            p2.texture_handle(),
            Some(handle_before),
            "must reuse the parked texture"
        );
        assert_eq!(pool.pending_count(), 0);
        assert_eq!(pool.pending_bytes().texture, 0);
        assert!(pool.outstanding_bytes().texture > 0);
    }

    #[test]
    fn unretired_epoch_blocks_reuse() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut pool = TransientPool::new();
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool.acquire_texture(&ctx, 16, 16, fmt, acc, flags).unwrap();
        let handle_before = p.texture_handle().unwrap();

        // Fabricate an unretired epoch via mark_referenced. The mock backend
        // retires submissions instantly, so a real submission cannot produce a
        // pending reference; this is the one tolerated mechanism-poke for
        // testing the gating logic.
        let future = ctx.gpu_progress() + 100;
        p.mark_referenced(ctx.backend_handle(), future);
        pool.recycle(p);

        let p2 = pool.acquire_texture(&ctx, 16, 16, fmt, acc, flags).unwrap();
        assert_ne!(
            p2.texture_handle(),
            Some(handle_before),
            "unretired parcel must not be re-issued"
        );
        assert_eq!(pool.pending_count(), 1, "blocked parcel stays parked");
    }

    #[test]
    fn descriptor_mismatch_blocks_reuse() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut pool = TransientPool::new();
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool.acquire_texture(&ctx, 16, 16, fmt, acc, flags).unwrap();
        let handle_before = p.texture_handle().unwrap();
        pool.recycle(p);

        // Different extent → fresh allocation.
        let p2 = pool.acquire_texture(&ctx, 32, 32, fmt, acc, flags).unwrap();
        assert_ne!(p2.texture_handle(), Some(handle_before));
        assert_eq!(pool.pending_count(), 1);
    }

    #[test]
    fn adopt_from_retained_pool_and_reuse() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut retained = RetainedPool::new(device.clone());
        let (fmt, acc, flags) = rgba_interpolated();
        let p = retained.acquire_texture(8, 8, fmt, acc, flags, None).unwrap();
        let handle_before = p.texture_handle().unwrap();

        retained.release(&ctx, p);
        assert_eq!(
            retained.bytes_by_kind().texture,
            0,
            "retained accounting drops at handoff"
        );
        assert_eq!(ctx.with_transient_pool(|t| t.pending_count()), 1);

        let p2 = ctx
            .with_transient_pool(|transient| transient.acquire_texture(&ctx, 8, 8, fmt, acc, flags))
            .unwrap();
        assert_eq!(p2.texture_handle(), Some(handle_before), "adopted parcel is reusable");
    }

    #[test]
    fn adopted_buffer_parcel_is_held_not_reissued() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut retained = RetainedPool::new(device.clone());
        let p = retained
            .acquire_buffer(
                64,
                crate::types::BufferKind::Scattered,
                None,
                crate::types::BufferFlags::empty(),
                None,
            )
            .unwrap();
        retained.release(&ctx, p);
        assert_eq!(ctx.with_transient_pool(|t| t.pending_count()), 1);
        assert!(ctx.with_transient_pool(|t| t.pending_bytes().buffer >= 64));

        // Ready (unreferenced) → drain drops it.
        let released = ctx.with_transient_pool(|t| t.drain_ready(&ctx));
        assert_eq!(released, 1);
        assert_eq!(ctx.with_transient_pool(|t| t.pending_count()), 0);
        assert_eq!(ctx.with_transient_pool(|t| t.pending_bytes().buffer), 0);
    }

    #[test]
    fn drain_ready_keeps_unretired_entries() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut pool = TransientPool::new();
        let (fmt, acc, flags) = rgba_interpolated();

        let ready = pool.acquire_texture(&ctx, 16, 16, fmt, acc, flags).unwrap();
        let blocked = pool.acquire_texture(&ctx, 16, 16, fmt, acc, flags).unwrap();
        // Fabricated unretired epoch — see unretired_epoch_blocks_reuse.
        blocked.mark_referenced(ctx.backend_handle(), ctx.gpu_progress() + 100);
        pool.recycle(ready);
        pool.recycle(blocked);
        assert_eq!(pool.pending_count(), 2);

        let released = pool.drain_ready(&ctx);
        assert_eq!(released, 1, "only the retired entry is dropped");
        assert_eq!(pool.pending_count(), 1);
    }

    #[test]
    fn dropping_outstanding_parcel_decrements_accounting() {
        let device = test_device();
        let ctx = device.create_context().unwrap();
        let mut pool = TransientPool::new();
        let (fmt, acc, flags) = rgba_interpolated();
        let p = pool.acquire_texture(&ctx, 16, 16, fmt, acc, flags).unwrap();
        assert!(pool.outstanding_bytes().texture > 0);
        drop(p);
        assert_eq!(pool.outstanding_bytes().texture, 0);
    }
}
