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

/// A parked resource awaiting epoch retirement.
struct PendingEntry {
    hold: RetainedHold,
    ready_after: ReferenceTable,
}

/// Epoch-gated recycling pool for transient parcels.
pub struct TransientPool {
    /// Bytes parked in recycle bins.
    pending: Arc<PoolBookkeeping>,
    /// Bytes held by clients through this pool (guard-decremented on drop).
    outstanding: Arc<PoolBookkeeping>,
    texture_bins: HashMap<TextureKey, Vec<PendingEntry>>,
    /// Non-reusable intake (buffer resources): held until ready, then dropped.
    holding: Vec<PendingEntry>,
}

impl TransientPool {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(PoolBookkeeping::new()),
            outstanding: Arc::new(PoolBookkeeping::new()),
            texture_bins: HashMap::new(),
            holding: Vec::new(),
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
                let RetainedHold::Texture(mut parcel) = entry.hold else {
                    unreachable!("texture bin holds texture parcels");
                };
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

    #[allow(dead_code)]
    pub(crate) fn recycle(&mut self, mut parcel: Parcel) {
        let ready_after = parcel.last_referenced();
        parcel.release_bookkeeping();
        self.park(StampedParcel {
            hold: RetainedHold::Texture(parcel),
            ready_after,
        });
    }

    pub(crate) fn adopt(&mut self, stamped: StampedParcel) {
        self.park(stamped);
    }

    fn park(&mut self, stamped: StampedParcel) {
        let StampedParcel { hold, ready_after } = stamped;
        let bytes = hold.byte_size();
        match hold.texture_descriptor() {
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
                    .push(PendingEntry { hold, ready_after });
            }
            None => {
                self.pending.add(ParcelType::Buffer, bytes);
                self.holding.push(PendingEntry { hold, ready_after });
            }
        }
    }

    pub fn drain_ready(&mut self, ctx: &Context) -> usize {
        let pending = &self.pending;
        let mut released = 0;
        for bin in self.texture_bins.values_mut() {
            bin.retain(|e| {
                if ctx.parcel_ready(&e.ready_after) {
                    pending.subtract(ParcelType::Texture, e.hold.byte_size());
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
                pending.subtract(ParcelType::Buffer, e.hold.byte_size());
                released += 1;
                false
            } else {
                true
            }
        });
        released
    }

    pub fn pending_bytes(&self) -> BytesByKind {
        self.pending.snapshot()
    }

    pub fn outstanding_bytes(&self) -> BytesByKind {
        self.outstanding.snapshot()
    }

    pub fn pending_count(&self) -> usize {
        self.texture_bins.values().map(Vec::len).sum::<usize>() + self.holding.len()
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
    fn adopted_buffer_is_held_not_reissued() {
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
        retained.release_buffer(&ctx, b);
        assert_eq!(ctx.with_transient_pool(|t| t.pending_count()), 1);
        assert!(ctx.with_transient_pool(|t| t.pending_bytes().buffer >= 64));

        let released = ctx.with_transient_pool(|t| t.drain_ready(&ctx));
        assert_eq!(released, 1);
        assert_eq!(ctx.with_transient_pool(|t| t.pending_count()), 0);
    }
}
