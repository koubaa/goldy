//! Pluggable allocation policies for [`DefaultVramAllocator`](crate::vram_allocator::DefaultVramAllocator).
//!
//! Policies run on the allocator path *before* backend allocation (`before_alloc` may fail)
//! and record commits/frees for accounting.
//!
//! # Test coverage
//!
//! Byte-level tracking and budget enforcement are tested here and in
//! [`goldy::vram_allocator`](crate::vram_allocator) (see the `allocation_policy` test
//! module).

use std::sync::atomic::{AtomicI64, Ordering};

use anyhow::Result;

use crate::buffer::Allocation;
use crate::texture::TextureBacking;
use crate::vram_allocator::{bytesize, ParcelType};

/// Request about to be handed to the backend allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocRequest {
    pub reserved_estimate: u64,
    pub committed_estimate: u64,
    pub kind: ParcelType,
}

/// Successful allocation (actual reserved / committed sizes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocCommit {
    pub reserved: u64,
    pub committed: u64,
    pub kind: ParcelType,
}

impl AllocCommit {
    pub(crate) fn from_buffer(buf: &Allocation) -> Self {
        Self {
            reserved: buf.allocated_size(),
            committed: buf.size(),
            kind: ParcelType::Buffer,
        }
    }

    pub(crate) fn from_texture(tex: &TextureBacking) -> Self {
        let byte_size = tex.byte_size() as u64;
        Self {
            reserved: byte_size,
            committed: byte_size,
            kind: ParcelType::Texture,
        }
    }
}

/// Free notification when a deed-holding parcel is dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocFreeEvent {
    pub reserved: u64,
    pub committed: u64,
    pub kind: ParcelType,
}

/// Policy hook for [`DefaultVramAllocator`](crate::vram_allocator::DefaultVramAllocator).
pub(crate) trait AllocationPolicy: Send + Sync {
    /// Called before backend allocation. May return `Err` to block the alloc.
    fn before_alloc(&self, req: &AllocRequest) -> Result<()>;

    /// Called after a successful backend allocation.
    fn after_alloc(&self, commit: &AllocCommit);

    /// Called when a deed-holding parcel is dropped.
    fn on_freed(&self, free: &AllocFreeEvent);

    /// Net bytes tracked by this policy (allocations minus frees).
    fn allocated_bytes(&self) -> u64 {
        0
    }

    /// `true` when this is the default no-op policy ([`NoPolicy`]).
    ///
    /// Used by [`DefaultVramAllocator`](crate::vram_allocator::DefaultVramAllocator) to
    /// reject a second [`Device::set_allocation_policy`](crate::device::Device::set_allocation_policy).
    fn is_noop(&self) -> bool {
        false
    }
}

/// No-op policy installed by default. Zero overhead on the hot path.
pub(crate) struct NoPolicy;

impl AllocationPolicy for NoPolicy {
    fn before_alloc(&self, _: &AllocRequest) -> Result<()> {
        Ok(())
    }

    fn after_alloc(&self, _: &AllocCommit) {}

    fn on_freed(&self, _: &AllocFreeEvent) {}

    fn is_noop(&self) -> bool {
        true
    }
}

/// Byte-level tracking with an optional budget enforced before GPU allocation.
pub struct BudgetPolicy {
    live_bytes: AtomicI64,
    budget_bytes: Option<u64>,
}

impl BudgetPolicy {
    /// Track allocations with no budget cap.
    pub fn new() -> Self {
        Self {
            live_bytes: AtomicI64::new(0),
            budget_bytes: None,
        }
    }

    /// Track allocations and reject those that would exceed `budget_bytes`.
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self {
            live_bytes: AtomicI64::new(0),
            budget_bytes: Some(budget_bytes),
        }
    }

    /// Net bytes currently tracked (allocations minus frees).
    pub fn allocated_bytes(&self) -> u64 {
        self.live_bytes.load(Ordering::Relaxed).max(0) as u64
    }

    /// Optional byte budget, if configured.
    pub fn budget(&self) -> Option<u64> {
        self.budget_bytes
    }
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocationPolicy for BudgetPolicy {
    fn before_alloc(&self, req: &AllocRequest) -> Result<()> {
        if let Some(cap) = self.budget_bytes {
            let current = self.live_bytes.load(Ordering::Relaxed) as u64;
            if current.saturating_add(req.reserved_estimate) > cap {
                anyhow::bail!(
                    "VRAM budget exceeded: {current} + {} > {cap} (budget={})",
                    req.reserved_estimate,
                    bytesize(cap),
                );
            }
        }
        Ok(())
    }

    fn after_alloc(&self, commit: &AllocCommit) {
        self.live_bytes.fetch_add(commit.reserved as i64, Ordering::Relaxed);
    }

    fn on_freed(&self, free: &AllocFreeEvent) {
        self.live_bytes.fetch_sub(free.reserved as i64, Ordering::Relaxed);
    }

    fn allocated_bytes(&self) -> u64 {
        BudgetPolicy::allocated_bytes(self)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    struct RecordingPolicy {
        before: AtomicU32,
        after: AtomicU32,
        freed: AtomicU32,
    }

    impl RecordingPolicy {
        fn new() -> Self {
            Self {
                before: AtomicU32::new(0),
                after: AtomicU32::new(0),
                freed: AtomicU32::new(0),
            }
        }
    }

    impl AllocationPolicy for RecordingPolicy {
        fn before_alloc(&self, _: &AllocRequest) -> Result<()> {
            self.before.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn after_alloc(&self, _: &AllocCommit) {
            self.after.fetch_add(1, Ordering::Relaxed);
        }

        fn on_freed(&self, _: &AllocFreeEvent) {
            self.freed.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn budget_policy_enforces_cap() {
        let policy = BudgetPolicy::with_budget(8192);
        policy
            .before_alloc(&AllocRequest {
                reserved_estimate: 4096,
                committed_estimate: 4096,
                kind: ParcelType::Buffer,
            })
            .unwrap();
        policy.after_alloc(&AllocCommit {
            reserved: 4096,
            committed: 4096,
            kind: ParcelType::Buffer,
        });
        assert_eq!(policy.allocated_bytes(), 4096);

        let err = policy.before_alloc(&AllocRequest {
            reserved_estimate: 8192,
            committed_estimate: 8192,
            kind: ParcelType::Buffer,
        });
        assert!(err.is_err());
        assert_eq!(policy.allocated_bytes(), 4096);

        policy.on_freed(&AllocFreeEvent {
            reserved: 4096,
            committed: 4096,
            kind: ParcelType::Buffer,
        });
        assert_eq!(policy.allocated_bytes(), 0);
    }

    #[test]
    fn device_alloc_path_invokes_policy_hooks_in_order() {
        use std::sync::Arc;

        use crate::backend::mock::MockBackend;
        use crate::device::Device;
        use crate::types::{BufferFlags, BufferKind, TextureFlags, TextureFormat, TextureKind};

        let device = Device::from_backend(Box::new(MockBackend::new())).unwrap();
        let policy = Arc::new(RecordingPolicy::new());
        device.set_allocation_policy(policy.clone()).unwrap();

        let buf = device
            .alloc_buffer(1024, BufferKind::Scattered, None, BufferFlags::empty())
            .unwrap();
        assert_eq!(policy.before.load(Ordering::Relaxed), 1);
        assert_eq!(policy.after.load(Ordering::Relaxed), 1);
        assert_eq!(policy.freed.load(Ordering::Relaxed), 0);

        drop(buf);
        assert_eq!(policy.freed.load(Ordering::Relaxed), 1);

        let tex = device
            .alloc_texture(
                8,
                8,
                TextureFormat::Rgba8Unorm,
                TextureKind::Interpolated,
                TextureFlags::COPY_DST,
            )
            .unwrap();
        assert_eq!(policy.before.load(Ordering::Relaxed), 2);
        assert_eq!(policy.after.load(Ordering::Relaxed), 2);
        drop(tex);
        assert_eq!(policy.freed.load(Ordering::Relaxed), 2);
    }
}
