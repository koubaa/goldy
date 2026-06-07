//! Pluggable allocation policies for [`DefaultVramAllocator`](crate::vram_allocator::DefaultVramAllocator).
//!
//! Policies run on the allocator path *before* backend allocation (`before_alloc` may fail)
//! and record commits/frees for accounting.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use anyhow::Result;

use crate::buffer::Buffer;
use crate::texture::Texture;
use crate::vram_allocator::{bytesize, ParcelType};

/// Request about to be handed to the backend allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocRequest {
    pub reserved_estimate: u64,
    pub committed_estimate: u64,
    pub kind: ParcelType,
}

/// Successful allocation (actual reserved / committed sizes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocCommit {
    pub reserved: u64,
    pub committed: u64,
    pub kind: ParcelType,
}

impl AllocCommit {
    pub fn from_buffer(buf: &Buffer) -> Self {
        Self {
            reserved: buf.allocated_size(),
            committed: buf.size(),
            kind: ParcelType::Buffer,
        }
    }

    pub fn from_texture(tex: &Texture) -> Self {
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
pub struct AllocFreeEvent {
    pub reserved: u64,
    pub committed: u64,
    pub kind: ParcelType,
}

/// Policy hook for [`DefaultVramAllocator`](crate::vram_allocator::DefaultVramAllocator).
pub trait AllocationPolicy: Send + Sync {
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

    /// Optional byte budget enforced by [`Self::before_alloc`].
    fn budget(&self) -> Option<u64> {
        None
    }
}

/// No-op policy installed by default. Zero overhead on the hot path.
pub struct NoPolicy;

impl AllocationPolicy for NoPolicy {
    fn before_alloc(&self, _: &AllocRequest) -> Result<()> {
        Ok(())
    }

    fn after_alloc(&self, _: &AllocCommit) {}

    fn on_freed(&self, _: &AllocFreeEvent) {}
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

    fn budget(&self) -> Option<u64> {
        self.budget_bytes
    }
}

/// Install `policy` on `allocator` when it is a [`DefaultVramAllocator`](crate::vram_allocator::DefaultVramAllocator).
pub fn set_on_default(
    allocator: &Arc<dyn crate::vram_allocator::VramAllocator>,
    policy: Arc<dyn AllocationPolicy>,
) -> bool {
    allocator.set_allocation_policy(policy)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
