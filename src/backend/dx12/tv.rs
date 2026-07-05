//! Per-space timeline tokens for the DX12 backend.
//!
//! Context compute submissions signal [`CtxTv`] on each context's fence, allocated from
//! [`CtxTimelineNext`] on [`super::types::Dx12SubmissionContext`]. Present / device-queue
//! work signals [`DeviceTv`] on the shared device fence, allocated from
//! [`DeviceTimelineNext`] on [`super::types::LogicalDevice`]. These are **not
//! interchangeable** — mixing them at compile time is a type error.
//!
//! The public API still uses [`crate::timeline::TimelineValue`] (`u64`); convert at the
//! DX12 backend boundary with [`CtxTv::from_public`] / [`DeviceTv::from_public`] on ingress
//! and [`CtxTv::to_public`] / [`DeviceTv::to_public`] on egress.

use crate::timeline::TimelineValue;
use std::sync::atomic::{AtomicU64, Ordering};

/// Timeline value signaled on a **per-context** fence (compute / standalone copy on ctx queue).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CtxTv(u64);

/// Timeline value signaled on the **device** fence (present copy, surface sync, teardown).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DeviceTv(u64);

impl CtxTv {
    pub const ZERO: Self = Self(0);

    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Ingress from the public [`TimelineValue`] API (context-scoped waits / submissions).
    #[inline]
    pub(crate) fn from_public(tv: TimelineValue) -> Self {
        Self(tv)
    }

    #[inline]
    pub fn to_public(self) -> TimelineValue {
        self.0
    }
}

impl DeviceTv {
    pub const ZERO: Self = Self(0);

    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Ingress from the public [`TimelineValue`] API (device-scoped waits / present).
    #[inline]
    pub(crate) fn from_public(tv: TimelineValue) -> Self {
        Self(tv)
    }

    #[inline]
    pub fn to_public(self) -> TimelineValue {
        self.0
    }
}

/// Per-context monotonic counter for [`CtxTv`] tokens on that context's fence.
#[derive(Debug)]
pub(crate) struct CtxTimelineNext {
    next: AtomicU64,
}

impl CtxTimelineNext {
    pub(crate) fn new(start: u64) -> Self {
        Self {
            next: AtomicU64::new(start),
        }
    }

    pub(crate) fn allocate(&self) -> CtxTv {
        CtxTv(self.next.fetch_add(1, Ordering::AcqRel))
    }

    pub(crate) fn horizon(&self) -> CtxTv {
        CtxTv(self.next.load(Ordering::Acquire).saturating_sub(1))
    }
}

/// Device-fence monotonic counter for [`DeviceTv`] tokens (present, surface sync, teardown).
#[derive(Debug)]
pub(crate) struct DeviceTimelineNext {
    next: AtomicU64,
}

impl DeviceTimelineNext {
    pub(crate) fn new(start: u64) -> Self {
        Self {
            next: AtomicU64::new(start),
        }
    }

    pub(crate) fn allocate_device(&self) -> DeviceTv {
        DeviceTv(self.next.fetch_add(1, Ordering::AcqRel))
    }

    pub(crate) fn device_horizon(&self) -> DeviceTv {
        DeviceTv(self.next.load(Ordering::Acquire).saturating_sub(1))
    }

    pub(crate) fn peek_next(&self) -> DeviceTv {
        DeviceTv(self.next.load(Ordering::Relaxed))
    }

    pub(crate) fn bump(&self) {
        self.next.fetch_add(1, Ordering::Relaxed);
    }
}

impl From<CtxTv> for TimelineValue {
    fn from(v: CtxTv) -> Self {
        v.to_public()
    }
}

impl From<DeviceTv> for TimelineValue {
    fn from(v: DeviceTv) -> Self {
        v.to_public()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctx_and_device_allocators_are_independent() {
        let ctx_a = CtxTimelineNext::new(1);
        let ctx_b = CtxTimelineNext::new(1);
        let device = DeviceTimelineNext::new(1);

        assert_eq!(ctx_a.allocate().raw(), 1);
        assert_eq!(ctx_b.allocate().raw(), 1);
        assert_eq!(device.allocate_device().raw(), 1);
        assert_eq!(ctx_a.allocate().raw(), 2);
        assert_eq!(device.allocate_device().raw(), 2);
        assert_eq!(ctx_a.horizon().raw(), 2);
        assert_eq!(device.device_horizon().raw(), 2);
    }
}
