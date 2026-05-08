//! Monotonic GPU timeline values for completion tracking.
//!
//! Each successful standalone submission or completed frame bracket is assigned a
//! [`TimelineValue`] that the GPU signals when that work finishes. Use
//! [`crate::Device::gpu_progress`] to query completion without blocking, and
//! [`crate::Device::wait_until`] / [`crate::Device::wait_until_timeout`] to block.

/// Monotonic completion handle returned from [`crate::Device::submit`],
/// [`crate::surface::Frame::present`], and related APIs.
pub type TimelineValue = u64;
