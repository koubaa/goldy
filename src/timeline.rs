//! Monotonic GPU timeline values for completion tracking.
//!
//! Each successful standalone submission or completed frame bracket is assigned a
//! [`TimelineValue`] that the GPU signals when that work finishes. Use
//! [`crate::Context::gpu_progress`] to query completion without blocking, and
//! [`crate::Context::wait_until`] / [`crate::Context::wait_until_timeout`] to block.
//!
//! ## Resource lifetime vs the timeline
//!
//! Destroying a [`crate::Buffer`], [`crate::Texture`], or similar may be **deferred** on GPU
//! backends: the handle becomes invalid immediately, but underlying GPU memory may be kept
//! alive until all work **already submitted** before the destroy has finished (the same
//! conservative rule as tagging with the latest scheduled timeline point). If you record
//! commands that use a resource and destroy it **before** submitting that recording, the
//! implementation cannot always detect the hazard — submit (or bracket a frame) before
//! dropping resources that must outlive those commands.

pub type TimelineValue = u64;
