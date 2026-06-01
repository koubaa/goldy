//! Submission/timeline context bound to a [`Device`].
//!
//! A [`Context`] holds an `Arc` clone of the device substrate so the device
//! outlives every context. Timeline read/wait APIs live here; submit and
//! reclamation remain on [`Device`] until the full context split lands.

use crate::device::Device;
use crate::error::GoldyError;
use crate::timeline::TimelineValue;
use std::sync::Arc;

/// GPU submission/timeline context for a single device.
///
/// Clone is cheap (`Arc` bump). Dropping the last `Context` releases its
/// `Device` handle; the substrate is torn down only when every `Device` and
/// `Context` is gone.
///
/// Future work will move the fence poller, command pool, and shutdown drain
/// into this type (goldy issue #179).
pub struct Context {
    pub(crate) inner: Arc<ContextInner>,
}

pub(crate) struct ContextInner {
    device: Device,
}

impl Clone for Context {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context").finish_non_exhaustive()
    }
}

impl Context {
    pub(crate) fn new(device: Device) -> Self {
        Self {
            inner: Arc::new(ContextInner { device }),
        }
    }

    /// The device this context is bound to.
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// Latest GPU completion counter on this context's timeline.
    pub fn gpu_progress(&self) -> TimelineValue {
        self.inner.device.gpu_progress_impl()
    }

    /// Block until the timeline reaches at least `value`.
    pub fn wait_until(&self, value: TimelineValue) -> Result<(), GoldyError> {
        self.inner.device.wait_until_impl(value)
    }

    /// Like [`wait_until`](Self::wait_until) but returns `Err(`[`GoldyError::SubmitTimeout`]`)` on timeout.
    pub fn wait_until_timeout(
        &self,
        value: TimelineValue,
        timeout_ms: u32,
    ) -> Result<(), GoldyError> {
        self.inner
            .device
            .wait_until_timeout_impl(value, timeout_ms)
    }

    /// Oldest timeline ticket not yet retired by the GPU, if work is still in flight.
    pub fn peek_oldest_in_flight(&self) -> Option<TimelineValue> {
        self.inner.device.peek_oldest_in_flight_impl()
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::mock::MockBackend;
    use crate::device::Device;
    use crate::task_graph::TaskGraph;
    use std::sync::Arc;

    fn test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn device_outlives_context() {
        let device = test_device();
        let ctx = device.create_context();
        assert_eq!(Arc::strong_count(&device.inner), 2);
        drop(device);
        assert_eq!(ctx.gpu_progress(), 0);
        assert_eq!(Arc::strong_count(&ctx.device().inner), 1);
    }

    #[test]
    fn device_inner_dropped_only_after_context() {
        let device = test_device();
        let weak = Arc::downgrade(&device.inner);
        let ctx = device.create_context();
        drop(device);
        assert!(weak.upgrade().is_some());
        drop(ctx);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn adapter_outlives_device() {
        let device = test_device();
        let adapter = device.adapter().clone();
        let weak = Arc::downgrade(&adapter.inner);
        drop(device);
        assert!(weak.upgrade().is_some());
    }

    #[test]
    fn context_wait_until_after_submit() {
        let device = test_device();
        let ctx = device.create_context();
        let mut graph = TaskGraph::new();
        let tv = device.submit(&mut graph).unwrap();
        ctx.wait_until(tv).unwrap();
        assert!(ctx.gpu_progress() >= tv);
    }
}
