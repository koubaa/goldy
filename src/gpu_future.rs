//! Non-blocking GPU compute submission.
//!
//! [`GpuFuture`] represents pending GPU work. Use it to pipeline CPU and GPU work,
//! poll for completion, or wait with a timeout to detect long-running or TDR-risk scenarios.

use crate::backend::DeviceHandle;
use crate::backend::{FenceToken, GpuBackend};
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A future representing pending GPU compute work.
///
/// Created by [`ComputeEncoder::submit`](crate::ComputeEncoder::submit). Use
/// [`is_complete`](GpuFuture::is_complete) to poll without blocking, or
/// [`wait`](GpuFuture::wait) / [`wait_timeout`](GpuFuture::wait_timeout) to block until done.
pub struct GpuFuture {
    pub(crate) backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) device: DeviceHandle,
    pub(crate) fence_token: FenceToken,
}

impl GpuFuture {
    /// Returns whether the GPU has finished executing the submitted work.
    pub fn is_complete(&self) -> bool {
        let backend = self.backend.lock().unwrap();
        backend.is_fence_complete(self.device, self.fence_token)
    }

    /// Block until the GPU has finished. Returns an error if the device was lost.
    pub fn wait(&self) -> Result<()> {
        let backend = self.backend.lock().unwrap();
        backend.wait_fence(self.device, self.fence_token)
    }

    /// Wait with a timeout. Returns:
    /// - `Ok(true)` if work completed before the timeout
    /// - `Ok(false)` if the timeout elapsed (GPU still busy)
    /// - `Err` if the device was lost
    pub fn wait_timeout(&self, timeout_ms: u32) -> Result<bool> {
        let backend = self.backend.lock().unwrap();
        backend.wait_fence_timeout(self.device, self.fence_token, timeout_ms)
    }
}
