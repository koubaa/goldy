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
///
/// Dropping a `GpuFuture` without calling [`wait`](GpuFuture::wait) is safe: the `Drop`
/// implementation blocks until the GPU work finishes and then destroys the fence. This ensures
/// the underlying `VkFence` is always freed before `vkDestroyDevice` is called.
pub struct GpuFuture {
    pub(crate) backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) device: DeviceHandle,
    /// `Some` while the fence is still live; `take`n to `None` once the fence has been
    /// waited on and destroyed so that `Drop` does not attempt a second wait.
    pub(crate) fence_token: Option<FenceToken>,
}

impl GpuFuture {
    /// Returns whether the GPU has finished executing the submitted work.
    pub fn is_complete(&self) -> bool {
        let Some(token) = self.fence_token else {
            return true;
        };
        let backend = self.backend.lock().unwrap();
        backend.is_fence_complete(self.device, token)
    }

    /// Block until the GPU has finished. Returns an error if the device was lost.
    pub fn wait(&mut self) -> Result<()> {
        let Some(token) = self.fence_token.take() else {
            return Ok(());
        };
        let mut backend = self.backend.lock().unwrap();
        backend.wait_fence(self.device, token)
    }

    /// Wait with a timeout. Returns:
    /// - `Ok(true)` if work completed before the timeout
    /// - `Ok(false)` if the timeout elapsed (GPU still busy)
    /// - `Err` if the device was lost
    pub fn wait_timeout(&mut self, timeout_ms: u32) -> Result<bool> {
        let Some(token) = self.fence_token else {
            return Ok(true);
        };
        let mut backend = self.backend.lock().unwrap();
        let result = backend.wait_fence_timeout(self.device, token, timeout_ms)?;
        if result {
            // Fence signaled — backend has already destroyed it, consume the token.
            self.fence_token = None;
        }
        Ok(result)
    }
}

impl Drop for GpuFuture {
    fn drop(&mut self) {
        let Some(token) = self.fence_token.take() else {
            return; // Already waited on.
        };
        // Best-effort: wait for the GPU work and destroy the fence. Errors here are
        // expected when the device was already destroyed (the drain in destroy_device
        // will have cleaned up the fence already).
        let Ok(mut backend) = self.backend.try_lock() else {
            // Mutex is held — this can happen during Drop on a panic path when the
            // backend lock is already held by the panicking thread. Leave the fence
            // in the pool; destroy_device's drain will handle it.
            tracing::warn!(
                token,
                "GpuFuture dropped while backend mutex is held; fence will be drained by destroy_device"
            );
            return;
        };
        let _ = backend.wait_fence(self.device, token);
    }
}
