//! Background fence / timeline polling for Vulkan and DX12 signal delivery.

use crate::signal::{Signal, SignalQueue};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Shared state for a device fence polling thread.
pub struct FencePollerState {
    pub shutdown: Arc<AtomicBool>,
    pub signal_queue: Arc<SignalQueue>,
    /// Highest epoch for which `BoundaryCrossed` was already posted.
    pub last_emitted_epoch: Arc<AtomicU64>,
    /// Returns the GPU-completed timeline value (Vulkan semaphore / DX12 fence).
    pub gpu_completed: Arc<dyn Fn() -> u64 + Send + Sync>,
}

/// Spawn a thread that watches GPU completion and posts [`Signal::BoundaryCrossed`].
pub fn spawn_fence_poller(state: FencePollerState) -> JoinHandle<()> {
    thread::spawn(move || {
        while !state.shutdown.load(Ordering::Relaxed) {
            let completed = (state.gpu_completed)();
            let mut last = state.last_emitted_epoch.load(Ordering::Acquire);
            while last < completed {
                last += 1;
                state.signal_queue.push(Signal::BoundaryCrossed { epoch: last });
                state.last_emitted_epoch.store(last, Ordering::Release);
            }
            // Avoid busy-spin; driver callbacks are coarse-grained.
            thread::sleep(Duration::from_millis(1));
        }
    })
}

/// Join a fence poller thread after setting shutdown.
pub fn join_fence_poller(shutdown: &AtomicBool, handle: Option<JoinHandle<()>>) {
    shutdown.store(true, Ordering::Relaxed);
    if let Some(h) = handle {
        let _ = h.join();
    }
}
