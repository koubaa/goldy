//! GPU backend signals and the delivery queue used by [`crate::Device::poll_signals`].
//!
//! Async signals (`BoundaryCrossed`, swapchain events) are pushed from driver callback
//! threads or backend internals into a per-device [`SignalQueue`]. Synchronous signals
//! (`Oversubscribed`) are accumulated on the calling thread and merged when the client
//! drains via `poll_signals`.

use crate::timeline::TimelineValue;
use std::cell::RefCell;
use std::sync::Mutex;

/// Reason reported with [`Signal::Oversubscribed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OversubscribedReason {
    BufferHeap,
    TextureHeap,
    BindlessSlots,
}

/// A non-blocking notification from the GPU backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Signal {
    /// GPU completion handler advanced the monotonic timeline to `epoch`.
    BoundaryCrossed { epoch: TimelineValue },
    /// A swapchain drawable was handed to the client (`Surface::begin` / acquire).
    SwapchainAcquired { image_index: u32 },
    /// Compositor / WSI released a drawable back to the swapchain pool.
    SwapchainReturned { image_index: u32 },
    /// An internal pool/heap could not satisfy an allocation without growing past budget.
    Oversubscribed {
        reason: OversubscribedReason,
        size_hint: u64,
    },
}

/// Thread-safe queue for async signals (producer: driver callback / fence thread).
#[derive(Debug, Default)]
pub struct SignalQueue {
    inner: Mutex<Vec<Signal>>,
}

impl SignalQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, signal: Signal) {
        if let Ok(mut q) = self.inner.lock() {
            q.push(signal);
        }
    }

    pub fn drain(&self) -> Vec<Signal> {
        self.inner
            .lock()
            .map(|mut q| std::mem::take(&mut *q))
            .unwrap_or_default()
    }
}

thread_local! {
    static SYNC_SIGNALS: RefCell<Vec<Signal>> = const { RefCell::new(Vec::new()) };
}

/// Push a synchronous signal (e.g. [`Signal::Oversubscribed`]) on the current thread.
pub fn push_sync_signal(signal: Signal) {
    SYNC_SIGNALS.with(|s| s.borrow_mut().push(signal));
}

fn drain_sync_signals() -> Vec<Signal> {
    SYNC_SIGNALS.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

/// Drain async queue and synchronous buffer into one vec (async first, then sync).
pub fn drain_all_signals(queue: &SignalQueue) -> Vec<Signal> {
    let mut out = queue.drain();
    out.append(&mut drain_sync_signals());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_queue_fifo() {
        let q = SignalQueue::new();
        q.push(Signal::BoundaryCrossed { epoch: 1 });
        q.push(Signal::SwapchainAcquired { image_index: 0 });
        let drained = q.drain();
        assert_eq!(drained.len(), 2);
        assert!(matches!(drained[0], Signal::BoundaryCrossed { epoch: 1 }));
    }

    #[test]
    fn sync_signals_merged_on_drain_all() {
        let q = SignalQueue::new();
        push_sync_signal(Signal::Oversubscribed {
            reason: OversubscribedReason::BufferHeap,
            size_hint: 1024,
        });
        q.push(Signal::BoundaryCrossed { epoch: 2 });
        let all = drain_all_signals(&q);
        assert_eq!(all.len(), 2);
        assert!(matches!(all[0], Signal::BoundaryCrossed { epoch: 2 }));
        assert!(matches!(
            all[1],
            Signal::Oversubscribed {
                reason: OversubscribedReason::BufferHeap,
                size_hint: 1024,
            }
        ));
    }
}
