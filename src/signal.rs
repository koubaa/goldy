//! GPU backend signals and the delivery queue used by [`crate::Context::poll_signals`].
//!
//! Async signals (`BoundaryCrossed`, swapchain events) are pushed from driver callback
//! threads or backend internals into a per-device [`SignalQueue`] (mutex-protected `Vec`;
//! low contention in practice). Synchronous signals (`Oversubscribed`) are accumulated on
//! the calling thread and merged when the client drains via `poll_signals`.

use crate::timeline::TimelineValue;
use std::cell::RefCell;
use std::sync::Mutex;

/// Reason reported with [`Signal::Oversubscribed`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OversubscribedReason {
    BufferHeap,
    TextureHeap,
    /// Reserved. Not yet emitted; bindless slot exhaustion currently panics.
    /// Will be wired when `ResourceRegistry` allocation is converted to `Result`.
    BindlessSlots,
}

/// A non-blocking notification from the GPU backend.
///
/// GPU completion (`BoundaryCrossed`) is serviced inside
/// [`crate::Context::poll_signals_and_service`] and is **not** returned to clients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Signal {
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

/// Internal + client signals carried on the backend queue before filtering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum QueuedSignal {
    /// GPU completion handler advanced the monotonic timeline to `epoch`.
    BoundaryCrossed {
        epoch: TimelineValue,
    },
    Client(Signal),
}

impl From<Signal> for QueuedSignal {
    fn from(signal: Signal) -> Self {
        QueuedSignal::Client(signal)
    }
}

/// Thread-safe queue for async signals (producer: driver callback / fence thread).
#[derive(Debug, Default)]
pub struct SignalQueue {
    inner: Mutex<Vec<QueuedSignal>>,
}

impl SignalQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push_queued(&self, signal: QueuedSignal) {
        if let Ok(mut q) = self.inner.lock() {
            q.push(signal);
        }
    }

    pub fn push(&self, signal: Signal) {
        self.push_queued(QueuedSignal::Client(signal));
    }

    pub(crate) fn push_boundary_crossed(&self, epoch: TimelineValue) {
        self.push_queued(QueuedSignal::BoundaryCrossed { epoch });
    }

    pub(crate) fn drain_queued(&self) -> Vec<QueuedSignal> {
        self.inner
            .lock()
            .map(|mut q| std::mem::take(&mut *q))
            .unwrap_or_default()
    }

    pub fn drain(&self) -> Vec<Signal> {
        self.drain_queued()
            .into_iter()
            .filter_map(|s| match s {
                QueuedSignal::Client(c) => Some(c),
                QueuedSignal::BoundaryCrossed { .. } => None,
            })
            .collect()
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

/// Drain async queue first, then thread-local synchronous signals.
///
/// Boundary-crossed events are returned in the queued list for internal servicing;
/// [`crate::Context::poll_signals_and_service`] filters them from the client-facing result.
/// `Oversubscribed` always trails because it fires on the calling thread after
/// the async queue snapshot is already taken.
pub(crate) fn drain_all_queued_signals(queue: &SignalQueue) -> Vec<QueuedSignal> {
    let mut out = queue.drain_queued();
    out.extend(drain_sync_signals().into_iter().map(QueuedSignal::Client));
    out
}

/// Drain async queue first, then thread-local synchronous signals (client-visible only).
pub fn drain_all_signals(queue: &SignalQueue) -> Vec<Signal> {
    drain_all_queued_signals(queue)
        .into_iter()
        .filter_map(|s| match s {
            QueuedSignal::Client(c) => Some(c),
            QueuedSignal::BoundaryCrossed { .. } => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_queue_fifo() {
        let q = SignalQueue::new();
        q.push_boundary_crossed(1);
        q.push(Signal::SwapchainAcquired { image_index: 0 });
        let drained = q.drain_queued();
        assert_eq!(drained.len(), 2);
        assert!(matches!(drained[0], QueuedSignal::BoundaryCrossed { epoch: 1 }));
    }

    #[test]
    fn oversubscribed_ordering() {
        let q = SignalQueue::new();
        push_sync_signal(Signal::Oversubscribed {
            reason: OversubscribedReason::BufferHeap,
            size_hint: 512,
        });
        q.push_boundary_crossed(3);
        let all = drain_all_queued_signals(&q);
        assert_eq!(all.len(), 2);
        assert!(matches!(all[0], QueuedSignal::BoundaryCrossed { epoch: 3 }));
        assert!(matches!(
            all[1],
            QueuedSignal::Client(Signal::Oversubscribed {
                reason: OversubscribedReason::BufferHeap,
                size_hint: 512,
            })
        ));
    }

    #[test]
    fn client_drain_filters_boundary_crossed() {
        let q = SignalQueue::new();
        q.push_boundary_crossed(3);
        q.push(Signal::SwapchainReturned { image_index: 1 });
        let client = drain_all_signals(&q);
        assert_eq!(client.len(), 1);
        assert!(matches!(client[0], Signal::SwapchainReturned { image_index: 1 }));
    }
}
