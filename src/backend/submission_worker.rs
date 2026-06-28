//! Per-device FIFO submission worker.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::Result;

pub(crate) trait PendingSubmit: Send {
    fn execute(self: Box<Self>) -> Result<()>;
}

enum WorkerMessage {
    Submit { tv: u64, work: Box<dyn PendingSubmit> },
    Flush { done: std::sync::mpsc::Sender<Result<()>> },
    Shutdown,
}

pub(crate) const SUBMISSION_QUEUE_CAPACITY: usize = 128;

pub(crate) struct SubmissionWorker {
    submitted_epoch: Arc<AtomicU64>,
    latched_error: Arc<Mutex<Option<anyhow::Error>>>,
    sender: std::sync::mpsc::SyncSender<WorkerMessage>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl SubmissionWorker {
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel(capacity.max(1));
        let submitted_epoch = Arc::new(AtomicU64::new(0));
        let latched_error: Arc<Mutex<Option<anyhow::Error>>> = Arc::new(Mutex::new(None));
        let epoch_worker = Arc::clone(&submitted_epoch);
        let err_worker = Arc::clone(&latched_error);
        let thread = thread::Builder::new()
            .name("goldy-submit".into())
            .spawn(move || worker_loop(receiver, epoch_worker, err_worker))
            .expect("spawn goldy submission worker");
        Self {
            submitted_epoch,
            latched_error,
            sender,
            thread: Mutex::new(Some(thread)),
        }
    }

    pub fn submitted_epoch(&self) -> &Arc<AtomicU64> {
        &self.submitted_epoch
    }

    pub fn check_error(&self) -> Result<()> {
        if let Some(err) = self.latched_error.lock().unwrap().take() {
            return Err(err);
        }
        Ok(())
    }

    pub fn enqueue(&self, tv: u64, work: Box<dyn PendingSubmit>) -> Result<()> {
        self.check_error()?;
        self.sender
            .send(WorkerMessage::Submit { tv, work })
            .map_err(|e| anyhow::anyhow!("submission worker channel closed: {e}"))
    }

    /// Advance the worker epoch for GPU work executed synchronously on the caller thread
    /// (e.g. DX12 device DIRECT-queue render submits) so [`Self::wait_submitted`] matches
    /// [`submission_horizon`](submission_horizon).
    pub fn record_synchronous_submit(&self, tv: u64) -> Result<()> {
        self.check_error()?;
        if tv == 0 {
            return Ok(());
        }
        self.submitted_epoch.fetch_max(tv, Ordering::Release);
        Ok(())
    }

    pub fn wait_submitted(&self, tv: u64) -> Result<()> {
        self.check_error()?;
        if tv == 0 {
            return Ok(());
        }
        while self.submitted_epoch.load(Ordering::Acquire) < tv {
            self.check_error()?;
            thread::yield_now();
        }
        Ok(())
    }

    pub fn wait_submitted_timeout(&self, tv: u64, timeout_ms: u32) -> Result<bool> {
        self.check_error()?;
        if tv == 0 {
            return Ok(true);
        }
        let deadline = Instant::now() + Duration::from_millis(timeout_ms as u64);
        while self.submitted_epoch.load(Ordering::Acquire) < tv {
            self.check_error()?;
            if Instant::now() >= deadline {
                return Ok(false);
            }
            thread::yield_now();
        }
        Ok(true)
    }

    /// Like [`wait_submitted`](Self::wait_submitted), but no-ops when `tv` was never scheduled.
    pub fn wait_submitted_if_scheduled(&self, tv: u64, scheduled_horizon: u64) -> Result<()> {
        if tv == 0 || tv > scheduled_horizon {
            return Ok(());
        }
        self.wait_submitted(tv)
    }

    /// Like [`wait_submitted_timeout`](Self::wait_submitted_timeout), but returns `false`
    /// immediately when `tv` was never scheduled.
    pub fn wait_submitted_if_scheduled_timeout(
        &self,
        tv: u64,
        scheduled_horizon: u64,
        timeout_ms: u32,
    ) -> Result<bool> {
        if tv == 0 {
            return Ok(true);
        }
        if tv > scheduled_horizon {
            return Ok(false);
        }
        self.wait_submitted_timeout(tv, timeout_ms)
    }

    pub fn flush(&self) -> Result<()> {
        self.check_error()?;
        let (tx, rx) = std::sync::mpsc::channel();
        self.sender
            .send(WorkerMessage::Flush { done: tx })
            .map_err(|e| anyhow::anyhow!("submission worker channel closed: {e}"))?;
        rx.recv()?.map_err(|e| e)?;
        self.check_error()
    }

    pub fn shutdown(&self) {
        let _ = self.sender.send(WorkerMessage::Shutdown);
        if let Some(handle) = self.thread.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SubmissionWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(
    receiver: std::sync::mpsc::Receiver<WorkerMessage>,
    submitted_epoch: Arc<AtomicU64>,
    latched_error: Arc<Mutex<Option<anyhow::Error>>>,
) {
    #[cfg(feature = "tracy")]
    crate::_tracy_client::set_thread_name!("goldy-submit");

    loop {
        let msg = {
            let _wait = crate::tracy_zone!("goldy.submit_worker.wait");
            match receiver.recv() {
                Ok(msg) => msg,
                Err(_) => break,
            }
        };
        match msg {
            WorkerMessage::Submit { tv, work } => {
                let _exec = crate::tracy_zone!("goldy.submit_worker.execute");
                if latched_error.lock().unwrap().is_some() {
                    // Still advance so wait_submitted(last_submitted_seq) cannot spin
                    // forever after an earlier execute failure skipped this job.
                    submitted_epoch.fetch_max(tv, Ordering::Release);
                    continue;
                }
                match work.execute() {
                    Ok(()) => {
                        submitted_epoch.fetch_max(tv, Ordering::Release);
                    }
                    Err(e) => {
                        submitted_epoch.fetch_max(tv, Ordering::Release);
                        *latched_error.lock().unwrap() = Some(e);
                    }
                }
            }
            WorkerMessage::Flush { done } => {
                let _flush = crate::tracy_zone!("goldy.submit_worker.flush");
                let res = if latched_error.lock().unwrap().is_some() {
                    Err(anyhow::anyhow!("submission worker latched error"))
                } else {
                    Ok(())
                };
                let _ = done.send(res);
            }
            WorkerMessage::Shutdown => break,
        }
    }
}

pub(crate) fn allocate_timeline_value(timeline_next: &AtomicU64) -> u64 {
    timeline_next.fetch_add(1, Ordering::Relaxed)
}

/// Highest timeline value pre-allocated on this device (may still be in the worker queue).
pub(crate) fn submission_horizon(timeline_next: &AtomicU64) -> u64 {
    timeline_next.load(Ordering::Relaxed).saturating_sub(1)
}
