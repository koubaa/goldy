//! Per-device FIFO submission worker.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
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

/// Wait for a pre-scheduled submission epoch without holding the backend mutex.
pub(crate) struct SubmissionEpochWait {
    worker: std::sync::Arc<SubmissionWorker>,
    tv: u64,
    horizon: u64,
}

impl SubmissionEpochWait {
    pub fn new(worker: std::sync::Arc<SubmissionWorker>, tv: u64, horizon: u64) -> Self {
        Self { worker, tv, horizon }
    }

    pub fn wait(self) -> Result<()> {
        self.worker.wait_submitted_if_scheduled(self.tv, self.horizon)?;
        self.worker.check_error()
    }
}

pub(crate) struct SubmissionWorker {
    submitted_epoch: Arc<AtomicU64>,
    latched_error: Arc<Mutex<Option<anyhow::Error>>>,
    wait_notify: Arc<(Mutex<()>, Condvar)>,
    sender: std::sync::mpsc::SyncSender<WorkerMessage>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl SubmissionWorker {
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel(capacity.max(1));
        let submitted_epoch = Arc::new(AtomicU64::new(0));
        let latched_error: Arc<Mutex<Option<anyhow::Error>>> = Arc::new(Mutex::new(None));
        let wait_notify = Arc::new((Mutex::new(()), Condvar::new()));
        let epoch_worker = Arc::clone(&submitted_epoch);
        let err_worker = Arc::clone(&latched_error);
        let notify_worker = Arc::clone(&wait_notify);
        let thread = thread::Builder::new()
            .name("goldy-submit".into())
            .spawn(move || worker_loop(receiver, epoch_worker, err_worker, notify_worker))
            .expect("spawn goldy submission worker");
        Self {
            submitted_epoch,
            latched_error,
            wait_notify,
            sender,
            thread: Mutex::new(Some(thread)),
        }
    }

    #[cfg(any(feature = "vulkan", feature = "dx12"))]
    pub fn submitted_epoch(&self) -> &Arc<AtomicU64> {
        &self.submitted_epoch
    }

    pub fn check_error(&self) -> Result<()> {
        if let Some(err) = self.latched_error.lock().unwrap().as_ref() {
            return Err(anyhow::anyhow!("{err:#}"));
        }
        Ok(())
    }

    pub fn enqueue(&self, tv: u64, work: Box<dyn PendingSubmit>) -> Result<()> {
        self.check_error()?;
        self.sender
            .send(WorkerMessage::Submit { tv, work })
            .map_err(|e| anyhow::anyhow!("submission worker channel closed: {e}"))
    }

    /// Run one submit job on the calling thread and advance the submitted epoch.
    ///
    /// Used by the mock backend for compute/transfer submits so unit tests do not spawn
    /// blocking worker waits (and backend mutex holds) under parallel `cargo test`.
    /// FIFO present-at-submit still uses [`Self::enqueue`] so ordering matches real backends.
    pub fn execute_immediately(&self, tv: u64, work: Box<dyn PendingSubmit>) -> Result<()> {
        self.check_error()?;
        match work.execute() {
            Ok(()) => {
                advance_submitted_epoch(&self.submitted_epoch, &self.wait_notify, tv);
                Ok(())
            }
            Err(e) => {
                advance_submitted_epoch(&self.submitted_epoch, &self.wait_notify, tv);
                *self.latched_error.lock().unwrap() = Some(e);
                notify_waiters(&self.wait_notify);
                self.check_error()
            }
        }
    }

    pub fn wait_submitted(&self, tv: u64) -> Result<()> {
        self.check_error()?;
        if tv == 0 {
            return Ok(());
        }
        wait_for_submitted_epoch(&self.submitted_epoch, &self.wait_notify, &self.latched_error, tv, None)?;
        Ok(())
    }

    pub fn wait_submitted_timeout(&self, tv: u64, timeout_ms: u32) -> Result<bool> {
        self.check_error()?;
        if tv == 0 {
            return Ok(true);
        }
        let deadline = Instant::now() + Duration::from_millis(timeout_ms as u64);
        wait_for_submitted_epoch(
            &self.submitted_epoch,
            &self.wait_notify,
            &self.latched_error,
            tv,
            Some(deadline),
        )
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
        let (tx, rx) = std::sync::mpsc::channel();
        self.sender
            .send(WorkerMessage::Flush { done: tx })
            .map_err(|e| anyhow::anyhow!("submission worker channel closed: {e}"))?;
        rx.recv()
            .map_err(|e| anyhow::anyhow!("submission worker flush response lost: {e}"))?
    }

    pub fn shutdown(&self) {
        let _ = self.flush();
        if self.sender.try_send(WorkerMessage::Shutdown).is_err() {
            let _ = self.sender.send(WorkerMessage::Shutdown);
        }
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

fn advance_submitted_epoch(submitted_epoch: &AtomicU64, wait_notify: &Arc<(Mutex<()>, Condvar)>, tv: u64) {
    submitted_epoch.fetch_max(tv, Ordering::Release);
    // Hold the wait mutex while notifying so a waiter cannot pass the epoch
    // check and then miss the notify before condvar.wait (lost wakeup).
    notify_waiters(wait_notify);
}

fn notify_waiters(wait_notify: &Arc<(Mutex<()>, Condvar)>) {
    let guard = wait_notify.0.lock().unwrap();
    wait_notify.1.notify_all();
    drop(guard);
}

fn wait_for_submitted_epoch(
    submitted_epoch: &AtomicU64,
    wait_notify: &Arc<(Mutex<()>, Condvar)>,
    latched_error: &Arc<Mutex<Option<anyhow::Error>>>,
    tv: u64,
    deadline: Option<Instant>,
) -> Result<bool> {
    while submitted_epoch.load(Ordering::Acquire) < tv {
        if let Some(err) = latched_error.lock().unwrap().as_ref() {
            return Err(anyhow::anyhow!("{err:#}"));
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                return Ok(false);
            }
        }
        let mut guard = wait_notify.0.lock().unwrap();
        while submitted_epoch.load(Ordering::Acquire) < tv {
            if let Some(err) = latched_error.lock().unwrap().as_ref() {
                drop(guard);
                return Err(anyhow::anyhow!("{err:#}"));
            }
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    drop(guard);
                    return Ok(false);
                }
            }
            match deadline {
                None => {
                    guard = wait_notify.1.wait(guard).unwrap();
                }
                Some(d) => {
                    let remaining = d.saturating_duration_since(Instant::now());
                    let (g, timeout) = wait_notify.1.wait_timeout(guard, remaining).unwrap();
                    guard = g;
                    if timeout.timed_out() && submitted_epoch.load(Ordering::Acquire) < tv && Instant::now() >= d {
                        drop(guard);
                        return Ok(false);
                    }
                }
            }
        }
        drop(guard);
    }
    Ok(true)
}

fn worker_loop(
    receiver: std::sync::mpsc::Receiver<WorkerMessage>,
    submitted_epoch: Arc<AtomicU64>,
    latched_error: Arc<Mutex<Option<anyhow::Error>>>,
    wait_notify: Arc<(Mutex<()>, Condvar)>,
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
                    // Still advance so wait_submitted(last_submitted_seq) cannot block
                    // forever after an earlier execute failure skipped this job.
                    advance_submitted_epoch(&submitted_epoch, &wait_notify, tv);
                    continue;
                }
                match work.execute() {
                    Ok(()) => {
                        advance_submitted_epoch(&submitted_epoch, &wait_notify, tv);
                    }
                    Err(e) => {
                        advance_submitted_epoch(&submitted_epoch, &wait_notify, tv);
                        *latched_error.lock().unwrap() = Some(e);
                        notify_waiters(&wait_notify);
                    }
                }
            }
            WorkerMessage::Flush { done } => {
                let _flush = crate::tracy_zone!("goldy.submit_worker.flush");
                let res = if let Some(e) = latched_error.lock().unwrap().take() {
                    Err(e)
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
    timeline_next.fetch_add(1, Ordering::AcqRel)
}

/// Highest timeline value pre-allocated on this device (may still be in the worker queue).
#[cfg(any(feature = "vulkan", feature = "dx12"))]
pub(crate) fn submission_horizon(timeline_next: &AtomicU64) -> u64 {
    timeline_next.load(Ordering::Acquire).saturating_sub(1)
}
