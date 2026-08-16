//! Serialize API-thread CUDA driver calls against THREAD_LOCAL graph capture.
//!
//! Capture runs on the submit worker (`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`). Concurrent
//! work on another stream (alloc, `device_ptr` waits, `cuArrayCreate`, pinned malloc)
//! yields `CUDA_ERROR_STREAM_CAPTURE_ISOLATION` and invalidates the in-flight graph.
//! The lock is reentrant on the same thread so materialize → alloc → bake can nest.

use std::cell::Cell;
use std::sync::{Mutex, MutexGuard};

static CAPTURE_ALLOC_GATE: Mutex<()> = Mutex::new(());

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

pub(super) struct CaptureAllocGate {
    _guard: Option<MutexGuard<'static, ()>>,
}

impl Drop for CaptureAllocGate {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

pub(super) fn lock_capture_alloc_gate() -> CaptureAllocGate {
    DEPTH.with(|d| {
        let depth = d.get();
        if depth == 0 {
            let guard = CAPTURE_ALLOC_GATE.lock().unwrap();
            d.set(1);
            CaptureAllocGate { _guard: Some(guard) }
        } else {
            d.set(depth + 1);
            CaptureAllocGate { _guard: None }
        }
    })
}
