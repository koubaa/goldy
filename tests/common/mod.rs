//! Common utilities for Goldy integration tests.

pub mod image;

use goldy::{Context, Device};

/// Submission/timeline context for tests (canonical home for `gpu_progress` / `wait_until`).
pub fn submission_context(device: &Device) -> Context {
    device.create_context()
}
