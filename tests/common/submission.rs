//! Submission/timeline context for integration tests (`gpu_progress` / `wait_until`).

use goldy::{Context, Device};

pub fn submission_context(device: &Device) -> Context {
    device.create_context().expect("context")
}
