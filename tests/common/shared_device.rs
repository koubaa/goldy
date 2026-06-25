//! Process-wide GPU device for integration tests that must not create/destroy
//! `VkInstance`/`VkDevice` per test (avoids validation-layer teardown races under
//! parallel `cargo test`).
//!
//! Tests that share the device must also take [`test_lock`] for the duration of
//! the test body so cross-submit / retention state does not leak between parallel
//! threads.

use goldy::{Device, DeviceDescriptor, Instance, RequestAdapterOptions};
use std::sync::{Arc, Mutex, OnceLock};

static SHARED_DEVICE: OnceLock<Arc<Device>> = OnceLock::new();
static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// One logical GPU device for the process. Each test should create its own [`Context`].
pub fn shared_device() -> Arc<Device> {
    Arc::clone(SHARED_DEVICE.get_or_init(|| {
        let instance = Instance::new().expect("instance");
        Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())
                .expect("adapter")
                .request_device(&DeviceDescriptor::default())
                .expect("device"),
        )
    }))
}

/// Serialize tests that borrow [`shared_device`]. Hold the guard for the whole test.
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}
