//! Process-wide GPU device for integration tests that must not create/destroy
//! `VkInstance`/`VkDevice` per test (avoids validation-layer teardown races under
//! parallel `cargo test`).
//!
//! Each test should create its own [`Context`]; the device itself is safe to use
//! concurrently across parallel tests.

use goldy::{Device, DeviceDescriptor, Instance, RequestAdapterOptions};
use std::sync::{Arc, OnceLock};

static SHARED_DEVICE: OnceLock<Arc<Device>> = OnceLock::new();

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
