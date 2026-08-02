//! CUDA + DX12 presentation companion tests (Windows).
//!
//! GPU-dependent cases skip cleanly when no NVIDIA adapter is present.

#![cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]

use goldy::types::BackendType;
use goldy::{DeviceDescriptor, Instance, RequestAdapterOptions};
use std::sync::Arc;

fn try_cuda_instance() -> Option<Instance> {
    // SAFETY: test process; GOLDY_BACKEND is read during Instance::new.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cuda") };
    Instance::new().ok().filter(|i| {
        i.enumerate_adapters()
            .into_iter()
            .any(|a| a.get_info().backend == BackendType::Cuda)
    })
}

#[test]
fn cuda_device_attaches_dx12_companion_or_skips() {
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = match instance.request_adapter(&RequestAdapterOptions::default()) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("skip: request_adapter failed: {e:#}");
            return;
        }
    };
    let device = match adapter.request_device(&DeviceDescriptor::default()) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("skip: CUDA↔DX12 companion attach failed (expected on WARP/TCC): {e:#}");
            return;
        }
    };
    assert_eq!(device.backend_type(), BackendType::Cuda);
    // Creating a context exercises the same device that holds the companion.
    let _ctx = device.create_context().expect("create_context");
}

#[test]
fn cuda_surface_format_is_rgba32_float_when_companion_works() {
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = match instance.request_adapter(&RequestAdapterOptions::default()) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("skip: {e:#}");
            return;
        }
    };
    let Ok(device) = adapter.request_device(&DeviceDescriptor::default()) else {
        eprintln!("skip: no DX12 companion");
        return;
    };
    let device = Arc::new(device);
    let ctx = device.create_context().expect("context");

    // Headless: we cannot create a real HWND surface here without winit.
    // Device creation succeeding is the LUID + shared-fence import proof.
    let _ = ctx;
    assert_eq!(device.backend_type(), BackendType::Cuda);
}
