//! Smoke test: create a D3D12 device on WARP when `GOLDY_DX12_ALLOW_WARP=1`.
#![cfg(all(target_os = "windows", feature = "dx12"))]

use goldy::{Instance, WARP_ADAPTER_ID};

#[test]
fn create_device_on_warp_adapter() {
    std::env::set_var("GOLDY_BACKEND", "dx12");
    std::env::set_var("GOLDY_DX12_ALLOW_WARP", "1");
    std::env::set_var("GOLDY_DX12_NO_DEBUG", "1");

    let instance = Instance::new().expect("Instance::new");
    let _device = instance
        .create_device_for_adapter(WARP_ADAPTER_ID)
        .expect("create_device_for_adapter(WARP)");
}
