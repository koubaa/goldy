//! Process-wide DXGI factory + adapter list (initialized once).
//!
//! Each [`crate::Device`] / [`super::Dx12Backend`] owns independent [`super::Dx12State`]
//! (resource tables, contexts, devices) for lock-free concurrent submit, but DXGI factory
//! creation with `DXGI_CREATE_FACTORY_DEBUG` must not run concurrently — parallel
//! `Instance::new` calls would otherwise hit `DXGI_ERROR_INVALID_CALL`.

use super::device;
use super::types::DxgiAdapterInfo;
use super::{env_allow_warp, env_enable_dred, env_enable_gbv, is_debug_mode, WARP_ADAPTER_ID};
use anyhow::{Context, Result};
use std::sync::{Arc, Once, OnceLock};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::{D3D12GetDebugInterface, ID3D12Debug, ID3D12Debug1};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIAdapter, IDXGIAdapter1, IDXGIFactory4, IDXGIFactory5, DXGI_ADAPTER_FLAG,
    DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_DEBUG, DXGI_CREATE_FACTORY_FLAGS,
    DXGI_FEATURE_PRESENT_ALLOW_TEARING,
};

static DEBUG_LAYER_INIT: Once = Once::new();

pub(crate) struct Dx12ProcessShared {
    pub factory: IDXGIFactory4,
    pub allow_tearing: bool,
    pub adapters: Vec<DxgiAdapterInfo>,
}

static PROCESS: OnceLock<Arc<Dx12ProcessShared>> = OnceLock::new();

pub(crate) fn process_shared() -> Result<Arc<Dx12ProcessShared>> {
    Ok(Arc::clone(PROCESS.get_or_init(|| {
        Arc::new(init_process_shared().unwrap_or_else(|e| panic!("Failed to initialize DX12 process resources: {e:#}")))
    })))
}

fn init_process_shared() -> Result<Dx12ProcessShared> {
    DEBUG_LAYER_INIT.call_once(|| {
        if is_debug_mode() {
            // Must be set before the first D3D12 DLL load (including GetDebugInterface).
            if env_enable_dred() {
                super::diagnostic::prepare_dred_env();
            }

            let mut debug_interface: Option<ID3D12Debug> = None;
            if unsafe { D3D12GetDebugInterface(&mut debug_interface) }.is_ok() {
                if let Some(d) = debug_interface {
                    unsafe { d.EnableDebugLayer() };
                    tracing::info!("D3D12 debug layer enabled");

                    if env_enable_dred() {
                        super::diagnostic::enable_dred_settings();
                    }

                    if env_enable_gbv() {
                        if let Ok(debug1) = d.cast::<ID3D12Debug1>() {
                            unsafe { debug1.SetEnableGPUBasedValidation(true) };
                            tracing::info!("D3D12 GPU-Based Validation (GBV) enabled");
                        } else {
                            tracing::warn!("ID3D12Debug1 not available — GPU-Based Validation unavailable");
                        }
                    }
                }
            }
        }
    });

    let factory_flags = if is_debug_mode() {
        DXGI_CREATE_FACTORY_DEBUG
    } else {
        DXGI_CREATE_FACTORY_FLAGS(0)
    };

    let factory: IDXGIFactory4 =
        unsafe { CreateDXGIFactory2(factory_flags) }.context("Failed to create DXGI factory")?;

    let allow_tearing = factory
        .cast::<IDXGIFactory5>()
        .ok()
        .and_then(|f5| {
            let mut allow: i32 = 0;
            let hr = unsafe {
                f5.CheckFeatureSupport(
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                    &mut allow as *mut _ as *mut _,
                    std::mem::size_of::<i32>() as u32,
                )
            };
            hr.ok().map(|()| allow != 0)
        })
        .unwrap_or(false);
    tracing::info!("DXGI tearing support: {allow_tearing}");

    let mut adapters = Vec::new();
    let mut adapter_index = 0u32;

    loop {
        let adapter_result: Result<IDXGIAdapter1, _> = unsafe { factory.EnumAdapters1(adapter_index) };
        match adapter_result {
            Ok(adapter) => {
                let desc = match unsafe { adapter.GetDesc1() } {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let flags = DXGI_ADAPTER_FLAG(desc.Flags as i32);
                if !flags.contains(DXGI_ADAPTER_FLAG_SOFTWARE) {
                    let name = String::from_utf16_lossy(&desc.Description)
                        .trim_end_matches('\0')
                        .to_string();
                    tracing::info!("  [{}] {}", adapter_index, name);

                    let supports_reserved_buffers = device::query_supports_reserved_buffers(&adapter);
                    adapters.push(DxgiAdapterInfo {
                        adapter,
                        desc,
                        adapter_id: adapter_index,
                        supports_reserved_buffers,
                    });
                }
                adapter_index += 1;
            }
            Err(_) => break,
        }
    }

    tracing::info!("Found {} hardware DXGI adapters", adapters.len());

    if env_allow_warp() {
        let warp_result: windows::core::Result<IDXGIAdapter> = unsafe { factory.EnumWarpAdapter() };
        match warp_result {
            Ok(warp) => match warp.cast::<IDXGIAdapter1>() {
                Ok(adapter) => match unsafe { adapter.GetDesc1() } {
                    Ok(desc) => {
                        let name = String::from_utf16_lossy(&desc.Description)
                            .trim_end_matches('\0')
                            .to_string();
                        tracing::info!("  [{}] {} (WARP)", WARP_ADAPTER_ID, name);
                        let supports_reserved_buffers = device::query_supports_reserved_buffers(&adapter);
                        adapters.push(DxgiAdapterInfo {
                            adapter,
                            desc,
                            adapter_id: WARP_ADAPTER_ID,
                            supports_reserved_buffers,
                        });
                    }
                    Err(e) => tracing::warn!("WARP GetDesc1 failed: {:?}", e),
                },
                Err(e) => tracing::warn!("WARP IDXGIAdapter cast failed: {:?}", e),
            },
            Err(e) => tracing::warn!("EnumWarpAdapter failed: {:?}", e),
        }
    }

    tracing::info!("Total {} DX12 adapters (including WARP if enabled)", adapters.len());

    Ok(Dx12ProcessShared {
        factory,
        allow_tearing,
        adapters,
    })
}
