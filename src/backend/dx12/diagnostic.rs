//! One-off diagnostics for the DX12 WARP path (e.g. which `d3d10warp.dll` loaded).

use std::sync::Once;

static WARP_LOG_ONCE: Once = Once::new();

/// After a D3D12 WARP device is created, log which `d3d10warp.dll` the loader
/// resolved (first call per process only). Used to confirm app-local
/// side-loading of the NuGet redistributable WARP on CI vs. the system copy
/// in `C:\Windows\System32`.
pub(crate) fn log_warp_module_path_once() {
    WARP_LOG_ONCE.call_once(|| {
        #[cfg(windows)]
        {
            match windows_impl::warp_module_path() {
                Some(path) => eprintln!("[WARP] d3d10warp.dll loaded from: {path}"),
                None => eprintln!(
                    "[WARP] d3d10warp.dll is not loaded in this process (unexpected after WARP device creation)"
                ),
            }
        }
        #[cfg(not(windows))]
        {
            // Non-Windows targets do not use WARP.
        }
    });
}

/// Set `D3D12_ENABLE_DRED=1` before the first D3D12 DLL load.
#[cfg(windows)]
pub(crate) fn prepare_dred_env() {
    if std::env::var("D3D12_ENABLE_DRED").is_err() {
        // SAFETY: `DEBUG_LAYER_INIT` runs once before any D3D12 device creation.
        unsafe { std::env::set_var("D3D12_ENABLE_DRED", "1") };
    }
}

#[cfg(not(windows))]
pub(crate) fn prepare_dred_env() {}

/// Enable DRED auto-breadcrumbs and page-fault capture via the settings interface.
/// Must run after `EnableDebugLayer` and before `D3D12CreateDevice`.
///
/// DRED settings are a separate COM object from [`ID3D12Debug`] — obtain them with their
/// own `D3D12GetDebugInterface` query (casting from `ID3D12Debug` fails on all platforms).
#[cfg(windows)]
pub(crate) fn enable_dred_settings() {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12GetDebugInterface, ID3D12DeviceRemovedExtendedDataSettings, ID3D12DeviceRemovedExtendedDataSettings1,
        D3D12_DRED_ENABLEMENT_FORCED_ON,
    };

    let mut settings1: Option<ID3D12DeviceRemovedExtendedDataSettings1> = None;
    if unsafe { D3D12GetDebugInterface(&mut settings1) }.is_ok() {
        if let Some(settings1) = settings1 {
            unsafe {
                settings1.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                settings1.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                settings1.SetWatsonDumpEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                settings1.SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
            tracing::info!("D3D12 DRED enabled (ID3D12DeviceRemovedExtendedDataSettings1)");
            return;
        }
    }

    let mut settings: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
    if unsafe { D3D12GetDebugInterface(&mut settings) }.is_ok() {
        if let Some(settings) = settings {
            unsafe {
                settings.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                settings.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                settings.SetWatsonDumpEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
            tracing::info!("D3D12 DRED enabled (ID3D12DeviceRemovedExtendedDataSettings)");
            return;
        }
    }

    tracing::warn!(
        target: "goldy::backend::dx12::diagnostic",
        "ID3D12DeviceRemovedExtendedDataSettings not available — DRED API setup skipped \
         (D3D12_ENABLE_DRED=1 was set; breadcrumbs may still be active on Win11 22H2+ / recent Agility SDK)"
    );
}

#[cfg(not(windows))]
pub(crate) fn enable_dred_settings() {}

/// First detection of fence `u64::MAX` / device removal: log DRED once and dump the del ring.
pub(crate) fn first_touch_device_removed(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device10,
    device_removed: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    _location: &str,
    _wait_value: u64,
    completed: u64,
) {
    if completed != u64::MAX {
        return;
    }
    if device_removed
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_ok()
    {
        log_dred_on_device_removed(device);
    }
}

#[cfg(not(windows))]
pub(crate) fn first_touch_device_removed(
    _device: &(),
    _device_removed: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    _location: &str,
    _wait_value: u64,
    _completed: u64,
) {
}

/// Log safe, scalar DRED metadata after device removal (TDR/hang).
///
/// Does **not** walk DRED linked-list pointers — those can be stale immediately after removal
/// and have caused access violations when queried from error paths such as `ResizeBuffers`.
/// Full breadcrumb dumps belong in WinDbg / PIX after attaching post-mortem.
#[cfg(windows)]
pub(crate) fn log_dred_on_device_removed(device: &windows::Win32::Graphics::Direct3D12::ID3D12Device10) {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D12::ID3D12DeviceRemovedExtendedData2;

    let reason = unsafe { device.GetDeviceRemovedReason() };
    tracing::error!(target: "goldy::dx12::dred", ?reason, "GetDeviceRemovedReason");

    if let Ok(dred) = device.cast::<ID3D12DeviceRemovedExtendedData2>() {
        let state = unsafe { dred.GetDeviceState() };
        tracing::error!(target: "goldy::dx12::dred", ?state, "DRED device state");

        let mut fault = Default::default();
        if unsafe { dred.GetPageFaultAllocationOutput2(&mut fault) }.is_ok() {
            tracing::error!(
                target: "goldy::dx12::dred",
                page_fault_va = fault.PageFaultVA,
                page_fault_flags = fault.PageFaultFlags.0,
                "DRED page fault (see WinDbg/PIX for allocation breadcrumb lists)"
            );
        }
    }

    tracing::error!(
        target: "goldy::dx12::dred",
        "DRED auto-breadcrumb linked lists omitted here (unsafe to walk from process after TDR); \
         attach WinDbg or enable WDDM DRED dump collection for full command history"
    );
}

#[cfg(not(windows))]
pub(crate) fn log_dred_on_device_removed(_device: &()) {}

#[cfg(windows)]
mod windows_impl {
    use std::ffi::{c_void, OsString};
    use std::os::windows::ffi::OsStringExt;

    unsafe extern "system" {
        fn GetModuleHandleW(lp_module_name: *const u16) -> *mut c_void;
        fn GetModuleFileNameW(h_module: *mut c_void, lp_filename: *mut u16, n_size: u32) -> u32;
    }

    pub(super) fn warp_module_path() -> Option<String> {
        let name: Vec<u16> = "d3d10warp.dll\0".encode_utf16().collect();
        let handle = unsafe { GetModuleHandleW(name.as_ptr()) };
        if handle.is_null() {
            return None;
        }
        let mut buf = vec![0u16; 1024];
        let len = unsafe { GetModuleFileNameW(handle, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        if len == 0 || len >= buf.len() {
            return None;
        }
        Some(OsString::from_wide(&buf[..len]).to_string_lossy().into_owned())
    }
}
