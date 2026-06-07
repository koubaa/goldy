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
