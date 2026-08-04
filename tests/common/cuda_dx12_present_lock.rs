//! Serialize CUDA+DX12 DXGI present across in-process and cross-binary tests.
//!
//! Parallel `Present` against the same adapter (multiple Win32 swapchains in one
//! cargo test run) has failed with opaque present HRESULTs under the default
//! libtest thread pool. A named Win32 mutex covers both threads in one binary
//! and concurrent integration-test processes.

#![cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]
#![allow(dead_code)] // included from several integration binaries; not every entry is used in each

use windows::core::w;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows::Win32::System::Threading::{CreateMutexW, ReleaseMutex, WaitForSingleObject};

const LOCK_NAME: windows::core::PCWSTR = w!("Local\\goldy-cuda-dx12-present-tests");
const WAIT_MS: u32 = 180_000;

pub struct CudaDx12PresentLock {
    handle: HANDLE,
}

impl CudaDx12PresentLock {
    pub fn acquire() -> Self {
        let handle = unsafe { CreateMutexW(None, false, LOCK_NAME) }.expect("CreateMutexW for CUDA present tests");
        let wait = unsafe { WaitForSingleObject(handle, WAIT_MS) };
        assert_ne!(
            wait, WAIT_TIMEOUT,
            "timed out waiting for CUDA/DX12 present test lock ({WAIT_MS} ms)"
        );
        assert_eq!(
            wait, WAIT_OBJECT_0,
            "WaitForSingleObject on CUDA present lock failed: {wait:?}"
        );
        Self { handle }
    }
}

impl Drop for CudaDx12PresentLock {
    fn drop(&mut self) {
        unsafe {
            let _ = ReleaseMutex(self.handle);
            let _ = CloseHandle(self.handle);
        }
    }
}
