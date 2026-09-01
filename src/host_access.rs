//! Page-protected host allocations for `GOLDY_VALIDATION=host_access`.
//!
//! GPU-owned CPU copies (CPU-backend parcels, later staging) live in their own
//! page-aligned mapping with a trailing guard page. The mapping is `PROT_NONE` /
//! `PAGE_NOACCESS` except during a legal CPU window (upload, kernel dispatch,
//! withdraw read). Stray pointers then fault instead of silently aliasing.
//!
//! This is a debug allocator: slower, not a complete detector (page granularity,
//! no native device-local VRAM), and intended to grow over time.

use anyhow::{Context, Result};
use std::ptr;
use std::ptr::NonNull;

/// Host mapping that is inaccessible except while [`GuardedPages::grant`] is held.
pub struct GuardedPages {
    ptr: NonNull<u8>,
    logical_len: usize,
    map_len: usize,
    cpu_refs: std::cell::Cell<u32>,
}

unsafe impl Send for GuardedPages {}
// Backend mutex serializes access; Cell is used only under that lock.
unsafe impl Sync for GuardedPages {}

impl GuardedPages {
    pub fn new(logical_len: usize) -> Result<Self> {
        let page = page_size();
        anyhow::ensure!(page > 0 && page.is_power_of_two(), "invalid page size {page}");
        let data_len = logical_len.max(1).div_ceil(page) * page;
        let map_len = data_len.checked_add(page).context("guarded mapping overflow")?;
        let ptr = map_rw(map_len)?;
        // Zero is already guaranteed for anonymous maps; lock the whole range including the guard.
        protect(ptr, map_len, Access::None)?;
        Ok(Self {
            ptr,
            logical_len,
            map_len,
            cpu_refs: std::cell::Cell::new(0),
        })
    }

    /// Allow CPU loads/stores. Nested grants are reference-counted.
    pub fn grant(&self) -> Result<()> {
        let refs = self.cpu_refs.get();
        if refs == 0 {
            let data_len = self.map_len.saturating_sub(page_size());
            protect(self.ptr, data_len, Access::ReadWrite)?;
        }
        self.cpu_refs.set(refs.saturating_add(1));
        Ok(())
    }

    pub fn revoke(&self) {
        let refs = self.cpu_refs.get();
        if refs == 0 {
            return;
        }
        let next = refs - 1;
        self.cpu_refs.set(next);
        if next == 0 {
            let data_len = self.map_len.saturating_sub(page_size());
            let _ = protect(self.ptr, data_len, Access::None);
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        debug_assert!(self.cpu_refs.get() > 0, "GuardedPages accessed without grant");
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.logical_len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        debug_assert!(self.cpu_refs.get() > 0, "GuardedPages accessed without grant");
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.logical_len) }
    }

    pub fn resize(&mut self, new_len: usize, preserve: bool) -> Result<()> {
        let mut next = Self::new(new_len)?;
        if preserve && self.logical_len > 0 && new_len > 0 {
            let n = self.logical_len.min(new_len);
            self.grant()?;
            next.grant()?;
            next.as_mut_slice()[..n].copy_from_slice(&self.as_slice()[..n]);
            next.revoke();
            self.revoke();
        }
        *self = next;
        Ok(())
    }
}

impl Drop for GuardedPages {
    fn drop(&mut self) {
        if self.cpu_refs.get() > 0 {
            self.cpu_refs.set(0);
            let data_len = self.map_len.saturating_sub(page_size());
            let _ = protect(self.ptr, data_len, Access::None);
        }
        unmap(self.ptr, self.map_len);
    }
}

#[derive(Clone, Copy)]
enum Access {
    None,
    ReadWrite,
}

fn page_size() -> usize {
    #[cfg(unix)]
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE).max(1) as usize
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        struct SystemInfo {
            w_processor_architecture: u16,
            w_reserved: u16,
            dw_page_size: u32,
            lp_minimum_application_address: *mut core::ffi::c_void,
            lp_maximum_application_address: *mut core::ffi::c_void,
            dw_active_processor_mask: usize,
            dw_number_of_processors: u32,
            dw_processor_type: u32,
            dw_allocation_granularity: u32,
            w_processor_level: u16,
            w_processor_revision: u16,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetSystemInfo(info: *mut SystemInfo);
        }
        unsafe {
            let mut info = std::mem::zeroed::<SystemInfo>();
            GetSystemInfo(&mut info);
            info.dw_page_size.max(1) as usize
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        4096
    }
}

fn map_rw(len: usize) -> Result<NonNull<u8>> {
    #[cfg(unix)]
    unsafe {
        let p = libc::mmap(
            ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if p == libc::MAP_FAILED {
            anyhow::bail!("mmap({len}) failed: {}", std::io::Error::last_os_error());
        }
        NonNull::new(p.cast::<u8>()).context("mmap returned null")
    }
    #[cfg(windows)]
    unsafe {
        const MEM_COMMIT: u32 = 0x1000;
        const MEM_RESERVE: u32 = 0x2000;
        const PAGE_READWRITE: u32 = 0x04;
        #[link(name = "kernel32")]
        extern "system" {
            fn VirtualAlloc(
                addr: *mut core::ffi::c_void,
                size: usize,
                alloc_type: u32,
                protect: u32,
            ) -> *mut core::ffi::c_void;
        }
        let p = VirtualAlloc(ptr::null_mut(), len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        NonNull::new(p.cast::<u8>())
            .with_context(|| format!("VirtualAlloc({len}) failed: {}", std::io::Error::last_os_error()))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = len;
        anyhow::bail!("host_access guarded pages are not supported on this OS")
    }
}

fn protect(ptr: NonNull<u8>, len: usize, access: Access) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    #[cfg(unix)]
    unsafe {
        let prot = match access {
            Access::None => libc::PROT_NONE,
            Access::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
        };
        if libc::mprotect(ptr.as_ptr().cast(), len, prot) != 0 {
            anyhow::bail!("mprotect failed: {}", std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(windows)]
    unsafe {
        const PAGE_NOACCESS: u32 = 0x01;
        const PAGE_READWRITE: u32 = 0x04;
        #[link(name = "kernel32")]
        extern "system" {
            fn VirtualProtect(
                addr: *mut core::ffi::c_void,
                size: usize,
                new_protect: u32,
                old_protect: *mut u32,
            ) -> i32;
        }
        let prot = match access {
            Access::None => PAGE_NOACCESS,
            Access::ReadWrite => PAGE_READWRITE,
        };
        let mut old = 0u32;
        if VirtualProtect(ptr.as_ptr().cast(), len, prot, &mut old) == 0 {
            anyhow::bail!("VirtualProtect failed: {}", std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (ptr, len, access);
        Ok(())
    }
}

fn unmap(ptr: NonNull<u8>, len: usize) {
    #[cfg(unix)]
    unsafe {
        libc::munmap(ptr.as_ptr().cast(), len);
    }
    #[cfg(windows)]
    unsafe {
        const MEM_RELEASE: u32 = 0x8000;
        #[link(name = "kernel32")]
        extern "system" {
            fn VirtualFree(addr: *mut core::ffi::c_void, size: usize, free_type: u32) -> i32;
        }
        VirtualFree(ptr.as_ptr().cast(), 0, MEM_RELEASE);
        let _ = len;
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (ptr, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grant_write_revoke_roundtrip() {
        let mut pages = GuardedPages::new(16).expect("map");
        pages.grant().expect("grant");
        pages.as_mut_slice()[..4].copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(&pages.as_slice()[..4], &[1, 2, 3, 4]);
        pages.revoke();
    }

    #[cfg(unix)]
    #[test]
    fn stray_touch_faults_after_revoke() {
        let mut pages = GuardedPages::new(8).expect("map");
        pages.grant().expect("grant");
        let ptr = pages.as_mut_slice().as_mut_ptr();
        unsafe {
            ptr.write(7);
        }
        pages.revoke();
        let pid = unsafe { libc::fork() };
        match pid {
            -1 => panic!("fork failed"),
            0 => {
                // Child: this should SIGSEGV / SIGBUS.
                unsafe {
                    std::ptr::read_volatile(ptr);
                }
                unsafe { libc::_exit(0) };
            }
            _ => {
                let mut status = 0;
                let w = unsafe { libc::waitpid(pid, &mut status, 0) };
                assert_eq!(w, pid);
                assert!(
                    libc::WIFSIGNALED(status)
                        && (libc::WTERMSIG(status) == libc::SIGSEGV || libc::WTERMSIG(status) == libc::SIGBUS),
                    "expected SIGSEGV/SIGBUS, status={status}"
                );
            }
        }
    }
}
