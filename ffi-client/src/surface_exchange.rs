use crate::context::Context;
use crate::error::{check, non_null_expect, Result};
use crate::exchange::Transaction;
use crate::scheme::{PresentLease, Scheme, SchemeRenderTargetLease};
use crate::sys::{self, GoldySurfaceExchange};
use crate::texture::Texture;
use crate::types::TextureFormat;

/// Window-surface exchange for present-on-scheme.
pub struct SurfaceExchange {
    ptr: *mut GoldySurfaceExchange,
}

impl SurfaceExchange {
    #[cfg(windows)]
    pub fn from_win32(ctx: &Context, hwnd: *mut std::ffi::c_void, depth: u32) -> Result<Self> {
        let ptr = non_null_expect(unsafe { sys::goldy_surface_exchange_create_win32(ctx.as_ptr(), hwnd, depth) });
        Ok(Self { ptr })
    }

    #[cfg(target_os = "macos")]
    pub fn from_appkit(ctx: &Context, ns_view: *mut std::ffi::c_void, depth: u32) -> Result<Self> {
        let ptr = non_null_expect(unsafe { sys::goldy_surface_exchange_create_appkit(ctx.as_ptr(), ns_view, depth) });
        Ok(Self { ptr })
    }

    #[cfg(target_os = "linux")]
    pub fn from_wayland(
        ctx: &Context,
        display: *mut std::ffi::c_void,
        surface: *mut std::ffi::c_void,
        depth: u32,
    ) -> Result<Self> {
        let ptr = non_null_expect(unsafe {
            sys::goldy_surface_exchange_create_wayland(ctx.as_ptr(), display, surface, depth)
        });
        Ok(Self { ptr })
    }

    pub fn size(&self) -> (u32, u32) {
        (unsafe { sys::goldy_surface_exchange_width(self.ptr) }, unsafe {
            sys::goldy_surface_exchange_height(self.ptr)
        })
    }

    pub fn format(&self) -> TextureFormat {
        unsafe { sys::goldy_surface_exchange_format(self.ptr).into() }
    }

    pub fn generation(&self) -> u64 {
        unsafe { sys::goldy_surface_exchange_generation(self.ptr) }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        check(unsafe { sys::goldy_surface_exchange_resize(self.ptr, width, height) })
    }

    pub fn lease(&self) -> Result<PresentLease> {
        let ptr = non_null_expect(unsafe { sys::goldy_surface_exchange_lease(self.ptr) });
        Ok(PresentLease::from_ptr(ptr))
    }

    pub fn bind_render_target(&self, scheme: &mut Scheme, source: &SchemeRenderTargetLease) -> Result<Transaction> {
        let ptr = non_null_expect(unsafe {
            sys::goldy_surface_exchange_bind_render_target(self.ptr, scheme.as_ptr(), source.as_ptr())
        });
        Ok(Transaction::from_ptr(ptr))
    }

    pub fn bind(&self, scheme: &mut Scheme, source: &Texture) -> Result<Transaction> {
        let ptr =
            non_null_expect(unsafe { sys::goldy_surface_exchange_bind(self.ptr, scheme.as_ptr(), source.as_ptr()) });
        Ok(Transaction::from_ptr(ptr))
    }

    pub fn bind_destination(&self, scheme: &mut Scheme) -> Result<(PresentLease, Transaction)> {
        let mut out = sys::GoldySurfaceExchangeBindDestinationOut {
            lease: std::ptr::null_mut(),
            transaction: std::ptr::null_mut(),
        };
        check(unsafe { sys::goldy_surface_exchange_bind_destination(self.ptr, scheme.as_ptr(), &mut out) })?;
        Ok((
            PresentLease::from_ptr(out.lease),
            Transaction::from_ptr(out.transaction),
        ))
    }
}

impl Drop for SurfaceExchange {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_surface_exchange_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
