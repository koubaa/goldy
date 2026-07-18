//! FFI bindings for [`goldy::SurfaceExchange`], [`goldy::Transaction`], and [`goldy::Claim`].

use crate::context::GoldyContext;
use crate::error::{set_last_error, set_last_error_from_anyhow, GoldyResult};
use crate::retained_pool::GoldyTexture;
use crate::scheme::{GoldyScheme, GoldySchemeRenderTargetLease, GoldySchemeSubmission};
use crate::types::GoldyTextureFormat;
use goldy::{Claim, SurfaceConfig, SurfaceExchange, Transaction};
use std::ptr;

/// Opaque handle to a window-surface exchange.
pub struct GoldySurfaceExchange {
    pub(crate) inner: SurfaceExchange,
}

/// Erased exchange transaction recorded in a scheme.
pub struct GoldyTransaction {
    pub(crate) inner: Transaction,
}

/// One submission's claim extracted from a transaction.
pub struct GoldyClaim {
    pub(crate) inner: Claim,
}

/// Opaque handle to a stable present lease from a surface exchange.
pub struct GoldyPresentLease {
    #[allow(dead_code)] // owned; dropped on destroy / Box drop
    pub(crate) inner: goldy::swapchain_pool::PresentLease,
}

/// Destroy a present lease handle.
///
/// # Safety
/// `lease` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_present_lease_destroy(lease: *mut GoldyPresentLease) {
    if !lease.is_null() {
        drop(Box::from_raw(lease));
    }
}

/// Destroy a surface exchange.
///
/// # Safety
/// `exchange` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_exchange_destroy(exchange: *mut GoldySurfaceExchange) {
    if !exchange.is_null() {
        drop(Box::from_raw(exchange));
    }
}

/// Current drawable width.
///
/// # Safety
/// `exchange` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_exchange_width(exchange: *const GoldySurfaceExchange) -> u32 {
    if exchange.is_null() {
        return 0;
    }
    (*exchange).inner.width()
}

/// Current drawable height.
///
/// # Safety
/// `exchange` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_exchange_height(exchange: *const GoldySurfaceExchange) -> u32 {
    if exchange.is_null() {
        return 0;
    }
    (*exchange).inner.height()
}

/// Surface format.
///
/// # Safety
/// `exchange` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_exchange_format(exchange: *const GoldySurfaceExchange) -> GoldyTextureFormat {
    if exchange.is_null() {
        return GoldyTextureFormat::Bgra8UnormSrgb;
    }
    (*exchange).inner.format().into()
}

/// Current backing generation (advances on resize / present-mode change).
///
/// # Safety
/// `exchange` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_exchange_generation(exchange: *const GoldySurfaceExchange) -> u64 {
    if exchange.is_null() {
        return 0;
    }
    (*exchange).inner.generation()
}

/// Resize the underlying swapchain.
///
/// # Safety
/// `exchange` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_exchange_resize(
    exchange: *mut GoldySurfaceExchange,
    width: u32,
    height: u32,
) -> GoldyResult {
    if exchange.is_null() {
        return GoldyResult::NullPointer;
    }
    match (*exchange).inner.resize(width, height) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}

/// Acquire a stable present lease (prefer exchange bind helpers for new code).
///
/// Returns a heap-allocated lease handle; destroy with [`goldy_present_lease_destroy`].
///
/// # Safety
/// `exchange` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_exchange_lease(exchange: *const GoldySurfaceExchange) -> *mut GoldyPresentLease {
    if exchange.is_null() {
        set_last_error("SurfaceExchange pointer is null");
        return ptr::null_mut();
    }
    let lease = (*exchange).inner.lease();
    Box::into_raw(Box::new(GoldyPresentLease { inner: lease }))
}

/// Record offscreen render-target → surface copy and return a transaction.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_exchange_bind_render_target(
    exchange: *const GoldySurfaceExchange,
    scheme: *mut GoldyScheme,
    src_lease: *const GoldySchemeRenderTargetLease,
) -> *mut GoldyTransaction {
    if exchange.is_null() || scheme.is_null() || src_lease.is_null() {
        set_last_error("SurfaceExchange, scheme, or render-target lease pointer is null");
        return ptr::null_mut();
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot bind_render_target while recording a node");
        return ptr::null_mut();
    }
    match (*exchange)
        .inner
        .bind_render_target(&mut (*scheme).inner, &(*src_lease).lease)
    {
        Ok(transaction) => Box::into_raw(Box::new(GoldyTransaction { inner: transaction })),
        Err(e) => {
            set_last_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

/// Record texture → surface copy and return a transaction.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_exchange_bind(
    exchange: *const GoldySurfaceExchange,
    scheme: *mut GoldyScheme,
    source: *const GoldyTexture,
) -> *mut GoldyTransaction {
    if exchange.is_null() || scheme.is_null() || source.is_null() {
        set_last_error("SurfaceExchange, scheme, or texture pointer is null");
        return ptr::null_mut();
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot bind while recording a node");
        return ptr::null_mut();
    }
    match (*exchange).inner.bind(&mut (*scheme).inner, &(*source).inner) {
        Ok(transaction) => Box::into_raw(Box::new(GoldyTransaction { inner: transaction })),
        Err(e) => {
            set_last_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

/// Out-parameters for [`goldy_surface_exchange_bind_destination`].
#[repr(C)]
pub struct GoldySurfaceExchangeBindDestinationOut {
    pub lease: *mut GoldyPresentLease,
    pub transaction: *mut GoldyTransaction,
}

/// Register present without a copy; scheme writes the drawable directly.
///
/// Writes lease and transaction into `out`.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_surface_exchange_bind_destination(
    exchange: *const GoldySurfaceExchange,
    scheme: *mut GoldyScheme,
    out: *mut GoldySurfaceExchangeBindDestinationOut,
) -> GoldyResult {
    if exchange.is_null() || scheme.is_null() || out.is_null() {
        return GoldyResult::NullPointer;
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot bind_destination while recording a node");
        return GoldyResult::InvalidArgument;
    }
    match (*exchange).inner.bind_destination(&mut (*scheme).inner) {
        Ok((lease, transaction)) => {
            (*out).lease = Box::into_raw(Box::new(GoldyPresentLease { inner: lease }));
            (*out).transaction = Box::into_raw(Box::new(GoldyTransaction { inner: transaction }));
            GoldyResult::Ok
        }
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}

/// Destroy a transaction handle.
///
/// # Safety
/// `transaction` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_transaction_destroy(transaction: *mut GoldyTransaction) {
    if !transaction.is_null() {
        drop(Box::from_raw(transaction));
    }
}

/// Binding id for this transaction within its scheme.
///
/// # Safety
/// `transaction` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_transaction_binding_id(transaction: *const GoldyTransaction) -> u32 {
    if transaction.is_null() {
        return 0;
    }
    (*transaction).inner.binding_id()
}

/// Backing generation snapshotted at claim time.
///
/// # Safety
/// `transaction` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_transaction_generation(transaction: *const GoldyTransaction) -> u64 {
    if transaction.is_null() {
        return 0;
    }
    (*transaction).inner.generation()
}

/// Extract this transaction's claim from a successful submission.
///
/// Returns a heap-allocated claim; destroy with [`goldy_claim_destroy`].
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_transaction_claim(
    transaction: *const GoldyTransaction,
    submission: *mut GoldySchemeSubmission,
) -> *mut GoldyClaim {
    if transaction.is_null() || submission.is_null() {
        set_last_error("Transaction or submission pointer is null");
        return ptr::null_mut();
    }
    match (*transaction).inner.claim(&mut (*submission).inner) {
        Ok(claim) => Box::into_raw(Box::new(GoldyClaim { inner: claim })),
        Err(e) => {
            set_last_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

/// Destroy a claim without consuming or discarding intentionally.
///
/// # Safety
/// `claim` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_claim_destroy(claim: *mut GoldyClaim) {
    if !claim.is_null() {
        drop(Box::from_raw(claim));
    }
}

/// Perform the claim's external handoff (for example present).
///
/// # Safety
/// `claim` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_claim_consume(claim: *mut GoldyClaim) -> GoldyResult {
    if claim.is_null() {
        return GoldyResult::NullPointer;
    }
    let boxed = Box::from_raw(claim);
    match boxed.inner.consume() {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}

/// Settle the claim without intentionally performing the external operation.
///
/// # Safety
/// `claim` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_claim_discard(claim: *mut GoldyClaim) -> GoldyResult {
    if claim.is_null() {
        return GoldyResult::NullPointer;
    }
    let boxed = Box::from_raw(claim);
    match boxed.inner.discard() {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}

#[cfg(target_os = "macos")]
pub mod appkit_exchange {
    use super::*;
    use raw_window_handle::{
        AppKitDisplayHandle, AppKitWindowHandle, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    struct AppKitWindow {
        ns_view: NonNull<c_void>,
    }

    impl HasWindowHandle for AppKitWindow {
        fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            let handle = AppKitWindowHandle::new(self.ns_view);
            Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(RawWindowHandle::AppKit(handle)) })
        }
    }

    impl HasDisplayHandle for AppKitWindow {
        fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            let handle = AppKitDisplayHandle::new();
            Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(RawDisplayHandle::AppKit(handle)) })
        }
    }

    /// Create a surface exchange from an AppKit `NSView` pointer.
    ///
    /// # Safety
    /// `ctx` must be valid. `ns_view` must be a valid `NSView*` for the window's content view.
    #[no_mangle]
    pub unsafe extern "C" fn goldy_surface_exchange_create_appkit(
        ctx: *const GoldyContext,
        ns_view: *mut c_void,
        depth: u32,
    ) -> *mut GoldySurfaceExchange {
        if ctx.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Context pointer is null"));
            return ptr::null_mut();
        }
        if ns_view.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("NSView pointer is null"));
            return ptr::null_mut();
        }
        let ns_view = match NonNull::new(ns_view) {
            Some(v) => v,
            None => {
                set_last_error_from_anyhow(&anyhow::anyhow!("NSView pointer is null"));
                return ptr::null_mut();
            }
        };
        let window = AppKitWindow { ns_view };
        match SurfaceExchange::new_with_depth(&(*ctx).inner, &window, depth, SurfaceConfig::default()) {
            Ok(exchange) => Box::into_raw(Box::new(GoldySurfaceExchange { inner: exchange })),
            Err(e) => {
                set_last_error(format!("{e}"));
                ptr::null_mut()
            }
        }
    }
}

#[cfg(windows)]
pub mod windows_exchange {
    use super::*;
    use raw_window_handle::{
        HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
    };
    use std::ffi::c_void;
    use std::num::NonZeroIsize;

    struct Win32Window {
        hwnd: NonZeroIsize,
    }

    impl HasWindowHandle for Win32Window {
        fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            let handle = Win32WindowHandle::new(self.hwnd);
            Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
        }
    }

    impl HasDisplayHandle for Win32Window {
        fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            let handle = WindowsDisplayHandle::new();
            Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(RawDisplayHandle::Windows(handle)) })
        }
    }

    /// Create a surface exchange from a Win32 HWND.
    ///
    /// # Safety
    /// `ctx` must be valid. `hwnd` must be a valid Win32 window handle.
    #[no_mangle]
    pub unsafe extern "C" fn goldy_surface_exchange_create_win32(
        ctx: *const GoldyContext,
        hwnd: *mut c_void,
        depth: u32,
    ) -> *mut GoldySurfaceExchange {
        if ctx.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Context pointer is null"));
            return ptr::null_mut();
        }
        if hwnd.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("HWND is null"));
            return ptr::null_mut();
        }
        let hwnd_nonzero = match NonZeroIsize::new(hwnd as isize) {
            Some(h) => h,
            None => {
                set_last_error_from_anyhow(&anyhow::anyhow!("HWND is zero"));
                return ptr::null_mut();
            }
        };
        let window = Win32Window { hwnd: hwnd_nonzero };
        match SurfaceExchange::new_with_depth(&(*ctx).inner, &window, depth, SurfaceConfig::default()) {
            Ok(exchange) => Box::into_raw(Box::new(GoldySurfaceExchange { inner: exchange })),
            Err(e) => {
                set_last_error(format!("{e}"));
                ptr::null_mut()
            }
        }
    }
}

#[cfg(target_os = "linux")]
pub mod wayland_exchange {
    use super::*;
    use raw_window_handle::{
        HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    struct WaylandWindow {
        display: NonNull<c_void>,
        surface: NonNull<c_void>,
    }

    impl HasWindowHandle for WaylandWindow {
        fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            let handle = WaylandWindowHandle::new(self.surface);
            Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(RawWindowHandle::Wayland(handle)) })
        }
    }

    impl HasDisplayHandle for WaylandWindow {
        fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            let handle = WaylandDisplayHandle::new(self.display);
            Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(RawDisplayHandle::Wayland(handle)) })
        }
    }

    /// Create a surface exchange from Wayland `wl_display` and `wl_surface` pointers.
    ///
    /// # Safety
    /// `ctx` must be valid. `display` and `surface` must be valid Wayland handles.
    #[no_mangle]
    pub unsafe extern "C" fn goldy_surface_exchange_create_wayland(
        ctx: *const GoldyContext,
        display: *mut c_void,
        surface: *mut c_void,
        depth: u32,
    ) -> *mut GoldySurfaceExchange {
        if ctx.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Context pointer is null"));
            return ptr::null_mut();
        }
        if display.is_null() || surface.is_null() {
            set_last_error_from_anyhow(&anyhow::anyhow!("Wayland display or surface pointer is null"));
            return ptr::null_mut();
        }
        let display = match NonNull::new(display) {
            Some(d) => d,
            None => {
                set_last_error_from_anyhow(&anyhow::anyhow!("Wayland display pointer is null"));
                return ptr::null_mut();
            }
        };
        let surface = match NonNull::new(surface) {
            Some(s) => s,
            None => {
                set_last_error_from_anyhow(&anyhow::anyhow!("Wayland surface pointer is null"));
                return ptr::null_mut();
            }
        };
        let window = WaylandWindow { display, surface };
        match SurfaceExchange::new_with_depth(&(*ctx).inner, &window, depth, SurfaceConfig::default()) {
            Ok(exchange) => Box::into_raw(Box::new(GoldySurfaceExchange { inner: exchange })),
            Err(e) => {
                set_last_error(format!("{e}"));
                ptr::null_mut()
            }
        }
    }
}
