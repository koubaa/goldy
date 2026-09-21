//! FFI bindings for [`goldy::MemoryExchange`] deposit.

use crate::context::GoldyContext;
use crate::error::{set_last_error, GoldyResult};
use crate::retained_pool::{GoldyParcel, GoldyTexture};
use crate::scheme::GoldyScheme;
use goldy::{DepositTarget, DepositTransaction, MemoryExchange};
use std::ptr;

/// Opaque CPU↔GPU memory exchange.
pub struct GoldyMemoryExchange {
    pub(crate) inner: MemoryExchange,
}

/// Stable deposit relationship recorded in one scheme.
pub struct GoldyDepositTransaction {
    pub(crate) inner: DepositTransaction,
}

/// Create a memory exchange bound to `ctx`.
///
/// # Safety
/// `ctx` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_memory_exchange_create(ctx: *const GoldyContext) -> *mut GoldyMemoryExchange {
    if ctx.is_null() {
        set_last_error("Context pointer is null");
        return ptr::null_mut();
    }
    Box::into_raw(Box::new(GoldyMemoryExchange {
        inner: MemoryExchange::new(&(*ctx).inner),
    }))
}

/// Destroy a memory exchange.
///
/// # Safety
/// `exchange` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_memory_exchange_destroy(exchange: *mut GoldyMemoryExchange) {
    if !exchange.is_null() {
        drop(Box::from_raw(exchange));
    }
}

/// Destination of a memory-exchange deposit (buffer range or texture region).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoldyDepositTargetKind {
    Buffer = 0,
    Texture = 1,
}

/// Tagged deposit destination matching [`GoldyDepositTarget`] in `goldy.h`.
#[repr(C)]
pub struct GoldyDepositTarget {
    pub kind: GoldyDepositTargetKind,
    pub buffer: *const GoldyParcel,
    pub dst_offset: u64,
    pub capacity: u64,
    pub texture: *const GoldyTexture,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub src_row_pitch: u32,
}

/// Bind a deposit into `target` (buffer range or texture region).
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_memory_exchange_bind_deposit(
    exchange: *const GoldyMemoryExchange,
    scheme: *mut GoldyScheme,
    target: *const GoldyDepositTarget,
) -> *mut GoldyDepositTransaction {
    if exchange.is_null() || scheme.is_null() || target.is_null() {
        set_last_error("MemoryExchange, scheme, or deposit target pointer is null");
        return ptr::null_mut();
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot bind_deposit while recording a node");
        return ptr::null_mut();
    }
    let target = &*target;
    let rust_target = match target.kind {
        GoldyDepositTargetKind::Buffer => {
            if target.buffer.is_null() {
                set_last_error("DepositTarget buffer pointer is null");
                return ptr::null_mut();
            }
            DepositTarget::Buffer {
                destination: &(*target.buffer).inner,
                dst_offset: target.dst_offset,
                capacity: target.capacity,
            }
        }
        GoldyDepositTargetKind::Texture => {
            if target.texture.is_null() {
                set_last_error("DepositTarget texture pointer is null");
                return ptr::null_mut();
            }
            DepositTarget::texture(
                &(*target.texture).inner,
                target.x,
                target.y,
                target.width,
                target.height,
                target.capacity,
                target.src_row_pitch,
            )
        }
    };
    match (*exchange).inner.bind_deposit(&mut (*scheme).inner, rust_target) {
        Ok(tx) => Box::into_raw(Box::new(GoldyDepositTransaction { inner: tx })),
        Err(e) => {
            set_last_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

/// Destroy a deposit transaction.
///
/// # Safety
/// `transaction` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_deposit_transaction_destroy(transaction: *mut GoldyDepositTransaction) {
    if !transaction.is_null() {
        drop(Box::from_raw(transaction));
    }
}

/// Staging capacity declared for this deposit.
///
/// # Safety
/// `transaction` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_deposit_transaction_capacity(transaction: *const GoldyDepositTransaction) -> u64 {
    if transaction.is_null() {
        return 0;
    }
    (*transaction).inner.capacity()
}

/// Stable declaration index within the owning scheme.
///
/// # Safety
/// `transaction` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_deposit_transaction_id(transaction: *const GoldyDepositTransaction) -> u32 {
    if transaction.is_null() {
        return 0;
    }
    (*transaction).inner.id()
}

/// Write `data` into deposit staging before submit. Submit claims the occurrence internally.
///
/// # Safety
/// All pointers must be valid. `data` must point to at least `data_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_deposit_transaction_write(
    transaction: *const GoldyDepositTransaction,
    offset: u64,
    data: *const u8,
    data_size: usize,
) -> GoldyResult {
    if transaction.is_null() || (data.is_null() && data_size > 0) {
        return GoldyResult::NullPointer;
    }
    let slice = if data_size == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(data, data_size)
    };
    match (*transaction).inner.write(offset, slice) {
        Ok(()) => GoldyResult::Ok,
        Err(e) => {
            set_last_error(format!("{e}"));
            GoldyResult::GpuError
        }
    }
}
