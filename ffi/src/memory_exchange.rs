//! FFI bindings for [`goldy::MemoryExchange`] withdraw / deposit.

use crate::context::GoldyContext;
use crate::error::{set_last_error, GoldyResult};
use crate::retained_pool::{GoldyParcel, GoldyTexture};
use crate::scheme::{GoldyScheme, GoldySchemeSubmission};
use goldy::{DepositTarget, DepositTransaction, MemoryExchange, WithdrawBytes, WithdrawClaim, WithdrawTransaction};
use std::ptr;

/// Opaque CPU↔GPU memory exchange.
pub struct GoldyMemoryExchange {
    pub(crate) inner: MemoryExchange,
}

/// Stable withdraw relationship recorded in one scheme.
pub struct GoldyWithdrawTransaction {
    pub(crate) inner: WithdrawTransaction,
}

/// Linear claim for one submission's memory withdrawal.
pub struct GoldyWithdrawClaim {
    pub(crate) inner: WithdrawClaim,
}

/// CPU-readable bytes from a consumed withdraw claim.
pub struct GoldyWithdrawBytes {
    pub(crate) inner: WithdrawBytes,
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

/// Bind a withdrawal over a buffer or texture deed parcel.
///
/// Returns a heap-allocated transaction; destroy with [`goldy_withdraw_transaction_destroy`].
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_memory_exchange_bind_withdraw(
    exchange: *const GoldyMemoryExchange,
    scheme: *mut GoldyScheme,
    parcel: *const GoldyParcel,
) -> *mut GoldyWithdrawTransaction {
    if exchange.is_null() || scheme.is_null() || parcel.is_null() {
        set_last_error("MemoryExchange, scheme, or parcel pointer is null");
        return ptr::null_mut();
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot bind_withdraw while recording a node");
        return ptr::null_mut();
    }
    match (*exchange).inner.bind_withdraw(&mut (*scheme).inner, &(*parcel).inner) {
        Ok(tx) => Box::into_raw(Box::new(GoldyWithdrawTransaction { inner: tx })),
        Err(e) => {
            set_last_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

/// Bind a withdrawal over a texture deed (same as parcel withdraw; texture is a parcel).
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_memory_exchange_bind_withdraw_texture(
    exchange: *const GoldyMemoryExchange,
    scheme: *mut GoldyScheme,
    texture: *const GoldyTexture,
) -> *mut GoldyWithdrawTransaction {
    if exchange.is_null() || scheme.is_null() || texture.is_null() {
        set_last_error("MemoryExchange, scheme, or texture pointer is null");
        return ptr::null_mut();
    }
    if (*scheme).has_active_recorder() {
        set_last_error("Cannot bind_withdraw while recording a node");
        return ptr::null_mut();
    }
    match (*exchange)
        .inner
        .bind_withdraw(&mut (*scheme).inner, &*(*texture).inner)
    {
        Ok(tx) => Box::into_raw(Box::new(GoldyWithdrawTransaction { inner: tx })),
        Err(e) => {
            set_last_error(format!("{e}"));
            ptr::null_mut()
        }
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

/// Destroy a withdraw transaction.
///
/// # Safety
/// `transaction` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_withdraw_transaction_destroy(transaction: *mut GoldyWithdrawTransaction) {
    if !transaction.is_null() {
        drop(Box::from_raw(transaction));
    }
}

/// Logical byte size of readable data for this withdrawal.
///
/// # Safety
/// `transaction` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_withdraw_transaction_byte_size(transaction: *const GoldyWithdrawTransaction) -> u64 {
    if transaction.is_null() {
        return 0;
    }
    (*transaction).inner.byte_size()
}

/// Extract this transaction's claim from a successful submission.
///
/// Returns a heap-allocated claim; settle with [`goldy_withdraw_claim_consume`] or
/// [`goldy_withdraw_claim_discard`], or destroy with [`goldy_withdraw_claim_destroy`].
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_withdraw_transaction_claim(
    transaction: *const GoldyWithdrawTransaction,
    submission: *mut GoldySchemeSubmission,
) -> *mut GoldyWithdrawClaim {
    if transaction.is_null() || submission.is_null() {
        set_last_error("WithdrawTransaction or submission pointer is null");
        return ptr::null_mut();
    }
    match (*transaction).inner.claim(&mut (*submission).inner) {
        Ok(claim) => Box::into_raw(Box::new(GoldyWithdrawClaim { inner: claim })),
        Err(e) => {
            set_last_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

/// Destroy a withdraw claim without consuming or discarding intentionally.
///
/// Drop recycles staging like discard when the claim is still unsettled.
///
/// # Safety
/// `claim` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_withdraw_claim_destroy(claim: *mut GoldyWithdrawClaim) {
    if !claim.is_null() {
        drop(Box::from_raw(claim));
    }
}

/// Wait for the submission, read staging into CPU bytes, and return RAII-managed bytes.
///
/// Takes ownership of `claim` (do not destroy it afterward). Destroy the result with
/// [`goldy_withdraw_bytes_destroy`].
///
/// # Safety
/// `claim` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_withdraw_claim_consume(claim: *mut GoldyWithdrawClaim) -> *mut GoldyWithdrawBytes {
    if claim.is_null() {
        set_last_error("WithdrawClaim pointer is null");
        return ptr::null_mut();
    }
    let boxed = Box::from_raw(claim);
    match boxed.inner.consume() {
        Ok(bytes) => Box::into_raw(Box::new(GoldyWithdrawBytes { inner: bytes })),
        Err(e) => {
            set_last_error(format!("{e}"));
            ptr::null_mut()
        }
    }
}

/// Settle without reading bytes; recycle staging. Takes ownership of `claim`.
///
/// # Safety
/// `claim` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_withdraw_claim_discard(claim: *mut GoldyWithdrawClaim) -> GoldyResult {
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

/// Byte length of consumed withdraw data.
///
/// # Safety
/// `bytes` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_withdraw_bytes_len(bytes: *const GoldyWithdrawBytes) -> u64 {
    if bytes.is_null() {
        return 0;
    }
    let bytes = &*bytes;
    let slice: &[u8] = &bytes.inner;
    slice.len() as u64
}

/// Pointer to consumed withdraw data (valid until [`goldy_withdraw_bytes_destroy`]).
///
/// # Safety
/// `bytes` must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_withdraw_bytes_data(bytes: *const GoldyWithdrawBytes) -> *const u8 {
    if bytes.is_null() {
        return ptr::null();
    }
    let bytes = &*bytes;
    let slice: &[u8] = &bytes.inner;
    slice.as_ptr()
}

/// Copy consumed withdraw data into `output` (must be exactly [`goldy_withdraw_bytes_len`] bytes).
///
/// # Safety
/// All pointers must be valid. `output` must point to at least `output_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn goldy_withdraw_bytes_copy(
    bytes: *const GoldyWithdrawBytes,
    output: *mut u8,
    output_size: usize,
) -> GoldyResult {
    if bytes.is_null() || output.is_null() {
        return GoldyResult::NullPointer;
    }
    let bytes = &*bytes;
    let src: &[u8] = &bytes.inner;
    if output_size != src.len() {
        set_last_error(format!(
            "withdraw bytes size mismatch: expected {}, got {output_size}",
            src.len()
        ));
        return GoldyResult::InvalidArgument;
    }
    let out = std::slice::from_raw_parts_mut(output, output_size);
    out.copy_from_slice(src);
    GoldyResult::Ok
}

/// Destroy consumed withdraw bytes (recycles staging).
///
/// # Safety
/// `bytes` must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_withdraw_bytes_destroy(bytes: *mut GoldyWithdrawBytes) {
    if !bytes.is_null() {
        drop(Box::from_raw(bytes));
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
