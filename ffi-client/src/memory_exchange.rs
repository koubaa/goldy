use crate::buffer::Buffer;
use crate::context::Context;
use crate::error::{check, non_null_expect, Result};
use crate::parcel::Parcel;
use crate::scheme::{Scheme, SchemeSubmission};
use crate::sys::{
    self, GoldyDepositTarget, GoldyDepositTargetKind, GoldyDepositTransaction, GoldyMemoryExchange, GoldyWithdrawBytes,
    GoldyWithdrawClaim, GoldyWithdrawTransaction,
};
use crate::texture::Texture;
use std::ops::Deref;

/// CPU-readable bytes from a consumed withdraw claim.
pub struct WithdrawBytes {
    ptr: *mut GoldyWithdrawBytes,
}

impl WithdrawBytes {
    pub fn len(&self) -> usize {
        unsafe { sys::goldy_withdraw_bytes_len(self.ptr) as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        let len = self.len();
        let data = unsafe { sys::goldy_withdraw_bytes_data(self.ptr) };
        if data.is_null() || len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice().to_vec()
    }
}

impl Deref for WithdrawBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl Drop for WithdrawBytes {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_withdraw_bytes_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Linear claim for one submission's memory withdrawal.
pub struct WithdrawClaim {
    ptr: *mut GoldyWithdrawClaim,
}

impl WithdrawClaim {
    /// Wait for the submission, read staging into CPU bytes.
    ///
    /// Takes ownership of this claim (do not drop afterward).
    pub fn consume(mut self) -> Result<WithdrawBytes> {
        let ptr = self.ptr;
        self.ptr = std::ptr::null_mut();
        let bytes = non_null_expect(unsafe { sys::goldy_withdraw_claim_consume(ptr) });
        Ok(WithdrawBytes { ptr: bytes })
    }

    /// Settle without reading bytes; recycle staging.
    pub fn discard(mut self) -> Result<()> {
        let ptr = self.ptr;
        self.ptr = std::ptr::null_mut();
        check(unsafe { sys::goldy_withdraw_claim_discard(ptr) })
    }
}

impl Drop for WithdrawClaim {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_withdraw_claim_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Stable withdraw relationship recorded in one [`Scheme`].
pub struct WithdrawTransaction {
    ptr: *mut GoldyWithdrawTransaction,
}

impl WithdrawTransaction {
    pub fn byte_size(&self) -> u64 {
        unsafe { sys::goldy_withdraw_transaction_byte_size(self.ptr) }
    }

    pub fn claim(&self, submission: &mut SchemeSubmission) -> Result<WithdrawClaim> {
        let ptr = non_null_expect(unsafe { sys::goldy_withdraw_transaction_claim(self.ptr, submission.as_mut_ptr()) });
        Ok(WithdrawClaim { ptr })
    }
}

impl Drop for WithdrawTransaction {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_withdraw_transaction_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Stable deposit relationship recorded in one [`Scheme`].
///
/// Write staging bytes before [`Scheme::submit`]; submit claims the occurrence internally.
pub struct DepositTransaction {
    ptr: *mut GoldyDepositTransaction,
}

impl DepositTransaction {
    pub fn capacity(&self) -> u64 {
        unsafe { sys::goldy_deposit_transaction_capacity(self.ptr) }
    }

    pub fn id(&self) -> u32 {
        unsafe { sys::goldy_deposit_transaction_id(self.ptr) }
    }

    pub fn write(&self, data: &[u8], offset: u64) -> Result<()> {
        check(unsafe { sys::goldy_deposit_transaction_write(self.ptr, offset, data.as_ptr(), data.len()) })
    }
}

impl Drop for DepositTransaction {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_deposit_transaction_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// CPU↔GPU memory exchange: withdrawals (readback) and deposits (upload).
pub struct MemoryExchange {
    ptr: *mut GoldyMemoryExchange,
}

impl MemoryExchange {
    pub fn new(ctx: &Context) -> Result<Self> {
        let ptr = non_null_expect(unsafe { sys::goldy_memory_exchange_create(ctx.as_ptr()) });
        Ok(Self { ptr })
    }

    pub fn bind_withdraw(&self, scheme: &mut Scheme, parcel: &Parcel) -> Result<WithdrawTransaction> {
        let ptr = non_null_expect(unsafe {
            sys::goldy_memory_exchange_bind_withdraw(self.ptr, scheme.as_ptr(), parcel.as_ptr())
        });
        Ok(WithdrawTransaction { ptr })
    }

    pub fn bind_withdraw_buffer(&self, scheme: &mut Scheme, buffer: &Buffer) -> Result<WithdrawTransaction> {
        let parcel = buffer.field(0)?;
        self.bind_withdraw(scheme, &parcel)
    }

    pub fn bind_withdraw_texture(&self, scheme: &mut Scheme, texture: &Texture) -> Result<WithdrawTransaction> {
        let ptr = non_null_expect(unsafe {
            sys::goldy_memory_exchange_bind_withdraw_texture(self.ptr, scheme.as_ptr(), texture.as_ptr())
        });
        Ok(WithdrawTransaction { ptr })
    }

    pub fn bind_deposit(&self, scheme: &mut Scheme, target: DepositTarget<'_>) -> Result<DepositTransaction> {
        let ffi_target = target.as_ffi();
        let ptr =
            non_null_expect(unsafe { sys::goldy_memory_exchange_bind_deposit(self.ptr, scheme.as_ptr(), &ffi_target) });
        Ok(DepositTransaction { ptr })
    }
}

/// Destination of a memory-exchange deposit (buffer range or texture region).
pub enum DepositTarget<'a> {
    Buffer {
        destination: &'a Parcel,
        dst_offset: u64,
        capacity: u64,
    },
    Texture {
        destination: &'a Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        capacity: u64,
        src_row_pitch: u32,
    },
}

impl<'a> DepositTarget<'a> {
    pub fn buffer(destination: &'a Parcel, capacity: u64) -> Self {
        Self::Buffer {
            destination,
            dst_offset: 0,
            capacity,
        }
    }

    pub fn buffer_at(destination: &'a Parcel, dst_offset: u64, capacity: u64) -> Self {
        Self::Buffer {
            destination,
            dst_offset,
            capacity,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn texture(
        destination: &'a Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        capacity: u64,
        src_row_pitch: u32,
    ) -> Self {
        Self::Texture {
            destination,
            x,
            y,
            width,
            height,
            capacity,
            src_row_pitch,
        }
    }

    fn as_ffi(&self) -> GoldyDepositTarget {
        match self {
            Self::Buffer {
                destination,
                dst_offset,
                capacity,
            } => GoldyDepositTarget {
                kind: GoldyDepositTargetKind::GOLDY_DEPOSIT_TARGET_BUFFER,
                buffer: destination.as_ptr(),
                dst_offset: *dst_offset,
                capacity: *capacity,
                texture: std::ptr::null(),
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                src_row_pitch: 0,
            },
            Self::Texture {
                destination,
                x,
                y,
                width,
                height,
                capacity,
                src_row_pitch,
            } => GoldyDepositTarget {
                kind: GoldyDepositTargetKind::GOLDY_DEPOSIT_TARGET_TEXTURE,
                buffer: std::ptr::null(),
                dst_offset: 0,
                capacity: *capacity,
                texture: destination.as_ptr(),
                x: *x,
                y: *y,
                width: *width,
                height: *height,
                src_row_pitch: *src_row_pitch,
            },
        }
    }
}

impl Drop for MemoryExchange {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_memory_exchange_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}
