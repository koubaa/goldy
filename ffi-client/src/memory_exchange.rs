use crate::buffer::Buffer;
use crate::context::Context;
use crate::error::{check, non_null_expect, Result};
use crate::parcel::Parcel;
use crate::scheme::{Scheme, SchemeSubmission};
use crate::sys::{
    self, GoldyDepositTransaction, GoldyMemoryExchange, GoldyWithdrawBytes, GoldyWithdrawClaim,
    GoldyWithdrawTransaction,
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
/// Write staging bytes before [`Scheme::submit`]; no claim afterward.
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

    pub fn write(&self, scheme: &mut Scheme, data: &[u8], offset: u64) -> Result<()> {
        check(unsafe {
            sys::goldy_deposit_transaction_write(self.ptr, scheme.as_ptr(), offset, data.as_ptr(), data.len())
        })
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

    pub fn bind_deposit_buffer(
        &self,
        scheme: &mut Scheme,
        destination: &Parcel,
        capacity: u64,
    ) -> Result<DepositTransaction> {
        let ptr = non_null_expect(unsafe {
            sys::goldy_memory_exchange_bind_deposit_buffer(self.ptr, scheme.as_ptr(), destination.as_ptr(), capacity)
        });
        Ok(DepositTransaction { ptr })
    }

    pub fn bind_deposit_texture(
        &self,
        scheme: &mut Scheme,
        destination: &Texture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        capacity: u64,
        src_row_pitch: u32,
    ) -> Result<DepositTransaction> {
        let ptr = non_null_expect(unsafe {
            sys::goldy_memory_exchange_bind_deposit_texture(
                self.ptr,
                scheme.as_ptr(),
                destination.as_ptr(),
                x,
                y,
                width,
                height,
                capacity,
                src_row_pitch,
            )
        });
        Ok(DepositTransaction { ptr })
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
