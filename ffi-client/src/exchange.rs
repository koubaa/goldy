use crate::error::{check, non_null_expect, Result};
use crate::scheme::SchemeSubmission;
use crate::sys::{self, GoldyClaim, GoldyTransaction};

/// Erased exchange transaction recorded in a scheme.
pub struct Transaction {
    ptr: *mut GoldyTransaction,
}

impl Transaction {
    pub(crate) fn from_ptr(ptr: *mut GoldyTransaction) -> Self {
        Self { ptr }
    }

    pub fn binding_id(&self) -> u32 {
        unsafe { sys::goldy_transaction_binding_id(self.ptr) }
    }

    pub fn generation(&self) -> u64 {
        unsafe { sys::goldy_transaction_generation(self.ptr) }
    }

    pub fn claim(&self, submission: &mut SchemeSubmission) -> Result<Claim> {
        let ptr = non_null_expect(unsafe { sys::goldy_transaction_claim(self.ptr, submission.as_mut_ptr()) });
        Ok(Claim::from_ptr(ptr))
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::goldy_transaction_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// One submission's claim extracted from a transaction.
pub struct Claim {
    ptr: *mut GoldyClaim,
    consumed: bool,
}

impl Claim {
    pub(crate) fn from_ptr(ptr: *mut GoldyClaim) -> Self {
        Self { ptr, consumed: false }
    }

    pub fn consume(mut self) -> Result<()> {
        self.consumed = true;
        let result = check(unsafe { sys::goldy_claim_consume(self.ptr) });
        self.ptr = std::ptr::null_mut();
        result
    }

    pub fn discard(mut self) -> Result<()> {
        self.consumed = true;
        let result = check(unsafe { sys::goldy_claim_discard(self.ptr) });
        self.ptr = std::ptr::null_mut();
        result
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if !self.consumed {
            let _ = unsafe { sys::goldy_claim_discard(self.ptr) };
        }
        unsafe { sys::goldy_claim_destroy(self.ptr) };
        self.ptr = std::ptr::null_mut();
    }
}
