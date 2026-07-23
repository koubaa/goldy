//! Upload helpers for integration tests (migration convenience).

use goldy::{Context, DepositTransaction, GoldyError, MemoryExchange, Parcel, Scheme, Submission};

/// Stage bytes through a bound deposit and submit the upload [`Scheme`].
pub fn upload_parcel(upload: &mut Scheme, deposit: &DepositTransaction, data: &[u8]) -> Result<Submission, GoldyError> {
    deposit.write(upload, 0, data)?;
    upload.submit()
}

/// Bind a reusable buffer deposit on an upload scheme.
pub fn bind_upload_deposit(
    ctx: &Context,
    upload: &mut Scheme,
    parcel: &Parcel,
    capacity: u64,
) -> Result<DepositTransaction, GoldyError> {
    MemoryExchange::new(ctx).bind_deposit_buffer(upload, parcel, capacity)
}

/// One-shot upload on an ephemeral scheme. Fine for tests that do not assert retained
/// resubmit stats on a separate reader/worker scheme touching the same parcel.
#[allow(dead_code)]
pub fn write_to_parcel(ctx: &Context, parcel: &Parcel, data: &[u8]) -> Result<Submission, GoldyError> {
    let mut upload = Scheme::new(ctx);
    let deposit = MemoryExchange::new(ctx).bind_deposit_buffer(&mut upload, parcel, data.len() as u64)?;
    upload_parcel(&mut upload, &deposit, data)
}
