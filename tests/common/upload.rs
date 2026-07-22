//! Upload helpers for integration tests (migration convenience).

use goldy::{Context, GoldyError, Parcel, Scheme, Submission};

/// Upload CPU bytes via a retained upload [`Scheme`] (reuse across frames).
///
/// Reusing one upload scheme avoids topology churn that dirties unrelated retained
/// reader/worker schemes when each upload would otherwise register a new foreign edge.
pub fn upload_parcel(upload: &mut Scheme, parcel: &Parcel, data: &[u8]) -> Result<Submission, GoldyError> {
    upload.write_parcel(parcel, 0, data.to_vec())?;
    upload.submit()
}

/// One-shot upload on an ephemeral scheme. Fine for tests that do not assert retained
/// resubmit stats on a separate reader/worker scheme touching the same parcel.
#[allow(dead_code)]
pub fn write_to_parcel(ctx: &Context, parcel: &Parcel, data: &[u8]) -> Result<Submission, GoldyError> {
    let mut upload = Scheme::new(ctx);
    upload_parcel(&mut upload, parcel, data)
}
