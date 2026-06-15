//! One-node upload scheme helper for integration tests (migration convenience).

use goldy::{Context, GoldyError, Parcel, Scheme, Submission};

/// Upload CPU bytes into a retained buffer [`Parcel`] via a property-only dispatch.
pub fn write_to_parcel(ctx: &Context, parcel: &Parcel, data: &[u8]) -> Result<Submission, GoldyError> {
    let mut upload = Scheme::new(ctx);
    upload.commit_write_parcel(parcel, 0, data.to_vec())?;
    upload.submit()
}
