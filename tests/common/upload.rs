//! One-node upload scheme helper for integration tests (migration convenience).

use goldy::{Context, GoldyError, Parcel, Scheme, TimelineValue};

/// Timeline ticket for a completed upload submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteToken(TimelineValue);

impl WriteToken {
    pub fn timeline_value(self) -> TimelineValue {
        self.0
    }
}

/// Upload CPU bytes into a retained buffer [`Parcel`] via a property-only dispatch.
pub fn write_to_parcel(ctx: &Context, parcel: &Parcel, data: &[u8]) -> Result<WriteToken, GoldyError> {
    let mut upload = Scheme::new(ctx);
    upload.commit_write_parcel(parcel, 0, data.to_vec())?;
    let tv = upload.submit()?;
    Ok(WriteToken(tv))
}
