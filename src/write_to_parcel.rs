//! Property-only CPU→GPU upload over a deed parcel (internal convenience).
//!
//! [`write_to_parcel`] packages the upload micro-scheme pattern: a single `WriteBuffer`
//! node submitted on `ctx`, serialized against any scheme that declares ownership of the
//! same parcel via queue order. In-crate callers only; integration tests duplicate this
//! in `tests/common/upload.rs`.

use crate::context::Context;
use crate::error::GoldyError;
use crate::parcel::Parcel;
use crate::scheme::Scheme;
use crate::timeline::TimelineValue;

/// Timeline ticket for a completed [`write_to_parcel`] submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WriteToken(TimelineValue);

impl WriteToken {
    pub(crate) fn timeline_value(self) -> TimelineValue {
        self.0
    }
}

/// Upload CPU bytes into a retained buffer [`Parcel`] via a property-only dispatch.
pub(crate) fn write_to_parcel(
    ctx: &Context,
    parcel: &Parcel,
    data: &[u8],
) -> Result<WriteToken, GoldyError> {
    let mut upload = Scheme::new(ctx);
    upload.commit_write_parcel(parcel, 0, data.to_vec())?;
    let tv = upload.submit()?;
    Ok(WriteToken(tv))
}
