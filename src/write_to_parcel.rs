//! Property-only CPU→GPU upload over a deed parcel.
//!
//! [`write_to_parcel`] packages the upload micro-scheme pattern: a single
//! `WriteBuffer` dispatch submitted on `ctx`, serialized against any scheme
//! that declares ownership of the same parcel via queue order.

use crate::context::Context;
use crate::error::GoldyError;
use crate::parcel::Parcel;
use crate::task_graph::TaskGraph;
use crate::timeline::TimelineValue;

/// Timeline ticket for a completed [`write_to_parcel`] submission.
///
/// Discardable — most callers need only the serialization guarantee that the
/// write landed before the next submission on the same context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteToken(TimelineValue);

impl WriteToken {
    /// The timeline value assigned to this upload submission.
    pub fn timeline_value(self) -> TimelineValue {
        self.0
    }
}

/// Upload CPU bytes into a retained buffer [`Parcel`] via a property-only dispatch.
///
/// Creates and submits a one-node micro-scheme on `ctx`. Same-context consumers
/// are serialized by queue order; the returned token is optional.
pub fn write_to_parcel(
    ctx: &Context,
    parcel: &Parcel,
    data: &[u8],
) -> Result<WriteToken, GoldyError> {
    let mut graph = TaskGraph::new();
    graph
        .write_parcel(parcel, 0, data.to_vec())
        .map_err(|e| ctx.classify(e))?;
    let tv = ctx.submit_pipelined(&mut graph)?;
    Ok(WriteToken(tv))
}
