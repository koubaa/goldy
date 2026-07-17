/// Typed error variants for the goldy public API.
///
/// Returned by [`crate::Scheme::submit`], [`crate::Context::wait_until`],
/// and [`crate::Context::wait_until_timeout`] so callers can distinguish recoverable
/// conditions (timeout) from permanent ones (device loss) without string-matching.
#[derive(Debug, thiserror::Error)]
pub enum GoldyError {
    /// The GPU device has been lost and cannot process further commands.
    ///
    /// Any resources associated with this device are now invalid. The caller
    /// should drop the [`Device`](crate::Device) and re-create from a new
    /// [`Instance`](crate::Instance) if recovery is desired.
    #[error("GPU device lost")]
    DeviceLost,

    /// The GPU or driver ran out of memory.
    ///
    /// The operation was not completed. The caller may attempt to free
    /// resources and retry, or treat this as fatal.
    #[error("GPU out of memory")]
    OutOfMemory,

    /// A fence or timeline wait exceeded the requested timeout.
    ///
    /// Returned by [`crate::Context::wait_until_timeout`] when the GPU has not
    /// reached the target [`TimelineValue`](crate::TimelineValue) within
    /// the specified `timeout_ms`. The device itself is still healthy.
    #[error("GPU submit timed out")]
    SubmitTimeout,

    /// An unexpected backend error that does not map to the typed variants above.
    #[error(transparent)]
    Backend(#[from] anyhow::Error),
}
