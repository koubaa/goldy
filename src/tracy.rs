//! Tracy profiler integration.
//!
//! Provides thin wrappers around the [`tracy-client`](https://docs.rs/tracy-client) crate
//! that compile to no-ops when the `tracy` feature is disabled.
//!
//! # Quick start
//!
//! 1. Build with `--features tracy`
//! 2. Launch the [Tracy profiler](https://github.com/wolfpld/tracy) GUI
//! 3. It will auto-discover the running application via broadcast
//!
//! # Macros
//!
//! | Macro | Purpose |
//! |-------|---------|
//! | [`tracy_zone!()`][macro@crate::tracy_zone] | CPU zone — measures wall time of the enclosing scope |
//! | [`tracy_frame_mark!()`][macro@crate::tracy_frame_mark] | Main frame boundary marker |
//! | [`tracy_plot!()`][macro@crate::tracy_plot] | Numeric value plotted over time |

/// Mark a CPU profiling zone that lasts until the binding is dropped.
///
/// ```rust,ignore
/// let _z = tracy_zone!("submit");
/// let _z = tracy_zone!("dispatch", "my_shader.slang");
/// ```
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! tracy_zone {
    ($name:expr) => {
        $crate::_tracy_client::span!($name, 0)
    };
    ($name:expr, $text:expr) => {{
        let mut span = $crate::_tracy_client::span!($name, 0);
        span.emit_text($text);
        span
    }};
}

#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! tracy_zone {
    ($name:expr) => {
        $crate::tracy::NoopZone
    };
    ($name:expr, $text:expr) => {
        $crate::tracy::NoopZone
    };
}

/// Mark the end of the main frame (Tracy's primary frame counter).
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! tracy_frame_mark {
    () => {
        $crate::_tracy_client::frame_mark()
    };
}

#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! tracy_frame_mark {
    () => {};
}

/// Plot a named numeric value over time in Tracy's plot view.
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! tracy_plot {
    ($name:expr, $value:expr) => {
        $crate::_tracy_client::plot!($name, $value)
    };
}

#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! tracy_plot {
    ($name:expr, $value:expr) => {};
}

/// No-op zone guard when Tracy is disabled.
#[cfg(not(feature = "tracy"))]
pub struct NoopZone;
