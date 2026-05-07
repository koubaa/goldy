//! Structured instrumentation for Goldy.
//!
//! Provides named observation points with structured context data.
//! Zero-cost when the `instrumentation` feature is disabled.
//!
//! ## Debug Paths
//!
//! The [`debug_log_path`] and [`shader_dump_path`] functions return
//! cross-platform paths relative to `CWD/.goldy_debug/` so that the same
//! instrumented binary works on macOS, Windows, and Linux without edits.
//!
//! # Observation Points
//!
//! Goldy uses hierarchical dot-notation for observation point names:
//!
//! | Category | Point Name | Emitted Data |
//! |----------|------------|--------------|
//! | **Slang** | `slang.library.load` | `path`, `success` |
//! | | `slang.compile.start` | `target`, `entry_points`, `bindless` |
//! | | `slang.compile.end` | `duration_ms`, `output_size`, `success` |
//! | | `slang.reflection.extract` | `parameter_blocks`, `fields` |
//! | **Shader** | `shader.module.create` | `backend`, `shader_type` |
//! | | `shader.pipeline.create` | `pipeline_type`, `bind_groups` |
//! | **Resource** | `resource.buffer.create` | `size`, `usage` |
//! | | `resource.texture.create` | `dimensions`, `format` |
//! | | `resource.bind_group.create` | `bindings_count` |
//! | **Render** | `render.frame.start` | `frame_id` |
//! | | `render.compute.dispatch` | `workgroups`, `pipeline` |
//! | | `render.draw` | `vertices`, `instances` |
//! | | `render.frame.end` | `frame_id`, `duration_ms` |
//!
//! # Usage
//!
//! ```rust,ignore
//! use goldy::{goldy_span, goldy_event};
//!
//! fn render(&mut self) {
//!     let _frame_span = goldy_span!("render.frame", frame_id = self.frame_count).entered();
//!     
//!     // ... rendering code ...
//!     
//!     goldy_event!("render.frame.end", frame_id = self.frame_count);
//! }
//! ```
//!
//! # Filtering
//!
//! Use environment variables to filter instrumentation output:
//! - `RUST_LOG=goldy=debug` - Enable all Goldy instrumentation
//! - `RUST_LOG=goldy::render=trace` - Enable only render-related points

pub mod debug_paths;

#[cfg(feature = "instrumentation")]
mod json_subscriber;

#[cfg(feature = "instrumentation")]
pub use json_subscriber::JsonFileLayer;

/// Target name for all Goldy instrumentation (enables filtering).
pub const TARGET: &str = "goldy";

/// Create a span for timing a section of code.
///
/// Spans automatically track entry/exit timing and support nested hierarchies.
///
/// # Example
///
/// ```rust,ignore
/// use goldy::goldy_span;
///
/// fn compile_shader(&self) {
///     let _span = goldy_span!("slang.compile", target = "metal").entered();
///     // ... compilation code ...
///     // Span automatically records duration when dropped
/// }
/// ```
#[cfg(feature = "instrumentation")]
#[macro_export]
macro_rules! goldy_span {
    ($name:expr $(, $($rest:tt)*)?) => {
        tracing::span!(
            target: $crate::instrumentation::TARGET,
            tracing::Level::DEBUG,
            $name
            $(, $($rest)*)?
        )
    };
}

/// No-op version when instrumentation is disabled.
#[cfg(not(feature = "instrumentation"))]
#[macro_export]
macro_rules! goldy_span {
    ($name:expr $(, $($rest:tt)*)?) => {
        $crate::instrumentation::NoopSpan
    };
}

/// Emit a structured event at an observation point.
///
/// Events are instantaneous markers with associated data.
///
/// # Example
///
/// ```rust,ignore
/// use goldy::goldy_event;
///
/// goldy_event!("slang.library.load",
///     path = %lib_path.display(),
///     success = true
/// );
/// ```
#[cfg(feature = "instrumentation")]
#[macro_export]
macro_rules! goldy_event {
    ($name:expr $(, $($rest:tt)*)?) => {
        tracing::event!(
            target: $crate::instrumentation::TARGET,
            tracing::Level::DEBUG,
            name = $name
            $(, $($rest)*)?
        )
    };
}

/// No-op version when instrumentation is disabled.
#[cfg(not(feature = "instrumentation"))]
#[macro_export]
macro_rules! goldy_event {
    ($name:expr $(, $($rest:tt)*)?) => {
        // No-op
    };
}

/// A no-op span guard that does nothing when instrumentation is disabled.
#[cfg(not(feature = "instrumentation"))]
pub struct NoopSpan;

#[cfg(not(feature = "instrumentation"))]
impl NoopSpan {
    /// No-op enter that returns self.
    #[inline]
    pub fn entered(self) -> Self {
        self
    }
}

/// Install a JSON file logger for Goldy instrumentation.
///
/// This creates a tracing subscriber that writes structured JSON logs to the
/// specified file path. Only events targeting "goldy" are captured.
///
/// # Example
///
/// ```rust,ignore
/// use goldy::instrumentation::install_json_logger;
///
/// // At application startup
/// install_json_logger("/tmp/goldy-debug.json")?;
///
/// // Now all goldy_span!/goldy_event! calls will be logged to the file
/// ```
#[cfg(feature = "instrumentation")]
pub fn install_json_logger(path: &str) -> std::io::Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    let json_layer = JsonFileLayer::new(path)?;

    tracing_subscriber::registry()
        .with(json_layer)
        .try_init()
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    Ok(())
}

/// No-op version when instrumentation is disabled.
#[cfg(not(feature = "instrumentation"))]
pub fn install_json_logger(_path: &str) -> std::io::Result<()> {
    Ok(())
}
