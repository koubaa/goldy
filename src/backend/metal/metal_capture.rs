//! Opt-in programmatic Metal GPU capture (`MTLCaptureManager`).
//!
//! Enable with `GOLDY_METAL_CAPTURE`:
//!
//! | Value | Behaviour |
//! |-------|-----------|
//! | `1` / `true` / `yes` | Capture to Xcode Developer Tools |
//! | `/path/to/out.gputrace` | Write a `.gputrace` document |
//! | `path,skip=N,frames=M` | Path (or `1`) plus warm-up / frame count |
//!
//! Defaults: `skip=60`, `frames=1`. Requires `METAL_CAPTURE_ENABLED=1` (set
//! automatically by the Metal backend when this env var is present).
//!
//! Example:
//! ```bash
//! GOLDY_METAL_CAPTURE=/tmp/ekrano.gputrace,skip=120,frames=1 \
//!   target/release/with_winit_bin --no-vsync
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::LazyLock;

use ::metal as mtl;

/// Parsed `GOLDY_METAL_CAPTURE` configuration.
#[derive(Clone, Debug)]
pub struct MetalCaptureConfig {
    /// `None` → Developer Tools destination; `Some(path)` → `.gputrace` file.
    pub output: Option<PathBuf>,
    /// Submits to skip before starting capture (warm-up).
    pub skip: u64,
    /// Number of submits to capture once warm-up is done.
    pub frames: u64,
}

impl MetalCaptureConfig {
    pub fn from_env() -> Option<Self> {
        let Ok(raw) = std::env::var("GOLDY_METAL_CAPTURE") else {
            return None;
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut output: Option<PathBuf> = None;
        let mut skip = 60_u64;
        let mut frames = 1_u64;
        let mut saw_destination = false;

        for part in trimmed.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some(rest) = part.strip_prefix("skip=").or_else(|| part.strip_prefix("skip:")) {
                if let Ok(n) = rest.trim().parse() {
                    skip = n;
                }
                continue;
            }
            if let Some(rest) = part.strip_prefix("frames=").or_else(|| part.strip_prefix("frames:")) {
                if let Ok(n) = rest.trim().parse::<u64>() {
                    frames = n.max(1);
                }
                continue;
            }
            let lower = part.to_ascii_lowercase();
            if matches!(lower.as_str(), "1" | "true" | "yes") {
                // Developer Tools destination (no file).
                output = None;
                saw_destination = true;
                continue;
            }
            output = Some(PathBuf::from(part));
            saw_destination = true;
        }

        if !saw_destination {
            // Only skip=/frames= tokens: still enable Developer Tools capture.
            output = None;
        }

        Some(Self { output, skip, frames })
    }
}

static CONFIG: LazyLock<Option<MetalCaptureConfig>> = LazyLock::new(MetalCaptureConfig::from_env);
static SUBMIT_COUNT: AtomicU64 = AtomicU64::new(0);
static CAPTURING: AtomicBool = AtomicBool::new(false);
static CAPTURES_DONE: AtomicU64 = AtomicU64::new(0);
static FINISHED: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn enabled() -> bool {
    CONFIG.is_some()
}

/// Call immediately before `new_command_buffer` on a capture-eligible submit.
///
/// Returns `true` if a capture session was started for this submit.
pub fn begin_submit(command_queue: &mtl::CommandQueueRef) -> bool {
    let Some(cfg) = CONFIG.as_ref() else {
        return false;
    };
    if FINISHED.load(Ordering::Relaxed) {
        return false;
    }

    let n = SUBMIT_COUNT.fetch_add(1, Ordering::Relaxed);
    if n < cfg.skip {
        return false;
    }
    if CAPTURES_DONE.load(Ordering::Relaxed) >= cfg.frames {
        FINISHED.store(true, Ordering::Relaxed);
        return false;
    }
    if CAPTURING.load(Ordering::Relaxed) {
        return true;
    }

    let manager = mtl::CaptureManager::shared();
    if manager.is_capturing() {
        tracing::warn!("GOLDY_METAL_CAPTURE: CaptureManager already capturing; skipping start");
        return false;
    }

    let descriptor = mtl::CaptureDescriptor::new();
    descriptor.set_capture_command_queue(command_queue);

    match &cfg.output {
        Some(path) => {
            if !manager.supports_destination(mtl::MTLCaptureDestination::GpuTraceDocument) {
                tracing::error!(
                    "GOLDY_METAL_CAPTURE: GpuTraceDocument destination unsupported \
                     (is METAL_CAPTURE_ENABLED=1 set?)"
                );
                FINISHED.store(true, Ordering::Relaxed);
                return false;
            }
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            // Metal refuses to overwrite an existing .gputrace.
            if path.exists() {
                let _ = std::fs::remove_dir_all(path);
                let _ = std::fs::remove_file(path);
            }
            descriptor.set_destination(mtl::MTLCaptureDestination::GpuTraceDocument);
            descriptor.set_output_url(path);
            tracing::info!(
                path = %path.display(),
                skip = cfg.skip,
                frames = cfg.frames,
                submit = n,
                "GOLDY_METAL_CAPTURE: starting GpuTraceDocument capture"
            );
        }
        None => {
            descriptor.set_destination(mtl::MTLCaptureDestination::DeveloperTools);
            tracing::info!(
                skip = cfg.skip,
                frames = cfg.frames,
                submit = n,
                "GOLDY_METAL_CAPTURE: starting DeveloperTools capture"
            );
        }
    }

    match manager.start_capture(&descriptor) {
        Ok(()) => {
            CAPTURING.store(true, Ordering::Relaxed);
            true
        }
        Err(e) => {
            tracing::error!("GOLDY_METAL_CAPTURE: start_capture failed: {e}");
            FINISHED.store(true, Ordering::Relaxed);
            false
        }
    }
}

/// Call after the command buffer for a capturing submit has been committed.
pub fn end_submit() {
    if !CAPTURING.swap(false, Ordering::Relaxed) {
        return;
    }
    let manager = mtl::CaptureManager::shared();
    manager.stop_capture();
    let done = CAPTURES_DONE.fetch_add(1, Ordering::Relaxed) + 1;
    if let Some(cfg) = CONFIG.as_ref() {
        tracing::info!(
            done,
            total = cfg.frames,
            path = ?cfg.output.as_ref().map(|p| p.display().to_string()),
            "GOLDY_METAL_CAPTURE: stopped capture"
        );
        if done >= cfg.frames {
            FINISHED.store(true, Ordering::Relaxed);
        }
    }
}

/// RAII helper: stops an in-progress capture when dropped.
///
/// Construct only after a successful [`begin_submit`] that returned `true`.
pub struct CaptureSession {
    active: bool,
}

impl CaptureSession {
    pub fn start(command_queue: &mtl::CommandQueueRef) -> Self {
        Self {
            active: begin_submit(command_queue),
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}

impl Drop for CaptureSession {
    fn drop(&mut self) {
        if self.active {
            end_submit();
            self.active = false;
        }
    }
}
