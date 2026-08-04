//! Shared helpers for interactive examples (run limits, perf reporting).

use goldy::{Device, RenderPipeline, RenderPipelineDesc, ShaderModule, SurfaceExchange};
use std::time::{Duration, Instant};
use winit::event_loop::ActiveEventLoop;

/// Rolling frame timestamps for windowed FPS (e.g. last 5s at exit).
#[allow(dead_code)]
pub struct FpsWindow {
    window: Duration,
    frames: Vec<Instant>,
}

#[allow(dead_code)]
impl FpsWindow {
    pub fn new(window_secs: f64) -> Self {
        Self {
            window: Duration::from_secs_f64(window_secs),
            frames: Vec::new(),
        }
    }

    pub fn record(&mut self, now: Instant) {
        self.prune(now);
        self.frames.push(now);
    }

    fn prune(&mut self, now: Instant) {
        let cutoff = now.checked_sub(self.window).unwrap_or(now);
        let keep_from = self.frames.partition_point(|t| *t < cutoff);
        if keep_from > 0 {
            self.frames.drain(..keep_from);
        }
    }

    /// Returns `(frames_in_window, window_span_secs, fps)` for the trailing window.
    pub fn stats(&mut self, now: Instant) -> Option<(u64, f64, f64)> {
        self.prune(now);
        let n = self.frames.len();
        if n == 0 {
            return None;
        }
        let span = now.duration_since(self.frames[0]).as_secs_f64();
        if span <= 0.0 {
            return None;
        }
        Some((n as u64, span, n as f64 / span))
    }
}

/// Build or rebuild a render pipeline using the surface's current format.
#[allow(dead_code)]
pub fn render_pipeline_for_surface(
    device: &Device,
    shader: &ShaderModule,
    surface: &SurfaceExchange,
    desc: RenderPipelineDesc,
) -> anyhow::Result<RenderPipeline> {
    Ok(RenderPipeline::new(
        device,
        shader,
        shader,
        &RenderPipelineDesc {
            target_format: surface.format(),
            ..desc
        },
    )?)
}

/// Run limit in seconds from `GOLDY_EXAMPLE_TIMEOUT` or `EXAMPLE_TIMEOUT`.
pub fn run_limit_secs() -> Option<f64> {
    for key in ["GOLDY_EXAMPLE_TIMEOUT", "EXAMPLE_TIMEOUT"] {
        if let Ok(raw) = std::env::var(key) {
            if let Ok(secs) = raw.parse::<f64>() {
                if secs > 0.0 {
                    return Some(secs);
                }
            }
        }
    }
    None
}

/// Exit the event loop once the run limit elapses so `Drop` can print `GOLDY_PERF`.
#[allow(dead_code)]
pub fn exit_if_timed_out(event_loop: &ActiveEventLoop, start: Instant) {
    if let Some(limit) = run_limit_secs() {
        if start.elapsed() >= Duration::from_secs_f64(limit) {
            event_loop.exit();
        }
    }
}
