//! Shared helpers for interactive examples (run limits, perf reporting).

use goldy::{Device, RenderPipeline, RenderPipelineDesc, ShaderModule, SwapchainPool};
use std::time::{Duration, Instant};
use winit::event_loop::ActiveEventLoop;

/// Build or rebuild a render pipeline using the swapchain's current format.
#[allow(dead_code)]
pub fn render_pipeline_for_swapchain(
    device: &Device,
    shader: &ShaderModule,
    swapchain: &SwapchainPool,
    desc: RenderPipelineDesc,
) -> anyhow::Result<RenderPipeline> {
    Ok(RenderPipeline::new(
        device,
        shader,
        shader,
        &RenderPipelineDesc {
            target_format: swapchain.format(),
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
pub fn exit_if_timed_out(event_loop: &ActiveEventLoop, start: Instant) {
    if let Some(limit) = run_limit_secs() {
        if start.elapsed() >= Duration::from_secs_f64(limit) {
            event_loop.exit();
        }
    }
}
