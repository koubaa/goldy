//! Shared helpers for interactive examples (run limits, perf reporting, book captures).
//!
//! Set `GOLDY_EXAMPLE_CAPTURE` to a raw-RGBA output path to render headlessly
//! into pixels that `scripts/record_example_captures.sh` stitches with ffmpeg.
//! Optional: `GOLDY_EXAMPLE_CAPTURE_FRAMES` (default 75), `GOLDY_EXAMPLE_CAPTURE_FPS`
//! (default 15), `GOLDY_EXAMPLE_CAPTURE_WIDTH` / `HEIGHT` (default 640×480).
//!
//! Capture is file I/O plus a retained RGBA texture. Present still goes through
//! [`SurfaceExchange`] / [`MemoryExchange`] in each example.

use goldy::{
    Device, RenderPipeline, RenderPipelineDesc, RetainedPool, ShaderModule, SurfaceExchange, Texture, TextureFlags,
    TextureFormat, TextureKind,
};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

/// Window attributes for examples that reveal only after the first frame is ready.
#[allow(dead_code)]
pub fn hidden_window(title: impl Into<String>, width: u32, height: u32) -> WindowAttributes {
    Window::default_attributes()
        .with_title(title.into())
        .with_inner_size(winit::dpi::LogicalSize::new(width, height))
        .with_visible(false)
}

/// Show a window after GPU init and an initial present path have completed.
pub fn reveal_window(window: &Window) {
    window.set_visible(true);
}

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

/// Build or rebuild a render pipeline for a colour-target format.
#[allow(dead_code)]
pub fn render_pipeline(
    device: &Device,
    shader: &ShaderModule,
    format: TextureFormat,
    desc: RenderPipelineDesc,
) -> anyhow::Result<RenderPipeline> {
    RenderPipeline::new(
        device,
        shader,
        shader,
        &RenderPipelineDesc {
            target_format: format,
            ..desc
        },
    )
}

/// Build or rebuild a render pipeline using the surface's current format.
#[allow(dead_code)]
pub fn render_pipeline_for_surface(
    device: &Device,
    shader: &ShaderModule,
    surface: &SurfaceExchange,
    desc: RenderPipelineDesc,
) -> anyhow::Result<RenderPipeline> {
    render_pipeline(device, shader, surface.format(), desc)
}

/// True when this process should dump frames instead of opening a window.
pub fn capture_requested() -> bool {
    match std::env::var("GOLDY_EXAMPLE_CAPTURE") {
        Ok(path) => !path.is_empty(),
        Err(_) => false,
    }
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.parse().ok())
        .filter(|n| *n > 0.0)
        .unwrap_or(default)
}

/// Packed-RGBA dump for mdBook clips (`GOLDY_EXAMPLE_CAPTURE`). Not a Goldy type.
pub struct CaptureDump {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    last: Option<Vec<u8>>,
    fps: f32,
    frames: u32,
    written: u32,
    width: u32,
    height: u32,
}

#[allow(dead_code)]
impl CaptureDump {
    fn capture_size() -> (u32, u32, u32, f32) {
        (
            env_u32("GOLDY_EXAMPLE_CAPTURE_WIDTH", 640),
            env_u32("GOLDY_EXAMPLE_CAPTURE_HEIGHT", 480),
            env_u32("GOLDY_EXAMPLE_CAPTURE_FRAMES", 75),
            env_f32("GOLDY_EXAMPLE_CAPTURE_FPS", 15.0),
        )
    }

    /// File dump from `GOLDY_EXAMPLE_CAPTURE`.
    pub fn from_env() -> anyhow::Result<Self> {
        let path = std::env::var("GOLDY_EXAMPLE_CAPTURE").expect("GOLDY_EXAMPLE_CAPTURE");
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let (width, height, frames, fps) = Self::capture_size();
        let file = File::create(&path)?;
        Ok(Self {
            path,
            writer: Some(BufWriter::new(file)),
            last: None,
            fps,
            frames,
            written: 0,
            width,
            height,
        })
    }

    /// In-memory frames only (e.g. `multi_window` panels before hstack).
    pub fn memory(width: u32, height: u32) -> Self {
        let fps = env_f32("GOLDY_EXAMPLE_CAPTURE_FPS", 15.0);
        Self {
            path: PathBuf::new(),
            writer: None,
            last: None,
            fps,
            frames: u32::MAX,
            written: 0,
            width,
            height,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn format() -> TextureFormat {
        TextureFormat::Rgba8Unorm
    }

    /// Virtual clock `written / fps` so clips are deterministic.
    pub fn time(&self) -> f32 {
        self.written as f32 / self.fps
    }

    pub fn dt(&self) -> f32 {
        1.0 / self.fps
    }

    pub fn finished(&self) -> bool {
        self.written >= self.frames
    }

    pub fn write_rgba(&mut self, pixels: &[u8]) -> anyhow::Result<()> {
        let expected = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        anyhow::ensure!(
            pixels.len() == expected,
            "capture frame is {} bytes, expected {expected} ({}x{} rgba)",
            pixels.len(),
            self.width,
            self.height
        );
        if let Some(file) = &mut self.writer {
            file.write_all(pixels)?;
        }
        self.last = Some(pixels.to_vec());
        self.written += 1;
        if let Some(file) = &mut self.writer {
            if self.written >= self.frames {
                file.flush()?;
                println!(
                    "GOLDY_CAPTURE: wrote {} frames ({}x{} rgba) to {}",
                    self.written,
                    self.width,
                    self.height,
                    self.path.display()
                );
            }
        }
        Ok(())
    }

    pub fn take_rgba(&mut self) -> Option<Vec<u8>> {
        self.last.take()
    }
}

/// Retained RGBA8 texture for `copy_to_texture` + [`goldy::MemoryExchange::bind_withdraw`].
#[allow(dead_code)]
pub fn capture_readback(pool: &mut RetainedPool, width: u32, height: u32) -> anyhow::Result<Texture> {
    pool.acquire_texture(
        width,
        height,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        None,
    )
}

/// Horizontal concat of equal-sized packed RGBA panels (for `multi_window` capture).
#[allow(dead_code)]
pub fn hstack_rgba(panels: &[&[u8]], width: u32, height: u32) -> anyhow::Result<Vec<u8>> {
    let row_bytes = width as usize * 4;
    let panel_bytes = row_bytes * height as usize;
    for (i, panel) in panels.iter().enumerate() {
        anyhow::ensure!(
            panel.len() == panel_bytes,
            "hstack panel {i} is {} bytes, expected {panel_bytes}",
            panel.len()
        );
    }
    let n = panels.len();
    let mut out = vec![0u8; panel_bytes.saturating_mul(n)];
    for y in 0..height as usize {
        for (i, panel) in panels.iter().enumerate() {
            let src = y * row_bytes;
            let dst = y * row_bytes * n + i * row_bytes;
            out[dst..dst + row_bytes].copy_from_slice(&panel[src..src + row_bytes]);
        }
    }
    Ok(out)
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
