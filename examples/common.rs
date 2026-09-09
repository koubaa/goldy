//! Shared helpers for interactive examples (run limits, perf reporting, book captures).
//!
//! Set `GOLDY_EXAMPLE_CAPTURE` to a raw-RGBA output path to render headlessly
//! into pixels that `scripts/record_example_captures.sh` stitches with ffmpeg.
//! Optional: `GOLDY_EXAMPLE_CAPTURE_FRAMES` (default 75), `GOLDY_EXAMPLE_CAPTURE_FPS`
//! (default 15), `GOLDY_EXAMPLE_CAPTURE_WIDTH` / `HEIGHT` (default 640×480).

use goldy::{
    Context, Device, Lease, LeaseRenderTarget, MemoryExchange, NodeAccess, PresentLease, RenderPipeline,
    RenderPipelineDesc, RetainedPool, Scheme, ShaderModule, Submission, SurfaceConfig, SurfaceExchange, Texture,
    TextureFlags, TextureFormat, TextureKind, Transaction, WithdrawTransaction,
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

enum FrameTicket {
    Present(Transaction),
    Withdraw(WithdrawTransaction),
}

enum CaptureWriter {
    File(BufWriter<File>),
    Memory,
}

#[allow(clippy::large_enum_variant)]
enum SinkKind {
    Window {
        surface: SurfaceExchange,
    },
    Capture {
        path: PathBuf,
        writer: CaptureWriter,
        last: Option<Vec<u8>>,
        fps: f32,
        frames: u32,
        written: u32,
        readback: Option<Texture>,
        compute: Option<Texture>,
    },
}

/// Window swapchain or a raw-RGBA dump that ffmpeg can stitch.
pub struct FrameSink {
    format: TextureFormat,
    width: u32,
    height: u32,
    kind: SinkKind,
    ticket: Option<FrameTicket>,
}

/// Colour target for compute-to-surface examples (`with_present` vs `with_parcel`).
pub enum ComputeColorTarget {
    Present(PresentLease),
    Texture,
}

#[allow(dead_code)]
impl FrameSink {
    /// Windowed present when `window` is `Some`; otherwise a capture dump from env.
    pub fn open(ctx: &Context, pool: &mut RetainedPool, window: Option<&Window>) -> anyhow::Result<Self> {
        match window {
            Some(window) => Self::from_window(ctx, window),
            None => Self::from_env(ctx, pool),
        }
    }

    fn from_window(ctx: &Context, window: &Window) -> anyhow::Result<Self> {
        let surface = SurfaceExchange::new(ctx, window, SurfaceConfig::default())?;
        let (width, height) = surface.size();
        Ok(Self {
            format: surface.format(),
            width: width.max(1),
            height: height.max(1),
            kind: SinkKind::Window { surface },
            ticket: None,
        })
    }

    fn capture_size() -> (u32, u32, u32, f32) {
        (
            env_u32("GOLDY_EXAMPLE_CAPTURE_WIDTH", 640),
            env_u32("GOLDY_EXAMPLE_CAPTURE_HEIGHT", 480),
            env_u32("GOLDY_EXAMPLE_CAPTURE_FRAMES", 75),
            env_f32("GOLDY_EXAMPLE_CAPTURE_FPS", 15.0),
        )
    }

    fn acquire_readback(pool: &mut RetainedPool, width: u32, height: u32) -> anyhow::Result<Texture> {
        pool.acquire_texture(
            width,
            height,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
            None,
        )
    }

    fn from_env(_ctx: &Context, pool: &mut RetainedPool) -> anyhow::Result<Self> {
        let path = std::env::var("GOLDY_EXAMPLE_CAPTURE").expect("GOLDY_EXAMPLE_CAPTURE");
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let (width, height, frames, fps) = Self::capture_size();
        let readback = Self::acquire_readback(pool, width, height)?;
        let file = File::create(&path)?;
        Ok(Self {
            format: TextureFormat::Rgba8Unorm,
            width,
            height,
            kind: SinkKind::Capture {
                path,
                writer: CaptureWriter::File(BufWriter::new(file)),
                last: None,
                fps,
                frames,
                written: 0,
                readback: Some(readback),
                compute: None,
            },
            ticket: None,
        })
    }

    /// Packed-RGBA file from env, no GPU resource — stitch panels with [`write_rgba`].
    pub fn rgba_file() -> anyhow::Result<Self> {
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
            format: TextureFormat::Rgba8Unorm,
            width,
            height,
            kind: SinkKind::Capture {
                path,
                writer: CaptureWriter::File(BufWriter::new(file)),
                last: None,
                fps,
                frames,
                written: 0,
                readback: None,
                compute: None,
            },
            ticket: None,
        })
    }

    /// Offscreen RGBA readback that does not write a file (e.g. `multi_window` panels).
    pub fn memory(pool: &mut RetainedPool, width: u32, height: u32) -> anyhow::Result<Self> {
        let fps = env_f32("GOLDY_EXAMPLE_CAPTURE_FPS", 15.0);
        let readback = Self::acquire_readback(pool, width, height)?;
        Ok(Self {
            format: TextureFormat::Rgba8Unorm,
            width,
            height,
            kind: SinkKind::Capture {
                path: PathBuf::new(),
                writer: CaptureWriter::Memory,
                last: None,
                fps,
                frames: u32::MAX,
                written: 0,
                readback: Some(readback),
                compute: None,
            },
            ticket: None,
        })
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn is_capture(&self) -> bool {
        matches!(self.kind, SinkKind::Capture { .. })
    }

    /// Wall-clock time in a window; virtual `written / fps` while capturing.
    pub fn time(&self, start: Instant) -> f32 {
        match &self.kind {
            SinkKind::Capture { fps, written, .. } => *written as f32 / *fps,
            SinkKind::Window { .. } => start.elapsed().as_secs_f32(),
        }
    }

    /// Frame delta: `1 / fps` while capturing, otherwise elapsed since `last`.
    pub fn dt(&self, last: Instant) -> f32 {
        match &self.kind {
            SinkKind::Capture { fps, .. } => 1.0 / *fps,
            SinkKind::Window { .. } => last.elapsed().as_secs_f32(),
        }
    }

    pub fn finished(&self) -> bool {
        match &self.kind {
            SinkKind::Capture { frames, written, .. } => *written >= *frames,
            SinkKind::Window { .. } => false,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        if let SinkKind::Window { surface } = &self.kind {
            surface.resize(width, height)?;
            self.width = width;
            self.height = height;
            self.format = surface.format();
        }
        Ok(())
    }

    /// After the offscreen pass: copy-to-present, or copy-to-readback + withdraw.
    pub fn bind_render_target(&mut self, scheme: &mut Scheme, rt: &Lease<LeaseRenderTarget>) -> anyhow::Result<()> {
        let ticket = match &mut self.kind {
            SinkKind::Window { surface } => FrameTicket::Present(surface.bind_render_target(scheme, rt)?),
            SinkKind::Capture { readback, .. } => {
                let readback = readback.as_ref().expect("capture readback texture");
                scheme.copy_to_texture(rt, readback)?;
                FrameTicket::Withdraw(MemoryExchange::new(scheme.context()).bind_withdraw(scheme, readback)?)
            }
        };
        self.ticket = Some(ticket);
        Ok(())
    }

    /// Windowed `bind_destination`, or a Direct compute colour target for capture.
    pub fn compute_color_target(
        &mut self,
        scheme: &mut Scheme,
        pool: &mut RetainedPool,
    ) -> anyhow::Result<ComputeColorTarget> {
        let (width, height, format) = (self.width, self.height, self.format);
        let (target, ticket) = match &mut self.kind {
            SinkKind::Window { surface } => {
                let (lease, tx) = surface.bind_destination(scheme)?;
                (ComputeColorTarget::Present(lease), Some(FrameTicket::Present(tx)))
            }
            SinkKind::Capture { compute, .. } => {
                if compute.is_none() {
                    *compute = Some(pool.acquire_texture(
                        width,
                        height,
                        format,
                        TextureKind::Direct,
                        TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
                        None,
                    )?);
                }
                (ComputeColorTarget::Texture, None)
            }
        };
        if let Some(ticket) = ticket {
            self.ticket = Some(ticket);
        }
        Ok(target)
    }

    pub fn compute_texture(&self) -> &Texture {
        match &self.kind {
            SinkKind::Capture { compute, .. } => compute.as_ref().expect("compute_color_target first"),
            SinkKind::Window { .. } => panic!("compute_texture is capture-only"),
        }
    }

    /// After recording the compute dispatch: bind a withdraw on the compute target.
    pub fn complete_compute(&mut self, scheme: &mut Scheme) -> anyhow::Result<()> {
        let ticket = match &self.kind {
            SinkKind::Capture { compute, .. } => {
                let tex = compute.as_ref().expect("compute_color_target first");
                Some(FrameTicket::Withdraw(
                    MemoryExchange::new(scheme.context()).bind_withdraw(scheme, tex)?,
                ))
            }
            SinkKind::Window { .. } => None,
        };
        if let Some(ticket) = ticket {
            self.ticket = Some(ticket);
        }
        Ok(())
    }

    /// Present or write one RGBA frame. Capture stops after `GOLDY_EXAMPLE_CAPTURE_FRAMES`.
    pub fn settle(&mut self, submission: &mut Submission) -> anyhow::Result<()> {
        let pixels = {
            let ticket = self
                .ticket
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("FrameSink has no bind ticket"))?;
            match ticket {
                FrameTicket::Present(tx) => {
                    tx.claim(submission)?.consume()?;
                    None
                }
                FrameTicket::Withdraw(tx) => Some(tx.claim(submission)?.consume()?),
            }
        };
        if let Some(pixels) = pixels {
            self.write_rgba(&pixels)?;
        }
        Ok(())
    }

    /// Append packed RGBA8 (`width * height * 4` bytes) as one capture frame.
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
        let (width, height) = (self.width, self.height);
        let SinkKind::Capture {
            writer,
            last,
            written,
            frames,
            path,
            ..
        } = &mut self.kind
        else {
            return Ok(());
        };
        match writer {
            CaptureWriter::File(file) => file.write_all(pixels)?,
            CaptureWriter::Memory => {}
        }
        *last = Some(pixels.to_vec());
        *written += 1;
        if let CaptureWriter::File(file) = writer {
            if *written >= *frames {
                file.flush()?;
                println!(
                    "GOLDY_CAPTURE: wrote {written} frames ({width}x{height} rgba) to {}",
                    path.display()
                );
            }
        }
        Ok(())
    }

    /// Last packed RGBA frame from a capture/`memory` sink.
    pub fn take_rgba(&mut self) -> Option<Vec<u8>> {
        match &mut self.kind {
            SinkKind::Capture { last, .. } => last.take(),
            SinkKind::Window { .. } => None,
        }
    }

    pub fn as_surface(&self) -> Option<&SurfaceExchange> {
        match &self.kind {
            SinkKind::Window { surface } => Some(surface),
            SinkKind::Capture { .. } => None,
        }
    }
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

#[allow(dead_code)]
pub fn bind_compute_node<'a>(
    node: goldy::SchemeNodeBuilder<'a>,
    target: &'a ComputeColorTarget,
    sink: &'a FrameSink,
) -> goldy::SchemeNodeBuilder<'a> {
    match target {
        ComputeColorTarget::Present(lease) => node.with_present(lease),
        ComputeColorTarget::Texture => node.with_parcel(sink.compute_texture(), NodeAccess::Write),
    }
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
