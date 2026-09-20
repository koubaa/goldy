//! Compute-to-Surface example — pure compute rendering without a graphics pipeline.
//!
//! Demonstrates present-on-scheme: a retained [`Scheme`] writes directly to a
//! drawable from [`SurfaceExchange::bind_destination`], then presents via
//! [`Transaction::claim`] and [`Claim::consume`].
//!
//! Run with: cargo run --example compute_to_surface

use anyhow::Result;
use goldy::{
    Buffer, BufferKind, DepositTarget, DepositTransaction, DeviceDescriptor, Instance, MemoryExchange, PresentMode,
    RequestAdapterOptions, Scheme, SurfaceConfig, SurfaceExchange, Texture, Transaction, WithdrawTransaction,
};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};
mod common;
use common::CaptureDump;

#[goldy::gpu]
struct Uniforms {
    width: u32,
    height: u32,
    time: f32,
}

#[goldy::compute(workgroup_size = [8, 8, 1])]
fn plasma(uniforms: &[Uniforms], output: goldy::gpu::DirectSpatial<goldy::gpu::Float4>) {
    let tid = goldy::gpu::global_id();
    let u: Uniforms = uniforms[0];
    if tid.x >= u.width || tid.y >= u.height {
        return;
    }

    let uv = goldy::gpu::float2(tid.x as f32 / u.width as f32, tid.y as f32 / u.height as f32);
    let mut p = uv * 2.0 - 1.0;
    p.x *= u.width as f32 / u.height as f32;

    let mut v = 0.0;
    v += goldy::gpu::sin(p.x * 6.0 + u.time);
    v += goldy::gpu::sin(p.y * 6.0 + u.time * 1.3);
    v += goldy::gpu::sin((p.x + p.y) * 4.0 + u.time * 0.7);
    v += goldy::gpu::sin(goldy::gpu::length(p) * 8.0 - u.time * 2.0);
    v *= 0.25;

    let col = goldy::gpu::float3(
        0.5 + 0.5 * goldy::gpu::sin(v * 3.14159 + 0.0),
        0.5 + 0.5 * goldy::gpu::sin(v * 3.14159 + 2.094),
        0.5 + 0.5 * goldy::gpu::sin(v * 3.14159 + 4.188),
    );
    output[tid.xy] = goldy::gpu::float4(col.x, col.y, col.z, 1.0);
}

const INITIAL_WIDTH: u32 = 800;
const INITIAL_HEIGHT: u32 = 600;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    if common::capture_requested() {
        let warmup = warm_gpu()?;
        let mut app = App {
            warmup: Some(warmup),
            state: None,
        };
        app.init(None)?;
        let state = app.state.as_mut().expect("capture state");
        while !state.capture.as_ref().is_none_or(CaptureDump::finished) {
            render_frame(state)?;
        }
        return Ok(());
    }

    println!("Goldy — Compute to Surface Example");
    println!("===================================");
    println!("Press V to toggle vsync, Escape to exit\n");

    println!("Initializing GPU...");
    let warmup = warm_gpu()?;
    println!("GPU ready.");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        warmup: Some(warmup),
        state: None,
    };
    event_loop.run_app(&mut app)?;

    Ok(())
}

/// Device, context, and compiled compute pipeline — everything except the window/surface.
struct GpuWarmup {
    ctx: goldy::Context,
    kernel: plasma::Kernel,
    device: Arc<goldy::Device>,
}

fn warm_gpu() -> Result<GpuWarmup> {
    let instance = Instance::new()?;
    let device = Arc::new(
        instance
            .request_adapter(&RequestAdapterOptions::default())?
            .request_device(&DeviceDescriptor::default())?,
    );
    let ctx = device.create_context()?;
    let kernel = plasma::Kernel::prepare(&device)?;
    Ok(GpuWarmup {
        ctx,
        kernel,
        device,
    })
}

#[derive(Default)]
struct App {
    warmup: Option<GpuWarmup>,
    state: Option<RenderState>,
}

struct RenderState {
    window: Option<Arc<Window>>,
    ctx: goldy::Context,
    surface: Option<SurfaceExchange>,
    present: Option<Transaction>,
    capture: Option<CaptureDump>,
    readback: Option<Texture>,
    withdraw: Option<WithdrawTransaction>,
    scheme: Scheme,
    compute_pipeline: plasma::Kernel,
    uniform_buffer: Buffer,
    upload_scheme: Scheme,
    uniform_deposit: DepositTransaction,
    start_time: Instant,
    vsync: bool,
    frame_count: u32,
}

fn record_scheme(
    scheme: &mut Scheme,
    kernel: &plasma::Kernel,
    uniform: &Buffer,
    width: u32,
    height: u32,
    surface: Option<&SurfaceExchange>,
    readback: Option<&Texture>,
) -> Result<(Option<Transaction>, Option<WithdrawTransaction>)> {
    if let Some(surface) = surface {
        let (lease, present) = surface.bind_destination(scheme)?;
        kernel
            .record(scheme, "compute", uniform, &lease)
            .over_2d(width, height);
        Ok((Some(present), None))
    } else {
        let target = readback.expect("capture readback");
        kernel
            .record(scheme, "compute", uniform, target)
            .over_2d(width, height);
        let withdraw = MemoryExchange::new(scheme.context()).bind_withdraw(scheme, target)?;
        Ok((None, Some(withdraw)))
    }
}

fn output_size(state: &RenderState) -> (u32, u32) {
    if let Some(surface) = &state.surface {
        surface.size()
    } else {
        state.capture.as_ref().expect("capture").size()
    }
}

fn rebuild_scheme(state: &mut RenderState, width: u32, height: u32) {
    let mut scheme = Scheme::new(&state.ctx);
    let (present, withdraw) = record_scheme(
        &mut scheme,
        &state.compute_pipeline,
        &state.uniform_buffer,
        width,
        height,
        state.surface.as_ref(),
        state.readback.as_ref(),
    )
    .expect("failed to record scheme");
    state.present = present;
    state.withdraw = withdraw;
    state.scheme = scheme;
}

impl App {
    fn init(&mut self, window: Option<Arc<Window>>) -> Result<()> {
        let warmup = self
            .warmup
            .take()
            .ok_or_else(|| anyhow::anyhow!("GPU warmup state missing"))?;
        let GpuWarmup {
            ctx,
            kernel: compute_pipeline,
            device,
            ..
        } = warmup;

        let (surface, capture, readback, width, height) = if let Some(window) = window.as_deref() {
            let surface = SurfaceExchange::new(&ctx, window, SurfaceConfig::default())?;
            let (width, height) = surface.size();
            (Some(surface), None, None, width, height)
        } else {
            let capture = CaptureDump::from_env()?;
            let (width, height) = capture.size();
            let readback = common::capture_readback(&device, width, height)?;
            (None, Some(capture), Some(readback), width, height)
        };

        let uniform_buffer = device.acquire_buffer_with_data(
            &[Uniforms {
                width,
                height,
                time: 0.0,
            }],
            BufferKind::Scattered,
        )?;

        let mut scheme = Scheme::new(&ctx);
        let (present, withdraw) = record_scheme(
            &mut scheme,
            &compute_pipeline,
            &uniform_buffer,
            width,
            height,
            surface.as_ref(),
            readback.as_ref(),
        )?;

        let mut upload_scheme = Scheme::new(&ctx);
        let uniform_deposit = MemoryExchange::new(&ctx).bind_deposit(
            &mut upload_scheme,
            DepositTarget::buffer_elements::<Uniforms>(&uniform_buffer, 1),
        )?;

        self.state = Some(RenderState {
            window,
            ctx,
            surface,
            present,
            capture,
            readback,
            withdraw,
            scheme,
            compute_pipeline,
            uniform_buffer,
            upload_scheme,
            uniform_deposit,
            start_time: Instant::now(),
            vsync: true,
            frame_count: 0,
        });

        Ok(())
    }
}

impl Drop for RenderState {
    fn drop(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 {
            self.frame_count as f64 / elapsed
        } else {
            0.0
        };
        println!(
            "GOLDY_PERF: frames={} elapsed={elapsed:.2}s avg_fps={fps:.1}",
            self.frame_count
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = common::hidden_window("Goldy — Compute to Surface", INITIAL_WIDTH, INITIAL_HEIGHT);

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        if let Err(e) = self.init(Some(window.clone())) {
            tracing::error!("Failed to initialize: {}", e);
            event_loop.exit();
            return;
        }

        if let Some(state) = &mut self.state {
            if let Err(e) = render_frame(state) {
                tracing::error!("First frame error: {e}");
            }
        }
        common::reveal_window(&window);
        window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            common::exit_if_timed_out(event_loop, state.start_time);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => match event.logical_key.as_ref() {
                Key::Named(NamedKey::Escape) => event_loop.exit(),
                Key::Character("v") => {
                    state.vsync = !state.vsync;
                    let mode = if state.vsync {
                        PresentMode::Fifo
                    } else {
                        PresentMode::Immediate
                    };
                    if let Some(surface) = &state.surface {
                        if let Err(e) = surface.set_present_mode(mode) {
                            eprintln!("Failed to set present mode: {e}");
                        } else {
                            println!(
                                "Vsync: {} (present mode: {:?})",
                                if state.vsync { "ON" } else { "OFF" },
                                mode
                            );
                        }
                    }
                }
                _ => {}
            },
            WindowEvent::Resized(new_size) if new_size.width > 0 && new_size.height > 0 => {
                if let Some(surface) = &state.surface {
                    let _ = surface.resize(new_size.width, new_size.height);
                    rebuild_scheme(state, new_size.width, new_size.height);
                }
                if let Some(window) = &state.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = render_frame(state) {
                    tracing::error!("Render error: {}", e);
                }
                if let Some(window) = &state.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn render_frame(state: &mut RenderState) -> Result<()> {
    state.frame_count += 1;

    let (width, height) = output_size(state);
    if width == 0 || height == 0 {
        return Ok(());
    }

    let elapsed = state
        .capture
        .as_ref()
        .map(CaptureDump::time)
        .unwrap_or_else(|| state.start_time.elapsed().as_secs_f32());
    let uniforms = Uniforms {
        width,
        height,
        time: elapsed,
    };

    state.uniform_deposit.write_data(0, &[uniforms])?;
    state.upload_scheme.submit()?;

    let mut submission = state.scheme.submit()?;
    if let Some(present) = &state.present {
        present.claim(&mut submission)?.consume()?;
    } else {
        let pixels = state.withdraw.as_ref().unwrap().claim(&mut submission)?.consume()?;
        state.capture.as_mut().unwrap().write_rgba(&pixels)?;
    }

    Ok(())
}
