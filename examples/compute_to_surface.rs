//! Compute-to-Surface example — pure compute rendering without a graphics pipeline.
//!
//! Demonstrates present-on-scheme: a retained [`Scheme`] writes directly to a
//! drawable from [`SurfaceExchange::bind_destination`], then presents via
//! [`Transaction::claim`] and [`Claim::consume`].
//!
//! Run with: cargo run --example compute_to_surface

use anyhow::Result;
use goldy::{
    task_graph::NodeAccess, Buffer, BufferKind, ComputePipeline, DeviceDescriptor, Instance, PresentMode,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, Transaction,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};
mod common;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    time: f32,
    _padding: f32,
}
impl goldy::StructuredBufferElement for Uniforms {}

const COMPUTE_SHADER: &str = r#"
import goldy_exp;

struct Uniforms {
    uint width;
    uint height;
    float time;
    float _padding;
};

[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(BufRO<Uniforms> uniforms_buf, DirectSpatial<float4> output, ThreadId tid) {
    Uniforms u = uniforms_buf[0];

    if (tid.x >= u.width || tid.y >= u.height)
        return;

    float2 uv = float2(float(tid.x) / float(u.width),
                       float(tid.y) / float(u.height));
    float2 p = uv * 2.0 - 1.0;
    p.x *= float(u.width) / float(u.height);

    float t = u.time;
    float v = 0.0;
    v += sin(p.x * 6.0 + t);
    v += sin(p.y * 6.0 + t * 1.3);
    v += sin((p.x + p.y) * 4.0 + t * 0.7);
    v += sin(length(p) * 8.0 - t * 2.0);
    v *= 0.25;

    float3 col = float3(0.5 + 0.5 * sin(v * 3.14159 + 0.0),
                        0.5 + 0.5 * sin(v * 3.14159 + 2.094),
                        0.5 + 0.5 * sin(v * 3.14159 + 4.188));
    output[tid.xy] = float4(col, 1.0);
}
"#;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    println!("Goldy — Compute to Surface Example");
    println!("===================================");
    println!("Press V to toggle vsync, Escape to exit\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[derive(Default)]
struct App {
    state: Option<RenderState>,
}

struct RenderState {
    window: Arc<Window>,
    ctx: goldy::Context,
    surface: SurfaceExchange,
    present: Transaction,
    scheme: Scheme,
    compute_pipeline: ComputePipeline,
    _retained_pool: RetainedPool,
    uniform_buffer: Buffer,
    start_time: std::time::Instant,
    vsync: bool,
    frame_count: u32,
}

fn record_scheme(
    scheme: &mut Scheme,
    surface: &SurfaceExchange,
    pipeline: &ComputePipeline,
    uniform: &Buffer,
    width: u32,
    height: u32,
) -> Result<Transaction> {
    let (lease, present_tx) = surface.bind_destination(scheme)?;
    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);
    scheme
        .node("compute", pipeline)
        .with_parcel(uniform, NodeAccess::Read)
        .with_present(&lease)
        .dispatch(wg_x, wg_y, 1);
    Ok(present_tx)
}

fn rebuild_scheme(state: &mut RenderState, width: u32, height: u32) {
    let mut scheme = Scheme::new(&state.ctx);
    state.present = record_scheme(
        &mut scheme,
        &state.surface,
        &state.compute_pipeline,
        &state.uniform_buffer,
        width,
        height,
    )
    .expect("failed to record scheme");
    state.scheme = scheme;
}

impl App {
    fn init(&mut self, window: Arc<Window>) -> Result<()> {
        let instance = Instance::new()?;
        let device = Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;

        let surface = SurfaceExchange::new_with_depth(
            &ctx,
            window.as_ref(),
            3,
            SurfaceConfig {
                present_mode: PresentMode::Fifo,
                depth_format: None,
            },
        )?;

        let shader = ShaderModule::from_slang(&device, COMPUTE_SHADER)?;
        let compute_pipeline = ComputePipeline::new(&device, &shader)?;

        let mut retained_pool = RetainedPool::new(device);
        let uniform_buffer = retained_pool.acquire_buffer_with_data(
            &[Uniforms {
                width: surface.width(),
                height: surface.height(),
                time: 0.0,
                _padding: 0.0,
            }],
            BufferKind::Scattered,
        )?;

        let mut scheme = Scheme::new(&ctx);
        let present = record_scheme(
            &mut scheme,
            &surface,
            &compute_pipeline,
            &uniform_buffer,
            surface.width(),
            surface.height(),
        )?;

        self.state = Some(RenderState {
            window,
            ctx,
            surface,
            present,
            scheme,
            compute_pipeline,
            _retained_pool: retained_pool,
            uniform_buffer,
            start_time: std::time::Instant::now(),
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
        let attrs = Window::default_attributes()
            .with_title("Goldy — Compute to Surface")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        if let Err(e) = self.init(window.clone()) {
            tracing::error!("Failed to initialize: {}", e);
            event_loop.exit();
            return;
        }

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
                    if let Err(e) = state.surface.set_present_mode(mode) {
                        eprintln!("Failed to set present mode: {e}");
                    } else {
                        println!(
                            "Vsync: {} (present mode: {:?})",
                            if state.vsync { "ON" } else { "OFF" },
                            mode
                        );
                    }
                }
                _ => {}
            },
            WindowEvent::Resized(new_size) if new_size.width > 0 && new_size.height > 0 => {
                let _ = state.surface.resize(new_size.width, new_size.height);
                rebuild_scheme(state, new_size.width, new_size.height);
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = render_frame(state) {
                    tracing::error!("Render error: {}", e);
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn render_frame(state: &mut RenderState) -> Result<()> {
    state.frame_count += 1;

    let (width, height) = state.surface.size();
    if width == 0 || height == 0 {
        return Ok(());
    }

    let elapsed = state.start_time.elapsed().as_secs_f32();
    let uniforms = Uniforms {
        width,
        height,
        time: elapsed,
        _padding: 0.0,
    };

    let mut upload = Scheme::new(&state.ctx);
    upload.commit_write_parcel(&state.uniform_buffer, 0, bytemuck::bytes_of(&uniforms).to_vec())?;
    upload.submit()?;

    let mut submission = state.scheme.submit()?;
    state.present.claim(&mut submission)?.consume()?;

    Ok(())
}
