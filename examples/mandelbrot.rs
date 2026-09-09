//! Mandelbrot example - interactive fractal explorer.
//!
//! Demonstrates retained scheme with offscreen render pass → copy-to-present.
//!
//! Run with: `cargo run --example mandelbrot`

use goldy::{
    shaders, Buffer, BufferKind, Color, DepositTransaction, DeviceDescriptor, Instance, Lease, LeaseRenderTarget,
    MemoryExchange, NodeAccess, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule, TargetLoad, TextureFormat,
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
use common::FrameSink;

/// Uniform buffer data (must match shader cbuffer layout)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    center: [f32; 2],
    zoom: f32,
    _padding: f32, // Align to 16 bytes
}
impl goldy::StructuredBufferElement for Uniforms {}

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    _retained_pool: Option<RetainedPool>,
    uniform: Option<Buffer>,
    window: Option<Arc<Window>>,
    sink: Option<FrameSink>,
    scene_rt: Option<Lease<LeaseRenderTarget>>,
    scheme: Option<Scheme>,
    upload_scheme: Option<Scheme>,
    uniform_deposit: Option<DepositTransaction>,
    center: [f32; 2],
    zoom: f32,
    start_time: std::time::Instant,
    frame_count: u32,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            ctx: None,
            device: None,
            pipeline: None,
            shader: None,
            _retained_pool: None,
            uniform: None,
            window: None,
            sink: None,
            scene_rt: None,
            scheme: None,
            upload_scheme: None,
            uniform_deposit: None,
            center: [-0.5, 0.0],
            zoom: 1.0,
            start_time: std::time::Instant::now(),
            frame_count: 0,
        })
    }

    fn create_pipeline(
        device: &goldy::Device,
        shader: &ShaderModule,
        format: TextureFormat,
    ) -> anyhow::Result<RenderPipeline> {
        common::render_pipeline(
            device,
            shader,
            format,
            RenderPipelineDesc {
                vertex_layout: goldy::VertexBufferLayout::empty(),
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        sink: &mut FrameSink,
        pipeline: &RenderPipeline,
        uniform: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
    ) -> anyhow::Result<()> {
        let mut pass = scheme.render_pass("mandelbrot", scene_rt, TargetLoad::Clear(Color::BLACK));
        pass.with_parcel(uniform, NodeAccess::Read);
        pass.set_pipeline(pipeline);
        pass.draw_fullscreen();
        pass.finish();
        sink.bind_render_target(scheme, scene_rt)
    }

    fn init_gpu(&mut self, window: Option<&Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let mut retained_pool = RetainedPool::new(device.clone());
        let mut sink = FrameSink::open(&ctx, &mut retained_pool, window)?;

        let shader = ShaderModule::from_slang(&device, shaders::MANDELBROT)?;

        let pipeline = Self::create_pipeline(&device, &shader, sink.format())?;

        let uniform = retained_pool.acquire_buffer_with_data(
            &[Uniforms {
                center: self.center,
                zoom: self.zoom,
                _padding: 0.0,
            }],
            BufferKind::Broadcast,
        )?;

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = sink.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), sink.format(), None)?;
        Self::record_scheme(&mut scheme, &mut sink, &pipeline, &uniform, &scene_rt)?;

        let mut upload_scheme = Scheme::new(&ctx);
        let uniform_deposit = MemoryExchange::new(&ctx).bind_deposit_buffer(
            &mut upload_scheme,
            &uniform,
            std::mem::size_of::<Uniforms>() as u64,
        )?;

        self.ctx = Some(ctx);
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self._retained_pool = Some(retained_pool);
        self.uniform = Some(uniform);
        self.sink = Some(sink);
        self.scene_rt = Some(scene_rt);
        self.scheme = Some(scheme);
        self.upload_scheme = Some(upload_scheme);
        self.uniform_deposit = Some(uniform_deposit);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        self.frame_count += 1;

        if let Some(window) = self.window.as_ref() {
            let size = window.inner_size();
            if size.width == 0 || size.height == 0 {
                return Ok(());
            }
        }

        let scheme = self.scheme.as_mut().unwrap();

        let uniforms = Uniforms {
            center: self.center,
            zoom: self.zoom,
            _padding: 0.0,
        };
        let upload = self.upload_scheme.as_mut().unwrap();
        self.uniform_deposit
            .unwrap()
            .write(upload, 0, bytemuck::bytes_of(&uniforms))?;
        upload.submit()?;

        let mut submission = scheme.submit()?;
        self.sink.as_mut().unwrap().settle(&mut submission)?;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(sink) = self.sink.as_mut() {
                let _ = sink.resize(new_size.width, new_size.height);
            }
            if let (Some(device), Some(sink), Some(shader)) = (&self.device, self.sink.as_ref(), &self.shader) {
                if let Ok(pipeline) = Self::create_pipeline(device, shader, sink.format()) {
                    self.pipeline = Some(pipeline);
                    if let (Some(ctx), Some(pipeline), Some(uniform), Some(sink)) = (
                        self.ctx.as_ref(),
                        self.pipeline.as_ref(),
                        self.uniform.as_ref(),
                        self.sink.as_mut(),
                    ) {
                        let mut scheme = Scheme::new(ctx);

                        let (width, height) = sink.size();

                        if let Ok(rt) = scheme.lease_render_target(width.max(1), height.max(1), sink.format(), None) {
                            if Self::record_scheme(&mut scheme, sink, pipeline, uniform, &rt).is_ok() {
                                self.scheme = Some(scheme);
                                self.scene_rt = Some(rt);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Drop for App {
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
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(common::hidden_window(
                        "Goldy - Mandelbrot (Scheme + Present, Arrows=pan, +/-=zoom, R=reset)",
                        800,
                        800,
                    ))
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.init_gpu(Some(window.as_ref())).unwrap();
            if let Err(e) = self.render_frame() {
                tracing::error!("First frame error: {e}");
            }
            common::reveal_window(&window);
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        common::exit_if_timed_out(event_loop, self.start_time);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                let pan = 0.1 / self.zoom;
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::ArrowUp) => self.center[1] += pan,
                    Key::Named(NamedKey::ArrowDown) => self.center[1] -= pan,
                    Key::Named(NamedKey::ArrowLeft) => self.center[0] -= pan,
                    Key::Named(NamedKey::ArrowRight) => self.center[0] += pan,
                    Key::Character(ref c) if c == "=" || c == "+" => self.zoom *= 1.5,
                    Key::Character(ref c) if c == "-" => self.zoom /= 1.5,
                    Key::Character(ref c) if c == "r" || c == "R" => {
                        self.center = [-0.5, 0.0];
                        self.zoom = 1.0;
                    }
                    _ => {}
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    tracing::error!("Render error: {}", e);
                }
            }
            WindowEvent::Resized(new_size) => {
                self.handle_resize(new_size);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    if common::capture_requested() {
        let mut app = App::new()?;
        app.init_gpu(None)?;
        while !app.sink.as_ref().is_none_or(common::FrameSink::finished) {
            app.render_frame()?;
        }
        return Ok(());
    }

    println!("Goldy Mandelbrot Example");
    println!("  Arrows - Pan");
    println!("  +/- - Zoom in/out");
    println!("  R - Reset view");
    println!("  Escape - Exit");
    let event_loop = EventLoop::new()?;
    // `Wait` idles until input arrives, which would also idle straight past a
    // configured run limit, so poll when one is set.
    event_loop.set_control_flow(if common::run_limit_secs().is_some() {
        ControlFlow::Poll
    } else {
        ControlFlow::Wait
    });
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
