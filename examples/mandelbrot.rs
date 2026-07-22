//! Mandelbrot example - interactive fractal explorer.
//!
//! Demonstrates retained scheme with offscreen render pass → copy-to-present.
//!
//! Run with: `cargo run --example mandelbrot`

use goldy::{
    shaders, Buffer, BufferFlags, BufferKind, Color, DeviceDescriptor, Instance, Lease, LeaseRenderTarget, NodeAccess,
    RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SurfaceConfig,
    SurfaceExchange, TargetLoad, Transaction,
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
    surface: Option<SurfaceExchange>,
    present: Option<Transaction>,
    scene_rt: Option<Lease<LeaseRenderTarget>>,
    scheme: Option<Scheme>,
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
            surface: None,
            present: None,
            scene_rt: None,
            scheme: None,
            center: [-0.5, 0.0],
            zoom: 1.0,
            start_time: std::time::Instant::now(),
            frame_count: 0,
        })
    }

    fn create_pipeline(
        device: &goldy::Device,
        shader: &ShaderModule,
        surface: &SurfaceExchange,
    ) -> anyhow::Result<RenderPipeline> {
        common::render_pipeline_for_surface(
            device,
            shader,
            surface,
            RenderPipelineDesc {
                vertex_layout: goldy::VertexBufferLayout::empty(),
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        surface: &SurfaceExchange,
        pipeline: &RenderPipeline,
        uniform: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
    ) -> anyhow::Result<Transaction> {
        let mut pass = scheme.render_pass("mandelbrot", scene_rt, TargetLoad::Clear(Color::BLACK));
        pass.with_parcel(uniform, NodeAccess::Read);
        pass.set_pipeline(pipeline);
        pass.draw_fullscreen();
        pass.finish();
        surface.bind_render_target(scheme, scene_rt).map_err(Into::into)
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let surface = SurfaceExchange::new(&ctx, window.as_ref(), SurfaceConfig::default())?;

        let shader = ShaderModule::from_slang(&device, shaders::MANDELBROT)?;

        let pipeline = Self::create_pipeline(&device, &shader, &surface)?;

        let mut retained_pool = RetainedPool::new(device.clone());
        let uniform = retained_pool.acquire_buffer_sized::<Uniforms>(1, BufferKind::Broadcast, BufferFlags::empty())?;

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = surface.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), surface.format(), None)?;
        let present = Self::record_scheme(&mut scheme, &surface, &pipeline, &uniform, &scene_rt)?;

        self.ctx = Some(ctx);
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self._retained_pool = Some(retained_pool);
        self.uniform = Some(uniform);
        self.surface = Some(surface);
        self.present = Some(present);
        self.scene_rt = Some(scene_rt);
        self.scheme = Some(scheme);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        self.frame_count += 1;

        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let ctx = self.ctx.as_ref().unwrap();
        let uniform = self.uniform.as_ref().unwrap();
        let scheme = self.scheme.as_mut().unwrap();

        let uniforms = Uniforms {
            center: self.center,
            zoom: self.zoom,
            _padding: 0.0,
        };
        let mut upload = Scheme::new(ctx);
        upload.write_parcel(uniform, 0, bytemuck::bytes_of(&uniforms).to_vec())?;
        upload.submit()?;

        let present = self.present.as_ref().unwrap();
        let mut submission = scheme.submit()?;
        present.claim(&mut submission)?.consume()?;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &self.surface {
                let _ = surface.resize(new_size.width, new_size.height);
            }
            if let (Some(device), Some(surface), Some(shader)) = (&self.device, &self.surface, &self.shader) {
                if let Ok(pipeline) = Self::create_pipeline(device, shader, surface) {
                    self.pipeline = Some(pipeline);
                    if let (Some(ctx), Some(pipeline), Some(uniform), Some(surface)) = (
                        self.ctx.as_ref(),
                        self.pipeline.as_ref(),
                        self.uniform.as_ref(),
                        self.surface.as_ref(),
                    ) {
                        let mut scheme = Scheme::new(ctx);

                        let (width, height) = surface.size();

                        if let Ok(rt) = scheme.lease_render_target(width.max(1), height.max(1), surface.format(), None)
                        {
                            if let Ok(present) = Self::record_scheme(&mut scheme, surface, pipeline, uniform, &rt) {
                                self.scheme = Some(scheme);
                                self.present = Some(present);
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
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Mandelbrot (Scheme + Present, Arrows=pan, +/-=zoom, R=reset)")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 800)),
                    )
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
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
    println!("Goldy Mandelbrot Example");
    println!("  Arrows - Pan");
    println!("  +/- - Zoom in/out");
    println!("  R - Reset view");
    println!("  Escape - Exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
