//! Plasma example - classic demoscene plasma effect.
//!
//! Demonstrates retained scheme with offscreen render pass → copy-to-present.
//!
//! Run with: `cargo run --example plasma`

use goldy::{
    shaders, Buffer, BufferFlags, BufferKind, Color, DepositTransaction, DeviceDescriptor, Instance, Lease,
    LeaseRenderTarget, MemoryExchange, NodeAccess, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions,
    RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, Texture, TextureFormat,
    Transaction, VertexBufferLayout, WithdrawTransaction,
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

/// Uniform buffer data (must match shader cbuffer layout)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    time: f32,
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
    capture: Option<CaptureDump>,
    readback: Option<Texture>,
    withdraw: Option<WithdrawTransaction>,
    scene_rt: Option<Lease<LeaseRenderTarget>>,
    scheme: Option<Scheme>,
    upload_scheme: Option<Scheme>,
    uniform_deposit: Option<DepositTransaction>,
    start_time: Instant,
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
            capture: None,
            readback: None,
            withdraw: None,
            scene_rt: None,
            scheme: None,
            upload_scheme: None,
            uniform_deposit: None,
            start_time: Instant::now(),
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
                vertex_layout: VertexBufferLayout::empty(),
                ..Default::default()
            },
        )
    }

    fn record_pass(
        scheme: &mut Scheme,
        pipeline: &RenderPipeline,
        uniform: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
    ) {
        let mut pass = scheme.render_pass("plasma", scene_rt, TargetLoad::Clear(Color::BLACK));
        pass.with_parcel(uniform, NodeAccess::Read);
        pass.set_pipeline(pipeline);
        pass.draw_fullscreen();
        pass.finish();
    }

    fn bind_frame(
        scheme: &mut Scheme,
        scene_rt: &Lease<LeaseRenderTarget>,
        surface: Option<&SurfaceExchange>,
        readback: Option<&Texture>,
    ) -> anyhow::Result<(Option<Transaction>, Option<WithdrawTransaction>)> {
        if let Some(surface) = surface {
            let present = surface.bind_render_target(scheme, scene_rt)?;
            Ok((Some(present), None))
        } else {
            let readback = readback.expect("capture readback");
            scheme.copy_to_texture(scene_rt, readback)?;
            let withdraw = MemoryExchange::new(scheme.context()).bind_withdraw(scheme, readback)?;
            Ok((None, Some(withdraw)))
        }
    }

    fn init_gpu(&mut self, window: Option<&Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let mut retained_pool = RetainedPool::new(device.clone());

        let (surface, capture, readback, format, width, height) = if let Some(window) = window {
            let surface = SurfaceExchange::new(&ctx, window, SurfaceConfig::default())?;
            let format = surface.format();
            let (width, height) = surface.size();
            (Some(surface), None, None, format, width, height)
        } else {
            let capture = CaptureDump::from_env()?;
            let (width, height) = capture.size();
            let readback = common::capture_readback(&mut retained_pool, width, height)?;
            (
                None,
                Some(capture),
                Some(readback),
                CaptureDump::format(),
                width,
                height,
            )
        };

        let shader = ShaderModule::from_slang(&device, shaders::PLASMA)?;

        let pipeline = Self::create_pipeline(&device, &shader, format)?;

        let uniform = retained_pool.acquire_buffer_sized::<Uniforms>(1, BufferKind::Broadcast, BufferFlags::empty())?;

        let mut scheme = Scheme::new(&ctx);
        let scene_rt = ctx.lease_render_target(width.max(1), height.max(1), format, None)?;
        Self::record_pass(&mut scheme, &pipeline, &uniform, &scene_rt);
        let (present, withdraw) = Self::bind_frame(&mut scheme, &scene_rt, surface.as_ref(), readback.as_ref())?;

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
        self.surface = surface;
        self.present = present;
        self.capture = capture;
        self.readback = readback;
        self.withdraw = withdraw;
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

        let time = self
            .capture
            .as_ref()
            .map(CaptureDump::time)
            .unwrap_or_else(|| self.start_time.elapsed().as_secs_f32());
        let uniforms = Uniforms { time };
        let upload = self.upload_scheme.as_mut().unwrap();
        self.uniform_deposit
            .unwrap()
            .write(upload, 0, bytemuck::bytes_of(&uniforms))?;
        upload.submit()?;

        let mut submission = scheme.submit()?;
        if let Some(present) = &self.present {
            present.claim(&mut submission)?.consume()?;
        } else {
            let pixels = self.withdraw.as_ref().unwrap().claim(&mut submission)?.consume()?;
            self.capture.as_mut().unwrap().write_rgba(&pixels)?;
        }
        Ok(())
    }

    fn capture_done(&self) -> bool {
        self.capture.as_ref().is_none_or(CaptureDump::finished)
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        let Some(surface) = self.surface.as_mut() else {
            return;
        };
        let _ = surface.resize(new_size.width, new_size.height);
        let format = surface.format();
        let (width, height) = surface.size();
        if let (Some(ctx), Some(device), Some(shader), Some(uniform)) = (
            self.ctx.as_ref(),
            self.device.as_ref(),
            self.shader.as_ref(),
            self.uniform.as_ref(),
        ) {
            if let Ok(pipeline) = Self::create_pipeline(device, shader, format) {
                self.pipeline = Some(pipeline);
                if let Some(pipeline) = self.pipeline.as_ref() {
                    let mut scheme = Scheme::new(ctx);
                    if let Ok(rt) = ctx.lease_render_target(width.max(1), height.max(1), format, None) {
                        Self::record_pass(&mut scheme, pipeline, uniform, &rt);
                        if let Ok((present, withdraw)) =
                            Self::bind_frame(&mut scheme, &rt, self.surface.as_ref(), self.readback.as_ref())
                        {
                            self.present = present;
                            self.withdraw = withdraw;
                            self.scheme = Some(scheme);
                            self.scene_rt = Some(rt);
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
                        "Goldy - Plasma Effect (Scheme + Present)",
                        800,
                        600,
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
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    tracing::error!("Render error: {}", e);
                }
                self.window.as_ref().unwrap().request_redraw();
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
        while !app.capture_done() {
            app.render_frame()?;
        }
        return Ok(());
    }

    println!("Goldy Plasma Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
