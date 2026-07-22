//! Checkerboard example - procedural texture with animation.
//!
//! Demonstrates retained scheme with offscreen render pass → copy-to-present.
//!
//! Run with: `cargo run --example checkerboard`
//!
//! Optional layout validation: `GOLDY_VALIDATE_LAYOUTS=1 cargo run --example checkerboard`

use goldy::{
    shaders, Buffer, BufferFlags, BufferKind, Color, DeviceDescriptor, Instance, LayoutCheckable, Lease,
    LeaseRenderTarget, NodeAccess, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, Transaction, VertexBufferLayout,
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

/// Uniform buffer data — fields must match `struct TimeUniforms` in `shaders/checkerboard.slang`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, LayoutCheckable)]
struct TimeUniforms {
    time: f32,
}
impl goldy::StructuredBufferElement for TimeUniforms {}

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
            scene_rt: None,
            scheme: None,
            start_time: Instant::now(),
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
                vertex_layout: VertexBufferLayout::empty(),
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
        let mut pass = scheme.render_pass("checkerboard", scene_rt, TargetLoad::Clear(Color::BLACK));
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

        let shader = ShaderModule::from_slang_with_options(
            &device,
            shaders::CHECKERBOARD,
            &[],
            &[],
            Default::default(),
            &[TimeUniforms::LAYOUT_CHECK],
        )?;

        let pipeline = Self::create_pipeline(&device, &shader, &surface)?;

        let mut retained_pool = RetainedPool::new(device.clone());
        let uniform =
            retained_pool.acquire_buffer_sized::<TimeUniforms>(1, BufferKind::Broadcast, BufferFlags::empty())?;

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

        let time = self.start_time.elapsed().as_secs_f32();
        let uniforms = TimeUniforms { time };
        let mut upload = Scheme::new(ctx);
        upload.commit_write_parcel(uniform, 0, bytemuck::bytes_of(&uniforms).to_vec())?;
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
                            .with_title("Goldy - Animated Checkerboard (Scheme + Present)")
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
    println!("Goldy Checkerboard Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
