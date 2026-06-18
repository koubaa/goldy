//! Triangle example - render a colored triangle in an interactive window.
//!
//! Demonstrates retained scheme with offscreen render pass → copy-to-present.
//!
//! Run with: cargo run --example triangle --features examples

use goldy::{
    shader::builtins, Buffer, BufferKind, Color, DeviceDescriptor, Grant, Instance, Lease, LeaseRenderTarget,
    NodeAccess, PresentGrant, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule, SwapchainPool, Vertex2D,
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

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    _retained_pool: Option<RetainedPool>,
    vertex_buffer: Option<Buffer>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    swapchain: Option<SwapchainPool>,
    screen: Option<goldy::PresentLease>,
    present: Option<PresentGrant>,
    scene_rt: Option<Lease<LeaseRenderTarget>>,
    scheme: Option<Scheme>,
    frame_count: u64,
    start_time: std::time::Instant,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            ctx: None,
            device: None,
            _retained_pool: None,
            vertex_buffer: None,
            pipeline: None,
            shader: None,
            window: None,
            swapchain: None,
            screen: None,
            present: None,
            scene_rt: None,
            scheme: None,
            frame_count: 0,
            start_time: std::time::Instant::now(),
        })
    }

    fn create_pipeline(
        device: &goldy::Device,
        shader: &ShaderModule,
        swapchain: &SwapchainPool,
    ) -> anyhow::Result<RenderPipeline> {
        common::render_pipeline_for_swapchain(
            device,
            shader,
            swapchain,
            RenderPipelineDesc {
                vertex_layout: Vertex2D::layout(),
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        pipeline: &RenderPipeline,
        vertex_buffer: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
        screen: &goldy::PresentLease,
        bg_color: Color,
    ) -> PresentGrant {
        let mut pass = scheme.render_pass("triangle", scene_rt);
        pass.with_parcel(vertex_buffer, NodeAccess::Read);
        pass.clear(bg_color);
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_buffer);
        pass.draw(0..3, 0..1);
        pass.finish();
        scheme.copy_to_present(scene_rt, screen);
        scheme.grant_present(screen)
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let swapchain = SwapchainPool::new(&ctx, window.as_ref(), 3)?;
        let screen = swapchain.lease();

        let vertices = [
            Vertex2D::new(0.0, -0.5, Color::RED),
            Vertex2D::new(-0.5, 0.5, Color::GREEN),
            Vertex2D::new(0.5, 0.5, Color::BLUE),
        ];
        let mut retained_pool = RetainedPool::new(device.clone());
        let vertex_buffer = retained_pool.acquire_buffer_with_data(&vertices, BufferKind::Scattered)?;

        let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
        let pipeline = Self::create_pipeline(&device, &shader, &swapchain)?;

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = swapchain.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)?;
        let bg_color = Color {
            r: 0.1,
            g: 0.1,
            b: 0.2,
            a: 1.0,
        };
        let present = Self::record_scheme(&mut scheme, &pipeline, &vertex_buffer, &scene_rt, &screen, bg_color);

        self.ctx = Some(ctx);
        self.device = Some(device);
        self._retained_pool = Some(retained_pool);
        self.vertex_buffer = Some(vertex_buffer);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.swapchain = Some(swapchain);
        self.screen = Some(screen);
        self.present = Some(present);
        self.scene_rt = Some(scene_rt);
        self.scheme = Some(scheme);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let scheme = self.scheme.as_mut().unwrap();
        let submission = scheme.submit()?;
        self.present.as_ref().unwrap().consume(&submission)?;

        self.frame_count += 1;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(swapchain) = &self.swapchain {
                let _ = swapchain.resize(new_size.width, new_size.height);
            }
            if let (Some(device), Some(swapchain), Some(shader), Some(scheme)) =
                (&self.device, &self.swapchain, &self.shader, self.scheme.as_mut())
            {
                if let Ok(pipeline) = Self::create_pipeline(device, shader, swapchain) {
                    self.pipeline = Some(pipeline);
                    if let (Some(pipeline), Some(vertex_buffer), Some(screen)) = (
                        self.pipeline.as_ref(),
                        self.vertex_buffer.as_ref(),
                        self.screen.as_ref(),
                    ) {
                        scheme.begin_rerecord();
                        let (width, height) = swapchain.size();
                        if let Ok(rt) =
                            scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)
                        {
                            let bg_color = Color {
                                r: 0.1,
                                g: 0.1,
                                b: 0.2,
                                a: 1.0,
                            };
                            let present = Self::record_scheme(scheme, pipeline, vertex_buffer, &rt, screen, bg_color);
                            self.present = Some(present);
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
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Animated Triangle (Scheme + Present)")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
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

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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

    println!("Goldy Triangle Example (Scheme + Present)");
    println!("Press Escape or close window to exit\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
