//! Triangle example - render a colored triangle in an interactive window.
//!
//! Demonstrates retained Scheme with offscreen render pass → copy-to-present.
//!
//! Run from `goldy/ffi-client`: `cargo run --example triangle`

use goldy_ffi_client::{
    shader::builtins, BufferKind, Color, Context, DepthFormat, DeviceDescriptor, Instance, NodeAccess,
    PresentGrant, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme,
    SchemeRenderTargetLease, ShaderModule, SwapchainPool, Vertex2D,
};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

fn swapchain_from_window(ctx: &Context, window: &Window) -> goldy_ffi_client::Result<SwapchainPool> {
    let handle = window
        .window_handle()
        .map_err(|e| goldy_ffi_client::GoldyError::from_message(format!("window handle: {e}")))?;
    match handle.as_raw() {
            #[cfg(windows)]
            RawWindowHandle::Win32(h) => SwapchainPool::from_win32(ctx, h.hwnd.get() as *mut _, 3),
            #[cfg(target_os = "macos")]
            RawWindowHandle::AppKit(h) => SwapchainPool::from_appkit(ctx, h.ns_view.as_ptr(), 3),
            #[cfg(target_os = "linux")]
            RawWindowHandle::Wayland(h) => SwapchainPool::from_wayland(
                ctx,
                h.display.as_ptr(),
                h.surface.as_ptr(),
                3,
            ),
        other => Err(goldy_ffi_client::GoldyError::from_message(format!(
            "unsupported window handle for swapchain pool: {other:?}"
        ))),
    }
}

fn record_scheme(
    scheme: &mut Scheme,
    pipeline: &RenderPipeline,
    vertex_buffer: &goldy_ffi_client::Parcel,
    scene_rt: &SchemeRenderTargetLease,
    screen: &goldy_ffi_client::PresentLease,
    bg_color: Color,
) -> goldy_ffi_client::Result<PresentGrant> {
    {
        let mut pass = scheme.render_pass("triangle", scene_rt);
        pass.with_parcel(vertex_buffer, NodeAccess::Read);
        pass.clear(bg_color);
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer_parcel(0, vertex_buffer);
        pass.draw(0..3, 0..1);
        pass.finish_recorded();
    }
    scheme.copy_to_present(scene_rt, screen)?;
    scheme.grant_present(screen)
}

struct App {
    instance: Instance,
    ctx: Option<Context>,
    device: Option<goldy_ffi_client::Device>,
    _retained_pool: Option<RetainedPool>,
    vertex_buffer: Option<goldy_ffi_client::Parcel>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    swapchain: Option<SwapchainPool>,
    screen: Option<goldy_ffi_client::PresentLease>,
    present: Option<PresentGrant>,
    scene_rt: Option<SchemeRenderTargetLease>,
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

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = self
            .instance
            .request_adapter(&RequestAdapterOptions::default())?
            .request_device(&DeviceDescriptor::default())?;
        let ctx = Context::new(&device)?;

        let vertices = [
            Vertex2D::new(0.0, -0.5, Color::RED),
            Vertex2D::new(-0.5, 0.5, Color::GREEN),
            Vertex2D::new(0.5, 0.5, Color::BLUE),
        ];
        let mut retained_pool = RetainedPool::new(&device)?;
        let vertex_buffer = retained_pool.acquire_buffer_with_data(&vertices, BufferKind::Scattered)?;

        let swapchain = swapchain_from_window(&ctx, window.as_ref())?;
        let screen = swapchain.lease()?;

        let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: Vertex2D::layout(),
                target_format: swapchain.format(),
                ..Default::default()
            },
        )?;

        let mut scheme = Scheme::new(&ctx)?;
        let (width, height) = swapchain.size();
        let scene_rt = scheme.lease_render_target(
            width.max(1),
            height.max(1),
            swapchain.format(),
            None::<DepthFormat>,
        )?;
        let bg_color = Color {
            r: 0.1,
            g: 0.1,
            b: 0.2,
            a: 1.0,
        };
        let present = record_scheme(
            &mut scheme,
            &pipeline,
            &vertex_buffer,
            &scene_rt,
            &screen,
            bg_color,
        )?;

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
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        if let Some(swapchain) = &mut self.swapchain {
            if let Err(e) = swapchain.resize(new_size.width, new_size.height) {
                tracing::error!("Failed to resize swapchain: {e}");
                return;
            }
            if let (
                Some(ctx),
                Some(swapchain),
                Some(shader),
                Some(vertex_buffer),
                Some(screen),
                Some(device),
            ) = (
                self.ctx.as_ref(),
                self.swapchain.as_ref(),
                self.shader.as_ref(),
                self.vertex_buffer.as_ref(),
                self.screen.as_ref(),
                self.device.as_ref(),
            ) {
                let pipeline = RenderPipeline::new(
                    device,
                    shader,
                    shader,
                    &RenderPipelineDesc {
                        vertex_layout: Vertex2D::layout(),
                        target_format: swapchain.format(),
                        ..Default::default()
                    },
                )
                .ok();
                if let Some(pipeline) = pipeline {
                    self.pipeline = Some(pipeline);
                }
                if let Some(pipeline) = self.pipeline.as_ref() {
                    let mut scheme = match Scheme::new(ctx) {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::error!("Failed to recreate scheme on resize: {e}");
                            return;
                        }
                    };
                    let (width, height) = swapchain.size();
                    if let Ok(rt) = scheme.lease_render_target(
                        width.max(1),
                        height.max(1),
                        swapchain.format(),
                        None::<DepthFormat>,
                    ) {
                        let bg_color = Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.2,
                            a: 1.0,
                        };
                        if let Ok(present) = record_scheme(
                            &mut scheme,
                            pipeline,
                            vertex_buffer,
                            &rt,
                            screen,
                            bg_color,
                        ) {
                            self.scheme = Some(scheme);
                            self.scene_rt = Some(rt);
                            self.present = Some(present);
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
                            .with_title("Goldy - Triangle (Scheme + Present)")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
                    )
                    .unwrap(),
            );
            self.window = Some(window.clone());
            if let Err(e) = self.init_gpu(&window) {
                tracing::error!("Failed to initialize GPU: {e}");
            }
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.window = None;
                self.swapchain = None;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    self.window = None;
                    self.swapchain = None;
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                if self.swapchain.is_none() {
                    return;
                }
                if let Err(e) = self.render_frame() {
                    tracing::error!("Render error: {e}");
                }
                if self.swapchain.is_some() {
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
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

    println!("Goldy Triangle Example (ffi-client / Scheme + Present)");
    println!("Press Escape or close window to exit\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
