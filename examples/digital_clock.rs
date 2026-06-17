//! Digital Clock example - render an animated 7-segment clock in a window.
//!
//! This example uses shared rendering code from `goldy::examples::digital_clock`,
//! demonstrating that the same logic can be used on both native and web platforms.
//! Uses retained scheme with offscreen render pass → copy-to-present.
//!
//! Run with: `cargo run --example digital_clock`

use goldy::{
    examples::digital_clock::{generate_clock_vertices, ClockState, ClockVertex, TimeData, SHADER_SOURCE},
    write_to_parcel, BufferFlags, BufferKind, Color, DeviceDescriptor, Grant, Instance, Lease, LeaseRenderTarget,
    NodeAccess, Parcel, PresentGrant, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule, SwapchainPool,
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

/// Upper bound on seven-segment clock vertices (8 glyphs × 7 segments × 6 verts).
const MAX_CLOCK_VERTICES: usize = 384;

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    _retained_pool: Option<RetainedPool>,
    vertex_parcel: Option<Parcel>,

    window: Option<Arc<Window>>,
    swapchain: Option<SwapchainPool>,
    screen: Option<goldy::PresentLease>,
    present: Option<PresentGrant>,
    scene_rt: Option<Lease<LeaseRenderTarget>>,
    scheme: Option<Scheme>,

    start_time: Instant,
    perf_start: Instant,
    frame_count: u32,
    clock_state: ClockState,
    recorded_vertex_count: u32,
    recorded_bg_color: Color,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let instance = Instance::new()?;
        Ok(Self {
            instance,
            ctx: None,
            device: None,
            pipeline: None,
            shader: None,
            window: None,
            swapchain: None,
            screen: None,
            present: None,
            scene_rt: None,
            scheme: None,
            start_time: Instant::now(),
            perf_start: Instant::now(),
            frame_count: 0,
            clock_state: ClockState::default(),
            _retained_pool: None,
            vertex_parcel: None,
            recorded_vertex_count: 0,
            recorded_bg_color: Color::BLACK,
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
                vertex_layout: ClockVertex::layout(),
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        pipeline: &RenderPipeline,
        vertex_parcel: &Parcel,
        vertex_count: u32,
        bg_color: Color,
        scene_rt: &Lease<LeaseRenderTarget>,
        screen: &goldy::PresentLease,
    ) -> PresentGrant {
        let mut pass = scheme.render_pass("digital_clock", scene_rt);
        pass.bind_parcel_mut(vertex_parcel, NodeAccess::Read);
        pass.clear(bg_color);
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_parcel);
        pass.draw(0..vertex_count, 0..1);
        pass.finish();
        scheme.copy_to_present(scene_rt, screen);
        scheme.grant_present(screen)
    }

    fn rerecord_scheme_if_needed(&mut self, vertex_count: u32, bg_color: Color) {
        if vertex_count == self.recorded_vertex_count && bg_color == self.recorded_bg_color {
            return;
        }
        if let (Some(scheme), Some(pipeline), Some(vertex_parcel), Some(swapchain), Some(screen)) = (
            self.scheme.as_mut(),
            self.pipeline.as_ref(),
            self.vertex_parcel.as_ref(),
            self.swapchain.as_ref(),
            self.screen.as_ref(),
        ) {
            scheme.begin_rerecord();
            let (width, height) = swapchain.size();
            if let Ok(rt) =
                scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)
            {
                let present = Self::record_scheme(
                    scheme,
                    pipeline,
                    vertex_parcel,
                    vertex_count,
                    bg_color,
                    &rt,
                    screen,
                );
                self.present = Some(present);
                self.recorded_vertex_count = vertex_count;
                self.recorded_bg_color = bg_color;
                self.scene_rt = Some(rt);
            }
        }
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

        let shader = ShaderModule::from_slang(&device, SHADER_SOURCE)?;
        let pipeline = Self::create_pipeline(&device, &shader, &swapchain)?;

        let mut retained_pool = RetainedPool::new(device.clone());
        let vertex_parcel = retained_pool.acquire_buffer_sized::<ClockVertex>(
            MAX_CLOCK_VERTICES as u64,
            BufferKind::Scattered,
            BufferFlags::empty(),
        )?;

        let bg_color = self.clock_state.background_color();
        let mut scheme = Scheme::new(&ctx);
        let (width, height) = swapchain.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)?;
        let present = Self::record_scheme(&mut scheme, &pipeline, &vertex_parcel, 1, bg_color, &scene_rt, &screen);

        self.ctx = Some(ctx);
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self._retained_pool = Some(retained_pool);
        self.vertex_parcel = Some(vertex_parcel);
        self.swapchain = Some(swapchain);
        self.screen = Some(screen);
        self.present = Some(present);
        self.scene_rt = Some(scene_rt);
        self.scheme = Some(scheme);
        self.recorded_vertex_count = 1;
        self.recorded_bg_color = bg_color;
        Ok(())
    }

    fn elapsed_secs(&self) -> u64 {
        if self.clock_state.paused {
            self.clock_state.accumulated_secs
        } else {
            self.start_time.elapsed().as_secs() + self.clock_state.accumulated_secs
        }
    }

    fn toggle_pause(&mut self) {
        let current = self.elapsed_secs();
        if self.clock_state.paused {
            self.start_time = Instant::now();
        }
        self.clock_state.toggle_pause(current);
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        self.frame_count += 1;

        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        let width = size.width;
        let height = size.height;

        if width == 0 || height == 0 {
            return Ok(());
        }

        let elapsed = self.elapsed_secs();
        let time = TimeData::from_elapsed_secs(elapsed);
        let color = self.clock_state.color();
        let bg_color = self.clock_state.background_color();
        let vertices = generate_clock_vertices(time, color, width, height);
        let vertex_count = vertices.len() as u32;

        self.rerecord_scheme_if_needed(vertex_count, bg_color);

        let ctx = self.ctx.as_ref().unwrap();
        let vertex_parcel = self.vertex_parcel.as_ref().unwrap();
        write_to_parcel(ctx, vertex_parcel, 0, bytemuck::cast_slice(&vertices))?;

        let scheme = self.scheme.as_mut().unwrap();
        let present = self.present.as_ref().unwrap();
        let submission = scheme.submit()?;
        present.consume(&submission)?;
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
                    if let (Some(pipeline), Some(vertex_parcel), Some(screen)) = (
                        self.pipeline.as_ref(),
                        self.vertex_parcel.as_ref(),
                        self.screen.as_ref(),
                    ) {
                        let bg_color = self.clock_state.background_color();
                        let vertex_count = self.recorded_vertex_count.max(1);
                        scheme.begin_rerecord();
                        let (width, height) = swapchain.size();
                        if let Ok(rt) =
                            scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)
                        {
                            let present = Self::record_scheme(
                                scheme,
                                pipeline,
                                vertex_parcel,
                                vertex_count,
                                bg_color,
                                &rt,
                                screen,
                            );
                            self.present = Some(present);
                            self.recorded_vertex_count = vertex_count;
                            self.recorded_bg_color = bg_color;
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
        let elapsed = self.perf_start.elapsed().as_secs_f64();
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
            let attrs = Window::default_attributes()
                .with_title("Goldy - Clock (Scheme + Present, Space: pause, Click: color)")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.window = Some(window.clone());

            if let Err(e) = self.init_gpu(&window) {
                tracing::error!("Failed to initialize GPU: {}", e);
            }
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        common::exit_if_timed_out(event_loop, self.perf_start);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => match event.logical_key {
                Key::Named(NamedKey::Escape) => event_loop.exit(),
                Key::Named(NamedKey::Space) => self.toggle_pause(),
                Key::Character(ref c) if c == "c" || c == "C" => self.clock_state.next_color(),
                _ => {}
            },
            WindowEvent::MouseInput { state, .. } if state.is_pressed() => {
                self.clock_state.next_color();
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    tracing::error!("Render error: {}", e);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
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

    println!("Goldy Clock Example (using shared rendering code, retained scheme)");
    println!("==================================================================");
    println!("Controls:");
    println!("  Space - Toggle pause");
    println!("  Click - Change color");
    println!("  Escape - Exit\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
