//! Spinning cube example - 3D wireframe cube.
//!
//! Demonstrates 3D projection via retained scheme with copy-to-present.
//!
//! Run with: cargo run --example spinning_cube

#![allow(deprecated)] // write_to_parcel migration deferred

use goldy::{
    write_to_parcel, Buffer, BufferFlags, BufferKind, Color, DeviceDescriptor, Grant, Instance, Lease,
    LeaseRenderTarget, NodeAccess, PresentGrant, PrimitiveTopology, RenderPipeline, RenderPipelineDesc,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SwapchainPool, Vertex2D,
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

const CUBE_VERTICES: [[f32; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

const CUBE_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

fn rotate_y(p: [f32; 3], angle: f32) -> [f32; 3] {
    let (s, c) = (angle.sin(), angle.cos());
    [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]
}

fn rotate_x(p: [f32; 3], angle: f32) -> [f32; 3] {
    let (s, c) = (angle.sin(), angle.cos());
    [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c]
}

fn project(p: [f32; 3], fov: f32) -> [f32; 2] {
    let z = p[2] + 4.0;
    let scale = fov / z;
    [p[0] * scale, p[1] * scale]
}

const MAX_LINE_VERTICES: usize = CUBE_EDGES.len() * 2;

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    _retained_pool: Option<RetainedPool>,
    vertex_parcel: Option<Buffer>,
    window: Option<Arc<Window>>,
    swapchain: Option<SwapchainPool>,
    screen: Option<goldy::PresentLease>,
    present: Option<PresentGrant>,
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
            window: None,
            swapchain: None,
            screen: None,
            present: None,
            scene_rt: None,
            scheme: None,
            start_time: Instant::now(),
            _retained_pool: None,
            vertex_parcel: None,
            frame_count: 0,
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
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        pipeline: &RenderPipeline,
        vertex_parcel: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
        screen: &goldy::PresentLease,
    ) -> PresentGrant {
        let mut pass = scheme.render_pass("spinning_cube", scene_rt);
        pass.with_parcel(vertex_parcel, NodeAccess::Read);
        pass.clear(Color {
            r: 0.02,
            g: 0.02,
            b: 0.05,
            a: 1.0,
        });
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_parcel);
        pass.draw(0..MAX_LINE_VERTICES as u32, 0..1);
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

        let shader = ShaderModule::from_slang(&device, goldy::shader::builtins::VERTEX_COLOR_2D)?;
        let pipeline = Self::create_pipeline(&device, &shader, &swapchain)?;

        let mut retained_pool = RetainedPool::new(device.clone());
        let vertex_parcel = retained_pool.acquire_buffer_sized::<Vertex2D>(
            MAX_LINE_VERTICES as u64,
            BufferKind::Scattered,
            BufferFlags::empty(),
        )?;

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = swapchain.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)?;
        let present = Self::record_scheme(&mut scheme, &pipeline, &vertex_parcel, &scene_rt, &screen);

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
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        self.frame_count += 1;

        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let time = self.start_time.elapsed().as_secs_f32();
        let transformed: Vec<[f32; 3]> = CUBE_VERTICES
            .iter()
            .map(|&v| rotate_x(rotate_y(v, time), time * 0.7))
            .collect();

        let mut vertices: Vec<Vertex2D> = Vec::new();
        for edge in &CUBE_EDGES {
            let p1 = project(transformed[edge[0]], 2.0);
            let p2 = project(transformed[edge[1]], 2.0);

            let z1 = transformed[edge[0]][2];
            let z2 = transformed[edge[1]][2];
            let avg_z = (z1 + z2) / 2.0;
            let brightness = (avg_z + 1.5) / 3.0;
            let color = Color {
                r: 0.2 + brightness * 0.8,
                g: 0.5 + brightness * 0.5,
                b: 1.0,
                a: 1.0,
            };

            vertices.push(Vertex2D::new(p1[0], p1[1], color));
            vertices.push(Vertex2D::new(p2[0], p2[1], color));
        }

        let ctx = self.ctx.as_ref().unwrap();
        write_to_parcel(
            ctx,
            self.vertex_parcel.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&vertices),
        )?;

        let scheme = self.scheme.as_mut().unwrap();
        let submission = scheme.submit()?;
        self.present.as_ref().unwrap().consume(&submission)?;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(swapchain) = &self.swapchain {
                let _ = swapchain.resize(new_size.width, new_size.height);
            }
            if let (Some(device), Some(swapchain), Some(shader)) = (&self.device, &self.swapchain, &self.shader) {
                if let Ok(pipeline) = Self::create_pipeline(device, shader, swapchain) {
                    self.pipeline = Some(pipeline);
                    if let (Some(ctx), Some(pipeline), Some(vertex_parcel), Some(screen)) = (
                        self.ctx.as_ref(),
                        self.pipeline.as_ref(),
                        self.vertex_parcel.as_ref(),
                        self.screen.as_ref(),
                    ) {
                        let mut scheme = Scheme::new(ctx);

                        let (width, height) = swapchain.size();

                        if let Ok(rt) =
                            scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)
                        {
                            let present = Self::record_scheme(&mut scheme, pipeline, vertex_parcel, &rt, screen);

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
                            .with_title("Goldy - Spinning Cube (Scheme + Present)")
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
    println!("Goldy Spinning Cube Example (Scheme + Present) - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
