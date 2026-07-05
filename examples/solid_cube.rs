//! Solid cube example - 3D filled cube with painter's algorithm.
//!
//! Demonstrates indexed rendering with 3D transformation via retained scheme.
//!
//! Run with: cargo run --example solid_cube

use goldy::{
    Buffer, BufferFlags, BufferKind, Color, DeviceDescriptor, Grant, IndexFormat, Instance, Lease, LeaseRenderTarget,
    NodeAccess, PresentGrant, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions,
    RetainedPool, Scheme, ShaderModule, SwapchainPool, Vertex2D,
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

#[derive(Clone, Copy)]
struct Vertex3D {
    position: [f32; 3],
    color: Color,
}

fn generate_cube_vertices() -> Vec<Vertex3D> {
    let face_colors = [
        Color {
            r: 1.0,
            g: 0.3,
            b: 0.3,
            a: 1.0,
        },
        Color {
            r: 0.3,
            g: 1.0,
            b: 0.3,
            a: 1.0,
        },
        Color {
            r: 0.3,
            g: 0.3,
            b: 1.0,
            a: 1.0,
        },
        Color {
            r: 1.0,
            g: 1.0,
            b: 0.3,
            a: 1.0,
        },
        Color {
            r: 1.0,
            g: 0.3,
            b: 1.0,
            a: 1.0,
        },
        Color {
            r: 0.3,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        },
    ];

    let faces: [[[f32; 3]; 4]; 6] = [
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
        [[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        [
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
        ],
        [[1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]],
        [[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
    ];

    let mut vertices = Vec::new();
    for (face_idx, face) in faces.iter().enumerate() {
        for &pos in face {
            vertices.push(Vertex3D {
                position: pos,
                color: face_colors[face_idx],
            });
        }
    }
    vertices
}

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

const MAX_CUBE_VERTICES: usize = 24;
const MAX_CUBE_INDICES: usize = 36;

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    _retained_pool: Option<RetainedPool>,
    vertex_parcel: Option<Buffer>,
    index_parcel: Option<Buffer>,
    window: Option<Arc<Window>>,
    swapchain: Option<SwapchainPool>,
    screen: Option<goldy::PresentLease>,
    present: Option<PresentGrant>,
    scene_rt: Option<Lease<LeaseRenderTarget>>,
    scheme: Option<Scheme>,
    start_time: Instant,
    cube_vertices: Vec<Vertex3D>,
    frame_count: u64,
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
            index_parcel: None,
            cube_vertices: generate_cube_vertices(),
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
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        pipeline: &RenderPipeline,
        vertex_parcel: &Buffer,
        index_parcel: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
        screen: &goldy::PresentLease,
    ) -> PresentGrant {
        let mut pass = scheme.render_pass("solid_cube", scene_rt);
        pass.with_parcel(vertex_parcel, NodeAccess::Read);
        pass.with_parcel(index_parcel, NodeAccess::Read);
        pass.clear(Color {
            r: 0.02,
            g: 0.02,
            b: 0.05,
            a: 1.0,
        });
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_parcel);
        pass.set_index_buffer(index_parcel, IndexFormat::Uint16);
        pass.draw_indexed(0..MAX_CUBE_INDICES as u32, 0, 0..1);
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
            MAX_CUBE_VERTICES as u64,
            BufferKind::Scattered,
            BufferFlags::empty(),
        )?;
        let index_parcel = retained_pool.acquire_buffer_sized::<u16>(
            MAX_CUBE_INDICES as u64,
            BufferKind::Scattered,
            BufferFlags::empty(),
        )?;

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = swapchain.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)?;
        let present = Self::record_scheme(
            &mut scheme,
            &pipeline,
            &vertex_parcel,
            &index_parcel,
            &scene_rt,
            &screen,
        );

        self.ctx = Some(ctx);
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self._retained_pool = Some(retained_pool);
        self.vertex_parcel = Some(vertex_parcel);
        self.index_parcel = Some(index_parcel);
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

        let time = self.start_time.elapsed().as_secs_f32();

        let rotated_3d: Vec<[f32; 3]> = self
            .cube_vertices
            .iter()
            .map(|v| rotate_x(rotate_y(v.position, time), time * 0.7))
            .collect();

        let mut face_depths: Vec<(usize, f32)> = (0..6)
            .map(|face_idx| {
                let base = face_idx * 4;
                let avg_z =
                    (rotated_3d[base][2] + rotated_3d[base + 1][2] + rotated_3d[base + 2][2] + rotated_3d[base + 3][2])
                        / 4.0;
                (face_idx, avg_z)
            })
            .collect();
        face_depths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut vertices = Vec::with_capacity(24);
        let mut sorted_indices = Vec::with_capacity(36);

        for (new_base, (face_idx, _)) in face_depths.iter().enumerate() {
            let old_base = face_idx * 4;
            let new_base = (new_base * 4) as u16;

            for i in 0..4 {
                let projected = project(rotated_3d[old_base + i], 2.0);
                vertices.push(Vertex2D::new(
                    projected[0],
                    projected[1],
                    self.cube_vertices[old_base + i].color,
                ));
            }

            sorted_indices.extend_from_slice(&[
                new_base,
                new_base + 1,
                new_base + 2,
                new_base,
                new_base + 2,
                new_base + 3,
            ]);
        }

        let ctx = self.ctx.as_ref().unwrap();
        let mut upload = Scheme::new(ctx);
        upload.commit_write_parcel(
            self.vertex_parcel.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&vertices).to_vec(),
        )?;
        upload.commit_write_parcel(
            self.index_parcel.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&sorted_indices).to_vec(),
        )?;
        upload.submit()?;

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
            if let (Some(device), Some(swapchain), Some(shader)) = (&self.device, &self.swapchain, &self.shader) {
                if let Ok(pipeline) = Self::create_pipeline(device, shader, swapchain) {
                    self.pipeline = Some(pipeline);
                    if let (Some(ctx), Some(pipeline), Some(vb), Some(ib), Some(screen)) = (
                        self.ctx.as_ref(),
                        self.pipeline.as_ref(),
                        self.vertex_parcel.as_ref(),
                        self.index_parcel.as_ref(),
                        self.screen.as_ref(),
                    ) {
                        let mut scheme = Scheme::new(ctx);

                        let (width, height) = swapchain.size();

                        if let Ok(rt) =
                            scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)
                        {
                            let present = Self::record_scheme(&mut scheme, pipeline, vb, ib, &rt, screen);

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
                            .with_title("Goldy - Solid Cube (Scheme + Present)")
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
    println!("Goldy Solid Cube Example (Scheme + Present) - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
