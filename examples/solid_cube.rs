//! Solid cube example - 3D filled cube with painter's algorithm.
//!
//! Demonstrates indexed rendering with 3D transformation.
//! For GPU depth testing, use RenderTarget::new_with_depth().
//!
//! Run with: cargo run --example solid_cube

use goldy::{
    Buffer, BufferKind, Color, DeviceDescriptor, IndexFormat, Instance, NodeAccess,
    PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions, ShaderModule,
    Surface, TaskGraph, Vertex2D,
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

// Cube vertices in 3D with associated face colors
#[derive(Clone, Copy)]
struct Vertex3D {
    position: [f32; 3],
    color: Color,
}

// Cube face definitions - 6 faces, 4 vertices each (24 unique vertices for proper face colors)
fn generate_cube_vertices() -> Vec<Vertex3D> {
    let face_colors = [
        Color {
            r: 1.0,
            g: 0.3,
            b: 0.3,
            a: 1.0,
        }, // Front - red
        Color {
            r: 0.3,
            g: 1.0,
            b: 0.3,
            a: 1.0,
        }, // Back - green
        Color {
            r: 0.3,
            g: 0.3,
            b: 1.0,
            a: 1.0,
        }, // Left - blue
        Color {
            r: 1.0,
            g: 1.0,
            b: 0.3,
            a: 1.0,
        }, // Right - yellow
        Color {
            r: 1.0,
            g: 0.3,
            b: 1.0,
            a: 1.0,
        }, // Top - magenta
        Color {
            r: 0.3,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }, // Bottom - cyan
    ];

    // Face vertices (counter-clockwise when viewed from outside)
    let faces: [[[f32; 3]; 4]; 6] = [
        // Front (z = -1)
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
        // Back (z = 1)
        [
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        // Left (x = -1)
        [
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
        ],
        // Right (x = 1)
        [
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ],
        // Top (y = 1)
        [
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        // Bottom (y = -1)
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
    let z = p[2] + 4.0; // Push back from camera
    let scale = fov / z;
    [p[0] * scale, p[1] * scale]
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct App {
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    scene_rt: Option<RenderTarget>,
    frame_graph: TaskGraph,
    start_time: Instant,
    vertex_buffers: Vec<Buffer>,
    cube_vertices: Vec<Vertex3D>,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let cube_vertices = generate_cube_vertices();

        Ok(Self {
            instance: Instance::new()?,
            device: None,
            pipeline: None,
            shader: None,
            window: None,
            surface: None,
            scene_rt: None,
            frame_graph: TaskGraph::new(),
            start_time: Instant::now(),
            vertex_buffers: Vec::with_capacity(MAX_FRAMES_IN_FLIGHT),
            cube_vertices,
        })
    }

    fn create_scene_rt(device: &goldy::Device, surface: &Surface) -> anyhow::Result<RenderTarget> {
        let (width, height) = surface.size();
        RenderTarget::new(device, width.max(1), height.max(1), surface.format()).map_err(Into::into)
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.request_adapter(&RequestAdapterOptions::default())?.request_device(&DeviceDescriptor::default())?);
        let ctx = device.create_context()?;
        let surface = Surface::new(&ctx, window.as_ref())?;

        let shader = ShaderModule::from_slang(&device, goldy::shader::builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: Vertex2D::layout(),
                target_format: surface.format(),
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
        )?;

        let scene_rt = Self::create_scene_rt(&device, &surface)?;

        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);
        self.scene_rt = Some(scene_rt);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let time = self.start_time.elapsed().as_secs_f32();

        // Transform cube vertices to 3D (rotated but not projected yet)
        let rotated_3d: Vec<[f32; 3]> = self
            .cube_vertices
            .iter()
            .map(|v| rotate_x(rotate_y(v.position, time), time * 0.7))
            .collect();

        // Calculate average Z for each face (4 vertices per face, 6 faces)
        // Painter's algorithm: sort faces back-to-front (largest Z = furthest = draw first)
        let mut face_depths: Vec<(usize, f32)> = (0..6)
            .map(|face_idx| {
                let base = face_idx * 4;
                let avg_z = (rotated_3d[base][2]
                    + rotated_3d[base + 1][2]
                    + rotated_3d[base + 2][2]
                    + rotated_3d[base + 3][2])
                    / 4.0;
                (face_idx, avg_z)
            })
            .collect();

        // Sort by Z descending (furthest first)
        face_depths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Build vertices and indices in sorted order
        let mut vertices = Vec::with_capacity(24);
        let mut sorted_indices = Vec::with_capacity(36);

        for (new_base, (face_idx, _)) in face_depths.iter().enumerate() {
            let old_base = face_idx * 4;
            let new_base = (new_base * 4) as u16;

            // Add vertices for this face
            for i in 0..4 {
                let projected = project(rotated_3d[old_base + i], 2.0);
                vertices.push(Vertex2D::new(
                    projected[0],
                    projected[1],
                    self.cube_vertices[old_base + i].color,
                ));
            }

            // Add indices for this face (2 triangles)
            sorted_indices.extend_from_slice(&[
                new_base,
                new_base + 1,
                new_base + 2,
                new_base,
                new_base + 2,
                new_base + 3,
            ]);
        }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let scene_rt = self.scene_rt.as_ref().unwrap();
        let vertex_buffer = device.as_ref().alloc_buffer_with_data(&vertices, BufferKind::Scattered)?;
        let index_buffer =
            device.as_ref().alloc_buffer_with_data(&sorted_indices, BufferKind::Scattered)?;

        if self.vertex_buffers.len() >= MAX_FRAMES_IN_FLIGHT {
            self.vertex_buffers.remove(0);
        }

        self.frame_graph.clear();

        let mut pass = self.frame_graph.render_pass("solid_cube", scene_rt);
        pass.bind_buffer_mut(&vertex_buffer, NodeAccess::Read);
        pass.bind_buffer_mut(&index_buffer, NodeAccess::Read);
        pass.clear(Color {
            r: 0.02,
            g: 0.02,
            b: 0.05,
            a: 1.0,
        });
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.set_index_buffer(&index_buffer, IndexFormat::Uint16);
        pass.draw_indexed(0..sorted_indices.len() as u32, 0, 0..1);
        pass.finish_recorded();

        let swapchain = self.frame_graph.declare_swapchain_output();
        self.frame_graph
            .copy_render_target_to_swapchain(scene_rt, swapchain);

        let frame = surface.begin()?;
        let frame = surface.submit_graph_to_frame(&mut self.frame_graph, frame)?;
        frame.present()?;
        self.vertex_buffers.push(vertex_buffer);
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &mut self.surface {
                let _ = surface.resize(new_size.width, new_size.height);
            }
            if let (Some(device), Some(surface)) = (&self.device, &self.surface) {
                if let Ok(rt) = Self::create_scene_rt(device, surface) {
                    self.scene_rt = Some(rt);
                }
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Solid Cube with Depth Buffer")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 800)),
                    )
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
            window.request_redraw();
        }
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
    println!("Goldy Solid Cube Example - Press Escape to exit");
    println!("Demonstrates indexed rendering with 3D transformations");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
