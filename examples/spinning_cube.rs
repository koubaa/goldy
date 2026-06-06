//! Spinning cube example - 3D wireframe cube.
//!
//! Demonstrates 3D projection and rotation matrices using Surface API.
//!
//! Run with: cargo run --example spinning_cube

use goldy::{
    Buffer, BufferKind, Color, CommandEncoder, DeviceDescriptor, Instance, PrimitiveTopology,
    RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, ShaderModule, Surface, Vertex2D,
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

// Cube vertices in 3D
const CUBE_VERTICES: [[f32; 3]; 8] = [
    [-1.0, -1.0, -1.0], // 0
    [1.0, -1.0, -1.0],  // 1
    [1.0, 1.0, -1.0],   // 2
    [-1.0, 1.0, -1.0],  // 3
    [-1.0, -1.0, 1.0],  // 4
    [1.0, -1.0, 1.0],   // 5
    [1.0, 1.0, 1.0],    // 6
    [-1.0, 1.0, 1.0],   // 7
];

// Edges of the cube (pairs of vertex indices)
const CUBE_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0], // Front face
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4], // Back face
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7], // Connecting edges
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
    start_time: Instant,
    vertex_buffers: Vec<Buffer>,
    frame_count: u32,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None,
            pipeline: None,
            shader: None,
            window: None,
            surface: None,
            start_time: Instant::now(),
            vertex_buffers: Vec::with_capacity(MAX_FRAMES_IN_FLIGHT),
            frame_count: 0,
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
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
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            },
        )?;
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);
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

        // Transform cube vertices
        let transformed: Vec<[f32; 3]> = CUBE_VERTICES
            .iter()
            .map(|&v| rotate_x(rotate_y(v, time), time * 0.7))
            .collect();

        // Project and create line vertices
        let mut vertices: Vec<Vertex2D> = Vec::new();
        for edge in &CUBE_EDGES {
            let p1 = project(transformed[edge[0]], 2.0);
            let p2 = project(transformed[edge[1]], 2.0);

            // Color based on depth (average Z of the two points)
            let z1 = transformed[edge[0]][2];
            let z2 = transformed[edge[1]][2];
            let avg_z = (z1 + z2) / 2.0;
            let brightness = (avg_z + 1.5) / 3.0; // Normalize to 0-1
            let color = Color {
                r: 0.2 + brightness * 0.8,
                g: 0.5 + brightness * 0.5,
                b: 1.0,
                a: 1.0,
            };

            vertices.push(Vertex2D::new(p1[0], p1[1], color));
            vertices.push(Vertex2D::new(p2[0], p2[1], color));
        }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let vertex_buffer = device
            .as_ref()
            .alloc_buffer_with_data(&vertices, BufferKind::Scattered)?;

        let frame = surface.begin()?;
        if self.vertex_buffers.len() >= MAX_FRAMES_IN_FLIGHT {
            self.vertex_buffers.remove(0);
        }

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color {
                r: 0.02,
                g: 0.02,
                b: 0.05,
                a: 1.0,
            });
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..vertices.len() as u32, 0..1);
        }

        frame.render(encoder)?;
        frame.present()?;
        self.vertex_buffers.push(vertex_buffer);
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &mut self.surface {
                let _ = surface.resize(new_size.width, new_size.height);
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
                            .with_title("Goldy - Spinning Cube (Surface API)")
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
    println!("Goldy Spinning Cube Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
