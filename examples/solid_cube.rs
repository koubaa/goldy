//! Solid cube example - 3D filled cube with painter's algorithm.
//!
//! Demonstrates indexed rendering with 3D transformation.
//! For GPU depth testing, use RenderTarget::new_with_depth().
//!
//! Run with: cargo run --example solid_cube

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, Surface,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule,
    Vertex2D, PrimitiveTopology, IndexFormat,
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
        Color { r: 1.0, g: 0.3, b: 0.3, a: 1.0 }, // Front - red
        Color { r: 0.3, g: 1.0, b: 0.3, a: 1.0 }, // Back - green
        Color { r: 0.3, g: 0.3, b: 1.0, a: 1.0 }, // Left - blue
        Color { r: 1.0, g: 1.0, b: 0.3, a: 1.0 }, // Right - yellow
        Color { r: 1.0, g: 0.3, b: 1.0, a: 1.0 }, // Top - magenta
        Color { r: 0.3, g: 1.0, b: 1.0, a: 1.0 }, // Bottom - cyan
    ];
    
    // Face vertices (counter-clockwise when viewed from outside)
    let faces: [[[f32; 3]; 4]; 6] = [
        // Front (z = -1)
        [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]],
        // Back (z = 1)
        [[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        // Left (x = -1)
        [[-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]],
        // Right (x = 1)
        [[1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]],
        // Top (y = 1)
        [[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
        // Bottom (y = -1)
        [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
    ];
    
    let mut vertices = Vec::new();
    for (face_idx, face) in faces.iter().enumerate() {
        for &pos in face {
            vertices.push(Vertex3D { position: pos, color: face_colors[face_idx] });
        }
    }
    vertices
}

// Generate indices for 6 faces (2 triangles per face)
fn generate_cube_indices() -> Vec<u16> {
    let mut indices = Vec::new();
    for face in 0..6 {
        let base = (face * 4) as u16;
        // Two triangles per face
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    indices
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
    device: Option<Arc<rag::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    start_time: Instant,
    vertex_buffers: Vec<Buffer>,
    index_buffer: Option<Buffer>,
    cube_vertices: Vec<Vertex3D>,
    cube_indices: Vec<u16>,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let cube_vertices = generate_cube_vertices();
        let cube_indices = generate_cube_indices();
        
        Ok(Self {
            instance: Instance::new()?,
            device: None, pipeline: None, shader: None,
            window: None, surface: None,
            start_time: Instant::now(),
            vertex_buffers: Vec::with_capacity(MAX_FRAMES_IN_FLIGHT),
            index_buffer: None,
            cube_vertices,
            cube_indices,
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);
        let surface = Surface::new(&device, window.as_ref())?;
        
        let shader = ShaderModule::from_slang(&device, rag::shader::builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: surface.format(),
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        })?;
        
        // Create index buffer once
        let index_buffer = Buffer::with_data(&device, &self.cube_indices, BufferUsage::INDEX)?;
        
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);
        self.index_buffer = Some(index_buffer);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 { return Ok(()); }

        let time = self.start_time.elapsed().as_secs_f32();

        // Transform cube vertices and project to 2D
        let vertices: Vec<Vertex2D> = self.cube_vertices
            .iter()
            .map(|v| {
                let rotated = rotate_x(rotate_y(v.position, time), time * 0.7);
                let projected = project(rotated, 2.0);
                Vertex2D::new(projected[0], projected[1], v.color)
            })
            .collect();

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let index_buffer = self.index_buffer.as_ref().unwrap();
        let vertex_buffer = Buffer::with_data(device.as_ref(), &vertices, BufferUsage::VERTEX)?;

        let frame = surface.acquire()?;
        if self.vertex_buffers.len() >= MAX_FRAMES_IN_FLIGHT {
            self.vertex_buffers.remove(0);
        }
        
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color { r: 0.02, g: 0.02, b: 0.05, a: 1.0 });
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.set_index_buffer(index_buffer, IndexFormat::Uint16);
            pass.draw_indexed(0..self.cube_indices.len() as u32, 0, 0..1);
        }

        frame.render(encoder)?;
        surface.present(frame)?;
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(event_loop.create_window(
                Window::default_attributes()
                    .with_title("RAG - Solid Cube with Depth Buffer")
                    .with_inner_size(winit::dpi::LogicalSize::new(800, 800))
            ).unwrap());
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) { event_loop.exit(); }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    eprintln!("Render error: {}", e);
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
    tracing_subscriber::fmt().with_env_filter("info").init();
    println!("RAG Solid Cube Example - Press Escape to exit");
    println!("Demonstrates indexed rendering with 3D transformations");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}

