//! Bouncing lines example - animated lines bouncing off walls.
//!
//! Demonstrates line primitive rendering using Surface API.
//!
//! Run with: cargo run --example bouncing_lines

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, Surface,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat,
    Vertex2D, PrimitiveTopology,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const NUM_LINES: usize = 20;

struct Line {
    x1: f32, y1: f32,
    x2: f32, y2: f32,
    vx1: f32, vy1: f32,
    vx2: f32, vy2: f32,
    color: Color,
}

impl Line {
    fn new(idx: usize) -> Self {
        let angle = (idx as f32 / NUM_LINES as f32) * std::f32::consts::PI * 2.0;
        let colors = [Color::RED, Color::GREEN, Color::BLUE, 
                      Color { r: 1.0, g: 1.0, b: 0.0, a: 1.0 },
                      Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 },
                      Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 }];
        Self {
            x1: angle.cos() * 0.3,
            y1: angle.sin() * 0.3,
            x2: -angle.cos() * 0.3,
            y2: -angle.sin() * 0.3,
            vx1: 0.01 * (idx as f32 * 0.7).cos(),
            vy1: 0.012 * (idx as f32 * 0.9).sin(),
            vx2: -0.011 * (idx as f32 * 1.1).cos(),
            vy2: 0.009 * (idx as f32 * 1.3).sin(),
            color: colors[idx % colors.len()],
        }
    }

    fn update(&mut self) {
        self.x1 += self.vx1;
        self.y1 += self.vy1;
        self.x2 += self.vx2;
        self.y2 += self.vy2;

        if self.x1 < -1.0 || self.x1 > 1.0 { self.vx1 = -self.vx1; }
        if self.y1 < -1.0 || self.y1 > 1.0 { self.vy1 = -self.vy1; }
        if self.x2 < -1.0 || self.x2 > 1.0 { self.vx2 = -self.vx2; }
        if self.y2 < -1.0 || self.y2 > 1.0 { self.vy2 = -self.vy2; }

        self.x1 = self.x1.clamp(-1.0, 1.0);
        self.y1 = self.y1.clamp(-1.0, 1.0);
        self.x2 = self.x2.clamp(-1.0, 1.0);
        self.y2 = self.y2.clamp(-1.0, 1.0);
    }

    fn vertices(&self) -> [Vertex2D; 2] {
        [
            Vertex2D::new(self.x1, self.y1, self.color),
            Vertex2D::new(self.x2, self.y2, self.color),
        ]
    }
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct App {
    instance: Instance,
    device: Option<Arc<rag::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    lines: Vec<Line>,
    vertex_buffers: Vec<Buffer>,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let lines: Vec<Line> = (0..NUM_LINES).map(Line::new).collect();
        Ok(Self {
            instance: Instance::new()?,
            device: None, pipeline: None, shader: None,
            window: None, surface: None,
            lines,
            vertex_buffers: Vec::with_capacity(MAX_FRAMES_IN_FLIGHT),
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);
        let shader = ShaderModule::from_slang(&device, rag::shader::builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Bgra8UnormSrgb,
            topology: PrimitiveTopology::LineList,
        })?;
        let surface = Surface::new(device.clone(), window.as_ref())?;
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 { return Ok(()); }

        // Update lines
        for line in &mut self.lines {
            line.update();
        }

        // Generate vertices
        let mut vertices: Vec<Vertex2D> = Vec::with_capacity(NUM_LINES * 2);
        for line in &self.lines {
            vertices.extend_from_slice(&line.vertices());
        }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let vertex_buffer = Buffer::with_data(device.as_ref(), &vertices, BufferUsage::VERTEX)?;

        let frame = surface.acquire()?;
        if self.vertex_buffers.len() >= MAX_FRAMES_IN_FLIGHT {
            self.vertex_buffers.remove(0);
        }
        
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color { r: 0.05, g: 0.05, b: 0.1, a: 1.0 });
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..vertices.len() as u32, 0..1);
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
                    .with_title("RAG - Bouncing Lines (Surface API)")
                    .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            ).unwrap());
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
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
    println!("RAG Bouncing Lines Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
