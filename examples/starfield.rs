//! Starfield example - classic 3D starfield flying through space.
//!
//! Demonstrates particle-like rendering with depth simulation using Surface API.
//!
//! Run with: cargo run --example starfield

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

const NUM_STARS: usize = 500;

struct Star {
    x: f32,
    y: f32,
    z: f32,
}

impl Star {
    fn new() -> Self {
        Self {
            x: (rand_f32() - 0.5) * 2.0,
            y: (rand_f32() - 0.5) * 2.0,
            z: rand_f32(),
        }
    }

    fn update(&mut self, speed: f32) {
        self.z -= speed;
        if self.z <= 0.0 {
            self.x = (rand_f32() - 0.5) * 2.0;
            self.y = (rand_f32() - 0.5) * 2.0;
            self.z = 1.0;
        }
    }

    fn to_vertex(&self) -> [Vertex2D; 6] {
        let size = 0.02 * (1.0 - self.z); // Bigger when closer
        let brightness = 1.0 - self.z;
        let color = Color {
            r: brightness,
            g: brightness,
            b: brightness,
            a: 1.0,
        };

        let x = self.x / self.z;
        let y = self.y / self.z;

        // Quad for the star
        [
            Vertex2D::new(x - size, y - size, color),
            Vertex2D::new(x + size, y - size, color),
            Vertex2D::new(x + size, y + size, color),
            Vertex2D::new(x - size, y - size, color),
            Vertex2D::new(x + size, y + size, color),
            Vertex2D::new(x - size, y + size, color),
        ]
    }
}

// Simple pseudo-random
static mut SEED: u32 = 12345;
fn rand_f32() -> f32 {
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        SEED as f32 / u32::MAX as f32
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
    stars: Vec<Star>,
    speed: f32,
    vertex_buffers: Vec<Buffer>,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let stars: Vec<Star> = (0..NUM_STARS).map(|_| Star::new()).collect();
        Ok(Self {
            instance: Instance::new()?,
            device: None, pipeline: None, shader: None,
            window: None, surface: None,
            stars,
            speed: 0.01,
            vertex_buffers: Vec::with_capacity(MAX_FRAMES_IN_FLIGHT),
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);
        let shader = ShaderModule::from_slang(&device, rag::shader::builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Bgra8UnormSrgb,
            topology: PrimitiveTopology::TriangleList,
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

        // Update stars
        for star in &mut self.stars {
            star.update(self.speed);
        }

        // Generate vertices
        let mut vertices: Vec<Vertex2D> = Vec::with_capacity(NUM_STARS * 6);
        for star in &self.stars {
            vertices.extend_from_slice(&star.to_vertex());
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
            pass.clear(Color::BLACK);
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
                    .with_title("RAG - Starfield (Surface API, Up/Down to change speed)")
                    .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
            ).unwrap());
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::ArrowUp) => self.speed = (self.speed + 0.005).min(0.1),
                    Key::Named(NamedKey::ArrowDown) => self.speed = (self.speed - 0.005).max(0.001),
                    _ => {}
                }
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
    println!("RAG Starfield Example");
    println!("  Up/Down - Change speed");
    println!("  Escape - Exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
