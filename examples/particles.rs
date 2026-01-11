//! Particles example - rain/snow particle system.
//!
//! Demonstrates many moving particles with physics using Surface API.
//!
//! Run with: cargo run --example particles

use goldy::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, Surface,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule,
    Vertex2D,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const NUM_PARTICLES: usize = 1000;

struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    size: f32,
    color: Color,
}

impl Particle {
    fn new_rain() -> Self {
        Self {
            x: random() * 2.0 - 1.0,
            y: -1.0 - random() * 0.5,
            vx: (random() - 0.5) * 0.002,
            vy: 0.01 + random() * 0.02,
            size: 0.002 + random() * 0.003,
            color: Color {
                r: 0.5 + random() * 0.2,
                g: 0.6 + random() * 0.2,
                b: 0.9 + random() * 0.1,
                a: 0.6 + random() * 0.4,
            },
        }
    }

    fn new_snow() -> Self {
        Self {
            x: random() * 2.0 - 1.0,
            y: -1.0 - random() * 0.5,
            vx: (random() - 0.5) * 0.005,
            vy: 0.002 + random() * 0.005,
            size: 0.003 + random() * 0.008,
            color: Color {
                r: 0.95 + random() * 0.05,
                g: 0.95 + random() * 0.05,
                b: 1.0,
                a: 0.7 + random() * 0.3,
            },
        }
    }

    fn update(&mut self, is_snow: bool) {
        self.x += self.vx;
        self.y += self.vy;
        
        if is_snow {
            self.vx += (random() - 0.5) * 0.001;
            self.vx = self.vx.clamp(-0.01, 0.01);
        }

        if self.y > 1.0 || self.x < -1.2 || self.x > 1.2 {
            if is_snow {
                *self = Self::new_snow();
            } else {
                *self = Self::new_rain();
            }
        }
    }

    fn vertices(&self) -> [Vertex2D; 6] {
        let s = self.size;
        let c = self.color;
        [
            Vertex2D::new(self.x - s, self.y - s * 3.0, c),
            Vertex2D::new(self.x + s, self.y - s * 3.0, c),
            Vertex2D::new(self.x + s, self.y + s * 3.0, c),
            Vertex2D::new(self.x - s, self.y - s * 3.0, c),
            Vertex2D::new(self.x + s, self.y + s * 3.0, c),
            Vertex2D::new(self.x - s, self.y + s * 3.0, c),
        ]
    }
}

static mut SEED: u32 = 42;
fn random() -> f32 {
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        SEED as f32 / u32::MAX as f32
    }
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct App {
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    particles: Vec<Particle>,
    is_snow: bool,
    vertex_buffers: Vec<Buffer>,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let particles: Vec<Particle> = (0..NUM_PARTICLES).map(|_| Particle::new_rain()).collect();
        Ok(Self {
            instance: Instance::new()?,
            device: None, pipeline: None, shader: None,
            window: None, surface: None,
            particles,
            is_snow: false,
            vertex_buffers: Vec::with_capacity(MAX_FRAMES_IN_FLIGHT),
        })
    }

    fn toggle_mode(&mut self) {
        self.is_snow = !self.is_snow;
        for p in &mut self.particles {
            if self.is_snow {
                *p = Particle::new_snow();
            } else {
                *p = Particle::new_rain();
            }
        }
        if let Some(w) = &self.window {
            w.set_title(&format!("Goldy - {} (Surface API, Space to toggle)", if self.is_snow { "Snow" } else { "Rain" }));
        }
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);
        let surface = Surface::new(&device, window.as_ref())?;
        let shader = ShaderModule::from_slang(&device, goldy::shader::builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: surface.format(),
            ..Default::default()
        })?;
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

        for p in &mut self.particles {
            p.update(self.is_snow);
        }

        let mut vertices: Vec<Vertex2D> = Vec::with_capacity(NUM_PARTICLES * 6);
        for p in &self.particles {
            vertices.extend_from_slice(&p.vertices());
        }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let vertex_buffer = Buffer::with_data(device.as_ref(), &vertices, BufferUsage::VERTEX)?;

        let bg = if self.is_snow {
            Color { r: 0.05, g: 0.05, b: 0.15, a: 1.0 }
        } else {
            Color { r: 0.02, g: 0.02, b: 0.05, a: 1.0 }
        };

        let frame = surface.acquire()?;
        if self.vertex_buffers.len() >= MAX_FRAMES_IN_FLIGHT {
            self.vertex_buffers.remove(0);
        }
        
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(bg);
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
                    .with_title("Goldy - Rain (Surface API, Space to toggle)")
                    .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
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
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) => self.toggle_mode(),
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
    println!("Goldy Particles Example");
    println!("  Space - Toggle rain/snow");
    println!("  Escape - Exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
