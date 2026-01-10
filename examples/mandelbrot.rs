//! Mandelbrot example - interactive fractal explorer.
//!
//! Demonstrates complex math in fragment shader with zoom/pan using Surface API.
//!
//! Run with: cargo run --example mandelbrot

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, Surface,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat,
    VertexBufferLayout, VertexAttribute, VertexFormat,
    shaders,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MandelbrotVertex {
    position: [f32; 2],
    center: [f32; 2],
    zoom: f32,
    uv: [f32; 2],
}

impl MandelbrotVertex {
    fn layout() -> VertexBufferLayout {
        VertexBufferLayout {
            stride: std::mem::size_of::<Self>() as u32,
            attributes: vec![
                VertexAttribute { location: 0, format: VertexFormat::Float32x2, offset: 0 },
                VertexAttribute { location: 1, format: VertexFormat::Float32x2, offset: 8 },
                VertexAttribute { location: 2, format: VertexFormat::Float32, offset: 16 },
                VertexAttribute { location: 3, format: VertexFormat::Float32x2, offset: 20 },
            ],
        }
    }
}

fn create_quad(center: [f32; 2], zoom: f32) -> [MandelbrotVertex; 6] {
    [
        MandelbrotVertex { position: [-1.0, -1.0], center, zoom, uv: [0.0, 1.0] },
        MandelbrotVertex { position: [1.0, -1.0], center, zoom, uv: [1.0, 1.0] },
        MandelbrotVertex { position: [1.0, 1.0], center, zoom, uv: [1.0, 0.0] },
        MandelbrotVertex { position: [-1.0, -1.0], center, zoom, uv: [0.0, 1.0] },
        MandelbrotVertex { position: [1.0, 1.0], center, zoom, uv: [1.0, 0.0] },
        MandelbrotVertex { position: [-1.0, 1.0], center, zoom, uv: [0.0, 0.0] },
    ]
}

struct App {
    instance: Instance,
    device: Option<Arc<rag::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    center: [f32; 2],
    zoom: f32,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None, pipeline: None, shader: None,
            window: None, surface: None,
            center: [-0.5, 0.0],
            zoom: 1.0,
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);
        let shader = ShaderModule::from_slang(&device, shaders::MANDELBROT)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: MandelbrotVertex::layout(),
            target_format: TextureFormat::Bgra8UnormSrgb,
            ..Default::default()
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

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();

        let vertices = create_quad(self.center, self.zoom);
        let vertex_buffer = Buffer::with_data(device.as_ref(), &vertices, BufferUsage::VERTEX)?;

        let frame = surface.acquire()?;
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..6, 0..1);
        }

        frame.render(encoder)?;
        surface.present(frame)?;
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
                    .with_title("RAG - Mandelbrot (Surface API, Arrows=pan, +/-=zoom, R=reset)")
                    .with_inner_size(winit::dpi::LogicalSize::new(800, 800))
            ).unwrap());
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                let pan = 0.1 / self.zoom;
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::ArrowUp) => self.center[1] += pan,
                    Key::Named(NamedKey::ArrowDown) => self.center[1] -= pan,
                    Key::Named(NamedKey::ArrowLeft) => self.center[0] -= pan,
                    Key::Named(NamedKey::ArrowRight) => self.center[0] += pan,
                    Key::Character(ref c) if c == "=" || c == "+" => self.zoom *= 1.5,
                    Key::Character(ref c) if c == "-" => self.zoom /= 1.5,
                    Key::Character(ref c) if c == "r" || c == "R" => {
                        self.center = [-0.5, 0.0];
                        self.zoom = 1.0;
                    }
                    _ => {}
                }
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    eprintln!("Render error: {}", e);
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
    tracing_subscriber::fmt().with_env_filter("info").init();
    println!("RAG Mandelbrot Example");
    println!("  Arrows - Pan");
    println!("  +/- - Zoom in/out");
    println!("  R - Reset view");
    println!("  Escape - Exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
