//! Checkerboard example - procedural texture with animation.
//!
//! Demonstrates procedural texturing in fragment shader using Surface API.
//!
//! Run with: cargo run --example checkerboard

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, Surface,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule,
    VertexBufferLayout, VertexAttribute, VertexFormat,
    shaders,
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CheckerVertex {
    position: [f32; 2],
    uv: [f32; 2],
    time: f32,
}

impl CheckerVertex {
    fn layout() -> VertexBufferLayout {
        VertexBufferLayout {
            stride: std::mem::size_of::<Self>() as u32,
            attributes: vec![
                VertexAttribute { location: 0, format: VertexFormat::Float32x2, offset: 0 },
                VertexAttribute { location: 1, format: VertexFormat::Float32x2, offset: 8 },
                VertexAttribute { location: 2, format: VertexFormat::Float32, offset: 16 },
            ],
        }
    }
}

fn create_quad(time: f32) -> [CheckerVertex; 6] {
    [
        CheckerVertex { position: [-1.0, -1.0], uv: [0.0, 1.0], time },
        CheckerVertex { position: [1.0, -1.0], uv: [1.0, 1.0], time },
        CheckerVertex { position: [1.0, 1.0], uv: [1.0, 0.0], time },
        CheckerVertex { position: [-1.0, -1.0], uv: [0.0, 1.0], time },
        CheckerVertex { position: [1.0, 1.0], uv: [1.0, 0.0], time },
        CheckerVertex { position: [-1.0, 1.0], uv: [0.0, 0.0], time },
    ]
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
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None, pipeline: None, shader: None,
            window: None, surface: None,
            start_time: Instant::now(),
            vertex_buffers: Vec::with_capacity(MAX_FRAMES_IN_FLIGHT),
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);
        let surface = Surface::new(device.clone(), window.as_ref())?;
        let shader = ShaderModule::from_slang(&device, shaders::CHECKERBOARD)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: CheckerVertex::layout(),
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

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let time = self.start_time.elapsed().as_secs_f32();

        let vertices = create_quad(time);
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
            pass.draw(0..6, 0..1);
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
                    .with_title("RAG - Animated Checkerboard (Surface API)")
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
    println!("RAG Checkerboard Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
