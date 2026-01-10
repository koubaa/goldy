//! Gradient example - animated color gradient.
//!
//! Demonstrates fragment shader with time-based animation.
//!
//! Run with: cargo run --example gradient

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, FrameOutput,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat,
    Vertex2D, VertexBufferLayout, VertexAttribute, VertexFormat,
    shaders,
};
use std::num::NonZeroU32;
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
struct GradientVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

impl GradientVertex {
    fn layout() -> VertexBufferLayout {
        VertexBufferLayout {
            stride: std::mem::size_of::<Self>() as u32,
            attributes: vec![
                VertexAttribute { location: 0, format: VertexFormat::Float32x2, offset: 0 },
                VertexAttribute { location: 1, format: VertexFormat::Float32x2, offset: 8 },
            ],
        }
    }
}

fn create_fullscreen_quad(time: f32) -> [GradientVertex; 6] {
    // Animate UV offset based on time
    let offset = time * 0.1;
    [
        GradientVertex { position: [-1.0, -1.0], uv: [0.0 + offset, 1.0] },
        GradientVertex { position: [1.0, -1.0], uv: [1.0 + offset, 1.0] },
        GradientVertex { position: [1.0, 1.0], uv: [1.0 + offset, 0.0] },
        GradientVertex { position: [-1.0, -1.0], uv: [0.0 + offset, 1.0] },
        GradientVertex { position: [1.0, 1.0], uv: [1.0 + offset, 0.0] },
        GradientVertex { position: [-1.0, 1.0], uv: [0.0 + offset, 0.0] },
    ]
}

struct App {
    instance: Instance,
    device: Option<rag::Device>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    start_time: Instant,
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
        })
    }

    fn init_gpu(&mut self) -> anyhow::Result<()> {
        let device = self.instance.create_device(DeviceType::DiscreteGpu)?;
        let shader = ShaderModule::from_slang(&device, shaders::GRADIENT)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: GradientVertex::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        })?;
        
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        let (width, height) = (size.width, size.height);
        if width == 0 || height == 0 { return Ok(()); }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let time = self.start_time.elapsed().as_secs_f32();

        let vertices = create_fullscreen_quad(time);
        let vertex_buffer = Buffer::with_data(device, &vertices, BufferUsage::VERTEX)?;

        let frame = FrameOutput::new(device, width, height, TextureFormat::Rgba8Unorm);
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..6, 0..1);
        }

        let output = frame.render(encoder)?;

        let surface = self.surface.as_mut().unwrap();
        surface.resize(NonZeroU32::new(width).unwrap(), NonZeroU32::new(height).unwrap())
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut buffer = surface.buffer_mut()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        for (i, pixel) in buffer.iter_mut().enumerate() {
            let offset = i * 4;
            if offset + 2 < output.len() {
                *pixel = ((output[offset] as u32) << 16) | ((output[offset + 1] as u32) << 8) | (output[offset + 2] as u32);
            }
        }
        buffer.present().map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(event_loop.create_window(
                Window::default_attributes()
                    .with_title("RAG - Animated Gradient")
                    .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            ).unwrap());
            let context = softbuffer::Context::new(window.clone()).unwrap();
            self.surface = Some(softbuffer::Surface::new(&context, window.clone()).unwrap());
            self.window = Some(window);
            self.init_gpu().unwrap();
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
                self.render_frame().ok();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    println!("RAG Gradient Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}

