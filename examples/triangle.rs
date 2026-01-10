//! Triangle example - render a colored triangle in an interactive window.
//!
//! Run with: cargo run --example triangle

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, FrameOutput,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat, Vertex2D,
    shader::builtins,
};
use std::num::NonZeroU32;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct App {
    // RAG resources
    instance: Instance,
    device: Option<rag::Device>,
    vertex_buffer: Option<Buffer>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    
    // Window resources
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    
    // Animation
    frame_count: u64,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let instance = Instance::new()?;
        Ok(Self {
            instance,
            device: None,
            vertex_buffer: None,
            pipeline: None,
            shader: None,
            window: None,
            surface: None,
            frame_count: 0,
        })
    }

    fn init_gpu(&mut self) -> anyhow::Result<()> {
        let device = self.instance.create_device(DeviceType::DiscreteGpu)?;
        
        // Create vertex buffer with a triangle
        let vertices = [
            Vertex2D::new(0.0, -0.5, Color::RED),
            Vertex2D::new(-0.5, 0.5, Color::GREEN),
            Vertex2D::new(0.5, 0.5, Color::BLUE),
        ];
        let vertex_buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;
        
        // Create shader and pipeline
        let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
        let pipeline_desc = RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        };
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &pipeline_desc)?;
        
        self.device = Some(device);
        self.vertex_buffer = Some(vertex_buffer);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        let width = size.width;
        let height = size.height;

        if width == 0 || height == 0 {
            return Ok(());
        }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();

        // Animate background color
        let t = (self.frame_count as f32 * 0.02).sin() * 0.5 + 0.5;
        let bg_color = Color {
            r: 0.1 + t * 0.1,
            g: 0.1 + t * 0.05,
            b: 0.2 + t * 0.1,
            a: 1.0,
        };

        // Create frame output and render
        let frame = FrameOutput::new(device, width, height, TextureFormat::Rgba8Unorm);
        
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(bg_color);
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, vertex_buffer);
            pass.draw(0..3, 0..1);
        }

        let output = frame.render(encoder)?;

        // Display in window using softbuffer
        let surface = self.surface.as_mut().unwrap();
        surface.resize(
            NonZeroU32::new(width).unwrap(),
            NonZeroU32::new(height).unwrap(),
        ).map_err(|e| anyhow::anyhow!("Failed to resize surface: {}", e))?;

        let mut buffer = surface.buffer_mut()
            .map_err(|e| anyhow::anyhow!("Failed to get buffer: {}", e))?;

        // Convert RGBA to softbuffer's format (0xAARRGGBB or 0x00RRGGBB)
        for (i, pixel) in buffer.iter_mut().enumerate() {
            let offset = i * 4;
            if offset + 3 < output.len() {
                let r = output[offset] as u32;
                let g = output[offset + 1] as u32;
                let b = output[offset + 2] as u32;
                *pixel = (r << 16) | (g << 8) | b;
            }
        }

        buffer.present()
            .map_err(|e| anyhow::anyhow!("Failed to present: {}", e))?;

        self.frame_count += 1;
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("RAG - Animated Triangle")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
            
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            
            // Create softbuffer surface
            let context = softbuffer::Context::new(window.clone()).unwrap();
            let surface = softbuffer::Surface::new(&context, window.clone()).unwrap();
            
            self.window = Some(window);
            self.surface = Some(surface);
            
            // Initialize GPU resources
            if let Err(e) = self.init_gpu() {
                eprintln!("Failed to initialize GPU: {}", e);
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    eprintln!("Render error: {}", e);
                }
                // Request another frame for animation
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(_) => {
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
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("RAG Window Example");
    println!("==================");
    println!("Press Escape or close window to exit\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}

