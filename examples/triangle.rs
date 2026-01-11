//! Triangle example - render a colored triangle in an interactive window.
//!
//! This example demonstrates the Surface API for zero-copy GPU presentation.
//!
//! Run with: cargo run --example triangle

use goldy::{
    shader::builtins, Buffer, BufferUsage, Color, CommandEncoder, DeviceType, Instance,
    RenderPipeline, RenderPipelineDesc, ShaderModule, Surface, Vertex2D,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct App {
    // Goldy resources
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    vertex_buffer: Option<Buffer>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,

    // Window and surface
    window: Option<Arc<Window>>,
    surface: Option<Surface>,

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

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);

        // Create vertex buffer with a triangle
        let vertices = [
            Vertex2D::new(0.0, -0.5, Color::RED),
            Vertex2D::new(-0.5, 0.5, Color::GREEN),
            Vertex2D::new(0.5, 0.5, Color::BLUE),
        ];
        let vertex_buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;

        // Create Surface for zero-copy presentation
        let surface = Surface::new(&device, window.as_ref())?;

        // Create shader and pipeline using surface's actual format
        let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
        let pipeline_desc = RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: surface.format(),
            ..Default::default()
        };
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &pipeline_desc)?;

        self.device = Some(device);
        self.vertex_buffer = Some(vertex_buffer);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);

        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();

        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();

        // Animate background color
        let t = (self.frame_count as f32 * 0.02).sin() * 0.5 + 0.5;
        let bg_color = Color {
            r: 0.1 + t * 0.1,
            g: 0.1 + t * 0.05,
            b: 0.2 + t * 0.1,
            a: 1.0,
        };

        // Acquire next frame from swapchain
        let frame = surface.acquire()?;

        // Build render commands
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(bg_color);
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, vertex_buffer);
            pass.draw(0..3, 0..1);
        }

        // Render to swapchain image (zero-copy - no CPU readback!)
        frame.render(encoder)?;

        // Present to screen
        surface.present(frame)?;

        self.frame_count += 1;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &mut self.surface {
                if let Err(e) = surface.resize(new_size.width, new_size.height) {
                    eprintln!("Failed to resize surface: {}", e);
                }
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Goldy - Animated Triangle (Surface API)")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

            let window = Arc::new(event_loop.create_window(attrs).unwrap());

            self.window = Some(window.clone());

            // Initialize GPU resources and create surface
            if let Err(e) = self.init_gpu(&window) {
                eprintln!("Failed to initialize GPU: {}", e);
            }
            window.request_redraw();
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
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("Goldy Surface API Example");
    println!("=======================");
    println!("Rendering triangle with zero-copy GPU presentation");
    println!("Press Escape or close window to exit\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
