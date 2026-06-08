//! Window example - render a colored triangle in an interactive window.
//!
//! This example uses the Surface API for zero-copy GPU presentation.
//!
//! Run with: cargo run --example window

use goldy::{
    shader::builtins, Buffer, BufferKind, Color, CommandEncoder, DeviceDescriptor, Instance, NodeAccess,
    RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions, ShaderModule, Surface, TaskGraph,
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

struct App {
    // Goldy resources
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    vertex_buffer: Option<Buffer>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,

    // Window resources
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    scene_rt: Option<RenderTarget>,
    frame_graph: TaskGraph,

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
            scene_rt: None,
            frame_graph: TaskGraph::new(),
            frame_count: 0,
        })
    }

    fn create_scene_rt(device: &goldy::Device, surface: &Surface) -> anyhow::Result<RenderTarget> {
        let (width, height) = surface.size();
        RenderTarget::new(device, width.max(1), height.max(1), surface.format()).map_err(Into::into)
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.request_adapter(&RequestAdapterOptions::default())?.request_device(&DeviceDescriptor::default())?);
        let ctx = device.create_context()?;

        // Create vertex buffer with a triangle
        let vertices = [
            Vertex2D::new(0.0, -0.5, Color::RED),
            Vertex2D::new(-0.5, 0.5, Color::GREEN),
            Vertex2D::new(0.5, 0.5, Color::BLUE),
        ];
        let vertex_buffer = device.alloc_buffer_with_data(&vertices, BufferKind::Scattered)?;

        // Create Surface for zero-copy presentation
        let surface = Surface::new(&ctx, window.as_ref())?;

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
        self.scene_rt = Some(Self::create_scene_rt(&device, &surface)?);

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
        let scene_rt = self.scene_rt.as_ref().unwrap();

        // Animate background color
        let t = (self.frame_count as f32 * 0.02).sin() * 0.5 + 0.5;
        let bg_color = Color {
            r: 0.1 + t * 0.1,
            g: 0.1 + t * 0.05,
            b: 0.2 + t * 0.1,
            a: 1.0,
        };

        self.frame_graph.clear();

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(bg_color);
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, vertex_buffer);
            pass.draw(0..3, 0..1);
        }

        self.frame_graph
            .render_pass("window", scene_rt)
            .bind_buffer(vertex_buffer, NodeAccess::Read)
            .finish_encoder(encoder);

        let swapchain = self.frame_graph.declare_swapchain_output();
        self.frame_graph
            .copy_render_target_to_swapchain(scene_rt, swapchain);

        let frame = surface.begin()?;
        let frame = surface.submit_graph_to_frame(&mut self.frame_graph, frame)?;
        frame.present()?;

        self.frame_count += 1;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &mut self.surface {
                let _ = surface.resize(new_size.width, new_size.height);
            }
            if let (Some(device), Some(surface)) = (&self.device, &self.surface) {
                if let Ok(rt) = Self::create_scene_rt(device, surface) {
                    self.scene_rt = Some(rt);
                }
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Goldy - Window (Surface API)")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.window = Some(window.clone());

            // Initialize GPU resources and create surface
            if let Err(e) = self.init_gpu(&window) {
                tracing::error!("Failed to initialize GPU: {}", e);
            }
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    tracing::error!("Render error: {}", e);
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
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    println!("Goldy Window Example (Surface API)");
    println!("=================================");
    println!("Press Escape or close window to exit\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
