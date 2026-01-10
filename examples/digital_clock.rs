//! Digital Clock example - render an animated 7-segment clock in a window.
//!
//! This example uses shared rendering code from `rag::examples::digital_clock`,
//! demonstrating that the same logic can be used on both native and web platforms.
//!
//! Run with: cargo run --example digital_clock

use rag::{
    Buffer, BufferUsage, CommandEncoder, DeviceType, FrameOutput,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat,
    examples::digital_clock::{
        ClockVertex, ClockState, TimeData, SHADER_SOURCE,
        generate_clock_vertices,
    },
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

struct App {
    instance: Instance,
    device: Option<rag::Device>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    
    start_time: Instant,
    clock_state: ClockState,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let instance = Instance::new()?;
        Ok(Self {
            instance,
            device: None,
            pipeline: None,
            shader: None,
            window: None,
            surface: None,
            start_time: Instant::now(),
            clock_state: ClockState::default(),
        })
    }

    fn init_gpu(&mut self) -> anyhow::Result<()> {
        let device = self.instance.create_device(DeviceType::DiscreteGpu)?;
        
        // Use the SHARED shader source from the examples module
        let shader = ShaderModule::from_slang(&device, SHADER_SOURCE)?;
        let pipeline_desc = RenderPipelineDesc {
            vertex_layout: ClockVertex::layout(), // Use shared vertex type
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        };
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &pipeline_desc)?;
        
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        
        Ok(())
    }

    fn elapsed_secs(&self) -> u64 {
        if self.clock_state.paused {
            self.clock_state.accumulated_secs
        } else {
            self.start_time.elapsed().as_secs() + self.clock_state.accumulated_secs
        }
    }

    fn toggle_pause(&mut self) {
        let current = self.elapsed_secs();
        if self.clock_state.paused {
            // Resuming - reset start time
            self.start_time = Instant::now();
        }
        self.clock_state.toggle_pause(current);
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

        let elapsed = self.elapsed_secs();
        let time = TimeData::from_elapsed_secs(elapsed);
        let color = self.clock_state.color();
        let bg_color = self.clock_state.background_color();

        // Generate clock vertices using SHARED function
        let vertices = generate_clock_vertices(time, color, width, height);
        
        // Convert ClockVertex to bytes for the buffer
        let vertex_data: &[u8] = bytemuck::cast_slice(&vertices);
        let vertex_buffer = Buffer::with_bytes(device, vertex_data, BufferUsage::VERTEX)?;

        // Render
        let frame = FrameOutput::new(device, width, height, TextureFormat::Rgba8Unorm);
        
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(bg_color);
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer_raw(0, &vertex_buffer);
            pass.draw(0..vertices.len() as u32, 0..1);
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

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("RAG - Clock (Space: pause, Click: color)")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            
            let context = softbuffer::Context::new(window.clone()).unwrap();
            let surface = softbuffer::Surface::new(&context, window.clone()).unwrap();
            
            self.window = Some(window);
            self.surface = Some(surface);
            
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
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Named(NamedKey::Space) => self.toggle_pause(),
                        Key::Character(ref c) if c == "c" || c == "C" => self.clock_state.next_color(),
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, .. } if state.is_pressed() => {
                self.clock_state.next_color();
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    eprintln!("Render error: {}", e);
                }
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

    println!("RAG Clock Example (using shared rendering code)");
    println!("================================================");
    println!("Controls:");
    println!("  Space - Toggle pause");
    println!("  Click - Change color");
    println!("  Escape - Exit\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
