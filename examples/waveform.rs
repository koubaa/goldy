//! Waveform example - animated audio waveform visualizer.
//!
//! Demonstrates procedural waveform generation and line rendering.
//!
//! Run with: cargo run --example waveform

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, FrameOutput,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat,
    Vertex2D, PrimitiveTopology,
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

const NUM_SAMPLES: usize = 200;
const NUM_CHANNELS: usize = 4;

fn generate_waveform(time: f32, channel: usize) -> Vec<f32> {
    let freq = 1.0 + channel as f32 * 0.5;
    let phase = channel as f32 * 0.7;
    
    (0..NUM_SAMPLES)
        .map(|i| {
            let x = i as f32 / NUM_SAMPLES as f32 * 6.0;
            let mut y = 0.0;
            
            // Superposition of different frequencies
            y += (x * freq + time * 2.0 + phase).sin() * 0.3;
            y += (x * freq * 2.3 + time * 1.7 + phase).sin() * 0.2;
            y += (x * freq * 3.7 + time * 0.9 + phase).cos() * 0.15;
            y += (x * freq * 5.1 + time * 2.3 + phase).sin() * 0.1;
            
            // Add some noise
            let noise = ((i as f32 * 1234.5 + time * 100.0).sin() * 43758.5453).fract() - 0.5;
            y += noise * 0.05;
            
            y.clamp(-1.0, 1.0)
        })
        .collect()
}

fn waveform_to_vertices(samples: &[f32], y_offset: f32, color: Color) -> Vec<Vertex2D> {
    let scale_y = 0.15;
    samples
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let x = (i as f32 / (NUM_SAMPLES - 1) as f32) * 1.9 - 0.95;
            let y = y_offset + sample * scale_y;
            Vertex2D::new(x, y, color)
        })
        .collect()
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
            device: None, pipeline: None, shader: None,
            window: None, surface: None,
            start_time: Instant::now(),
        })
    }

    fn init_gpu(&mut self) -> anyhow::Result<()> {
        let device = self.instance.create_device(DeviceType::DiscreteGpu)?;
        let shader = ShaderModule::from_slang(&device, rag::shader::builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            topology: PrimitiveTopology::LineStrip,
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

        // Colors for each channel
        let colors = [
            Color { r: 1.0, g: 0.3, b: 0.3, a: 1.0 },
            Color { r: 0.3, g: 1.0, b: 0.3, a: 1.0 },
            Color { r: 0.3, g: 0.5, b: 1.0, a: 1.0 },
            Color { r: 1.0, g: 0.8, b: 0.2, a: 1.0 },
        ];

        // Y offsets for each channel
        let y_offsets = [0.6, 0.2, -0.2, -0.6];

        let frame = FrameOutput::new(device, width, height, TextureFormat::Rgba8Unorm);
        let mut encoder = CommandEncoder::new();
        
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color { r: 0.02, g: 0.02, b: 0.08, a: 1.0 });
            pass.set_pipeline(pipeline);

            // Draw each channel
            for ch in 0..NUM_CHANNELS {
                let samples = generate_waveform(time, ch);
                let vertices = waveform_to_vertices(&samples, y_offsets[ch], colors[ch]);
                let vb = Buffer::with_data(device, &vertices, BufferUsage::VERTEX)?;
                pass.set_vertex_buffer(0, &vb);
                pass.draw(0..vertices.len() as u32, 0..1);
            }
        }

        let output = frame.render(encoder)?;
        let surface = self.surface.as_mut().unwrap();
        surface.resize(NonZeroU32::new(width).unwrap(), NonZeroU32::new(height).unwrap())
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut buffer = surface.buffer_mut()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        for (i, pixel) in buffer.iter_mut().enumerate() {
            let o = i * 4;
            if o + 2 < output.len() {
                *pixel = ((output[o] as u32) << 16) | ((output[o + 1] as u32) << 8) | (output[o + 2] as u32);
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
                    .with_title("RAG - Waveform Visualizer")
                    .with_inner_size(winit::dpi::LogicalSize::new(1024, 600))
            ).unwrap());
            let ctx = softbuffer::Context::new(window.clone()).unwrap();
            self.surface = Some(softbuffer::Surface::new(&ctx, window.clone()).unwrap());
            self.window = Some(window);
            self.init_gpu().unwrap();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) { event_loop.exit(); }
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
    println!("RAG Waveform Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}

