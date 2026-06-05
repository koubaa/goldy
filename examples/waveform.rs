//! Waveform example - animated audio waveform visualizer.
//!
//! Demonstrates procedural waveform generation and line rendering using Surface API.
//!
//! Run with: cargo run --example waveform

use goldy::{
    Buffer, Color, CommandEncoder, BufferKind, DeviceDescriptor, Instance, PrimitiveTopology,
    RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, ShaderModule, Surface, Vertex2D,
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
            let noise = ((i as f32 * 1234.5 + time * 100.0).sin() * 43_758.547_f32).fract() - 0.5;
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

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct App {
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    start_time: Instant,
    frame_count: u32,
    // Each frame may have multiple channel buffers
    frame_buffers: Vec<Vec<Buffer>>,
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
            frame_count: 0,
            frame_buffers: Vec::with_capacity(MAX_FRAMES_IN_FLIGHT),
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let surface = Surface::new(&ctx, window.as_ref())?;
        let shader = ShaderModule::from_slang(&device, goldy::shader::builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: Vertex2D::layout(),
                target_format: surface.format(),
                topology: PrimitiveTopology::LineStrip,
                ..Default::default()
            },
        )?;
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        self.frame_count += 1;

        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let time = self.start_time.elapsed().as_secs_f32();

        // Colors for each channel
        let colors = [
            Color {
                r: 1.0,
                g: 0.3,
                b: 0.3,
                a: 1.0,
            },
            Color {
                r: 0.3,
                g: 1.0,
                b: 0.3,
                a: 1.0,
            },
            Color {
                r: 0.3,
                g: 0.5,
                b: 1.0,
                a: 1.0,
            },
            Color {
                r: 1.0,
                g: 0.8,
                b: 0.2,
                a: 1.0,
            },
        ];

        // Y offsets for each channel
        let y_offsets = [0.6, 0.2, -0.2, -0.6];

        // Pre-create all buffers for this frame
        let mut channel_buffers = Vec::with_capacity(NUM_CHANNELS);
        let mut vertex_counts = Vec::with_capacity(NUM_CHANNELS);
        for ch in 0..NUM_CHANNELS {
            let samples = generate_waveform(time, ch);
            let vertices = waveform_to_vertices(&samples, y_offsets[ch], colors[ch]);
            vertex_counts.push(vertices.len() as u32);
            channel_buffers.push(
                device
                    .as_ref()
                    .alloc_buffer_with_data(&vertices, BufferKind::Scattered)?,
            );
        }

        let frame = surface.begin()?;

        // Drop oldest frame's buffers now that GPU is done
        if self.frame_buffers.len() >= MAX_FRAMES_IN_FLIGHT {
            self.frame_buffers.remove(0);
        }

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color {
                r: 0.02,
                g: 0.02,
                b: 0.08,
                a: 1.0,
            });
            pass.set_pipeline(pipeline);

            // Draw each channel
            for (ch, vb) in channel_buffers.iter().enumerate() {
                pass.set_vertex_buffer(0, vb);
                pass.draw(0..vertex_counts[ch], 0..1);
            }
        }

        frame.render(encoder)?;
        frame.present()?;

        // Keep this frame's buffers alive
        self.frame_buffers.push(channel_buffers);

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

impl Drop for App {
    fn drop(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 {
            self.frame_count as f64 / elapsed
        } else {
            0.0
        };
        println!(
            "GOLDY_PERF: frames={} elapsed={elapsed:.2}s avg_fps={fps:.1}",
            self.frame_count
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Waveform Visualizer (Surface API)")
                            .with_inner_size(winit::dpi::LogicalSize::new(1024, 600)),
                    )
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
            window.request_redraw();
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
                if let Err(e) = self.render_frame() {
                    tracing::error!("Render error: {}", e);
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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    println!("Goldy Waveform Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
