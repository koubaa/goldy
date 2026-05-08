//! Plasma example - classic demoscene plasma effect.
//!
//! Run with: cargo run --example plasma

use goldy::{
    shaders, Buffer, Color, CommandEncoder, DataAccess, DeviceType, Instance, RenderPipeline,
    RenderPipelineDesc, ShaderModule, Surface, VertexBufferLayout,
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

/// Uniform buffer data (must match shader cbuffer layout)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    time: f32,
}

struct App {
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    uniform_buffer: Option<Buffer>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    start_time: Instant,
    frame_count: u32,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None,
            pipeline: None,
            shader: None,
            uniform_buffer: None,
            window: None,
            surface: None,
            start_time: Instant::now(),
            frame_count: 0,
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);

        // Create surface first to get the correct format
        let surface = Surface::new(&device, window.as_ref())?;

        // Create shader
        let shader = ShaderModule::from_slang(&device, shaders::PLASMA)?;

        // Create pipeline - no vertex buffer needed, shader uses SV_VertexID
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        // Create uniform buffer
        let uniform_buffer = Buffer::new(
            device.as_ref(),
            std::mem::size_of::<Uniforms>() as u64,
            DataAccess::Broadcast,
        )?;

        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.uniform_buffer = Some(uniform_buffer);
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

        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let uniform_buffer = self.uniform_buffer.as_ref().unwrap();

        // Update uniform buffer with current time
        let time = self.start_time.elapsed().as_secs_f32();
        let uniforms = Uniforms { time };
        uniform_buffer.write_data(0, &[uniforms])?;

        // Acquire frame
        let frame = surface.begin()?;

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.set_pipeline(pipeline);
            // Pass buffer indices via push constants
            pass.bind_resources(&[uniform_buffer]);
            // No vertex buffer needed - shader uses SV_VertexID
            pass.draw_fullscreen();
        }

        frame.render(encoder)?;
        frame.present()?;

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
                            .with_title("Goldy - Plasma Effect")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
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
    println!("Goldy Plasma Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
