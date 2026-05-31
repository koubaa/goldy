//! Starfield example - classic 3D starfield flying through space.
//!
//! Demonstrates GPU compute + graphics integration using Surface API.
//! The compute shader updates star positions, the graphics shader renders them.
//!
//! Run with: cargo run --example starfield

use anyhow::Result;
use goldy::{
    Buffer, Color, CommandEncoder, ComputeEncoder, ComputePipeline, DataAccess, DeviceDescriptor,
    Instance, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions,
    ShaderModule, Surface, VertexBufferLayout,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const NUM_STARS: u32 = 500;

/// Star types for different celestial objects
const STAR_TYPE_NORMAL: f32 = 0.0;
const STAR_TYPE_GALAXY: f32 = 1.0;
const STAR_TYPE_QUASAR: f32 = 2.0;
const STAR_TYPE_WHITE_DWARF: f32 = 3.0;

/// Star structure matching the shader layout
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Star {
    x: f32,
    y: f32,
    z: f32,
    star_type: f32, // 0=normal, 1=galaxy, 2=quasar, 3=white dwarf
}

/// Uniform parameters for the compute shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct StarfieldParams {
    speed: f32,
    frame: f32,
    _pad1: f32,
    _pad2: f32,
}
impl goldy::StructuredBufferElement for Star {}
impl goldy::StructuredBufferElement for StarfieldParams {}

// Simple pseudo-random for initialization
static mut SEED: u32 = 12345;
fn rand_f32() -> f32 {
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        SEED as f32 / u32::MAX as f32
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    println!("Goldy Starfield Example");
    println!("  Up/Down - Change speed");
    println!("  Escape - Exit");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[derive(Default)]
struct App {
    state: Option<RenderState>,
}

struct RenderState {
    window: Arc<Window>,
    device: Arc<goldy::Device>,
    surface: Surface,
    // Compute resources
    compute_pipeline: ComputePipeline,
    // Buffers
    star_buffer: Buffer,
    params_buffer: Buffer,
    // Graphics resources
    render_pipeline: RenderPipeline,
    // State
    speed: f32,
    frame_count: f32,
    start_time: std::time::Instant,
}

impl RenderState {
    fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let surface = Surface::new(&device, window.as_ref())?;

        // Compute shader for star movement
        let compute_shader =
            ShaderModule::from_slang(&device, include_str!("../shaders/starfield_update.slang"))?;

        // Render shader for visualization
        let render_shader =
            ShaderModule::from_slang(&device, include_str!("../shaders/starfield_render.slang"))?;

        // Create star buffer with initial random positions and types
        // Start with z close to 1.0 so stars are immediately visible (small projection)
        let mut stars = Vec::with_capacity(NUM_STARS as usize);
        for _ in 0..NUM_STARS {
            // Assign star types with different probabilities:
            // 70% normal stars, 15% galaxies, 5% quasars, 10% white dwarfs
            let type_roll = rand_f32();
            let star_type = if type_roll < 0.70 {
                STAR_TYPE_NORMAL
            } else if type_roll < 0.85 {
                STAR_TYPE_GALAXY
            } else if type_roll < 0.90 {
                STAR_TYPE_QUASAR
            } else {
                STAR_TYPE_WHITE_DWARF
            };

            stars.push(Star {
                x: (rand_f32() - 0.5) * 0.8, // Smaller spread so projection stays on screen
                y: (rand_f32() - 0.5) * 0.8,
                z: 0.5 + rand_f32() * 0.5, // z from 0.5 to 1.0 (closer to 1.0 = smaller projection)
                star_type,
            });
        }

        let star_buffer = Buffer::with_data(&device, &stars, DataAccess::Scattered)?;

        // Create params buffer
        let initial_params = StarfieldParams {
            speed: 0.01, // Per-frame speed, same as original CPU version
            frame: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let params_buffer = Buffer::with_data(&device, &[initial_params], DataAccess::Broadcast)?;

        // Create compute pipeline
        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;

        // Create render pipeline
        let render_pipeline = RenderPipeline::new(
            &device,
            &render_shader,
            &render_shader,
            &RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::TriangleList,
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        println!("Created starfield with {} stars", NUM_STARS);

        Ok(Self {
            window,
            device,
            surface,
            compute_pipeline,
            star_buffer,
            params_buffer,
            render_pipeline,
            speed: 0.01,
            frame_count: 0.0,
            start_time: std::time::Instant::now(),
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1.0;

        // Update params buffer with current speed and frame
        let params = StarfieldParams {
            speed: self.speed,
            frame: self.frame_count,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        self.params_buffer.write_data(0, &[params])?;

        // Run compute pass to update stars
        let mut compute_encoder = ComputeEncoder::new();
        {
            let mut pass = compute_encoder.begin_compute_pass();
            pass.set_pipeline(&self.compute_pipeline);
            // Pass buffer indices via push constants
            pass.bind_resources(&[&self.star_buffer, &self.params_buffer]);
            let workgroups = NUM_STARS.div_ceil(64);
            pass.dispatch(workgroups, 1, 1);
        }
        compute_encoder.dispatch(&self.device)?;

        // Render stars
        let frame = self.surface.begin()?;

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.set_pipeline(&self.render_pipeline);
            // Pass buffer indices via push constants
            // Render shader only needs the star buffer (read-only)
            pass.bind_resources(&[&self.star_buffer]);
            // Draw 6 vertices (quad) per star instance
            pass.draw(0..6, 0..NUM_STARS);
        }

        frame.render(encoder)?;
        frame.present()?;

        self.window.request_redraw();
        Ok(())
    }

    fn change_speed(&mut self, delta: f32) {
        self.speed = (self.speed + delta).clamp(0.001, 0.1);
        if let Some(w) = Some(&self.window) {
            w.set_title(&format!("Goldy - Starfield (speed: {:.1})", self.speed));
        }
    }
}

impl Drop for RenderState {
    fn drop(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 {
            self.frame_count as f64 / elapsed
        } else {
            0.0
        };
        println!(
            "GOLDY_PERF: frames={} elapsed={elapsed:.2}s avg_fps={fps:.1}",
            self.frame_count as u64
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Starfield")
                            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768)),
                    )
                    .expect("Failed to create window"),
            );

            match RenderState::new(window.clone()) {
                Ok(state) => {
                    self.state = Some(state);
                    window.request_redraw();
                }
                Err(e) => {
                    tracing::error!("Failed to create render state: {}", e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if let Some(state) = &mut self.state {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Named(NamedKey::ArrowUp) => state.change_speed(0.005),
                        Key::Named(NamedKey::ArrowDown) => state.change_speed(-0.005),
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    if size.width > 0 && size.height > 0 {
                        state.surface.resize(size.width, size.height).ok();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    if let Err(e) = state.render() {
                        tracing::error!("Render error: {}", e);
                    }
                }
            }
            _ => {}
        }
    }
}
