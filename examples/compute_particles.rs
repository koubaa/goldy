//! GPU Particle Simulation Example
//!
//! This example demonstrates compute + graphics integration:
//! 1. Compute shader updates particle positions/velocities
//! 2. Graphics shader renders particles as colored quads (instanced)
//!
//! Run with: `cargo run --example compute_particles`

use anyhow::Result;
use goldy::{
    Buffer, Color, CommandEncoder, ComputeEncoder, ComputePipeline, DataAccess, DeviceType,
    Instance, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, ShaderModule, Surface,
    VertexBufferLayout,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const NUM_PARTICLES: u32 = 1024;

/// Particle structure matching the shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    position: [f32; 2],
    velocity: [f32; 2],
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

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
    // Buffer
    particle_buffer: Buffer,
    // Graphics resources
    render_pipeline: RenderPipeline,
    // Frame counter for debug
    frame_count: u32,
}

impl RenderState {
    fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(instance.create_device(DeviceType::DiscreteGpu)?);
        let surface = Surface::new(&device, window.as_ref())?;

        // Compute shader for particle simulation
        let compute_shader =
            ShaderModule::from_slang(&device, include_str!("../shaders/particle_update.slang"))?;

        // Render shader for visualization
        let render_shader =
            ShaderModule::from_slang(&device, include_str!("../shaders/particle_render.slang"))?;

        // Create particle buffer with initial scattered positions
        let mut particles = Vec::with_capacity(NUM_PARTICLES as usize);
        for i in 0..NUM_PARTICLES {
            // Distribute particles in a spiral pattern
            let t = i as f32 / NUM_PARTICLES as f32;
            let angle = t * std::f32::consts::TAU * 5.0; // 5 rotations
            let radius = 0.1 + t * 0.6; // Spiral outward

            // Add some randomness via deterministic noise
            let noise_x = ((i * 17) % 100) as f32 / 100.0 - 0.5;
            let noise_y = ((i * 31) % 100) as f32 / 100.0 - 0.5;

            particles.push(Particle {
                position: [
                    radius * angle.cos() + noise_x * 0.1,
                    radius * angle.sin() + noise_y * 0.1,
                ],
                velocity: [
                    angle.sin() * 0.3 + noise_x * 0.2,
                    -angle.cos() * 0.3 + noise_y * 0.2,
                ],
            });
        }

        let particle_buffer = Buffer::with_data(&device, &particles, DataAccess::Scattered)?;

        // Create compute pipeline
        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;

        // Create render pipeline
        let render_pipeline = RenderPipeline::new(
            &device,
            &render_shader,
            &render_shader,
            &RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(), // Shader uses SV_VertexID, not vertex attributes
                topology: PrimitiveTopology::TriangleList,
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        println!(
            "Created compute particles example with {} particles",
            NUM_PARTICLES
        );
        println!("Press Escape or close window to exit");

        Ok(Self {
            window,
            device,
            surface,
            compute_pipeline,
            particle_buffer,
            render_pipeline,
            frame_count: 0,
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;

        // Run compute pass to update particles
        let mut compute_encoder = ComputeEncoder::new();
        {
            let mut pass = compute_encoder.begin_compute_pass();
            pass.set_pipeline(&self.compute_pipeline);
            // Pass buffer indices via push constants
            pass.set_push_constants(&[&self.particle_buffer]);
            // Dispatch enough workgroups to cover all particles (64 threads per group)
            let workgroups = NUM_PARTICLES.div_ceil(64);
            pass.dispatch(workgroups, 1, 1);
        }
        compute_encoder.dispatch(&self.device)?;

        // Render particles
        let frame = self.surface.acquire()?;

        // Dark blue-purple background
        let bg_color = Color {
            r: 0.03,
            g: 0.02,
            b: 0.08,
            a: 1.0,
        };

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(bg_color);
            pass.set_pipeline(&self.render_pipeline);
            // Pass buffer indices via push constants
            pass.set_push_constants(&[&self.particle_buffer]);
            // Draw 6 vertices (quad) per particle instance
            pass.draw(0..6, 0..NUM_PARTICLES);
        }

        frame.render(encoder)?;
        self.surface.present(frame)?;

        // Request redraw for continuous animation
        self.window.request_redraw();

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Compute Particles")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
                    )
                    .expect("Failed to create window"),
            );

            match RenderState::new(window.clone()) {
                Ok(state) => {
                    self.state = Some(state);
                    // Request initial redraw to start the render loop
                    window.request_redraw();
                }
                Err(e) => {
                    tracing::error!("Failed to create render state: {:?}", e);
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
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    event_loop.exit();
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
