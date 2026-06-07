//! GPU Particle Simulation Example
//!
//! This example demonstrates compute + graphics integration:
//! 1. Compute shader updates particle positions/velocities
//! 2. Graphics shader renders particles as colored quads (instanced)
//!
//! Run with: `cargo run --example compute_particles`

use anyhow::Result;
use goldy::{
    Buffer, BufferFlags, BufferKind, Color, CommandEncoder, ComputePipeline, DeviceDescriptor, Instance, NodeAccess,
    PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, ResourceAccess, ShaderModule,
    Surface, TaskGraph, VertexBufferLayout,
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
impl goldy::StructuredBufferElement for Particle {}

/// Per-frame simulation parameters passed to the compute shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    delta_time: f32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

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
    context: goldy::Context,
    surface: Surface,
    compute_pipeline: ComputePipeline,
    particle_buffer: Buffer,
    params_buffer: Buffer,
    render_pipeline: RenderPipeline,
    frame_count: u32,
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
}

impl RenderState {
    fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let surface = Surface::new(&ctx, window.as_ref())?;

        // Compute shader for particle simulation
        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/particle_update.slang"))?;

        // Render shader for visualization
        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/particle_render.slang"))?;

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
                velocity: [angle.sin() * 0.3 + noise_x * 0.2, -angle.cos() * 0.3 + noise_y * 0.2],
            });
        }

        let particle_buffer = device.alloc_buffer_with_data(&particles, BufferKind::Scattered)?;

        // Per-frame simulation params (dt written each frame before dispatch)
        let params_buffer = device.alloc_buffer(
            std::mem::size_of::<SimParams>() as u64,
            BufferKind::Broadcast,
            None,
            BufferFlags::empty(),
        )?;

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

        println!("Created compute particles example with {} particles", NUM_PARTICLES);
        println!("Press Escape or close window to exit");

        Ok(Self {
            window,
            context: ctx,
            surface,
            compute_pipeline,
            particle_buffer,
            params_buffer,
            render_pipeline,
            frame_count: 0,
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;

        let dt = self.last_frame_time.elapsed().as_secs_f32().min(0.05);
        self.last_frame_time = std::time::Instant::now();
        self.params_buffer.write_data(0, &[SimParams { delta_time: dt }])?;

        // Run compute pass to update particles
        let workgroups = NUM_PARTICLES.div_ceil(64);
        let mut graph = TaskGraph::new();
        graph
            .node("update_particles", &self.compute_pipeline)
            .bind_buffer(&self.particle_buffer, NodeAccess::ReadWrite)
            .bind_buffer(&self.params_buffer, NodeAccess::Read)
            .bind_resources_raw_slice(&[
                self.particle_buffer.resource_index(ResourceAccess::Write).unwrap(),
                self.params_buffer.resource_index(ResourceAccess::Read).unwrap(),
            ])
            .dispatch(workgroups, 1, 1);
        graph.dispatch(&self.context)?;

        // Render particles
        let frame = self.surface.begin()?;

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
            pass.bind_resources(&[&self.particle_buffer]);
            // Draw 6 vertices (quad) per particle instance
            pass.draw(0..6, 0..NUM_PARTICLES);
        }

        frame.render(encoder)?;
        frame.present()?;

        // Request redraw for continuous animation
        self.window.request_redraw();

        Ok(())
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
            self.frame_count
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
