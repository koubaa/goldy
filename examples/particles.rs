//! Particles example - rain/snow particle system.
//!
//! Demonstrates GPU compute + graphics in a single task graph:
//! compute update → offscreen render pass → swapchain blit → present.
//!
//! Run with: cargo run --example particles

use anyhow::Result;
use goldy::{
    Buffer, BufferKind, Color, ComputePipeline, DeviceDescriptor, Instance, NodeAccess,
    PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions, ResourceAccess,
    ShaderModule, Surface, TaskGraph, VertexBufferLayout,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const NUM_PARTICLES: u32 = 1000;

/// Particle structure matching the shader layout
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    position: [f32; 2],
    velocity: [f32; 2],
    size: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

/// Uniform parameters for the shaders
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleParams {
    is_snow: f32,
    frame: f32,
    _pad1: f32,
    _pad2: f32,
}
impl goldy::StructuredBufferElement for Particle {}
impl goldy::StructuredBufferElement for ParticleParams {}

// Simple pseudo-random for initialization
static mut SEED: u32 = 42;
fn random() -> f32 {
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
    println!("Goldy Particles Example");
    println!("  Space - Toggle rain/snow");
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
    scene_rt: RenderTarget,
    frame_graph: TaskGraph,
    compute_pipeline: ComputePipeline,
    particle_buffer: Buffer,
    params_buffer: Buffer,
    render_pipeline: RenderPipeline,
    is_snow: bool,
    frame_count: f32,
    start_time: std::time::Instant,
}

impl RenderState {
    fn create_scene_rt(device: &goldy::Device, surface: &Surface) -> Result<RenderTarget> {
        let (width, height) = surface.size();
        Ok(RenderTarget::new(
            device,
            width.max(1),
            height.max(1),
            surface.format(),
        )?)
    }

    fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let surface = Surface::new(&ctx, window.as_ref())?;
        let scene_rt = Self::create_scene_rt(&device, &surface)?;

        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/rain_snow_update.slang"))?;
        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/rain_snow_render.slang"))?;

        let particles = Self::create_particles(false);
        let particle_buffer = device.alloc_buffer_with_data(&particles, BufferKind::Scattered)?;

        let initial_params = ParticleParams {
            is_snow: 0.0,
            frame: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let params_buffer = device.alloc_buffer_with_data(&[initial_params], BufferKind::Broadcast)?;

        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;

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

        println!("Created rain/snow simulation with {} particles", NUM_PARTICLES);

        Ok(Self {
            window,
            device,
            surface,
            scene_rt,
            frame_graph: TaskGraph::new(),
            compute_pipeline,
            particle_buffer,
            params_buffer,
            render_pipeline,
            is_snow: false,
            frame_count: 0.0,
            start_time: std::time::Instant::now(),
        })
    }

    fn create_particles(is_snow: bool) -> Vec<Particle> {
        let mut particles = Vec::with_capacity(NUM_PARTICLES as usize);
        for _ in 0..NUM_PARTICLES {
            let x = random() * 2.0 - 1.0;
            let y = random() * 2.2 - 1.0;

            let (vx, vy, size) = if is_snow {
                (
                    (random() - 0.5) * 0.005,
                    -(0.002 + random() * 0.005),
                    0.003 + random() * 0.008,
                )
            } else {
                (
                    (random() - 0.5) * 0.002,
                    -(0.01 + random() * 0.02),
                    0.002 + random() * 0.003,
                )
            };

            particles.push(Particle {
                position: [x, y],
                velocity: [vx, vy],
                size,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
            });
        }
        particles
    }

    fn toggle_mode(&mut self) -> Result<()> {
        self.is_snow = !self.is_snow;

        let particles = Self::create_particles(self.is_snow);
        self.particle_buffer.write_data(0, &particles)?;

        self.window.set_title(&format!(
            "Goldy - {} (Space to toggle)",
            if self.is_snow { "Snow" } else { "Rain" }
        ));

        Ok(())
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1.0;

        let params = ParticleParams {
            is_snow: if self.is_snow { 1.0 } else { 0.0 },
            frame: self.frame_count,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        self.params_buffer.write_data(0, &[params])?;

        let bg_color = if self.is_snow {
            Color {
                r: 0.05,
                g: 0.05,
                b: 0.15,
                a: 1.0,
            }
        } else {
            Color {
                r: 0.02,
                g: 0.02,
                b: 0.05,
                a: 1.0,
            }
        };

        self.frame_graph.clear();
        self.frame_graph
            .node("update_particles", &self.compute_pipeline)
            .bind_buffer(&self.particle_buffer, NodeAccess::ReadWrite)
            .bind_buffer(&self.params_buffer, NodeAccess::Read)
            .bind_resources_raw_slice(&[
                self.particle_buffer.resource_index(ResourceAccess::Write).unwrap(),
                self.params_buffer.resource_index(ResourceAccess::Read).unwrap(),
            ])
            .dispatch(NUM_PARTICLES.div_ceil(64), 1, 1);

        let mut pass = self.frame_graph.render_pass("particles", &self.scene_rt);
        pass.bind_buffer_mut(&self.particle_buffer, NodeAccess::Read);
        pass.bind_buffer_mut(&self.params_buffer, NodeAccess::Read);
        pass.clear(bg_color);
        pass.set_pipeline(&self.render_pipeline);
        pass.bind_resources(&[&self.particle_buffer, &self.params_buffer]);
        pass.draw_quads(NUM_PARTICLES);
        pass.finish_recorded();

        let swapchain = self.frame_graph.declare_swapchain_output();
        self.frame_graph
            .copy_render_target_to_swapchain(&self.scene_rt, swapchain);

        let frame = self.surface.begin()?;
        let frame = self.surface.submit_graph_to_frame(&mut self.frame_graph, frame)?;
        frame.present()?;

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
                            .with_title("Goldy - Rain (Space to toggle)")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
                    )
                    .expect("Failed to create window"),
            );

            match RenderState::new(window.clone()) {
                Ok(state) => {
                    self.state = Some(state);
                    window.request_redraw();
                }
                Err(e) => {
                    tracing::error!("Failed to create render state: {:#}", e);
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
                        Key::Named(NamedKey::Space) => {
                            if let Err(e) = state.toggle_mode() {
                                tracing::error!("Failed to toggle mode: {}", e);
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    if size.width > 0 && size.height > 0 {
                        state.surface.resize(size.width, size.height).ok();
                        if let Ok(rt) = RenderState::create_scene_rt(&state.device, &state.surface) {
                            state.scene_rt = rt;
                        }
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
