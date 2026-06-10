//! GPU Particle Simulation Example
//!
//! Demonstrates compute + graphics in a single task graph:
//! particle update dispatch → offscreen render → swapchain blit.
//!
//! Run with: `cargo run --example compute_particles`

use anyhow::Result;
use goldy::{
    BufferFlags, BufferKind, Color, ComputePipeline, DeviceDescriptor, Instance, NodeAccess, Parcel, PrimitiveTopology,
    RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions, ResourceAccess, RetainedPool,
    ShaderModule, Surface, TaskGraph, VertexBufferLayout,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};
mod common;

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
    device: Arc<goldy::Device>,
    surface: Surface,
    scene_rt: RenderTarget,
    frame_graph: TaskGraph,
    compute_pipeline: ComputePipeline,
    _retained_pool: RetainedPool,
    particle_buffer: Parcel,
    params_buffer: Parcel,
    render_pipeline: RenderPipeline,
    frame_count: u32,
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
}

impl RenderState {
    fn create_scene_rt(device: &goldy::Device, surface: &Surface) -> Result<RenderTarget> {
        let (width, height) = surface.size();
        RenderTarget::new(device, width.max(1), height.max(1), surface.format())
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

        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/particle_update.slang"))?;
        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/particle_render.slang"))?;

        let mut particles = Vec::with_capacity(NUM_PARTICLES as usize);
        for i in 0..NUM_PARTICLES {
            let t = i as f32 / NUM_PARTICLES as f32;
            let angle = t * std::f32::consts::TAU * 5.0;
            let radius = 0.1 + t * 0.6;

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

        let mut retained_pool = RetainedPool::new(device.clone());
        let particle_buffer = retained_pool.acquire_buffer_with_data(&particles, BufferKind::Scattered)?;
        let params_buffer =
            retained_pool.acquire_buffer_sized::<SimParams>(1, BufferKind::Broadcast, BufferFlags::empty())?;

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

        println!("Created compute particles example with {} particles", NUM_PARTICLES);
        println!("Press Escape or close window to exit");

        Ok(Self {
            window,
            device,
            surface,
            scene_rt,
            frame_graph: TaskGraph::new(),
            compute_pipeline,
            _retained_pool: retained_pool,
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

        let workgroups = NUM_PARTICLES.div_ceil(64);

        self.frame_graph.clear();
        self.frame_graph.write_parcel(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&SimParams { delta_time: dt }).to_vec(),
        )?;
        self.frame_graph
            .node("update_particles", &self.compute_pipeline)
            .bind_parcel(&self.particle_buffer, NodeAccess::ReadWrite)
            .bind_parcel(&self.params_buffer, NodeAccess::Read)
            .bind_resources_raw_slice(&[
                self.particle_buffer.resource_index(ResourceAccess::Write).unwrap(),
                self.params_buffer.resource_index(ResourceAccess::Read).unwrap(),
            ])
            .dispatch(workgroups, 1, 1);

        let bg_color = Color {
            r: 0.03,
            g: 0.02,
            b: 0.08,
            a: 1.0,
        };

        let mut pass = self.frame_graph.render_pass("particles", &self.scene_rt);
        pass.bind_parcel_mut(&self.particle_buffer, NodeAccess::Read);
        pass.clear(bg_color);
        pass.set_pipeline(&self.render_pipeline);
        pass.bind_resources_raw(&[self.particle_buffer.resource_index(ResourceAccess::Read).unwrap()]);
        pass.draw(0..6, 0..NUM_PARTICLES);
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
                    window.request_redraw();
                }
                Err(e) => {
                    tracing::error!("Failed to create render state: {:?}", e);
                    event_loop.exit();
                }
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            common::exit_if_timed_out(event_loop, state.start_time);
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
