//! GPU Particle Simulation Example
//!
//! Demonstrates retained scheme with compute dispatch → offscreen render → copy-to-present.
//!
//! Run with: `cargo run --example compute_particles`

use anyhow::Result;
use goldy::{
    Buffer, BufferFlags, BufferKind, Color, ComputePipeline, DeviceDescriptor, Instance, Lease, LeaseRenderTarget,
    NodeAccess, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, Transaction, VertexBufferLayout,
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    position: [f32; 2],
    velocity: [f32; 2],
}
impl goldy::StructuredBufferElement for Particle {}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    delta_time: f32,
}
impl goldy::StructuredBufferElement for SimParams {}

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
    ctx: goldy::Context,
    surface: SurfaceExchange,
    present: Transaction,
    scheme: Scheme,
    scene_rt: Lease<LeaseRenderTarget>,
    compute_pipeline: ComputePipeline,
    render_shader: ShaderModule,
    render_pipeline: RenderPipeline,
    _retained_pool: RetainedPool,
    particle_buffer: Buffer,
    params_buffer: Buffer,
    frame_count: u32,
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
}

impl RenderState {
    fn create_render_pipeline(
        device: &goldy::Device,
        render_shader: &ShaderModule,
        surface: &SurfaceExchange,
    ) -> Result<RenderPipeline> {
        common::render_pipeline_for_surface(
            device,
            render_shader,
            surface,
            RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
        )
        .map_err(Into::into)
    }

    fn record_scheme(
        scheme: &mut Scheme,
        surface: &SurfaceExchange,
        compute_pipeline: &ComputePipeline,
        render_pipeline: &RenderPipeline,
        particle_buffer: &Buffer,
        params_buffer: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
    ) -> anyhow::Result<Transaction> {
        scheme
            .node("update_particles", compute_pipeline)
            .with_parcel(particle_buffer, NodeAccess::ReadWrite)
            .with_parcel(params_buffer, NodeAccess::Read)
            .dispatch(NUM_PARTICLES.div_ceil(64), 1, 1);

        let bg_color = Color {
            r: 0.03,
            g: 0.02,
            b: 0.08,
            a: 1.0,
        };

        let mut pass = scheme.render_pass("particles", scene_rt, TargetLoad::Clear(bg_color));
        pass.with_parcel(particle_buffer, NodeAccess::Read);
        pass.set_pipeline(render_pipeline);
        pass.draw(0..6, 0..NUM_PARTICLES);
        pass.finish();

        surface.bind_render_target(scheme, scene_rt).map_err(Into::into)
    }

    fn rerecord_scheme(&mut self) {
        let mut scheme = Scheme::new(&self.ctx);
        let (width, height) = self.surface.size();
        if let Ok(rt) = scheme.lease_render_target(width.max(1), height.max(1), self.surface.format(), None) {
            self.scene_rt = rt;
            if let Ok(present) = Self::record_scheme(
                &mut scheme,
                &self.surface,
                &self.compute_pipeline,
                &self.render_pipeline,
                &self.particle_buffer,
                &self.params_buffer,
                &self.scene_rt,
            ) {
                self.present = present;
                self.scheme = scheme;
            }
        }
    }

    fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let surface = SurfaceExchange::new(&ctx, window.as_ref(), SurfaceConfig::default())?;

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
        let render_pipeline = Self::create_render_pipeline(&device, &render_shader, &surface)?;

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = surface.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), surface.format(), None)?;
        let present = Self::record_scheme(
            &mut scheme,
            &surface,
            &compute_pipeline,
            &render_pipeline,
            &particle_buffer,
            &params_buffer,
            &scene_rt,
        )?;

        println!("Created compute particles example with {NUM_PARTICLES} particles (Scheme + Present)");
        println!("Press Escape or close window to exit");

        Ok(Self {
            window,
            device,
            ctx,
            surface,
            present,
            scheme,
            scene_rt,
            compute_pipeline,
            render_shader,
            render_pipeline,
            _retained_pool: retained_pool,
            particle_buffer,
            params_buffer,
            frame_count: 0,
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;

        let dt = self.last_frame_time.elapsed().as_secs_f32().min(0.05);
        self.last_frame_time = std::time::Instant::now();

        let mut upload = Scheme::new(&self.ctx);
        upload.write_parcel(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&SimParams { delta_time: dt }).to_vec(),
        )?;
        upload.submit()?;

        let mut submission = self.scheme.submit()?;
        self.present.claim(&mut submission)?.consume()?;

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
                    tracing::error!("Failed to create render state: {e:?}");
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
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    if size.width > 0 && size.height > 0 {
                        state.surface.resize(size.width, size.height).ok();
                        if let Ok(pipeline) =
                            RenderState::create_render_pipeline(&state.device, &state.render_shader, &state.surface)
                        {
                            state.render_pipeline = pipeline;
                        }
                        state.rerecord_scheme();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    if let Err(e) = state.render() {
                        tracing::error!("Render error: {e}");
                    }
                }
            }
            _ => {}
        }
    }
}
