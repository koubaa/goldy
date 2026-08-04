//! Particles example - rain/snow particle system.
//!
//! Demonstrates retained scheme with compute dispatch → offscreen render → copy-to-present.
//!
//! Run with: `cargo run --example particles`

use anyhow::Result;
use goldy::{
    Buffer, BufferFlags, BufferKind, Color, ComputePipeline, DepositTransaction, DeviceDescriptor, Instance, Lease,
    LeaseRenderTarget, MemoryExchange, NodeAccess, PrimitiveTopology, RenderPipeline, RenderPipelineDesc,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, Transaction,
    VertexBufferLayout,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};
mod common;

const NUM_PARTICLES: u32 = 1000;

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
    upload_scheme: Scheme,
    params_deposit: DepositTransaction,
    is_snow: bool,
    frame_count: f32,
    start_time: std::time::Instant,
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
        bg_color: Color,
    ) -> anyhow::Result<Transaction> {
        scheme
            .node("update_particles", compute_pipeline)
            .with_parcel(particle_buffer, NodeAccess::ReadWrite)
            .with_parcel(params_buffer, NodeAccess::Read)
            .dispatch(NUM_PARTICLES.div_ceil(64), 1, 1);

        let mut pass = scheme.render_pass("particles", scene_rt, TargetLoad::Clear(bg_color));
        pass.with_parcel(particle_buffer, NodeAccess::Read);
        pass.with_parcel(params_buffer, NodeAccess::Read);
        pass.set_pipeline(render_pipeline);
        pass.draw(0..6, 0..NUM_PARTICLES);
        pass.finish();

        surface.bind_render_target(scheme, scene_rt).map_err(Into::into)
    }

    fn background_color(is_snow: bool) -> Color {
        if is_snow {
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
        }
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
                Self::background_color(self.is_snow),
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

        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/rain_snow_update.slang"))?;
        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/rain_snow_render.slang"))?;

        let particles = Self::create_particles(false);
        let mut retained_pool = RetainedPool::new(device.clone());
        let particle_buffer = retained_pool.acquire_buffer_with_data(&particles, BufferKind::Scattered)?;
        let params_buffer =
            retained_pool.acquire_buffer_sized::<ParticleParams>(1, BufferKind::Broadcast, BufferFlags::empty())?;

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
            Self::background_color(false),
        )?;

        let mut upload_scheme = Scheme::new(&ctx);
        let params_deposit = MemoryExchange::new(&ctx).bind_deposit_buffer(
            &mut upload_scheme,
            &params_buffer,
            std::mem::size_of::<ParticleParams>() as u64,
        )?;

        println!("Created rain/snow simulation with {NUM_PARTICLES} particles (Scheme + Present)");

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
            upload_scheme,
            params_deposit,
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
        let particle_capacity = (NUM_PARTICLES as u64) * std::mem::size_of::<Particle>() as u64;
        let mut particle_upload = Scheme::new(&self.ctx);
        let particle_deposit = MemoryExchange::new(&self.ctx).bind_deposit_buffer(
            &mut particle_upload,
            &self.particle_buffer,
            particle_capacity,
        )?;
        particle_deposit.write(&mut particle_upload, 0, bytemuck::cast_slice(&particles))?;
        particle_upload.submit()?;

        self.window.set_title(&format!(
            "Goldy - {} (Space to toggle)",
            if self.is_snow { "Snow" } else { "Rain" }
        ));

        self.rerecord_scheme();
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

        self.params_deposit
            .write(&mut self.upload_scheme, 0, bytemuck::bytes_of(&params))?;
        self.upload_scheme.submit()?;

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
            self.frame_count as u64
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(common::hidden_window("Goldy - Rain (Space to toggle)", 800, 600))
                    .expect("Failed to create window"),
            );

            match RenderState::new(window.clone()) {
                Ok(mut state) => {
                    if let Err(e) = state.render() {
                        tracing::error!("First frame error: {e}");
                    }
                    common::reveal_window(&window);
                    self.state = Some(state);
                    window.request_redraw();
                }
                Err(e) => {
                    tracing::error!("Failed to create render state: {e:#}");
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
                if let Some(state) = &mut self.state {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Named(NamedKey::Space) => {
                            if let Err(e) = state.toggle_mode() {
                                tracing::error!("Failed to toggle mode: {e}");
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
