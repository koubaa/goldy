//! Starfield example - classic 3D starfield flying through space.
//!
//! Demonstrates retained scheme with compute dispatch → offscreen render → copy-to-present.
//!
//! Run with: `cargo run --example starfield`

use anyhow::Result;
use goldy::{
    Buffer, BufferFlags, BufferKind, Color, ComputePipeline, DepositTransaction, DeviceDescriptor, Instance, Lease,
    LeaseRenderTarget, MemoryExchange, NodeAccess, PrimitiveTopology, RenderPipeline, RenderPipelineDesc,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, Texture,
    TextureFormat, Transaction, VertexBufferLayout, WithdrawTransaction,
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
use common::CaptureDump;

const NUM_STARS: u32 = 500;

const STAR_TYPE_NORMAL: f32 = 0.0;
const STAR_TYPE_GALAXY: f32 = 1.0;
const STAR_TYPE_QUASAR: f32 = 2.0;
const STAR_TYPE_WHITE_DWARF: f32 = 3.0;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Star {
    x: f32,
    y: f32,
    z: f32,
    star_type: f32,
}

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

    if common::capture_requested() {
        let mut state = RenderState::new(None)?;
        while !state.capture_done() {
            state.render()?;
        }
        return Ok(());
    }

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
    window: Option<Arc<Window>>,
    device: Arc<goldy::Device>,
    ctx: goldy::Context,
    surface: Option<SurfaceExchange>,
    capture: Option<CaptureDump>,
    readback: Option<Texture>,
    present: Option<Transaction>,
    withdraw: Option<WithdrawTransaction>,
    scheme: Scheme,
    scene_rt: Lease<LeaseRenderTarget>,
    compute_pipeline: ComputePipeline,
    render_shader: ShaderModule,
    render_pipeline: RenderPipeline,
    _retained_pool: RetainedPool,
    star_buffer: Buffer,
    params_buffer: Buffer,
    upload_scheme: Scheme,
    params_deposit: DepositTransaction,
    speed: f32,
    frame_count: f32,
    start_time: std::time::Instant,
}

impl RenderState {
    fn create_render_pipeline(
        device: &goldy::Device,
        render_shader: &ShaderModule,
        format: TextureFormat,
    ) -> Result<RenderPipeline> {
        common::render_pipeline(
            device,
            render_shader,
            format,
            RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
        )
    }

    fn bind_frame(
        scheme: &mut Scheme,
        scene_rt: &Lease<LeaseRenderTarget>,
        surface: Option<&SurfaceExchange>,
        readback: Option<&Texture>,
    ) -> anyhow::Result<(Option<Transaction>, Option<WithdrawTransaction>)> {
        if let Some(surface) = surface {
            let present = surface.bind_render_target(scheme, scene_rt)?;
            Ok((Some(present), None))
        } else {
            let readback = readback.expect("capture readback");
            scheme.copy_to_texture(scene_rt, readback)?;
            let withdraw = MemoryExchange::new(scheme.context()).bind_withdraw(scheme, readback)?;
            Ok((None, Some(withdraw)))
        }
    }

    fn record_scheme(
        scheme: &mut Scheme,
        compute_pipeline: &ComputePipeline,
        render_pipeline: &RenderPipeline,
        star_buffer: &Buffer,
        params_buffer: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
    ) {
        scheme
            .node("update_stars", compute_pipeline)
            .with_parcel(star_buffer, NodeAccess::ReadWrite)
            .with_parcel(params_buffer, NodeAccess::Read)
            .dispatch(NUM_STARS.div_ceil(64), 1, 1);

        let mut pass = scheme.render_pass("starfield", scene_rt, TargetLoad::Clear(Color::BLACK));
        pass.with_parcel(star_buffer, NodeAccess::Read);
        pass.set_pipeline(render_pipeline);
        pass.draw(0..6, 0..NUM_STARS);
        pass.finish();
    }

    fn target(&self) -> (TextureFormat, u32, u32) {
        if let Some(surface) = &self.surface {
            let (width, height) = surface.size();
            (surface.format(), width, height)
        } else {
            let capture = self.capture.as_ref().expect("capture dump");
            let (width, height) = capture.size();
            (CaptureDump::format(), width, height)
        }
    }

    fn capture_done(&self) -> bool {
        self.capture.as_ref().is_none_or(CaptureDump::finished)
    }

    fn rerecord_scheme(&mut self) {
        let mut scheme = Scheme::new(&self.ctx);
        let (format, width, height) = self.target();
        if let Ok(rt) = scheme.lease_render_target(width.max(1), height.max(1), format, None) {
            self.scene_rt = rt;
            Self::record_scheme(
                &mut scheme,
                &self.compute_pipeline,
                &self.render_pipeline,
                &self.star_buffer,
                &self.params_buffer,
                &self.scene_rt,
            );
            if let Ok((present, withdraw)) = Self::bind_frame(
                &mut scheme,
                &self.scene_rt,
                self.surface.as_ref(),
                self.readback.as_ref(),
            ) {
                self.present = present;
                self.withdraw = withdraw;
                self.scheme = scheme;
            }
        }
    }

    fn new(window: Option<Arc<Window>>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let mut retained_pool = RetainedPool::new(device.clone());

        let (surface, capture, readback, format, width, height) = if let Some(window) = window.as_deref() {
            let surface = SurfaceExchange::new(&ctx, window, SurfaceConfig::default())?;
            let format = surface.format();
            let (width, height) = surface.size();
            (Some(surface), None, None, format, width, height)
        } else {
            let capture = CaptureDump::from_env()?;
            let (width, height) = capture.size();
            let readback = common::capture_readback(&mut retained_pool, width, height)?;
            (
                None,
                Some(capture),
                Some(readback),
                CaptureDump::format(),
                width,
                height,
            )
        };

        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/starfield_update.slang"))?;
        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/starfield_render.slang"))?;

        let mut stars = Vec::with_capacity(NUM_STARS as usize);
        for _ in 0..NUM_STARS {
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
                x: (rand_f32() - 0.5) * 0.8,
                y: (rand_f32() - 0.5) * 0.8,
                z: 0.5 + rand_f32() * 0.5,
                star_type,
            });
        }

        let star_buffer = retained_pool.acquire_buffer_with_data(&stars, BufferKind::Scattered)?;
        let params_buffer =
            retained_pool.acquire_buffer_sized::<StarfieldParams>(1, BufferKind::Broadcast, BufferFlags::empty())?;

        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;
        let render_pipeline = Self::create_render_pipeline(&device, &render_shader, format)?;

        let mut scheme = Scheme::new(&ctx);
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), format, None)?;
        Self::record_scheme(
            &mut scheme,
            &compute_pipeline,
            &render_pipeline,
            &star_buffer,
            &params_buffer,
            &scene_rt,
        );
        let (present, withdraw) = Self::bind_frame(&mut scheme, &scene_rt, surface.as_ref(), readback.as_ref())?;

        let mut upload_scheme = Scheme::new(&ctx);
        let params_deposit = MemoryExchange::new(&ctx).bind_deposit_buffer(
            &mut upload_scheme,
            &params_buffer,
            std::mem::size_of::<StarfieldParams>() as u64,
        )?;

        println!("Created starfield with {NUM_STARS} stars (Scheme + Present)");

        Ok(Self {
            window,
            device,
            ctx,
            surface,
            capture,
            readback,
            present,
            withdraw,
            scheme,
            scene_rt,
            compute_pipeline,
            render_shader,
            render_pipeline,
            _retained_pool: retained_pool,
            star_buffer,
            params_buffer,
            upload_scheme,
            params_deposit,
            speed: 0.01,
            frame_count: 0.0,
            start_time: std::time::Instant::now(),
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1.0;

        let params = StarfieldParams {
            speed: self.speed,
            frame: self.frame_count,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        self.params_deposit
            .write(&mut self.upload_scheme, 0, bytemuck::bytes_of(&params))?;
        self.upload_scheme.submit()?;

        let mut submission = self.scheme.submit()?;
        if let Some(present) = &self.present {
            present.claim(&mut submission)?.consume()?;
        } else {
            let pixels = self.withdraw.as_ref().unwrap().claim(&mut submission)?.consume()?;
            self.capture.as_mut().unwrap().write_rgba(&pixels)?;
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
        Ok(())
    }

    fn change_speed(&mut self, delta: f32) {
        self.speed = (self.speed + delta).clamp(0.001, 0.1);
        if let Some(window) = &self.window {
            window.set_title(&format!("Goldy - Starfield (speed: {:.1})", self.speed));
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
                    .create_window(common::hidden_window("Goldy - Starfield", 1024, 768))
                    .expect("Failed to create window"),
            );

            match RenderState::new(Some(window.clone())) {
                Ok(mut state) => {
                    if let Err(e) = state.render() {
                        tracing::error!("First frame error: {e}");
                    }
                    common::reveal_window(&window);
                    self.state = Some(state);
                    window.request_redraw();
                }
                Err(e) => {
                    tracing::error!("Failed to create render state: {e}");
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
                        Key::Named(NamedKey::ArrowUp) => state.change_speed(0.005),
                        Key::Named(NamedKey::ArrowDown) => state.change_speed(-0.005),
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    if size.width > 0 && size.height > 0 {
                        let Some(surface) = state.surface.as_ref() else {
                            return;
                        };
                        let (prev_w, prev_h) = surface.size();
                        if size.width == prev_w && size.height == prev_h {
                            return;
                        }
                        let _ = surface.resize(size.width, size.height);
                        let format = surface.format();
                        if let Ok(pipeline) =
                            RenderState::create_render_pipeline(&state.device, &state.render_shader, format)
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
