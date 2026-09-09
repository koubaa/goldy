//! Conway's Game of Life — two retained schemes, alternating resubmit.
//!
//! Ping-pong cell grids live in one retained record buffer (fields `"a"` / `"b"`).
//! Orientation AB reads `a` and writes `b`; BA is the swap. Each is recorded once
//! (and again on resize). Simulation steps alternate which scheme submits. Idle
//! redraws skip submit and leave the last present on the surface.
//!
//! Run with: `cargo run --example game_of_life`

use anyhow::Result;
use goldy::{
    field, Buffer, ComputePipeline, Context, DeviceDescriptor, Init, Instance, Lease, LeaseRenderTarget,
    MemoryExchange, NodeAccess, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions,
    RetainedPool, Scheme, ShaderModule, Submission, SurfaceConfig, SurfaceExchange, TargetLoad, Texture, TextureFormat,
    Transaction, VertexBufferLayout, WithdrawTransaction,
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

const GRID_WIDTH: u32 = 128;
const GRID_HEIGHT: u32 = 128;
const CELL_COUNT: u32 = GRID_WIDTH * GRID_HEIGHT;

fn record_scheme(
    scheme: &mut Scheme,
    cells: &Buffer,
    read_field: &str,
    write_field: &str,
    compute_pipeline: &ComputePipeline,
    render_pipeline: &RenderPipeline,
    scene_rt: &Lease<LeaseRenderTarget>,
) {
    scheme
        .node("game_of_life", compute_pipeline)
        .with_parcel(&cells[read_field], NodeAccess::Read)
        .with_parcel(&cells[write_field], NodeAccess::Overwrite)
        .dispatch(GRID_WIDTH.div_ceil(8), GRID_HEIGHT.div_ceil(8), 1);

    let mut pass = scheme.render_pass("game_of_life_render", scene_rt, TargetLoad::Discard);
    pass.with_parcel(&cells[write_field], NodeAccess::Read);
    pass.set_pipeline(render_pipeline);
    pass.draw(0..3, 0..1);
    pass.finish();
}

struct FrameBind<'a> {
    format: TextureFormat,
    width: u32,
    height: u32,
    surface: Option<&'a SurfaceExchange>,
    readback: Option<&'a Texture>,
}

struct Recorded {
    scheme: Scheme,
    present: Option<Transaction>,
    withdraw: Option<WithdrawTransaction>,
}

fn bind_frame(
    scheme: &mut Scheme,
    scene_rt: &Lease<LeaseRenderTarget>,
    bind: &FrameBind<'_>,
) -> anyhow::Result<(Option<Transaction>, Option<WithdrawTransaction>)> {
    if let Some(surface) = bind.surface {
        let present = surface.bind_render_target(scheme, scene_rt)?;
        Ok((Some(present), None))
    } else {
        let readback = bind.readback.expect("capture readback");
        scheme.copy_to_texture(scene_rt, readback)?;
        let withdraw = MemoryExchange::new(scheme.context()).bind_withdraw(scheme, readback)?;
        Ok((None, Some(withdraw)))
    }
}

fn build_scheme(
    ctx: &Context,
    cells: &Buffer,
    read_field: &str,
    write_field: &str,
    compute_pipeline: &ComputePipeline,
    render_pipeline: &RenderPipeline,
    bind: &FrameBind<'_>,
) -> anyhow::Result<Recorded> {
    let mut scheme = Scheme::new(ctx);
    let scene_rt = ctx.lease_render_target(bind.width.max(1), bind.height.max(1), bind.format, None)?;
    record_scheme(
        &mut scheme,
        cells,
        read_field,
        write_field,
        compute_pipeline,
        render_pipeline,
        &scene_rt,
    );
    let (present, withdraw) = bind_frame(&mut scheme, &scene_rt, bind)?;
    Ok(Recorded {
        scheme,
        present,
        withdraw,
    })
}

fn build_schemes(
    ctx: &Context,
    cells: &Buffer,
    compute_pipeline: &ComputePipeline,
    render_pipeline: &RenderPipeline,
    bind: &FrameBind<'_>,
) -> anyhow::Result<(Recorded, Recorded)> {
    Ok((
        build_scheme(ctx, cells, "a", "b", compute_pipeline, render_pipeline, bind)?,
        build_scheme(ctx, cells, "b", "a", compute_pipeline, render_pipeline, bind)?,
    ))
}

fn create_initial_state() -> Vec<u32> {
    let mut cells = vec![0u32; CELL_COUNT as usize];

    let gun = [
        (1, 5),
        (1, 6),
        (2, 5),
        (2, 6),
        (11, 5),
        (11, 6),
        (11, 7),
        (12, 4),
        (12, 8),
        (13, 3),
        (13, 9),
        (14, 3),
        (14, 9),
        (15, 6),
        (16, 4),
        (16, 8),
        (17, 5),
        (17, 6),
        (17, 7),
        (18, 6),
        (21, 3),
        (21, 4),
        (21, 5),
        (22, 3),
        (22, 4),
        (22, 5),
        (23, 2),
        (23, 6),
        (25, 1),
        (25, 2),
        (25, 6),
        (25, 7),
        (35, 3),
        (35, 4),
        (36, 3),
        (36, 4),
    ];

    let offset_x = 10;
    let offset_y = 10;
    for (x, y) in gun.iter() {
        let px = (x + offset_x) as u32;
        let py = (y + offset_y) as u32;
        if px < GRID_WIDTH && py < GRID_HEIGHT {
            cells[(py * GRID_WIDTH + px) as usize] = 1;
        }
    }

    let mut rng = 42u64;
    for y in 60..100 {
        for x in 60..100 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32).is_multiple_of(4) {
                cells[(y * GRID_WIDTH + x) as usize] = 1;
            }
        }
    }

    cells
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
    ctx: Context,
    surface: Option<SurfaceExchange>,
    capture: Option<CaptureDump>,
    readback: Option<Texture>,
    scheme_ab: Scheme,
    scheme_ba: Scheme,
    present_ab: Option<Transaction>,
    present_ba: Option<Transaction>,
    withdraw_ab: Option<WithdrawTransaction>,
    withdraw_ba: Option<WithdrawTransaction>,
    compute_pipeline: ComputePipeline,
    render_pipeline: RenderPipeline,
    _retained_pool: RetainedPool,
    cells: Buffer,
    use_buffer_a: bool,
    frame_count: u32,
    last_update: std::time::Instant,
    start_time: std::time::Instant,
}

impl RenderState {
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

    fn record_orientations(&mut self) -> Result<()> {
        let (format, width, height) = self.target();
        let bind = FrameBind {
            format,
            width,
            height,
            surface: self.surface.as_ref(),
            readback: self.readback.as_ref(),
        };
        let (ab, ba) = build_schemes(
            &self.ctx,
            &self.cells,
            &self.compute_pipeline,
            &self.render_pipeline,
            &bind,
        )?;
        self.scheme_ab = ab.scheme;
        self.present_ab = ab.present;
        self.withdraw_ab = ab.withdraw;
        self.scheme_ba = ba.scheme;
        self.present_ba = ba.present;
        self.withdraw_ba = ba.withdraw;
        Ok(())
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

        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/game_of_life.slang"))?;
        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/game_of_life_render.slang"))?;

        let initial_state = create_initial_state();
        let cells = retained_pool.acquire_record([
            field("a", Init::data(&initial_state)),
            field("b", Init::data(&initial_state)),
        ])?;

        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;
        let render_pipeline = RenderPipeline::new(
            &device,
            &render_shader,
            &render_shader,
            &RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::default(),
                topology: PrimitiveTopology::TriangleList,
                target_format: format,
                ..Default::default()
            },
        )?;

        let bind = FrameBind {
            format,
            width,
            height,
            surface: surface.as_ref(),
            readback: readback.as_ref(),
        };
        let (ab, ba) = build_schemes(&ctx, &cells, &compute_pipeline, &render_pipeline, &bind)?;

        println!("Game of Life initialized: {}x{} grid", GRID_WIDTH, GRID_HEIGHT);
        println!("Features Gosper Glider Gun + random cells");
        println!("Press Escape or close window to exit");

        Ok(Self {
            window,
            ctx,
            surface,
            capture,
            readback,
            scheme_ab: ab.scheme,
            scheme_ba: ba.scheme,
            present_ab: ab.present,
            present_ba: ba.present,
            withdraw_ab: ab.withdraw,
            withdraw_ba: ba.withdraw,
            compute_pipeline,
            render_pipeline,
            _retained_pool: retained_pool,
            cells,
            use_buffer_a: true,
            frame_count: 0,
            last_update: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
        })
    }

    fn settle(
        present: Option<&Transaction>,
        withdraw: Option<&WithdrawTransaction>,
        capture: Option<&mut CaptureDump>,
        submission: &mut Submission,
    ) -> Result<()> {
        if let Some(present) = present {
            present.claim(submission)?.consume()?;
        } else {
            let pixels = withdraw.expect("capture withdraw").claim(submission)?.consume()?;
            capture.expect("capture dump").write_rgba(&pixels)?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<()> {
        if self.use_buffer_a {
            let mut submission = self.scheme_ab.submit()?;
            Self::settle(
                self.present_ab.as_ref(),
                self.withdraw_ab.as_ref(),
                self.capture.as_mut(),
                &mut submission,
            )?;
        } else {
            let mut submission = self.scheme_ba.submit()?;
            Self::settle(
                self.present_ba.as_ref(),
                self.withdraw_ba.as_ref(),
                self.capture.as_mut(),
                &mut submission,
            )?;
        }
        self.use_buffer_a = !self.use_buffer_a;
        Ok(())
    }

    fn render(&mut self) -> Result<()> {
        let now = std::time::Instant::now();
        let should_step =
            self.capture.is_some() || self.frame_count == 0 || now.duration_since(self.last_update).as_millis() > 33;

        if should_step {
            self.last_update = now;
            self.step()?;
            self.frame_count += 1;
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
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
                    .create_window(common::hidden_window("Game of Life", 800, 800))
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
                    tracing::error!("Failed to create render state: {:#}", e);
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
                        let Some(surface) = state.surface.as_ref() else {
                            return;
                        };
                        if let Err(e) = surface.resize(size.width, size.height) {
                            tracing::error!("Failed to resize surface: {e}");
                            return;
                        }
                        if let Err(e) = state.record_orientations() {
                            tracing::error!("Failed to rerecord schemes: {e}");
                            return;
                        }
                        state.last_update = std::time::Instant::now()
                            .checked_sub(std::time::Duration::from_millis(34))
                            .unwrap_or(state.last_update);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    if let Err(e) = state.render() {
                        tracing::error!("Render error: {:#}", e);
                    }
                }
            }
            _ => {}
        }
    }
}
