//! Conway's Game of Life — two retained schemes, alternating resubmit (ffi-client).
//!
//! Ping-pong cell grids live in one retained record buffer (fields `"a"` / `"b"`).
//! Orientation AB reads `a` and writes `b`; BA is the swap. Each is recorded once
//! (and again on resize). Simulation steps alternate which scheme submits. Idle
//! redraws skip submit and leave the last present on the surface.
//!
//! Run from `goldy/ffi-client`: `cargo run --example game_of_life`

use goldy_ffi_client::{
    Buffer, ComputePipeline, Context, DepthFormat, DeviceDescriptor, Instance, NodeAccess, PrimitiveTopology,
    RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, Scheme, SchemeRenderTargetLease,
    ShaderModule, SurfaceExchange, TargetLoad, Transaction,
};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const GRID_WIDTH: u32 = 128;
const GRID_HEIGHT: u32 = 128;
const CELL_COUNT: usize = (GRID_WIDTH * GRID_HEIGHT) as usize;

const COMPUTE_SHADER: &str = include_str!("../../shaders/game_of_life.slang");
const RENDER_SHADER: &str = include_str!("../../shaders/game_of_life_render.slang");

fn field_unit(name: &str) -> u32 {
    match name {
        "a" => 0,
        "b" => 1,
        other => panic!("unknown field {other:?}"),
    }
}

fn surface_from_window(ctx: &Context, window: &Window) -> goldy_ffi_client::Result<SurfaceExchange> {
    let handle = window
        .window_handle()
        .map_err(|e| goldy_ffi_client::GoldyError::from_message(format!("window handle: {e}")))?;
    match handle.as_raw() {
        #[cfg(windows)]
        RawWindowHandle::Win32(h) => SurfaceExchange::from_win32(ctx, h.hwnd.get() as *mut _, 3),
        #[cfg(target_os = "macos")]
        RawWindowHandle::AppKit(h) => SurfaceExchange::from_appkit(ctx, h.ns_view.as_ptr(), 3),
        #[cfg(target_os = "linux")]
        RawWindowHandle::Wayland(h) => SurfaceExchange::from_wayland(ctx, h.display.as_ptr(), h.surface.as_ptr(), 3),
        other => Err(goldy_ffi_client::GoldyError::from_message(format!(
            "unsupported window handle for surface exchange: {other:?}"
        ))),
    }
}

fn demo_frame_limit() -> Option<u32> {
    std::env::var("GOLDY_DEMO_FRAMES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .map(|n| n.max(1))
}

fn create_initial_state() -> Vec<u32> {
    let mut cells = vec![0u32; CELL_COUNT];

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

fn record_scheme(
    scheme: &mut Scheme,
    surface: &SurfaceExchange,
    cells: &Buffer,
    read_field: &str,
    write_field: &str,
    compute_pipeline: &ComputePipeline,
    render_pipeline: &RenderPipeline,
    scene_rt: &SchemeRenderTargetLease,
) -> goldy_ffi_client::Result<Transaction> {
    let read = cells.field(field_unit(read_field))?;
    let write = cells.field(field_unit(write_field))?;
    {
        let mut node = scheme.compute_node("game_of_life", compute_pipeline);
        node.with_parcel(&read, NodeAccess::Read);
        node.with_parcel(&write, NodeAccess::Overwrite);
        node.dispatch(GRID_WIDTH.div_ceil(8), GRID_HEIGHT.div_ceil(8), 1);
    }
    {
        let mut pass = scheme.render_pass("game_of_life_render", scene_rt, TargetLoad::Discard);
        pass.with_parcel(&write, NodeAccess::Read);
        pass.set_pipeline(render_pipeline);
        pass.draw_fullscreen();
        pass.finish_recorded();
    }
    surface.bind_render_target(scheme, scene_rt)
}

fn build_scheme(
    ctx: &Context,
    surface: &SurfaceExchange,
    cells: &Buffer,
    read_field: &str,
    write_field: &str,
    compute_pipeline: &ComputePipeline,
    render_pipeline: &RenderPipeline,
) -> goldy_ffi_client::Result<(Scheme, Transaction)> {
    let mut scheme = Scheme::new(ctx)?;
    let (width, height) = surface.size();
    let scene_rt = ctx.lease_render_target(width.max(1), height.max(1), surface.format(), None::<DepthFormat>)?;
    let present = record_scheme(
        &mut scheme,
        surface,
        cells,
        read_field,
        write_field,
        compute_pipeline,
        render_pipeline,
        &scene_rt,
    )?;
    Ok((scheme, present))
}

fn build_schemes(
    ctx: &Context,
    surface: &SurfaceExchange,
    cells: &Buffer,
    compute_pipeline: &ComputePipeline,
    render_pipeline: &RenderPipeline,
) -> goldy_ffi_client::Result<((Scheme, Transaction), (Scheme, Transaction))> {
    Ok((
        build_scheme(ctx, surface, cells, "a", "b", compute_pipeline, render_pipeline)?,
        build_scheme(ctx, surface, cells, "b", "a", compute_pipeline, render_pipeline)?,
    ))
}

struct RenderState {
    window: Arc<Window>,
    ctx: Context,
    surface: SurfaceExchange,
    scheme_ab: Scheme,
    scheme_ba: Scheme,
    present_ab: Transaction,
    present_ba: Transaction,
    compute_pipeline: ComputePipeline,
    render_pipeline: RenderPipeline,
    cells: Buffer,
    use_buffer_a: bool,
    frame_count: u32,
    last_update: std::time::Instant,
    start_time: std::time::Instant,
}

impl RenderState {
    fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let instance = Instance::new()?;
        let device = instance
            .request_adapter(&RequestAdapterOptions::default())?
            .request_device(&DeviceDescriptor::default())?;
        let ctx = Context::new(&device)?;
        let surface = surface_from_window(&ctx, window.as_ref())?;

        let initial = create_initial_state();
        let cells = device.acquire_record_pod(&[("a", &initial), ("b", &initial)])?;

        let compute_shader = ShaderModule::from_slang(&device, COMPUTE_SHADER)?;
        let render_shader = ShaderModule::from_slang(&device, RENDER_SHADER)?;
        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;
        let render_pipeline = RenderPipeline::new(
            &device,
            &render_shader,
            &render_shader,
            &RenderPipelineDesc {
                topology: PrimitiveTopology::TriangleList,
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        let ((scheme_ab, present_ab), (scheme_ba, present_ba)) =
            build_schemes(&ctx, &surface, &cells, &compute_pipeline, &render_pipeline)?;

        println!("Game of Life initialized: {GRID_WIDTH}x{GRID_HEIGHT} grid (ffi-client / Scheme)");
        println!("Features Gosper Glider Gun + random cells");
        println!("Press Escape or close window to exit");

        Ok(Self {
            window,
            ctx,
            surface,
            scheme_ab,
            scheme_ba,
            present_ab,
            present_ba,
            compute_pipeline,
            render_pipeline,
            cells,
            use_buffer_a: true,
            frame_count: 0,
            last_update: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
        })
    }

    fn record_orientations(&mut self) -> goldy_ffi_client::Result<()> {
        let (ab, ba) = build_schemes(
            &self.ctx,
            &self.surface,
            &self.cells,
            &self.compute_pipeline,
            &self.render_pipeline,
        )?;
        (self.scheme_ab, self.present_ab) = ab;
        (self.scheme_ba, self.present_ba) = ba;
        Ok(())
    }

    fn step(&mut self) -> goldy_ffi_client::Result<()> {
        let (scheme, present) = if self.use_buffer_a {
            (&mut self.scheme_ab, &self.present_ab)
        } else {
            (&mut self.scheme_ba, &self.present_ba)
        };
        let mut submission = scheme.submit()?;
        present.claim(&mut submission)?.consume()?;
        self.use_buffer_a = !self.use_buffer_a;
        Ok(())
    }

    fn render(&mut self) -> goldy_ffi_client::Result<()> {
        let now = std::time::Instant::now();
        let should_step = self.frame_count == 0 || now.duration_since(self.last_update).as_millis() > 33;

        if should_step {
            self.last_update = now;
            self.step()?;
            self.frame_count += 1;
        }

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

#[derive(Default)]
struct App {
    state: Option<RenderState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Game of Life (ffi-client / Scheme)")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 800)),
                    )
                    .expect("Failed to create window"),
            );

            match RenderState::new(window.clone()) {
                Ok(state) => {
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
                        if let Err(e) = state.surface.resize(size.width, size.height) {
                            tracing::error!("Failed to resize surface exchange: {e}");
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
                        tracing::error!("Render error: {e}");
                    }
                    if demo_frame_limit().is_some_and(|n| state.frame_count >= n) {
                        event_loop.exit();
                        return;
                    }
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

    println!("Goldy Game of Life (ffi-client / Scheme + SurfaceExchange)");
    println!("========================================\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::default())?;
    Ok(())
}
