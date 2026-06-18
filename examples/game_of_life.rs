//! Conway's Game of Life — hybrid Scheme (compute + render + present).
//!
//! Ping-pong cell grids live in one retained record buffer (fields `"a"` / `"b"`).
//! Each simulation step runs an ephemeral compute scheme; the display scheme is
//! rebuilt when the active field flips.
//!
//! Run with: `cargo run --example game_of_life`

use anyhow::Result;
use goldy::{
    field, Buffer, Color, ComputePipeline, Context, DeviceDescriptor, Grant, Init, Instance, Lease, LeaseRenderTarget,
    NodeAccess, PresentGrant, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions,
    ResourceAccess, RetainedPool, Scheme, ShaderModule, ShaderResourceSlot, SwapchainPool, VertexBufferLayout,
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

const GRID_WIDTH: u32 = 128;
const GRID_HEIGHT: u32 = 128;
const CELL_COUNT: u32 = GRID_WIDTH * GRID_HEIGHT;

fn run_compute_step(
    ctx: &Context,
    cells: &Buffer,
    read_field: &str,
    write_field: &str,
    pipeline: &ComputePipeline,
) -> Result<()> {
    let mut scheme = Scheme::new(ctx);
    scheme
        .node("game_of_life", pipeline)
        .with_parcel(&cells[read_field], NodeAccess::Read)
        .with_parcel(&cells[write_field], NodeAccess::Write)
        .with_views(&[
            cells[read_field]
                .handle(ResourceAccess::ReadWrite)
                .expect("read field UAV"),
            cells[write_field]
                .handle(ResourceAccess::Write)
                .expect("write field UAV"),
        ])
        .dispatch(GRID_WIDTH.div_ceil(8), GRID_HEIGHT.div_ceil(8), 1);
    scheme.submit()?;
    Ok(())
}

fn record_display_scheme(
    scheme: &mut Scheme,
    cells: &Buffer,
    current_field: &str,
    render_pipeline: &RenderPipeline,
    scene_rt: &Lease<LeaseRenderTarget>,
    screen: &goldy::PresentLease,
) -> PresentGrant {
    let current = &cells[current_field];
    let mut pass = scheme.render_pass("game_of_life_render", scene_rt);
    pass.with_parcel(current, NodeAccess::Read);
    pass.clear(Color::BLACK);
    pass.set_pipeline(render_pipeline);
    pass.with_shader_resources(&[ShaderResourceSlot::Parcel {
        parcel: current,
        access: NodeAccess::ReadWrite,
    }]);
    pass.draw(0..3, 0..1);
    pass.finish();
    scheme.copy_to_present(scene_rt, screen);
    scheme.grant_present(screen)
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
    ctx: Context,
    swapchain: SwapchainPool,
    screen: goldy::PresentLease,
    scene_rt: Lease<LeaseRenderTarget>,
    display_scheme: Scheme,
    present: PresentGrant,
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
    fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let swapchain = SwapchainPool::new(&ctx, window.as_ref(), 3)?;
        let screen = swapchain.lease();

        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/game_of_life.slang"))?;
        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/game_of_life_render.slang"))?;

        let initial_state = create_initial_state();
        let mut retained_pool = RetainedPool::new(device.clone());
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
                target_format: swapchain.format(),
                ..Default::default()
            },
        )?;

        let mut display_scheme = Scheme::new(&ctx);
        let (width, height) = swapchain.size();
        let scene_rt = display_scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)?;
        let present = record_display_scheme(&mut display_scheme, &cells, "a", &render_pipeline, &scene_rt, &screen);

        println!("Game of Life initialized: {}x{} grid", GRID_WIDTH, GRID_HEIGHT);
        println!("Features Gosper Glider Gun + random cells");
        println!("Press Escape or close window to exit");

        Ok(Self {
            window,
            ctx,
            swapchain,
            screen,
            scene_rt,
            display_scheme,
            present,
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

    fn rebuild_display_scheme(&mut self) -> Result<()> {
        let current_field = if self.use_buffer_a { "a" } else { "b" };
        let mut display_scheme = Scheme::new(&self.ctx);
        let (width, height) = self.swapchain.size();
        self.scene_rt =
            display_scheme.lease_render_target(width.max(1), height.max(1), self.swapchain.format(), None)?;
        self.present = record_display_scheme(
            &mut display_scheme,
            &self.cells,
            current_field,
            &self.render_pipeline,
            &self.scene_rt,
            &self.screen,
        );
        self.display_scheme = display_scheme;
        Ok(())
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;

        let now = std::time::Instant::now();
        let should_update = now.duration_since(self.last_update).as_millis() > 33;

        if should_update {
            self.last_update = now;

            let (read_field, write_field) = if self.use_buffer_a { ("a", "b") } else { ("b", "a") };
            run_compute_step(&self.ctx, &self.cells, read_field, write_field, &self.compute_pipeline)?;
            self.use_buffer_a = !self.use_buffer_a;
            self.rebuild_display_scheme()?;
        }

        let submission = self.display_scheme.submit()?;
        self.present.consume(&submission)?;

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
                            .with_title("Game of Life")
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
                        if let Err(e) = state.swapchain.resize(size.width, size.height) {
                            tracing::error!("Failed to resize swapchain: {e}");
                            return;
                        }
                        if let Err(e) = state.rebuild_display_scheme() {
                            tracing::error!("Failed to rebuild display scheme: {e}");
                        }
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
