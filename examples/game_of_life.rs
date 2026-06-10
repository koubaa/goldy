//! Conway's Game of Life - Compute + Graphics Example
//!
//! Demonstrates a unified task graph:
//! 1. Compute shader running cellular automaton rules (ping-pong buffers)
//! 2. Graphics shader rendering the grid to an offscreen target
//! 3. Swapchain blit and present
//!
//! Ping-pong cell grids live in one retained mosaic parcel (two sub-views, one backing
//! allocation). When the transient pool lands, this example is a candidate to re-migrate.
//!
//! Run with: `cargo run --example game_of_life`

use anyhow::Result;
use goldy::{
    Color, ComputePipeline, DeviceDescriptor, Instance, MosaicSlot, NodeAccess, Parcel, PrimitiveTopology,
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

const GRID_WIDTH: u32 = 128;
const GRID_HEIGHT: u32 = 128;
const CELL_COUNT: u32 = GRID_WIDTH * GRID_HEIGHT;

const SLOT_A: MosaicSlot = MosaicSlot(0);
const SLOT_B: MosaicSlot = MosaicSlot(1);

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
    render_pipeline: RenderPipeline,
    _retained_pool: RetainedPool,
    /// Mosaic parcel: slot A and slot B are ping-pong cell grids in one backing buffer.
    cells: Parcel,
    // State: true = A is current (read from A, write to B)
    use_buffer_a: bool,
    frame_count: u32,
    last_update: std::time::Instant,
    start_time: std::time::Instant,
}

/// Create initial pattern (glider gun + some random cells)
fn create_initial_state() -> Vec<u32> {
    let mut cells = vec![0u32; CELL_COUNT as usize];

    // Gosper Glider Gun (creates infinite gliders)
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

    // Place glider gun
    let offset_x = 10;
    let offset_y = 10;
    for (x, y) in gun.iter() {
        let px = (x + offset_x) as u32;
        let py = (y + offset_y) as u32;
        if px < GRID_WIDTH && py < GRID_HEIGHT {
            cells[(py * GRID_WIDTH + px) as usize] = 1;
        }
    }

    // Add some random cells in the lower right
    let seed = 42u64;
    let mut rng = seed;
    for y in 60..100 {
        for x in 60..100 {
            // Simple LCG for deterministic randomness
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32).is_multiple_of(4) {
                cells[(y * GRID_WIDTH + x) as usize] = 1;
            }
        }
    }

    cells
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

        // Load shaders
        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/game_of_life.slang"))?;

        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/game_of_life_render.slang"))?;

        // One retained mosaic: two ping-pong views in a single backing allocation.
        let initial_state = create_initial_state();
        let mut retained_pool = RetainedPool::new(device.clone());
        let mut mosaic = retained_pool.mosaic();
        mosaic.emplace::<u32>(&initial_state);
        mosaic.emplace::<u32>(&initial_state);
        let cells = mosaic.build()?;

        // Create compute pipeline
        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;

        // Create render pipeline
        let render_pipeline = RenderPipeline::new(
            &device,
            &render_shader,
            &render_shader,
            &RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::default(),
                topology: PrimitiveTopology::TriangleList,
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        println!("Game of Life initialized: {}x{} grid", GRID_WIDTH, GRID_HEIGHT);
        println!("Features Gosper Glider Gun + random cells");
        println!("Press Escape or close window to exit");

        Ok(Self {
            window,
            device,
            surface,
            scene_rt,
            frame_graph: TaskGraph::new(),
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

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;

        // Update simulation ~30 times per second
        let now = std::time::Instant::now();
        let should_update = now.duration_since(self.last_update).as_millis() > 33;

        self.frame_graph.clear();

        if should_update {
            self.last_update = now;

            let (read_slot, write_slot) = if self.use_buffer_a {
                (SLOT_A, SLOT_B)
            } else {
                (SLOT_B, SLOT_A)
            };
            let read_view = self.cells.view(read_slot);
            let write_view = self.cells.view(write_slot);

            self.frame_graph
                .node("game_of_life", &self.compute_pipeline)
                .bind_buffer_view(read_view, NodeAccess::Read)
                .bind_buffer_view(write_view, NodeAccess::Write)
                .bind_resources_raw_slice(&[
                    read_view.resource_index(ResourceAccess::ReadWrite).unwrap(),
                    write_view.resource_index(ResourceAccess::Write).unwrap(),
                ])
                .dispatch(GRID_WIDTH.div_ceil(8), GRID_HEIGHT.div_ceil(8), 1);

            self.use_buffer_a = !self.use_buffer_a;
        }

        let current_slot = if self.use_buffer_a { SLOT_A } else { SLOT_B };
        let current_view = self.cells.view(current_slot);

        let mut pass = self.frame_graph.render_pass("game_of_life_render", &self.scene_rt);
        pass.bind_buffer_view_mut(current_view, NodeAccess::Read);
        pass.clear(Color::BLACK);
        pass.set_pipeline(&self.render_pipeline);
        pass.bind_resources_raw(&[current_view.resource_index(ResourceAccess::ReadWrite).unwrap()]);
        pass.draw(0..3, 0..1);
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
                            .with_title("Game of Life")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 800)),
                    )
                    .expect("Failed to create window"),
            );

            match RenderState::new(window.clone()) {
                Ok(state) => {
                    self.state = Some(state);
                    window.request_redraw(); // Trigger initial render
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
                    state.window.request_redraw(); // Continue render loop
                }
            }
            _ => {}
        }
    }
}
