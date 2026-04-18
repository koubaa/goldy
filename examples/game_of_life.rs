//! Conway's Game of Life - Compute + Graphics Example
//!
//! This example demonstrates:
//! 1. Compute shader running cellular automaton rules
//! 2. Graphics shader rendering the grid
//! 3. Ping-pong buffer technique for in-place updates
//!
//! Run with: `cargo run --example game_of_life`

use anyhow::Result;
use goldy::{
    BufferPool, BufferView, Color, CommandEncoder, ComputeEncoder, ComputePipeline, DeviceType,
    Instance, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, ShaderModule, Surface,
    VertexBufferLayout,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const GRID_WIDTH: u32 = 128;
const GRID_HEIGHT: u32 = 128;
const CELL_COUNT: u32 = GRID_WIDTH * GRID_HEIGHT;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

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
    // Compute resources
    compute_pipeline: ComputePipeline,
    // Graphics resources
    render_pipeline: RenderPipeline,
    // Single pool holding both ping-pong buffers as views.
    // One GPU allocation instead of two.
    _pool: BufferPool,
    view_a: BufferView,
    view_b: BufferView,
    // State: true = A is current (read from A, write to B)
    use_buffer_a: bool,
    frame_count: u32,
    last_update: std::time::Instant,
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
    fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::new()?;

        let device = Arc::new(instance.create_device(DeviceType::DiscreteGpu)?);
        let surface = Surface::new(&device, window.as_ref())?;

        // Load shaders
        let compute_shader =
            ShaderModule::from_slang(&device, include_str!("../shaders/game_of_life.slang"))?;

        let render_shader = ShaderModule::from_slang(
            &device,
            include_str!("../shaders/game_of_life_render.slang"),
        )?;

        // Allocate both ping-pong buffers from a single pool (one GPU allocation).
        let initial_state = create_initial_state();
        let cell_bytes = (CELL_COUNT as usize) * std::mem::size_of::<u32>();
        let mut pool = BufferPool::new(&device, (cell_bytes * 2 + 512) as u64)?;

        let view_a = pool.alloc::<u32>(CELL_COUNT as u64)?;
        let view_b = pool.alloc::<u32>(CELL_COUNT as u64)?;

        // Write initial state into both views via the backing buffer.
        pool.backing_buffer().write_data(0, &initial_state)?;
        pool.backing_buffer()
            .write_data((cell_bytes as u64 + 255) & !255, &initial_state)?;

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

        println!(
            "Game of Life initialized: {}x{} grid",
            GRID_WIDTH, GRID_HEIGHT
        );
        println!("Features Gosper Glider Gun + random cells");
        println!("Press Escape or close window to exit");

        Ok(Self {
            window,
            device,
            surface,
            compute_pipeline,
            render_pipeline,
            _pool: pool,
            view_a,
            view_b,
            use_buffer_a: true,
            frame_count: 0,
            last_update: std::time::Instant::now(),
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;

        // Update simulation ~30 times per second
        let now = std::time::Instant::now();
        let should_update = now.duration_since(self.last_update).as_millis() > 33;

        if should_update {
            self.last_update = now;

            // Run compute pass with ping-pong buffers
            let mut compute_encoder = ComputeEncoder::new();
            {
                let mut pass = compute_encoder.begin_compute_pass();
                pass.set_pipeline(&self.compute_pipeline);

                // Pass buffer indices via push constants.
                // Order matters: [current_state, next_state] matching shader slots.
                let (read_handle, write_handle) = if self.use_buffer_a {
                    // A -> B: read from A, write to B
                    (
                        self.view_a.bindless_handle().unwrap(),
                        self.view_b.bindless_handle().unwrap(),
                    )
                } else {
                    // B -> A: read from B, write to A
                    (
                        self.view_b.bindless_handle().unwrap(),
                        self.view_a.bindless_handle().unwrap(),
                    )
                };
                pass.set_push_constants_typed(&[read_handle, write_handle]);

                // Dispatch workgroups (8x8 threads per group)
                let workgroups_x = GRID_WIDTH.div_ceil(8);
                let workgroups_y = GRID_HEIGHT.div_ceil(8);
                pass.dispatch(workgroups_x, workgroups_y, 1);
            }

            compute_encoder.dispatch(&self.device)?;

            // Toggle buffer for next frame
            self.use_buffer_a = !self.use_buffer_a;
        }

        // Render
        let frame = self.surface.acquire()?;

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.set_pipeline(&self.render_pipeline);

            // Read from the view that is now "current" (after the ping-pong swap).
            let current_handle = if self.use_buffer_a {
                self.view_a.bindless_handle().unwrap()
            } else {
                self.view_b.bindless_handle().unwrap()
            };
            pass.set_push_constants_typed(&[current_handle]);

            // Draw fullscreen triangle
            pass.draw(0..3, 0..1);
        }

        frame.render(encoder)?;
        frame.present()?;

        self.window.request_redraw();

        Ok(())
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
                    state.window.request_redraw(); // Continue render loop
                }
            }
            _ => {}
        }
    }
}
