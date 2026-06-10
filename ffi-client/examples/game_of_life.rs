//! Conway's Game of Life — hybrid TaskGraph via goldy-ffi-client.
//!
//! Mirrors `goldy/examples/game_of_life.rs`: compute ping-pong + offscreen render + swapchain blit.
//!
//! Run from `goldy/ffi-client`: `cargo run --example game_of_life`

use goldy_ffi_client::{
    Color, ComputePipeline, DeviceDescriptor, Instance, MosaicSlot, NodeAccess, PrimitiveTopology, RenderPipeline,
    RenderPipelineDesc, RenderTarget, RequestAdapterOptions, ResourceAccess, ResourceCategory, ResourceHandle,
    RetainedPool, ShaderModule, Surface, TaskGraph,
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

const SLOT_A: MosaicSlot = MosaicSlot(0);
const SLOT_B: MosaicSlot = MosaicSlot(1);

const COMPUTE_SHADER: &str = include_str!("../../shaders/game_of_life.slang");
const RENDER_SHADER: &str = include_str!("../../shaders/game_of_life_render.slang");

fn surface_from_window(device: &goldy_ffi_client::Device, window: &Window) -> goldy_ffi_client::Result<Surface> {
    let handle = window
        .window_handle()
        .map_err(|e| goldy_ffi_client::GoldyError::from_message(format!("window handle: {e}")))?;
    unsafe {
        match handle.as_raw() {
            #[cfg(windows)]
            RawWindowHandle::Win32(h) => Surface::from_win32(device, h.hwnd.get() as *mut _),
            #[cfg(target_os = "macos")]
            RawWindowHandle::AppKit(h) => Surface::from_appkit(device, h.ns_view.as_ptr()),
            other => Err(goldy_ffi_client::GoldyError::from_message(format!(
                "unsupported window handle for surface creation: {other:?}"
            ))),
        }
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

struct App {
    instance: Instance,
    device: Option<goldy_ffi_client::Device>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    scene_rt: Option<RenderTarget>,
    frame_graph: TaskGraph,
    compute_pipeline: Option<ComputePipeline>,
    render_pipeline: Option<RenderPipeline>,
    _retained_pool: Option<RetainedPool>,
    cells: Option<goldy_ffi_client::Parcel>,
    use_buffer_a: bool,
    frame_count: u32,
    last_update: std::time::Instant,
    start_time: std::time::Instant,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None,
            window: None,
            surface: None,
            scene_rt: None,
            frame_graph: TaskGraph::new(),
            compute_pipeline: None,
            render_pipeline: None,
            _retained_pool: None,
            cells: None,
            use_buffer_a: true,
            frame_count: 0,
            last_update: std::time::Instant::now(),
            start_time: std::time::Instant::now(),
        })
    }

    fn create_scene_rt(device: &goldy_ffi_client::Device, surface: &Surface) -> anyhow::Result<RenderTarget> {
        let (width, height) = surface.size();
        RenderTarget::new(device, width.max(1), height.max(1), surface.format()).map_err(Into::into)
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = self
            .instance
            .request_adapter(&RequestAdapterOptions::default())?
            .request_device(&DeviceDescriptor::default())?;

        let surface = surface_from_window(&device, window.as_ref())?;
        let scene_rt = Self::create_scene_rt(&device, &surface)?;

        let initial = create_initial_state();
        let mut retained_pool = RetainedPool::new(&device)?;
        let mut mosaic = retained_pool.mosaic()?;
        mosaic.emplace_pod::<u32>(&initial)?;
        mosaic.emplace_pod::<u32>(&initial)?;
        let cells = mosaic.build(&mut retained_pool)?;

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

        println!("Game of Life initialized: {GRID_WIDTH}x{GRID_HEIGHT} grid (ffi-client)");
        println!("Press Escape or close window to exit");

        self.device = Some(device);
        self.surface = Some(surface);
        self.scene_rt = Some(scene_rt);
        self.compute_pipeline = Some(compute_pipeline);
        self.render_pipeline = Some(render_pipeline);
        self._retained_pool = Some(retained_pool);
        self.cells = Some(cells);
        self.use_buffer_a = true;

        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        self.frame_count += 1;

        let now = std::time::Instant::now();
        let should_update = now.duration_since(self.last_update).as_millis() > 33;

        let compute_pipeline = self.compute_pipeline.as_ref().unwrap();
        let render_pipeline = self.render_pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let scene_rt = self.scene_rt.as_ref().unwrap();
        let cells = self.cells.as_ref().unwrap();

        self.frame_graph.clear();

        if should_update {
            self.last_update = now;

            let (read_slot, write_slot) = if self.use_buffer_a {
                (SLOT_A, SLOT_B)
            } else {
                (SLOT_B, SLOT_A)
            };

            let read_idx = cells.mosaic_view_resource_index(read_slot, ResourceAccess::ReadWrite)?;
            let write_idx = cells.mosaic_view_resource_index(write_slot, ResourceAccess::Write)?;

            let mut node = self.frame_graph.compute_node("game_of_life", compute_pipeline);
            node.bind_parcel_view(cells, read_slot, NodeAccess::Read);
            node.bind_parcel_view(cells, write_slot, NodeAccess::Write);
            node.bind_resources_raw(&[read_idx, write_idx]);
            node.dispatch(GRID_WIDTH.div_ceil(8), GRID_HEIGHT.div_ceil(8), 1);

            self.use_buffer_a = !self.use_buffer_a;
        }

        let current_slot = if self.use_buffer_a { SLOT_A } else { SLOT_B };
        let render_idx = cells.mosaic_view_resource_index(current_slot, ResourceAccess::ReadWrite)?;

        let mut pass = self.frame_graph.render_pass("game_of_life_render", scene_rt);
        pass.bind_parcel_view_mut(cells, current_slot, NodeAccess::Read);
        pass.clear(Color::BLACK);
        pass.set_pipeline(render_pipeline);
        pass.bind_resources_typed(&[ResourceHandle {
            category: ResourceCategory::Scattered,
            index: render_idx,
        }]);
        pass.draw_fullscreen();
        pass.finish_recorded();

        let swapchain = self.frame_graph.declare_swapchain_output();
        self.frame_graph.copy_render_target_to_swapchain(scene_rt, swapchain);

        let frame = surface.begin()?;
        let frame = surface.submit_graph_to_frame(&mut self.frame_graph, frame)?;
        frame.present()?;

        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &mut self.surface {
                if let Err(e) = surface.resize(new_size.width, new_size.height) {
                    tracing::error!("Failed to resize surface: {e}");
                }
            }
            if let (Some(device), Some(surface)) = (&self.device, &self.surface) {
                match Self::create_scene_rt(device, surface) {
                    Ok(rt) => self.scene_rt = Some(rt),
                    Err(e) => tracing::error!("Failed to resize scene render target: {e:#}"),
                }
            }
        }
    }
}

impl Drop for App {
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
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Game of Life (ffi-client)")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 800)),
                    )
                    .expect("Failed to create window"),
            );

            match self.init_gpu(&window) {
                Ok(()) => {
                    self.window = Some(window.clone());
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
            WindowEvent::CloseRequested => {
                self.window = None;
                self.surface = None;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    self.window = None;
                    self.surface = None;
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                self.handle_resize(size);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if self.surface.is_none() {
                    return;
                }
                if let Err(e) = self.render_frame() {
                    tracing::error!("Render error: {e}");
                }
                if demo_frame_limit().is_some_and(|n| self.frame_count >= n) {
                    self.window = None;
                    self.surface = None;
                    event_loop.exit();
                    return;
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
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

    println!("Goldy Game of Life (ffi-client TaskGraph)");
    println!("=========================================\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
