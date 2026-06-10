//! Digital Clock example - render an animated 7-segment clock in a window.
//!
//! This example uses shared rendering code from `goldy::examples::digital_clock`,
//! demonstrating that the same logic can be used on both native and web platforms.
//! Now uses the Surface API for zero-copy presentation.
//!
//! Run with: cargo run --example digital_clock

use goldy::{
    examples::digital_clock::{generate_clock_vertices, ClockState, ClockVertex, TimeData, SHADER_SOURCE},
    BufferFlags, BufferKind, DeviceDescriptor, Instance, NodeAccess, Parcel, RenderPipeline, RenderPipelineDesc,
    RenderTarget, RequestAdapterOptions, RetainedPool, ShaderModule, Surface, TaskGraph,
};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};
mod common;

/// Upper bound on seven-segment clock vertices (8 glyphs × 7 segments × 6 verts).
const MAX_CLOCK_VERTICES: usize = 384;

struct App {
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    _retained_pool: Option<RetainedPool>,
    vertex_parcel: Option<Parcel>,

    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    scene_rt: Option<RenderTarget>,
    frame_graph: TaskGraph,

    start_time: Instant,
    perf_start: Instant,
    frame_count: u32,
    clock_state: ClockState,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let instance = Instance::new()?;
        Ok(Self {
            instance,
            device: None,
            pipeline: None,
            shader: None,
            window: None,
            surface: None,
            scene_rt: None,
            frame_graph: TaskGraph::new(),
            start_time: Instant::now(),
            perf_start: Instant::now(),
            frame_count: 0,
            clock_state: ClockState::default(),
            _retained_pool: None,
            vertex_parcel: None,
        })
    }

    fn create_scene_rt(device: &goldy::Device, surface: &Surface) -> anyhow::Result<RenderTarget> {
        let (width, height) = surface.size();
        RenderTarget::new(device, width.max(1), height.max(1), surface.format())
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;

        // Create surface first to get the correct format
        let surface = Surface::new(&ctx, window.as_ref())?;

        // Use the SHARED shader source from the examples module
        let shader = ShaderModule::from_slang(&device, SHADER_SOURCE)?;
        let pipeline_desc = RenderPipelineDesc {
            vertex_layout: ClockVertex::layout(), // Use shared vertex type
            target_format: surface.format(),
            ..Default::default()
        };
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &pipeline_desc)?;

        let scene_rt = Self::create_scene_rt(&device, &surface)?;

        let mut retained_pool = RetainedPool::new(device.clone());
        let stride = std::mem::size_of::<ClockVertex>() as u32;
        let vertex_parcel = retained_pool.acquire_buffer(
            (MAX_CLOCK_VERTICES * stride as usize) as u64,
            BufferKind::Scattered,
            Some(stride),
            BufferFlags::empty(),
            None,
        )?;

        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self._retained_pool = Some(retained_pool);
        self.vertex_parcel = Some(vertex_parcel);
        self.surface = Some(surface);
        self.scene_rt = Some(scene_rt);

        Ok(())
    }

    fn elapsed_secs(&self) -> u64 {
        if self.clock_state.paused {
            self.clock_state.accumulated_secs
        } else {
            self.start_time.elapsed().as_secs() + self.clock_state.accumulated_secs
        }
    }

    fn toggle_pause(&mut self) {
        let current = self.elapsed_secs();
        if self.clock_state.paused {
            // Resuming - reset start time
            self.start_time = Instant::now();
        }
        self.clock_state.toggle_pause(current);
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        self.frame_count += 1;

        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        let width = size.width;
        let height = size.height;

        if width == 0 || height == 0 {
            return Ok(());
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let scene_rt = self.scene_rt.as_ref().unwrap();
        let vertex_parcel = self.vertex_parcel.as_ref().unwrap();

        let elapsed = self.elapsed_secs();
        let time = TimeData::from_elapsed_secs(elapsed);
        let color = self.clock_state.color();
        let bg_color = self.clock_state.background_color();

        let vertices = generate_clock_vertices(time, color, width, height);

        self.frame_graph.clear();
        self.frame_graph.write_parcel(
            vertex_parcel,
            0,
            bytemuck::cast_slice(&vertices).to_vec(),
        )?;

        let mut pass = self.frame_graph.render_pass("digital_clock", scene_rt);
        pass.bind_parcel_mut(vertex_parcel, NodeAccess::Read);
        pass.clear(bg_color);
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_parcel);
        pass.draw(0..vertices.len() as u32, 0..1);
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
                let _ = surface.resize(new_size.width, new_size.height);
            }
            if let (Some(device), Some(surface)) = (&self.device, &self.surface) {
                if let Ok(rt) = Self::create_scene_rt(device, surface) {
                    self.scene_rt = Some(rt);
                }
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        let elapsed = self.perf_start.elapsed().as_secs_f64();
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
            let attrs = Window::default_attributes()
                .with_title("Goldy - Clock (Surface API, Space: pause, Click: color)")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.window = Some(window.clone());

            if let Err(e) = self.init_gpu(&window) {
                tracing::error!("Failed to initialize GPU: {}", e);
            }
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        common::exit_if_timed_out(event_loop, self.perf_start);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => match event.logical_key {
                Key::Named(NamedKey::Escape) => event_loop.exit(),
                Key::Named(NamedKey::Space) => self.toggle_pause(),
                Key::Character(ref c) if c == "c" || c == "C" => self.clock_state.next_color(),
                _ => {}
            },
            WindowEvent::MouseInput { state, .. } if state.is_pressed() => {
                self.clock_state.next_color();
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    tracing::error!("Render error: {}", e);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(new_size) => {
                self.handle_resize(new_size);
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

    println!("Goldy Clock Example (using shared rendering code, Surface API)");
    println!("=============================================================");
    println!("Controls:");
    println!("  Space - Toggle pause");
    println!("  Click - Change color");
    println!("  Escape - Exit\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
