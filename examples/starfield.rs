//! Starfield example - classic 3D starfield flying through space.
//!
//! Demonstrates GPU compute + graphics in a single task graph:
//! star update dispatch → offscreen render → swapchain blit.
//!
//! Run with: cargo run --example starfield

use anyhow::Result;
use goldy::{
    Buffer, BufferKind, Color, CommandEncoder, ComputePipeline, DeviceDescriptor, Instance, NodeAccess,
    PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions, ResourceAccess,
    ShaderModule, Surface, TaskGraph, VertexBufferLayout,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const NUM_STARS: u32 = 500;

/// Star types for different celestial objects
const STAR_TYPE_NORMAL: f32 = 0.0;
const STAR_TYPE_GALAXY: f32 = 1.0;
const STAR_TYPE_QUASAR: f32 = 2.0;
const STAR_TYPE_WHITE_DWARF: f32 = 3.0;

/// Star structure matching the shader layout
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Star {
    x: f32,
    y: f32,
    z: f32,
    star_type: f32,
}

/// Uniform parameters for the compute shader
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
    window: Arc<Window>,
    device: Arc<goldy::Device>,
    surface: Surface,
    scene_rt: RenderTarget,
    frame_graph: TaskGraph,
    compute_pipeline: ComputePipeline,
    star_buffer: Buffer,
    params_buffer: Buffer,
    render_pipeline: RenderPipeline,
    speed: f32,
    frame_count: f32,
    start_time: std::time::Instant,
}

impl RenderState {
    fn create_scene_rt(device: &goldy::Device, surface: &Surface) -> Result<RenderTarget> {
        let (width, height) = surface.size();
        Ok(RenderTarget::new(
            device,
            width.max(1),
            height.max(1),
            surface.format(),
        )?)
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

        let star_buffer = device.alloc_buffer_with_data(&stars, BufferKind::Scattered)?;

        let initial_params = StarfieldParams {
            speed: 0.01,
            frame: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let params_buffer = device.alloc_buffer_with_data(&[initial_params], BufferKind::Broadcast)?;

        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;

        let render_pipeline = RenderPipeline::new(
            &device,
            &render_shader,
            &render_shader,
            &RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::TriangleList,
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        println!("Created starfield with {} stars", NUM_STARS);

        Ok(Self {
            window,
            device,
            surface,
            scene_rt,
            frame_graph: TaskGraph::new(),
            compute_pipeline,
            star_buffer,
            params_buffer,
            render_pipeline,
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
        self.params_buffer.write_data(0, &[params])?;

        self.frame_graph.clear();
        self.frame_graph
            .node("update_stars", &self.compute_pipeline)
            .bind_buffer(&self.star_buffer, NodeAccess::ReadWrite)
            .bind_buffer(&self.params_buffer, NodeAccess::Read)
            .bind_resources_raw_slice(&[
                self.star_buffer.resource_index(ResourceAccess::Write).unwrap(),
                self.params_buffer.resource_index(ResourceAccess::Read).unwrap(),
            ])
            .dispatch(NUM_STARS.div_ceil(64), 1, 1);

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.set_pipeline(&self.render_pipeline);
            pass.bind_resources(&[&self.star_buffer]);
            pass.draw(0..6, 0..NUM_STARS);
        }

        self.frame_graph
            .render_pass("starfield", &self.scene_rt)
            .bind_buffer(&self.star_buffer, NodeAccess::Read)
            .finish_encoder(encoder);

        let swapchain = self.frame_graph.declare_swapchain_output();
        self.frame_graph
            .copy_render_target_to_swapchain(&self.scene_rt, swapchain);

        let frame = self.surface.begin()?;
        let frame = self.surface.submit_graph_to_frame(&mut self.frame_graph, frame)?;
        frame.present()?;

        self.window.request_redraw();
        Ok(())
    }

    fn change_speed(&mut self, delta: f32) {
        self.speed = (self.speed + delta).clamp(0.001, 0.1);
        if let Some(w) = Some(&self.window) {
            w.set_title(&format!("Goldy - Starfield (speed: {:.1})", self.speed));
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
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Starfield")
                            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768)),
                    )
                    .expect("Failed to create window"),
            );

            match RenderState::new(window.clone()) {
                Ok(state) => {
                    self.state = Some(state);
                    window.request_redraw();
                }
                Err(e) => {
                    tracing::error!("Failed to create render state: {}", e);
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
                }
            }
            _ => {}
        }
    }
}
