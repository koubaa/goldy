//! Instancing example - render many objects efficiently.
//!
//! Demonstrates GPU compute + graphics in a single task graph:
//! instance update dispatch → offscreen render → swapchain blit.
//!
//! Run with: cargo run --example instancing

use anyhow::Result;
use goldy::{
    Buffer, BufferKind, Color, ComputePipeline, DeviceDescriptor, Instance, Instance2D, NodeAccess, PrimitiveTopology,
    RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions, ResourceAccess, ShaderModule, Surface,
    TaskGraph, VertexBufferLayout,
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

const GRID_SIZE: u32 = 20;
const QUAD_SIZE: f32 = 0.03;
const NUM_QUADS: u32 = GRID_SIZE * GRID_SIZE;

/// Animation parameters for compute shader
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AnimParams {
    time: f32,
    delta_time: f32,
    total_instances: u32,
    _pad: u32,
}
impl goldy::StructuredBufferElement for AnimParams {}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    println!("Goldy Instancing Example - {} quads (GPU-driven)", NUM_QUADS);
    println!("Press Escape to exit");

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
    render_pipeline: RenderPipeline,
    instance_buffer: Buffer,
    params_buffer: Buffer,
    start_time: Instant,
    last_time: f32,
    frame_count: u32,
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

        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/instancing_update.slang"))?;
        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/instancing_render.slang"))?;

        let mut instances = Vec::with_capacity(NUM_QUADS as usize);
        for i in 0..GRID_SIZE {
            for j in 0..GRID_SIZE {
                let nx = (i as f32 / (GRID_SIZE - 1) as f32) * 2.0 - 1.0;
                let ny = (j as f32 / (GRID_SIZE - 1) as f32) * 2.0 - 1.0;
                let cx = nx * 0.85;
                let cy = ny * 0.85;

                instances.push(Instance2D::new(cx, cy, 0.0, QUAD_SIZE, [1.0, 1.0, 1.0, 1.0]));
            }
        }

        let instance_buffer = device.alloc_buffer_with_data(&instances, BufferKind::Scattered)?;

        let params = AnimParams {
            time: 0.0,
            delta_time: 0.016,
            total_instances: NUM_QUADS,
            _pad: 0,
        };
        let params_buffer = device.alloc_buffer_with_data(&[params], BufferKind::Broadcast)?;

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

        println!(
            "Created instancing example with {} quads (GPU compute + graphics)",
            NUM_QUADS
        );

        Ok(Self {
            window,
            device,
            surface,
            scene_rt,
            frame_graph: TaskGraph::new(),
            compute_pipeline,
            render_pipeline,
            instance_buffer,
            params_buffer,
            start_time: Instant::now(),
            last_time: 0.0,
            frame_count: 0,
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;

        let time = self.start_time.elapsed().as_secs_f32();
        let delta_time = time - self.last_time;
        self.last_time = time;

        let params = AnimParams {
            time,
            delta_time,
            total_instances: NUM_QUADS,
            _pad: 0,
        };
        self.params_buffer.write_data(0, &[params])?;

        self.frame_graph.clear();
        self.frame_graph
            .node("update_instances", &self.compute_pipeline)
            .bind_buffer(&self.instance_buffer, NodeAccess::ReadWrite)
            .bind_buffer(&self.params_buffer, NodeAccess::Read)
            .bind_resources_raw_slice(&[
                self.instance_buffer.resource_index(ResourceAccess::Write).unwrap(),
                self.params_buffer.resource_index(ResourceAccess::Read).unwrap(),
            ])
            .dispatch(NUM_QUADS.div_ceil(64), 1, 1);

        let bg_color = Color {
            r: 0.02,
            g: 0.02,
            b: 0.04,
            a: 1.0,
        };

        let mut pass = self.frame_graph.render_pass("instancing", &self.scene_rt);
        pass.bind_buffer_mut(&self.instance_buffer, NodeAccess::Read);
        pass.clear(bg_color);
        pass.set_pipeline(&self.render_pipeline);
        pass.bind_resources(&[&self.instance_buffer]);
        pass.draw_quads(NUM_QUADS);
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
                            .with_title(format!("Goldy - Instancing ({} quads, GPU-driven)", NUM_QUADS))
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
                }
            }
            _ => {}
        }
    }
}
