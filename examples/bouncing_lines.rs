//! Bouncing lines example - animated lines bouncing off walls.
//!
//! Demonstrates GPU compute + graphics in a single task graph:
//! line physics dispatch → offscreen render → swapchain blit.
//!
//! Run with: cargo run --example bouncing_lines

use anyhow::Result;
use goldy::{
    BufferKind, Color, ComputePipeline, DeviceDescriptor, Instance, NodeAccess, Parcel, PrimitiveTopology,
    RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions, ResourceAccess, RetainedPool,
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
mod common;

const NUM_LINES: u32 = 20;

/// Line structure matching the shader layout
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Line {
    p1: [f32; 2],
    v1: [f32; 2],
    p2: [f32; 2],
    v2: [f32; 2],
    color_index: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
impl goldy::StructuredBufferElement for Line {}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    println!("Goldy Bouncing Lines Example");
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
    _retained_pool: RetainedPool,
    line_buffer: Parcel,
    render_pipeline: RenderPipeline,
    frame_count: u32,
    start_time: std::time::Instant,
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

        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/bouncing_lines_update.slang"))?;
        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/bouncing_lines_render.slang"))?;

        let mut lines = Vec::with_capacity(NUM_LINES as usize);
        for idx in 0..NUM_LINES {
            let angle = (idx as f32 / NUM_LINES as f32) * std::f32::consts::PI * 2.0;
            lines.push(Line {
                p1: [angle.cos() * 0.3, angle.sin() * 0.3],
                v1: [0.01 * (idx as f32 * 0.7).cos(), 0.012 * (idx as f32 * 0.9).sin()],
                p2: [-angle.cos() * 0.3, -angle.sin() * 0.3],
                v2: [-0.011 * (idx as f32 * 1.1).cos(), 0.009 * (idx as f32 * 1.3).sin()],
                color_index: idx,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            });
        }

        let mut retained_pool = RetainedPool::new(device.clone());
        let line_buffer = retained_pool.acquire_buffer_with_data(&lines, BufferKind::Scattered)?;

        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;

        let render_pipeline = RenderPipeline::new(
            &device,
            &render_shader,
            &render_shader,
            &RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::LineList,
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        println!("Created bouncing lines with {} lines", NUM_LINES);

        Ok(Self {
            window,
            device,
            surface,
            scene_rt,
            frame_graph: TaskGraph::new(),
            compute_pipeline,
            _retained_pool: retained_pool,
            line_buffer,
            render_pipeline,
            frame_count: 0,
            start_time: std::time::Instant::now(),
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;

        self.frame_graph.clear();
        self.frame_graph
            .node("update_lines", &self.compute_pipeline)
            .bind_parcel(&self.line_buffer, NodeAccess::ReadWrite)
            .bind_resources_raw_slice(&[self.line_buffer.resource_index(ResourceAccess::Write).unwrap()])
            .dispatch(NUM_LINES.div_ceil(64).max(1), 1, 1);

        let bg_color = Color {
            r: 0.05,
            g: 0.05,
            b: 0.1,
            a: 1.0,
        };

        let mut pass = self.frame_graph.render_pass("bouncing_lines", &self.scene_rt);
        pass.bind_parcel_mut(&self.line_buffer, NodeAccess::Read);
        pass.clear(bg_color);
        pass.set_pipeline(&self.render_pipeline);
        // `Scattered<Line>` in the vertex shader expects a UAV bindless slot on DX12.
        pass.bind_resources_raw(&[self
            .line_buffer
            .resource_index(ResourceAccess::ReadWrite)
            .unwrap()]);
        pass.draw(0..2, 0..NUM_LINES);
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
                            .with_title("Goldy - Bouncing Lines")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
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
                }
            }
            _ => {}
        }
    }
}
