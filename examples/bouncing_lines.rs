//! Bouncing lines example - animated lines bouncing off walls.
//!
//! Demonstrates retained scheme with compute dispatch → offscreen render → copy-to-present.
//!
//! Run with: cargo run --example bouncing_lines

use anyhow::Result;
use goldy::{
    types::ResourceAccess, Buffer, BufferKind, Color, ComputePipeline, DeviceDescriptor, Grant, Instance, Lease,
    LeaseRenderTarget, NodeAccess, PresentGrant, PrimitiveTopology, RenderPipeline, RenderPipelineDesc,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SwapchainPool, VertexBufferLayout,
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
    swapchain: SwapchainPool,
    screen: goldy::PresentLease,
    present: PresentGrant,
    scheme: Scheme,
    scene_rt: Lease<LeaseRenderTarget>,
    compute_pipeline: ComputePipeline,
    _retained_pool: RetainedPool,
    line_buffer: Buffer,
    render_shader: ShaderModule,
    render_pipeline: RenderPipeline,
    frame_count: u32,
    start_time: std::time::Instant,
}

impl RenderState {
    fn create_render_pipeline(
        device: &goldy::Device,
        render_shader: &ShaderModule,
        swapchain: &SwapchainPool,
    ) -> Result<RenderPipeline> {
        common::render_pipeline_for_swapchain(
            device,
            render_shader,
            swapchain,
            RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            },
        )
        .map_err(Into::into)
    }

    fn record_scheme(
        scheme: &mut Scheme,
        compute_pipeline: &ComputePipeline,
        render_pipeline: &RenderPipeline,
        line_buffer: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
        screen: &goldy::PresentLease,
    ) -> PresentGrant {
        scheme
            .node("update_lines", compute_pipeline)
            .with_parcel(line_buffer, NodeAccess::ReadWrite)
            .with_views(&[line_buffer.handle(ResourceAccess::Write).unwrap()])
            .dispatch(NUM_LINES.div_ceil(64).max(1), 1, 1);

        let bg_color = Color {
            r: 0.05,
            g: 0.05,
            b: 0.1,
            a: 1.0,
        };

        let mut pass = scheme.render_pass("bouncing_lines", scene_rt);
        pass.with_parcel(line_buffer, NodeAccess::Read);
        pass.clear(bg_color);
        pass.set_pipeline(render_pipeline);
        pass.with_views(&[line_buffer.handle(ResourceAccess::ReadWrite).unwrap()]);
        pass.draw(0..2, 0..NUM_LINES);
        pass.finish();

        scheme.copy_to_present(scene_rt, screen);
        scheme.grant_present(screen)
    }

    fn rerecord_scheme(&mut self) {
        let mut scheme = Scheme::new(&self.ctx);
        let (width, height) = self.swapchain.size();
        if let Ok(rt) = scheme.lease_render_target(width.max(1), height.max(1), self.swapchain.format(), None) {
            self.scene_rt = rt;
            self.present = Self::record_scheme(
                &mut scheme,
                &self.compute_pipeline,
                &self.render_pipeline,
                &self.line_buffer,
                &self.scene_rt,
                &self.screen,
            );
            self.scheme = scheme;
        }
    }

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
        let render_pipeline = Self::create_render_pipeline(&device, &render_shader, &swapchain)?;

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = swapchain.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)?;
        let present = Self::record_scheme(
            &mut scheme,
            &compute_pipeline,
            &render_pipeline,
            &line_buffer,
            &scene_rt,
            &screen,
        );

        println!("Created bouncing lines with {} lines", NUM_LINES);

        Ok(Self {
            window,
            device,
            swapchain,
            screen,
            present,
            scheme,
            scene_rt,
            compute_pipeline,
            _retained_pool: retained_pool,
            line_buffer,
            render_shader,
            render_pipeline,
            frame_count: 0,
            start_time: std::time::Instant::now(),
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;

        let submission = self.scheme.submit()?;
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
                        state.swapchain.resize(size.width, size.height).ok();
                        if let Ok(pipeline) =
                            RenderState::create_render_pipeline(&state.device, &state.render_shader, &state.swapchain)
                        {
                            state.render_pipeline = pipeline;
                        }
                        state.rerecord_scheme();
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
