//! Bouncing lines example - animated lines bouncing off walls.
//!
//! Demonstrates retained scheme with compute dispatch → offscreen render → copy-to-present.
//!
//! Run with: cargo run --example bouncing_lines

use anyhow::Result;
use goldy::{
    Buffer, BufferKind, Color, ComputePipeline, DeviceDescriptor, Instance, Lease, LeaseRenderTarget, MemoryExchange,
    NodeAccess, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, Texture, TextureFormat, Transaction, VertexBufferLayout,
    WithdrawTransaction,
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
use common::CaptureDump;

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

    if common::capture_requested() {
        let mut state = RenderState::new(None)?;
        while !state.capture_done() {
            state.render()?;
        }
        return Ok(());
    }

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
    window: Option<Arc<Window>>,
    device: Arc<goldy::Device>,
    ctx: goldy::Context,
    surface: Option<SurfaceExchange>,
    capture: Option<CaptureDump>,
    readback: Option<Texture>,
    present: Option<Transaction>,
    withdraw: Option<WithdrawTransaction>,
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
        format: TextureFormat,
    ) -> Result<RenderPipeline> {
        common::render_pipeline(
            device,
            render_shader,
            format,
            RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            },
        )
    }

    fn bind_frame(
        scheme: &mut Scheme,
        scene_rt: &Lease<LeaseRenderTarget>,
        surface: Option<&SurfaceExchange>,
        readback: Option<&Texture>,
    ) -> anyhow::Result<(Option<Transaction>, Option<WithdrawTransaction>)> {
        if let Some(surface) = surface {
            let present = surface.bind_render_target(scheme, scene_rt)?;
            Ok((Some(present), None))
        } else {
            let readback = readback.expect("capture readback");
            scheme.copy_to_texture(scene_rt, readback)?;
            let withdraw = MemoryExchange::new(scheme.context()).bind_withdraw(scheme, readback)?;
            Ok((None, Some(withdraw)))
        }
    }

    fn record_scheme(
        scheme: &mut Scheme,
        compute_pipeline: &ComputePipeline,
        render_pipeline: &RenderPipeline,
        line_buffer: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
    ) {
        scheme
            .node("update_lines", compute_pipeline)
            .with_parcel(line_buffer, NodeAccess::ReadWrite)
            .dispatch(NUM_LINES.div_ceil(64).max(1), 1, 1);

        let bg_color = Color {
            r: 0.05,
            g: 0.05,
            b: 0.1,
            a: 1.0,
        };

        let mut pass = scheme.render_pass("bouncing_lines", scene_rt, TargetLoad::Clear(bg_color));
        pass.with_parcel(line_buffer, NodeAccess::Read);
        pass.set_pipeline(render_pipeline);
        pass.draw(0..2, 0..NUM_LINES);
        pass.finish();
    }

    fn target(&self) -> (TextureFormat, u32, u32) {
        if let Some(surface) = &self.surface {
            let (width, height) = surface.size();
            (surface.format(), width, height)
        } else {
            let capture = self.capture.as_ref().expect("capture dump");
            let (width, height) = capture.size();
            (CaptureDump::format(), width, height)
        }
    }

    fn capture_done(&self) -> bool {
        self.capture.as_ref().is_none_or(CaptureDump::finished)
    }

    fn rerecord_scheme(&mut self) {
        let mut scheme = Scheme::new(&self.ctx);
        let (format, width, height) = self.target();
        if let Ok(rt) = scheme.lease_render_target(width.max(1), height.max(1), format, None) {
            self.scene_rt = rt;
            Self::record_scheme(
                &mut scheme,
                &self.compute_pipeline,
                &self.render_pipeline,
                &self.line_buffer,
                &self.scene_rt,
            );
            if let Ok((present, withdraw)) = Self::bind_frame(
                &mut scheme,
                &self.scene_rt,
                self.surface.as_ref(),
                self.readback.as_ref(),
            ) {
                self.present = present;
                self.withdraw = withdraw;
                self.scheme = scheme;
            }
        }
    }

    fn new(window: Option<Arc<Window>>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let mut retained_pool = RetainedPool::new(device.clone());

        let (surface, capture, readback, format, width, height) = if let Some(window) = window.as_deref() {
            let surface = SurfaceExchange::new(&ctx, window, SurfaceConfig::default())?;
            let format = surface.format();
            let (width, height) = surface.size();
            (Some(surface), None, None, format, width, height)
        } else {
            let capture = CaptureDump::from_env()?;
            let (width, height) = capture.size();
            let readback = common::capture_readback(&mut retained_pool, width, height)?;
            (
                None,
                Some(capture),
                Some(readback),
                CaptureDump::format(),
                width,
                height,
            )
        };

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

        let line_buffer = retained_pool.acquire_buffer_with_data(&lines, BufferKind::Scattered)?;

        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;
        let render_pipeline = Self::create_render_pipeline(&device, &render_shader, format)?;

        let mut scheme = Scheme::new(&ctx);
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), format, None)?;
        Self::record_scheme(
            &mut scheme,
            &compute_pipeline,
            &render_pipeline,
            &line_buffer,
            &scene_rt,
        );
        let (present, withdraw) = Self::bind_frame(&mut scheme, &scene_rt, surface.as_ref(), readback.as_ref())?;

        println!("Created bouncing lines with {} lines", NUM_LINES);

        Ok(Self {
            window,
            device,
            ctx,
            surface,
            capture,
            readback,
            present,
            withdraw,
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

        let mut submission = self.scheme.submit()?;
        if let Some(present) = &self.present {
            present.claim(&mut submission)?.consume()?;
        } else {
            let pixels = self.withdraw.as_ref().unwrap().claim(&mut submission)?.consume()?;
            self.capture.as_mut().unwrap().write_rgba(&pixels)?;
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
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
                    .create_window(common::hidden_window("Goldy - Bouncing Lines", 800, 600))
                    .expect("Failed to create window"),
            );

            match RenderState::new(Some(window.clone())) {
                Ok(mut state) => {
                    if let Err(e) = state.render() {
                        tracing::error!("First frame error: {e}");
                    }
                    common::reveal_window(&window);
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
                        let Some(surface) = state.surface.as_ref() else {
                            return;
                        };
                        let (prev_w, prev_h) = surface.size();
                        if size.width == prev_w && size.height == prev_h {
                            return;
                        }
                        let _ = surface.resize(size.width, size.height);
                        let format = surface.format();
                        if let Ok(pipeline) =
                            RenderState::create_render_pipeline(&state.device, &state.render_shader, format)
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
