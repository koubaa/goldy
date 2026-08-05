//! Instancing example - render many objects efficiently.
//!
//! Demonstrates retained scheme with compute dispatch → offscreen render → copy-to-present.
//!
//! Run with: cargo run --example instancing

use anyhow::Result;
use goldy::{
    Buffer, BufferFlags, BufferKind, Color, ComputePipeline, DepositTransaction, DeviceDescriptor, Instance, Lease,
    LeaseRenderTarget, MemoryExchange, NodeAccess, PrimitiveTopology, RenderPipeline, RenderPipelineDesc,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, Transaction,
    VertexBufferLayout,
};

mod instance2d;
use instance2d::Instance2D;
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

const GRID_SIZE: u32 = 20;
const QUAD_SIZE: f32 = 0.03;
const NUM_QUADS: u32 = GRID_SIZE * GRID_SIZE;

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
    println!("Goldy Instancing Example - {} quads (Scheme + Present)", NUM_QUADS);
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
    ctx: goldy::Context,
    surface: SurfaceExchange,
    present: Transaction,
    scheme: Scheme,
    scene_rt: Lease<LeaseRenderTarget>,
    compute_pipeline: ComputePipeline,
    render_shader: ShaderModule,
    render_pipeline: RenderPipeline,
    _retained_pool: RetainedPool,
    instance_buffer: Buffer,
    params_buffer: Buffer,
    upload_scheme: Scheme,
    params_deposit: DepositTransaction,
    start_time: Instant,
    last_time: f32,
    frame_count: u32,
}

impl RenderState {
    fn create_render_pipeline(
        device: &goldy::Device,
        render_shader: &ShaderModule,
        surface: &SurfaceExchange,
    ) -> Result<RenderPipeline> {
        common::render_pipeline_for_surface(
            device,
            render_shader,
            surface,
            RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
        )
        .map_err(Into::into)
    }

    fn record_scheme(
        scheme: &mut Scheme,
        surface: &SurfaceExchange,
        compute_pipeline: &ComputePipeline,
        render_pipeline: &RenderPipeline,
        instance_buffer: &Buffer,
        params_buffer: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
    ) -> anyhow::Result<Transaction> {
        scheme
            .node("update_instances", compute_pipeline)
            .with_parcel(instance_buffer, NodeAccess::ReadWrite)
            .with_parcel(params_buffer, NodeAccess::Read)
            .dispatch(NUM_QUADS.div_ceil(64), 1, 1);

        let bg_color = Color {
            r: 0.02,
            g: 0.02,
            b: 0.04,
            a: 1.0,
        };

        let mut pass = scheme.render_pass("instancing", scene_rt, TargetLoad::Clear(bg_color));
        pass.with_parcel(&instance_buffer, NodeAccess::Read);
        pass.set_pipeline(render_pipeline);
        pass.draw(0..6, 0..NUM_QUADS);
        pass.finish();

        surface.bind_render_target(scheme, scene_rt).map_err(Into::into)
    }

    fn rerecord_scheme(&mut self) {
        let mut scheme = Scheme::new(&self.ctx);
        let (width, height) = self.surface.size();
        if let Ok(rt) = scheme.lease_render_target(width.max(1), height.max(1), self.surface.format(), None) {
            self.scene_rt = rt;
            if let Ok(present) = Self::record_scheme(
                &mut scheme,
                &self.surface,
                &self.compute_pipeline,
                &self.render_pipeline,
                &self.instance_buffer,
                &self.params_buffer,
                &self.scene_rt,
            ) {
                self.present = present;
                self.scheme = scheme;
            }
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
        let surface = SurfaceExchange::new(&ctx, window.as_ref(), SurfaceConfig::default())?;

        let compute_shader = ShaderModule::from_slang(&device, include_str!("../shaders/instancing_update.slang"))?;
        let render_shader = ShaderModule::from_slang(&device, include_str!("../shaders/instancing_render.slang"))?;

        let mut instances = Vec::with_capacity(NUM_QUADS as usize);
        for i in 0..GRID_SIZE {
            for j in 0..GRID_SIZE {
                let nx = (i as f32 / (GRID_SIZE - 1) as f32) * 2.0 - 1.0;
                let ny = (j as f32 / (GRID_SIZE - 1) as f32) * 2.0 - 1.0;
                instances.push(Instance2D::new(
                    nx * 0.85,
                    ny * 0.85,
                    0.0,
                    QUAD_SIZE,
                    [1.0, 1.0, 1.0, 1.0],
                ));
            }
        }

        let mut retained_pool = RetainedPool::new(device.clone());
        let instance_buffer = retained_pool.acquire_buffer_with_data(&instances, BufferKind::Scattered)?;
        let params_buffer =
            retained_pool.acquire_buffer_sized::<AnimParams>(1, BufferKind::Broadcast, BufferFlags::empty())?;

        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;
        let render_pipeline = Self::create_render_pipeline(&device, &render_shader, &surface)?;

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = surface.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), surface.format(), None)?;
        let present = Self::record_scheme(
            &mut scheme,
            &surface,
            &compute_pipeline,
            &render_pipeline,
            &instance_buffer,
            &params_buffer,
            &scene_rt,
        )?;

        let mut upload_scheme = Scheme::new(&ctx);
        let params_deposit = MemoryExchange::new(&ctx).bind_deposit_buffer(
            &mut upload_scheme,
            &params_buffer,
            std::mem::size_of::<AnimParams>() as u64,
        )?;

        println!(
            "Created instancing example with {} quads (GPU compute + graphics)",
            NUM_QUADS
        );

        Ok(Self {
            window,
            device,
            ctx,
            surface,
            present,
            scheme,
            scene_rt,
            compute_pipeline,
            render_shader,
            render_pipeline,
            _retained_pool: retained_pool,
            instance_buffer,
            params_buffer,
            upload_scheme,
            params_deposit,
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

        self.params_deposit
            .write(&mut self.upload_scheme, 0, bytemuck::bytes_of(&params))?;
        self.upload_scheme.submit()?;

        let mut submission = self.scheme.submit()?;
        self.present.claim(&mut submission)?.consume()?;

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
                    .create_window(common::hidden_window(
                        format!("Goldy - Instancing ({} quads, Scheme + Present)", NUM_QUADS),
                        800,
                        800,
                    ))
                    .expect("Failed to create window"),
            );

            match RenderState::new(window.clone()) {
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
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    if size.width > 0 && size.height > 0 {
                        state.surface.resize(size.width, size.height).ok();
                        if let Ok(pipeline) =
                            RenderState::create_render_pipeline(&state.device, &state.render_shader, &state.surface)
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
