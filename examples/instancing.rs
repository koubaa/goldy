//! Instancing example - render many objects efficiently.
//!
//! Demonstrates retained scheme with compute dispatch → offscreen render → copy-to-present.
//!
//! Run with: cargo run --example instancing

#![allow(deprecated)] // write_to_parcel migration deferred

use anyhow::Result;
use goldy::{
    types::ResourceAccess, write_to_parcel, Buffer, BufferFlags, BufferKind, Color, ComputePipeline, DeviceDescriptor,
    Grant, Instance, Instance2D, Lease, LeaseRenderTarget, NodeAccess, PresentGrant, PrimitiveTopology, RenderPipeline,
    RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SwapchainPool, VertexBufferLayout,
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
    swapchain: SwapchainPool,
    screen: goldy::PresentLease,
    present: PresentGrant,
    scheme: Scheme,
    scene_rt: Lease<LeaseRenderTarget>,
    compute_pipeline: ComputePipeline,
    render_shader: ShaderModule,
    render_pipeline: RenderPipeline,
    _retained_pool: RetainedPool,
    instance_buffer: Buffer,
    params_buffer: Buffer,
    start_time: Instant,
    last_time: f32,
    frame_count: u32,
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
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
        )
        .map_err(Into::into)
    }

    fn record_scheme(
        scheme: &mut Scheme,
        compute_pipeline: &ComputePipeline,
        render_pipeline: &RenderPipeline,
        instance_buffer: &Buffer,
        params_buffer: &Buffer,
        scene_rt: &Lease<LeaseRenderTarget>,
        screen: &goldy::PresentLease,
    ) -> PresentGrant {
        scheme
            .node("update_instances", compute_pipeline)
            .with_parcel(instance_buffer, NodeAccess::ReadWrite)
            .with_parcel(params_buffer, NodeAccess::Read)
            .with_views(&[
                instance_buffer.handle(ResourceAccess::Write).unwrap(),
                params_buffer.handle(ResourceAccess::Read).unwrap(),
            ])
            .dispatch(NUM_QUADS.div_ceil(64), 1, 1);

        let bg_color = Color {
            r: 0.02,
            g: 0.02,
            b: 0.04,
            a: 1.0,
        };

        let mut pass = scheme.render_pass("instancing", scene_rt);
        // Graph dependency only — do not push Read SRV into set_pipeline; after compute
        // UAV writes, WARP reads zeros through the SRV slot (see scheme_compute_integration).
        pass.with_buffer_dependency(&instance_buffer, NodeAccess::Read);
        pass.clear(bg_color);
        pass.set_pipeline(render_pipeline);
        pass.with_views(&[instance_buffer.handle(ResourceAccess::ReadWrite).unwrap()]);
        pass.draw(0..6, 0..NUM_QUADS);
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
                &self.instance_buffer,
                &self.params_buffer,
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
        let render_pipeline = Self::create_render_pipeline(&device, &render_shader, &swapchain)?;

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = swapchain.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), swapchain.format(), None)?;
        let present = Self::record_scheme(
            &mut scheme,
            &compute_pipeline,
            &render_pipeline,
            &instance_buffer,
            &params_buffer,
            &scene_rt,
            &screen,
        );

        println!(
            "Created instancing example with {} quads (GPU compute + graphics)",
            NUM_QUADS
        );

        Ok(Self {
            window,
            device,
            ctx,
            swapchain,
            screen,
            present,
            scheme,
            scene_rt,
            compute_pipeline,
            render_shader,
            render_pipeline,
            _retained_pool: retained_pool,
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

        write_to_parcel(&self.ctx, &self.params_buffer, 0, bytemuck::bytes_of(&params))?;

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
                            .with_title(format!("Goldy - Instancing ({} quads, Scheme + Present)", NUM_QUADS))
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
