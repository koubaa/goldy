//! Depth quads example - two fullscreen quads whose depths cross periodically.
//!
//! Depth-tested rendering via an offscreen [`RenderTarget`] with a depth attachment
//! (`RenderTarget::new_with_depth`), then copy-to-present through a retained scheme.
//!
//! Run with: cargo run --example depth_quads

use bytemuck::{Pod, Zeroable};
use goldy::{
    write_to_parcel, BufferFlags, BufferKind, Color, CompareFunction, DepthFormat, DepthStencilState, DeviceDescriptor,
    Grant, Instance, NodeAccess, Parcel, PresentGrant, RenderPipeline, RenderPipelineDesc, RenderTarget,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SwapchainPool, VertexAttribute, VertexBufferLayout,
    VertexFormat,
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

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct DepthVertex {
    position: [f32; 3],
    color: [f32; 4],
}
impl goldy::StructuredBufferElement for DepthVertex {}

impl DepthVertex {
    const fn new(x: f32, y: f32, z: f32, r: f32, g: f32, b: f32) -> Self {
        Self {
            position: [x, y, z],
            color: [r, g, b, 1.0],
        }
    }
}

fn depth_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout {
        stride: std::mem::size_of::<DepthVertex>() as u32,
        attributes: vec![
            VertexAttribute {
                location: 0,
                format: VertexFormat::Float32x3,
                offset: 0,
            },
            VertexAttribute {
                location: 1,
                format: VertexFormat::Float32x4,
                offset: 12,
            },
        ],
    }
}

#[allow(clippy::too_many_arguments)]
fn quad_verts(x0: f32, y0: f32, x1: f32, y1: f32, z: f32, r: f32, g: f32, b: f32) -> [DepthVertex; 6] {
    let tl = DepthVertex::new(x0, y1, z, r, g, b);
    let bl = DepthVertex::new(x0, y0, z, r, g, b);
    let br = DepthVertex::new(x1, y0, z, r, g, b);
    let tr = DepthVertex::new(x1, y1, z, r, g, b);
    [tl, bl, br, tl, br, tr]
}

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    _retained_pool: Option<RetainedPool>,
    warm_parcel: Option<Parcel>,
    cool_parcel: Option<Parcel>,
    swapchain: Option<SwapchainPool>,
    screen: Option<goldy::PresentLease>,
    present: Option<PresentGrant>,
    scene_rt: Option<RenderTarget>,
    scheme: Option<Scheme>,
    window: Option<Arc<Window>>,
    frame_count: u64,
    start_time: std::time::Instant,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            ctx: None,
            device: None,
            pipeline: None,
            _retained_pool: None,
            warm_parcel: None,
            cool_parcel: None,
            swapchain: None,
            screen: None,
            present: None,
            scene_rt: None,
            scheme: None,
            window: None,
            frame_count: 0,
            start_time: std::time::Instant::now(),
        })
    }

    fn create_scene_rt(device: &goldy::Device, swapchain: &SwapchainPool) -> anyhow::Result<RenderTarget> {
        let (width, height) = swapchain.size();
        RenderTarget::new_with_depth(
            device,
            width.max(1),
            height.max(1),
            swapchain.format(),
            Some(DepthFormat::Depth32Float),
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        pipeline: &RenderPipeline,
        warm_parcel: &Parcel,
        cool_parcel: &Parcel,
        scene_rt: &RenderTarget,
        screen: &goldy::PresentLease,
    ) -> PresentGrant {
        let mut pass = scheme.render_pass("depth_quads", scene_rt);
        pass.bind_parcel_mut(warm_parcel, NodeAccess::Read);
        pass.bind_parcel_mut(cool_parcel, NodeAccess::Read);
        pass.clear(Color::BLACK);
        pass.clear_depth(1.0);
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, warm_parcel);
        pass.draw(0..6, 0..1);
        pass.set_vertex_buffer(0, cool_parcel);
        pass.draw(0..6, 0..1);
        pass.finish();
        scheme.copy_to_present(scene_rt, screen);
        scheme.grant_present(screen)
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let swapchain = SwapchainPool::new(&ctx, window.as_ref(), 3)?;
        let screen = swapchain.lease();

        let shader = ShaderModule::from_slang(&device, include_str!("../shaders/depth_test.slang"))?;
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: depth_vertex_layout(),
                target_format: swapchain.format(),
                depth_stencil: Some(DepthStencilState {
                    format: DepthFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::Less,
                }),
                ..Default::default()
            },
        )?;

        let scene_rt = Self::create_scene_rt(&device, &swapchain)?;

        let mut retained_pool = RetainedPool::new(device.clone());
        let warm_parcel =
            retained_pool.acquire_buffer_sized::<DepthVertex>(6, BufferKind::Scattered, BufferFlags::empty())?;
        let cool_parcel =
            retained_pool.acquire_buffer_sized::<DepthVertex>(6, BufferKind::Scattered, BufferFlags::empty())?;

        let mut scheme = Scheme::new(&ctx);
        let present = Self::record_scheme(&mut scheme, &pipeline, &warm_parcel, &cool_parcel, &scene_rt, &screen);

        self.ctx = Some(ctx);
        self.device = Some(device);
        self.pipeline = Some(pipeline);
        self._retained_pool = Some(retained_pool);
        self.warm_parcel = Some(warm_parcel);
        self.cool_parcel = Some(cool_parcel);
        self.swapchain = Some(swapchain);
        self.screen = Some(screen);
        self.present = Some(present);
        self.scene_rt = Some(scene_rt);
        self.scheme = Some(scheme);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let size = self.window.as_ref().unwrap().inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let t = self.frame_count as f32 * 0.04;
        let warm_z = t.sin() * 0.4 + 0.5;
        let cool_z = (t * 1.3 + 1.0).sin() * 0.4 + 0.5;

        let warm_verts = quad_verts(-1.0, -1.0, 1.0, 1.0, warm_z, 0.95, 0.35, 0.1);
        let cool_verts = quad_verts(-1.0, -1.0, 1.0, 1.0, cool_z, 0.1, 0.6, 0.95);

        let winner = if warm_z < cool_z { "WARM wins" } else { "COOL wins" };
        if let Some(window) = &self.window {
            window.set_title(&format!(
                "Depth Quads  |  warm z={:.3}  cool z={:.3}  →  {}",
                warm_z, cool_z, winner
            ));
        }

        let ctx = self.ctx.as_ref().unwrap();
        write_to_parcel(
            ctx,
            self.warm_parcel.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&warm_verts),
        )?;
        write_to_parcel(
            ctx,
            self.cool_parcel.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&cool_verts),
        )?;

        let scheme = self.scheme.as_mut().unwrap();
        let submission = scheme.submit()?;
        self.present.as_ref().unwrap().consume(&submission)?;

        self.frame_count += 1;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(swapchain) = &self.swapchain {
                let _ = swapchain.resize(new_size.width, new_size.height);
            }
            if let (Some(device), Some(swapchain)) = (&self.device, &self.swapchain) {
                if let Ok(rt) = Self::create_scene_rt(device, swapchain) {
                    if let (Some(scheme), Some(pipeline), Some(warm), Some(cool), Some(screen)) = (
                        self.scheme.as_mut(),
                        self.pipeline.as_ref(),
                        self.warm_parcel.as_ref(),
                        self.cool_parcel.as_ref(),
                        self.screen.as_ref(),
                    ) {
                        scheme.begin_rerecord();
                        let present = Self::record_scheme(scheme, pipeline, warm, cool, &rt, screen);
                        self.present = Some(present);
                    }
                    self.scene_rt = Some(rt);
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
                            .with_title("Goldy - Depth Quads (Scheme + Present)")
                            .with_inner_size(winit::dpi::LogicalSize::new(900, 600)),
                    )
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        common::exit_if_timed_out(event_loop, self.start_time);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    event_loop.exit();
                }
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

    println!("Goldy Depth Quads Example (Scheme + Present)");
    println!("Press Escape or close window to exit.\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
