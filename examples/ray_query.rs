//! Compute ray query — a TLAS of one triangle, primary rays into the swapchain.
//!
//! Skips (exit 0) when `DeviceCapabilities::ray_query` is false, or on WebGPU
//! (Slang WGSL has no `TraceRayInline`).
//!
//! Run with: cargo run --example ray_query --features examples

use anyhow::Result;
use goldy::{
    task_graph::NodeAccess, types::{BackendType, BufferFlags}, AccelInstance, AccelerationStructure, Buffer,
    BufferKind, ComputePipeline, DepositTransaction, DeviceDescriptor, Instance, MemoryExchange, PresentMode,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, Transaction,
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

const RAY_SHADER: &str = r#"
import goldy_exp;

struct Uniforms {
    uint width;
    uint height;
    float time;
    float _padding;
};

[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(BufRO<Uniforms> uniforms_buf, Accel scene, DirectSpatial<float4> output, ThreadId tid) {
    Uniforms u = uniforms_buf[0];
    if (tid.x >= u.width || tid.y >= u.height)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(u.width, u.height);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    RayDesc ray;
    ray.Origin = float3(0.0, 0.0, -2.0);
    ray.TMin = 0.001;
    ray.Direction = normalize(float3(ndc.x, ndc.y, 1.0));
    ray.TMax = 100.0;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    q.Proceed();

    float3 col = float3(0.05, 0.06, 0.12);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float2 bary = q.CommittedTriangleBarycentrics();
        col = float3(bary.x, bary.y, 1.0 - bary.x - bary.y);
        col += 0.15 * sin(u.time);
    }
    output[tid.xy] = float4(col, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    time: f32,
    _padding: f32,
}
impl goldy::StructuredBufferElement for Uniforms {}

const INITIAL_WIDTH: u32 = 800;
const INITIAL_HEIGHT: u32 = 600;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    println!("Goldy — Compute Ray Query");
    println!("=========================");
    println!("Press Escape to exit\n");

    let warmup = warm_gpu()?;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        warmup: Some(warmup),
        state: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct GpuWarmup {
    ctx: goldy::Context,
    compute_pipeline: ComputePipeline,
    retained_pool: RetainedPool,
    verts: Buffer,
    blas: AccelerationStructure,
    tlas: AccelerationStructure,
}

fn warm_gpu() -> Result<GpuWarmup> {
    let instance = Instance::new()?;
    let device = Arc::new(
        instance
            .request_adapter(&RequestAdapterOptions::default())?
            .request_device(&DeviceDescriptor::default())?,
    );
    if !device.capabilities().ray_query {
        println!("skip: DeviceCapabilities::ray_query is false on this adapter");
        std::process::exit(0);
    }
    if device.backend_type() == BackendType::WebGpu {
        println!("skip: WebGPU Slang path has no TraceRayInline");
        std::process::exit(0);
    }
    let ctx = device.create_context()?;
    let shader = ShaderModule::from_slang(&device, RAY_SHADER)?;
    let compute_pipeline = ComputePipeline::new(&device, &shader)?;
    let mut retained_pool = RetainedPool::new(device.clone());
    let positions: [[f32; 3]; 3] = [[0.0, 0.5, 0.0], [-0.7, -0.5, 0.0], [0.7, -0.5, 0.0]];
    let verts = retained_pool.acquire_buffer_with_data_and_flags(
        &positions,
        BufferKind::Scattered,
        BufferFlags::ACCEL_INPUT,
    )?;
    let blas = AccelerationStructure::blas_triangles(&device, 1, 3, 12)?;
    let tlas = AccelerationStructure::tlas(&device, 1)?;
    Ok(GpuWarmup {
        ctx,
        compute_pipeline,
        retained_pool,
        verts,
        blas,
        tlas,
    })
}

#[derive(Default)]
struct App {
    warmup: Option<GpuWarmup>,
    state: Option<RenderState>,
}

struct RenderState {
    window: Arc<Window>,
    ctx: goldy::Context,
    surface: SurfaceExchange,
    present: Transaction,
    scheme: Scheme,
    compute_pipeline: ComputePipeline,
    _retained_pool: RetainedPool,
    verts: Buffer,
    blas: AccelerationStructure,
    tlas: AccelerationStructure,
    uniform_buffer: Buffer,
    upload_scheme: Scheme,
    uniform_deposit: DepositTransaction,
    start_time: std::time::Instant,
    frame_count: u32,
}

fn record_scheme(
    scheme: &mut Scheme,
    surface: &SurfaceExchange,
    pipeline: &ComputePipeline,
    uniform: &Buffer,
    verts: &Buffer,
    blas: &AccelerationStructure,
    tlas: &AccelerationStructure,
    width: u32,
    height: u32,
) -> Result<Transaction> {
    scheme.build_blas(blas, verts.whole(), 3, 12, None)?;
    let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    scheme.build_tlas(
        tlas,
        &[AccelInstance {
            blas,
            transform: identity,
            mask: 0xFF,
            custom_index: 0,
        }],
    )?;
    let (lease, present_tx) = surface.bind_destination(scheme)?;
    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);
    scheme
        .node("rays", pipeline)
        .with_parcel(uniform, NodeAccess::Read)
        .with_parcel(tlas, NodeAccess::Read)
        .with_present(&lease)
        .dispatch(wg_x, wg_y, 1);
    Ok(present_tx)
}

fn rebuild_scheme(state: &mut RenderState, width: u32, height: u32) {
    let mut scheme = Scheme::new(&state.ctx);
    state.present = record_scheme(
        &mut scheme,
        &state.surface,
        &state.compute_pipeline,
        &state.uniform_buffer,
        &state.verts,
        &state.blas,
        &state.tlas,
        width,
        height,
    )
    .expect("failed to record scheme");
    state.scheme = scheme;
}

impl App {
    fn init(&mut self, window: Arc<Window>) -> Result<()> {
        let warmup = self
            .warmup
            .take()
            .ok_or_else(|| anyhow::anyhow!("GPU warmup state missing"))?;
        let GpuWarmup {
            ctx,
            compute_pipeline,
            mut retained_pool,
            verts,
            blas,
            tlas,
        } = warmup;

        let surface = SurfaceExchange::new_with_depth(
            &ctx,
            window.as_ref(),
            3,
            SurfaceConfig {
                present_mode: PresentMode::Fifo,
                depth_format: None,
            },
        )?;

        let uniform_buffer = retained_pool.acquire_buffer_with_data(
            &[Uniforms {
                width: surface.width(),
                height: surface.height(),
                time: 0.0,
                _padding: 0.0,
            }],
            BufferKind::Scattered,
        )?;

        let mut scheme = Scheme::new(&ctx);
        let present = record_scheme(
            &mut scheme,
            &surface,
            &compute_pipeline,
            &uniform_buffer,
            &verts,
            &blas,
            &tlas,
            surface.width(),
            surface.height(),
        )?;

        let mut upload_scheme = Scheme::new(&ctx);
        let uniform_deposit = MemoryExchange::new(&ctx).bind_deposit_buffer(
            &mut upload_scheme,
            &uniform_buffer,
            std::mem::size_of::<Uniforms>() as u64,
        )?;

        self.state = Some(RenderState {
            window,
            ctx,
            surface,
            present,
            scheme,
            compute_pipeline,
            _retained_pool: retained_pool,
            verts,
            blas,
            tlas,
            uniform_buffer,
            upload_scheme,
            uniform_deposit,
            start_time: std::time::Instant::now(),
            frame_count: 0,
        });
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
        if self.state.is_some() {
            return;
        }
        let attrs = common::hidden_window("Goldy — Compute Ray Query", INITIAL_WIDTH, INITIAL_HEIGHT);
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        if let Err(e) = self.init(window.clone()) {
            tracing::error!("Failed to initialize: {}", e);
            event_loop.exit();
            return;
        }
        if let Some(state) = &mut self.state {
            if let Err(e) = render_frame(state) {
                tracing::error!("First frame error: {e}");
            }
        }
        common::reveal_window(&window);
        window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            common::exit_if_timed_out(event_loop, state.start_time);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key.as_ref(), Key::Named(NamedKey::Escape)) {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(new_size) if new_size.width > 0 && new_size.height > 0 => {
                let _ = state.surface.resize(new_size.width, new_size.height);
                rebuild_scheme(state, new_size.width, new_size.height);
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = render_frame(state) {
                    tracing::error!("Render error: {}", e);
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn render_frame(state: &mut RenderState) -> Result<()> {
    state.frame_count += 1;
    let (width, height) = state.surface.size();
    if width == 0 || height == 0 {
        return Ok(());
    }
    let uniforms = Uniforms {
        width,
        height,
        time: state.start_time.elapsed().as_secs_f32(),
        _padding: 0.0,
    };
    state
        .uniform_deposit
        .write(&mut state.upload_scheme, 0, bytemuck::bytes_of(&uniforms))?;
    state.upload_scheme.submit()?;
    let mut submission = state.scheme.submit()?;
    state.present.claim(&mut submission)?.consume()?;
    Ok(())
}
