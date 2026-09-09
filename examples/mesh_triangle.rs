//! Mesh-shader triangle — `[goldy_mesh]` + `dispatch_mesh` with automatic payload linking.
//!
//! `MeshOutput` and `FsIn` use different struct names; Goldy links them by `SV_Position` / `COLOR`.
//!
//! Skips (exit 0) when `DeviceCapabilities::mesh_shaders` is false.
//!
//! Run with: cargo run --example mesh_triangle --features examples

use goldy::{
    Color, DeviceDescriptor, Instance, Lease, LeaseRenderTarget, MemoryExchange, MeshPipeline, RequestAdapterOptions,
    RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, Texture, TextureFormat,
    Transaction, WithdrawTransaction,
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
use common::{CaptureDump, FpsWindow};

const MESH_SLANG: &str = r#"
import goldy_exp;

struct MeshOutput {
    float4 pos : SV_Position;
    float4 color : COLOR;
};

[goldy_mesh]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void mesh_main(out vertices MeshOutput verts[3], out indices uint3 tris[1]) {
    SetMeshOutputCounts(3, 1);
    verts[0] = { float4(0.0, -0.5, 0.0, 1.0), float4(1.0, 0.0, 0.0, 1.0) };
    verts[1] = { float4(-0.5, 0.5, 0.0, 1.0), float4(0.0, 1.0, 0.0, 1.0) };
    verts[2] = { float4(0.5, 0.5, 0.0, 1.0), float4(0.0, 0.0, 1.0, 1.0) };
    tris[0] = uint3(0, 1, 2);
}

struct FsIn {
    float4 pos : SV_Position;
    float4 color : COLOR;
};

[goldy_fragment]
float4 fs_main(FsIn input) : SV_Target {
    return input.color;
}
"#;

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    _retained_pool: Option<RetainedPool>,
    pipeline: Option<MeshPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<SurfaceExchange>,
    present: Option<Transaction>,
    capture: Option<CaptureDump>,
    readback: Option<Texture>,
    withdraw: Option<WithdrawTransaction>,
    scene_rt: Option<Lease<LeaseRenderTarget>>,
    scheme: Option<Scheme>,
    frame_count: u64,
    perf_start: Option<Instant>,
    fps_window: FpsWindow,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            ctx: None,
            device: None,
            _retained_pool: None,
            pipeline: None,
            shader: None,
            window: None,
            surface: None,
            present: None,
            capture: None,
            readback: None,
            withdraw: None,
            scene_rt: None,
            scheme: None,
            frame_count: 0,
            perf_start: None,
            fps_window: FpsWindow::new(5.0),
        })
    }

    fn fps_window_secs() -> f64 {
        5.0
    }

    fn soak_secs() -> f64 {
        common::run_limit_secs().unwrap_or(60.0)
    }

    fn create_pipeline(
        device: &goldy::Device,
        shader: &ShaderModule,
        format: TextureFormat,
    ) -> anyhow::Result<MeshPipeline> {
        MeshPipeline::builder(device)
            .mesh(shader)
            .fragment(shader)
            .target_format(format)
            .build()
    }

    fn record_pass(scheme: &mut Scheme, pipeline: &MeshPipeline, scene_rt: &Lease<LeaseRenderTarget>, bg_color: Color) {
        let mut pass = scheme.render_pass("mesh", scene_rt, TargetLoad::Clear(bg_color));
        pass.set_mesh_pipeline(pipeline);
        pass.dispatch_mesh(1, 1, 1);
        pass.finish();
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

    fn init_gpu(&mut self, window: Option<&Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        if !device.capabilities().mesh_shaders {
            println!("skip: DeviceCapabilities::mesh_shaders is false on this adapter");
            std::process::exit(0);
        }
        let ctx = device.create_context()?;
        let mut retained_pool = RetainedPool::new(device.clone());

        let (surface, capture, readback, format, width, height) = if let Some(window) = window {
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

        let shader = ShaderModule::from_slang(&device, MESH_SLANG)?;
        let pipeline = Self::create_pipeline(&device, &shader, format)?;

        let mut scheme = Scheme::new(&ctx);
        let scene_rt = ctx.lease_render_target(width.max(1), height.max(1), format, None)?;
        let bg_color = Color {
            r: 0.1,
            g: 0.1,
            b: 0.2,
            a: 1.0,
        };
        Self::record_pass(&mut scheme, &pipeline, &scene_rt, bg_color);
        let (present, withdraw) = Self::bind_frame(&mut scheme, &scene_rt, surface.as_ref(), readback.as_ref())?;

        self.ctx = Some(ctx);
        self.device = Some(device);
        self._retained_pool = Some(retained_pool);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.surface = surface;
        self.present = present;
        self.capture = capture;
        self.readback = readback;
        self.withdraw = withdraw;
        self.scene_rt = Some(scene_rt);
        self.scheme = Some(scheme);
        self.perf_start = Some(Instant::now());
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        if let Some(window) = self.window.as_ref() {
            let size = window.inner_size();
            if size.width == 0 || size.height == 0 {
                return Ok(());
            }
        }
        let scheme = self.scheme.as_mut().unwrap();
        let mut submission = scheme.submit()?;
        if let Some(present) = &self.present {
            present.claim(&mut submission)?.consume()?;
        } else {
            let pixels = self.withdraw.as_ref().unwrap().claim(&mut submission)?.consume()?;
            self.capture.as_mut().unwrap().write_rgba(&pixels)?;
        }
        self.frame_count += 1;
        if self.perf_start.is_some() {
            self.fps_window.record(Instant::now());
        }
        Ok(())
    }

    fn capture_done(&self) -> bool {
        self.capture.as_ref().is_none_or(CaptureDump::finished)
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        let Some(surface) = self.surface.as_mut() else {
            return;
        };
        let _ = surface.resize(new_size.width, new_size.height);
        let format = surface.format();
        let (width, height) = surface.size();
        if let (Some(ctx), Some(device), Some(shader)) = (self.ctx.as_ref(), self.device.as_ref(), self.shader.as_ref())
        {
            if let Ok(pipeline) = Self::create_pipeline(device, shader, format) {
                self.pipeline = Some(pipeline);
                if let Some(pipeline) = self.pipeline.as_ref() {
                    let mut scheme = Scheme::new(ctx);
                    if let Ok(rt) = ctx.lease_render_target(width.max(1), height.max(1), format, None) {
                        let bg_color = Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.2,
                            a: 1.0,
                        };
                        Self::record_pass(&mut scheme, pipeline, &rt, bg_color);
                        if let Ok((present, withdraw)) =
                            Self::bind_frame(&mut scheme, &rt, self.surface.as_ref(), self.readback.as_ref())
                        {
                            self.present = present;
                            self.withdraw = withdraw;
                            self.scheme = Some(scheme);
                            self.scene_rt = Some(rt);
                        }
                    }
                }
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        let Some(perf_start) = self.perf_start else {
            return;
        };
        let now = Instant::now();
        let elapsed = perf_start.elapsed().as_secs_f64();
        let (window_frames, window_secs, fps) = self.fps_window.stats(now).unwrap_or((0, 0.0, 0.0));
        println!(
            "GOLDY_PERF: frames={} elapsed={elapsed:.2}s last_{:.0}s_fps={fps:.1} (window_frames={window_frames} window_secs={window_secs:.2} present=Auto soak={:.0}s)",
            self.frame_count,
            Self::fps_window_secs(),
            Self::soak_secs()
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(common::hidden_window("Goldy - Mesh Triangle", 800, 600))
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.init_gpu(Some(window.as_ref())).unwrap();
            if let Err(e) = self.render_frame() {
                tracing::error!("First frame error: {e}");
            }
            common::reveal_window(&window);
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        common::exit_if_timed_out(event_loop, self.perf_start.unwrap_or_else(Instant::now));
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
                self.window.as_ref().unwrap().request_redraw();
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

    if common::capture_requested() {
        let mut app = App::new()?;
        app.init_gpu(None)?;
        while !app.capture_done() {
            app.render_frame()?;
        }
        return Ok(());
    }

    println!("Goldy Mesh Triangle (set_mesh_pipeline + dispatch_mesh)");
    println!("Press Escape or close window to exit.\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
