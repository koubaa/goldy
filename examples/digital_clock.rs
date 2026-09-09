//! Digital Clock example - render an animated 7-segment clock in a window.
//!
//! Uses retained scheme with offscreen render pass → copy-to-present.
//!
//! Run with: `cargo run --example digital_clock`

mod digital_clock_shared;

use digital_clock_shared::{generate_clock_vertices, ClockState, ClockVertex, TimeData};
use goldy::{
    Buffer, BufferFlags, BufferKind, Color, DepositTransaction, DeviceDescriptor, Instance, Lease, LeaseRenderTarget,
    MemoryExchange, NodeAccess, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule, TargetLoad, TextureFormat, VertexBufferLayout, VertexFormat,
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
use common::FrameSink;

/// Upper bound on seven-segment clock vertices (8 glyphs × 7 segments × 6 verts).
const MAX_CLOCK_VERTICES: usize = 384;

const SHADER_SOURCE: &str = r#"
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
"#;

fn clock_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout::from_formats::<ClockVertex>(&[VertexFormat::Float32x2, VertexFormat::Float32x4])
}

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    _retained_pool: Option<RetainedPool>,
    vertex_parcel: Option<Buffer>,
    upload_scheme: Option<Scheme>,
    vertex_deposit: Option<DepositTransaction>,

    window: Option<Arc<Window>>,
    sink: Option<FrameSink>,
    scene_rt: Option<Lease<LeaseRenderTarget>>,
    scheme: Option<Scheme>,

    start_time: Instant,
    perf_start: Instant,
    frame_count: u32,
    clock_state: ClockState,
    recorded_vertex_count: u32,
    recorded_bg_color: Color,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let instance = Instance::new()?;
        Ok(Self {
            instance,
            ctx: None,
            device: None,
            pipeline: None,
            shader: None,
            window: None,
            sink: None,
            scene_rt: None,
            scheme: None,
            start_time: Instant::now(),
            perf_start: Instant::now(),
            frame_count: 0,
            clock_state: ClockState::default(),
            _retained_pool: None,
            vertex_parcel: None,
            upload_scheme: None,
            vertex_deposit: None,
            recorded_vertex_count: 0,
            recorded_bg_color: Color::BLACK,
        })
    }

    fn create_pipeline(
        device: &goldy::Device,
        shader: &ShaderModule,
        format: TextureFormat,
    ) -> anyhow::Result<RenderPipeline> {
        common::render_pipeline(
            device,
            shader,
            format,
            RenderPipelineDesc {
                vertex_layout: clock_vertex_layout(),
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        sink: &mut FrameSink,
        pipeline: &RenderPipeline,
        vertex_parcel: &Buffer,
        vertex_count: u32,
        bg_color: Color,
        scene_rt: &Lease<LeaseRenderTarget>,
    ) -> anyhow::Result<()> {
        let mut pass = scheme.render_pass("digital_clock", scene_rt, TargetLoad::Clear(bg_color));
        pass.with_parcel(vertex_parcel, NodeAccess::Read);
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_parcel);
        pass.draw(0..vertex_count, 0..1);
        pass.finish();
        sink.bind_render_target(scheme, scene_rt)
    }

    fn rerecord_scheme_if_needed(&mut self, vertex_count: u32, bg_color: Color) {
        if vertex_count == self.recorded_vertex_count && bg_color == self.recorded_bg_color {
            return;
        }
        if let (Some(ctx), Some(pipeline), Some(vertex_parcel), Some(sink)) = (
            self.ctx.as_ref(),
            self.pipeline.as_ref(),
            self.vertex_parcel.as_ref(),
            self.sink.as_mut(),
        ) {
            let mut scheme = Scheme::new(ctx);
            let (width, height) = sink.size();
            if let Ok(rt) = ctx.lease_render_target(width.max(1), height.max(1), sink.format(), None) {
                if Self::record_scheme(&mut scheme, sink, pipeline, vertex_parcel, vertex_count, bg_color, &rt).is_ok()
                {
                    self.scheme = Some(scheme);
                    self.recorded_vertex_count = vertex_count;
                    self.recorded_bg_color = bg_color;
                    self.scene_rt = Some(rt);
                }
            }
        }
    }

    fn init_gpu(&mut self, window: Option<&Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let mut retained_pool = RetainedPool::new(device.clone());
        let mut sink = FrameSink::open(&ctx, &mut retained_pool, window)?;

        let shader = ShaderModule::from_slang(&device, SHADER_SOURCE)?;
        let pipeline = Self::create_pipeline(&device, &shader, sink.format())?;

        let vertex_parcel = retained_pool.acquire_buffer_sized::<ClockVertex>(
            MAX_CLOCK_VERTICES as u64,
            BufferKind::Scattered,
            BufferFlags::empty(),
        )?;

        let bg_color = self.clock_state.background_color();
        let mut scheme = Scheme::new(&ctx);
        let (width, height) = sink.size();
        let scene_rt = ctx.lease_render_target(width.max(1), height.max(1), sink.format(), None)?;
        Self::record_scheme(
            &mut scheme,
            &mut sink,
            &pipeline,
            &vertex_parcel,
            1,
            bg_color,
            &scene_rt,
        )?;

        self.ctx = Some(ctx);
        let ctx = self.ctx.as_ref().unwrap();
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self._retained_pool = Some(retained_pool);
        self.vertex_parcel = Some(vertex_parcel);
        let vertex_parcel = self.vertex_parcel.as_ref().unwrap();
        let mut upload_scheme = Scheme::new(ctx);
        let vertex_deposit = MemoryExchange::new(ctx).bind_deposit_buffer(
            &mut upload_scheme,
            vertex_parcel,
            (MAX_CLOCK_VERTICES * std::mem::size_of::<ClockVertex>()) as u64,
        )?;
        self.upload_scheme = Some(upload_scheme);
        self.vertex_deposit = Some(vertex_deposit);
        self.sink = Some(sink);
        self.scene_rt = Some(scene_rt);
        self.scheme = Some(scheme);
        self.recorded_vertex_count = 1;
        self.recorded_bg_color = bg_color;
        Ok(())
    }

    fn elapsed_secs(&self) -> u64 {
        if let Some(sink) = self.sink.as_ref() {
            if sink.is_capture() {
                return sink.time(self.start_time) as u64;
            }
        }
        if self.clock_state.paused {
            self.clock_state.accumulated_secs
        } else {
            self.start_time.elapsed().as_secs() + self.clock_state.accumulated_secs
        }
    }

    fn toggle_pause(&mut self) {
        let current = self.elapsed_secs();
        if self.clock_state.paused {
            self.start_time = Instant::now();
        }
        self.clock_state.toggle_pause(current);
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        self.frame_count += 1;

        let (width, height) = if let Some(window) = self.window.as_ref() {
            let size = window.inner_size();
            (size.width, size.height)
        } else {
            self.sink.as_ref().unwrap().size()
        };

        if width == 0 || height == 0 {
            return Ok(());
        }

        let elapsed = self.elapsed_secs();
        let time = TimeData::from_elapsed_secs(elapsed);
        let color = self.clock_state.color();
        let bg_color = self.clock_state.background_color();
        let vertices = generate_clock_vertices(time, color, width, height);
        let vertex_count = vertices.len() as u32;

        self.rerecord_scheme_if_needed(vertex_count, bg_color);

        let upload = self.upload_scheme.as_mut().unwrap();
        self.vertex_deposit
            .unwrap()
            .write(upload, 0, bytemuck::cast_slice(&vertices))?;
        upload.submit()?;

        let scheme = self.scheme.as_mut().unwrap();
        let mut submission = scheme.submit()?;
        self.sink.as_mut().unwrap().settle(&mut submission)?;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(sink) = self.sink.as_mut() {
                let _ = sink.resize(new_size.width, new_size.height);
            }
            if let (Some(ctx), Some(device), Some(sink), Some(shader), Some(vertex_parcel)) = (
                self.ctx.as_ref(),
                self.device.as_ref(),
                self.sink.as_mut(),
                self.shader.as_ref(),
                self.vertex_parcel.as_ref(),
            ) {
                if let Ok(pipeline) = Self::create_pipeline(device, shader, sink.format()) {
                    self.pipeline = Some(pipeline);
                    if let Some(pipeline) = self.pipeline.as_ref() {
                        let bg_color = self.clock_state.background_color();
                        let vertex_count = self.recorded_vertex_count.max(1);
                        let mut scheme = Scheme::new(ctx);
                        let (width, height) = sink.size();
                        if let Ok(rt) = ctx.lease_render_target(width.max(1), height.max(1), sink.format(), None) {
                            if Self::record_scheme(
                                &mut scheme,
                                sink,
                                pipeline,
                                vertex_parcel,
                                vertex_count,
                                bg_color,
                                &rt,
                            )
                            .is_ok()
                            {
                                self.scheme = Some(scheme);
                                self.recorded_vertex_count = vertex_count;
                                self.recorded_bg_color = bg_color;
                                self.scene_rt = Some(rt);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        let elapsed = self.perf_start.elapsed().as_secs_f64();
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
            let attrs = common::hidden_window(
                "Goldy - Clock (Scheme + Present, Space: pause, Click: color)",
                1280,
                720,
            );

            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.window = Some(window.clone());

            if let Err(e) = self.init_gpu(Some(window.as_ref())) {
                tracing::error!("Failed to initialize GPU: {}", e);
            } else if let Err(e) = self.render_frame() {
                tracing::error!("First frame error: {e}");
            }
            common::reveal_window(&window);
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        common::exit_if_timed_out(event_loop, self.perf_start);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => match event.logical_key {
                Key::Named(NamedKey::Escape) => event_loop.exit(),
                Key::Named(NamedKey::Space) => self.toggle_pause(),
                Key::Character(ref c) if c == "c" || c == "C" => self.clock_state.next_color(),
                _ => {}
            },
            WindowEvent::MouseInput { state, .. } if state.is_pressed() => {
                self.clock_state.next_color();
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

    if common::capture_requested() {
        let mut app = App::new()?;
        app.init_gpu(None)?;
        while !app.sink.as_ref().is_none_or(common::FrameSink::finished) {
            app.render_frame()?;
        }
        return Ok(());
    }

    println!("Goldy Clock Example (retained scheme)");
    println!("==================================================================");
    println!("Controls:");
    println!("  Space - Toggle pause");
    println!("  Click - Change color");
    println!("  Escape - Exit\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
