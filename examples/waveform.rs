//! Waveform example - animated audio waveform visualizer.
//!
//! Demonstrates retained scheme with offscreen render pass → copy-to-present.
//!
//! Run with: `cargo run --example waveform`

use goldy::{
    Buffer, BufferFlags, BufferKind, Color, DepositTransaction, DeviceDescriptor, Instance, Lease, LeaseRenderTarget,
    MemoryExchange, NodeAccess, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions,
    RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange, TargetLoad, Transaction, Vertex2D,
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

const NUM_SAMPLES: usize = 200;
const NUM_CHANNELS: usize = 4;

fn generate_waveform(time: f32, channel: usize) -> Vec<f32> {
    let freq = 1.0 + channel as f32 * 0.5;
    let phase = channel as f32 * 0.7;

    (0..NUM_SAMPLES)
        .map(|i| {
            let x = i as f32 / NUM_SAMPLES as f32 * 6.0;
            let mut y = 0.0;

            // Superposition of different frequencies
            y += (x * freq + time * 2.0 + phase).sin() * 0.3;
            y += (x * freq * 2.3 + time * 1.7 + phase).sin() * 0.2;
            y += (x * freq * 3.7 + time * 0.9 + phase).cos() * 0.15;
            y += (x * freq * 5.1 + time * 2.3 + phase).sin() * 0.1;

            // Add some noise
            let noise = ((i as f32 * 1234.5 + time * 100.0).sin() * 43_758.547_f32).fract() - 0.5;
            y += noise * 0.05;

            y.clamp(-1.0, 1.0)
        })
        .collect()
}

fn waveform_to_vertices(samples: &[f32], y_offset: f32, color: Color) -> Vec<Vertex2D> {
    let scale_y = 0.15;
    samples
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let x = (i as f32 / (NUM_SAMPLES - 1) as f32) * 1.9 - 0.95;
            let y = y_offset + sample * scale_y;
            Vertex2D::new(x, y, color)
        })
        .collect()
}

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    _retained_pool: Option<RetainedPool>,
    channel_parcels: Option<[Buffer; NUM_CHANNELS]>,
    upload_scheme: Option<Scheme>,
    channel_deposits: Option<[DepositTransaction; NUM_CHANNELS]>,
    window: Option<Arc<Window>>,
    surface: Option<SurfaceExchange>,
    present: Option<Transaction>,
    scene_rt: Option<Lease<LeaseRenderTarget>>,
    scheme: Option<Scheme>,
    start_time: Instant,
    frame_count: u32,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            ctx: None,
            device: None,
            pipeline: None,
            shader: None,
            window: None,
            surface: None,
            present: None,
            scene_rt: None,
            scheme: None,
            start_time: Instant::now(),
            frame_count: 0,
            _retained_pool: None,
            channel_parcels: None,
            upload_scheme: None,
            channel_deposits: None,
        })
    }

    fn create_pipeline(
        device: &goldy::Device,
        shader: &ShaderModule,
        surface: &SurfaceExchange,
    ) -> anyhow::Result<RenderPipeline> {
        common::render_pipeline_for_surface(
            device,
            shader,
            surface,
            RenderPipelineDesc {
                vertex_layout: Vertex2D::layout(),
                topology: PrimitiveTopology::LineStrip,
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        surface: &SurfaceExchange,
        pipeline: &RenderPipeline,
        channel_parcels: &[Buffer; NUM_CHANNELS],
        scene_rt: &Lease<LeaseRenderTarget>,
    ) -> anyhow::Result<Transaction> {
        let mut pass = scheme.render_pass(
            "waveform",
            scene_rt,
            TargetLoad::Clear(Color {
                r: 0.02,
                g: 0.02,
                b: 0.08,
                a: 1.0,
            }),
        );
        for parcel in channel_parcels {
            pass.with_parcel(parcel, NodeAccess::Read);
        }

        pass.set_pipeline(pipeline);
        for parcel in channel_parcels {
            pass.set_vertex_buffer(0, parcel);
            pass.draw(0..NUM_SAMPLES as u32, 0..1);
        }
        pass.finish();
        surface.bind_render_target(scheme, scene_rt).map_err(Into::into)
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let surface = SurfaceExchange::new(&ctx, window.as_ref(), SurfaceConfig::default())?;

        let shader = ShaderModule::from_slang(&device, goldy::shader::builtins::VERTEX_COLOR_2D)?;
        let pipeline = Self::create_pipeline(&device, &shader, &surface)?;

        let mut retained_pool = RetainedPool::new(device.clone());
        let channel_parcels = std::array::from_fn(|_| {
            retained_pool
                .acquire_buffer_sized::<Vertex2D>(NUM_SAMPLES as u64, BufferKind::Scattered, BufferFlags::empty())
                .expect("waveform channel parcel")
        });

        let mut scheme = Scheme::new(&ctx);
        let (width, height) = surface.size();
        let scene_rt = scheme.lease_render_target(width.max(1), height.max(1), surface.format(), None)?;
        let present = Self::record_scheme(&mut scheme, &surface, &pipeline, &channel_parcels, &scene_rt)?;

        self.ctx = Some(ctx);
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self._retained_pool = Some(retained_pool);
        self.channel_parcels = Some(channel_parcels);
        let channel_parcels = self.channel_parcels.as_ref().unwrap();
        let mut upload_scheme = Scheme::new(&ctx);
        let memory = MemoryExchange::new(&ctx);
        let channel_capacity = channel_parcels[0].byte_size();
        let channel_deposits = std::array::from_fn(|ch| {
            memory
                .bind_deposit_buffer(&mut upload_scheme, &channel_parcels[ch], channel_capacity)
                .expect("bind channel deposit")
        });
        self.upload_scheme = Some(upload_scheme);
        self.channel_deposits = Some(channel_deposits);
        self.surface = Some(surface);
        self.present = Some(present);
        self.scene_rt = Some(scene_rt);
        self.scheme = Some(scheme);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        self.frame_count += 1;

        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let ctx = self.ctx.as_ref().unwrap();
        let channel_parcels = self.channel_parcels.as_ref().unwrap();
        let scheme = self.scheme.as_mut().unwrap();
        let time = self.start_time.elapsed().as_secs_f32();

        let colors = [
            Color {
                r: 1.0,
                g: 0.3,
                b: 0.3,
                a: 1.0,
            },
            Color {
                r: 0.3,
                g: 1.0,
                b: 0.3,
                a: 1.0,
            },
            Color {
                r: 0.3,
                g: 0.5,
                b: 1.0,
                a: 1.0,
            },
            Color {
                r: 1.0,
                g: 0.8,
                b: 0.2,
                a: 1.0,
            },
        ];
        let y_offsets = [0.6, 0.2, -0.2, -0.6];

        let upload = self.upload_scheme.as_mut().unwrap();
        let channel_deposits = self.channel_deposits.as_ref().unwrap();
        for ch in 0..NUM_CHANNELS {
            let samples = generate_waveform(time, ch);
            let vertices = waveform_to_vertices(&samples, y_offsets[ch], colors[ch]);
            channel_deposits[ch].write(upload, 0, bytemuck::cast_slice(&vertices))?;
        }
        upload.submit()?;

        let present = self.present.as_ref().unwrap();
        let mut submission = scheme.submit()?;
        present.claim(&mut submission)?.consume()?;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &self.surface {
                let _ = surface.resize(new_size.width, new_size.height);
            }
            if let (Some(device), Some(surface), Some(shader)) = (&self.device, &self.surface, &self.shader) {
                if let Ok(pipeline) = Self::create_pipeline(device, shader, surface) {
                    self.pipeline = Some(pipeline);
                    if let (Some(ctx), Some(pipeline), Some(channel_parcels), Some(surface)) = (
                        self.ctx.as_ref(),
                        self.pipeline.as_ref(),
                        self.channel_parcels.as_ref(),
                        self.surface.as_ref(),
                    ) {
                        let mut scheme = Scheme::new(ctx);

                        let (width, height) = surface.size();

                        if let Ok(rt) = scheme.lease_render_target(width.max(1), height.max(1), surface.format(), None)
                        {
                            if let Ok(present) =
                                Self::record_scheme(&mut scheme, surface, pipeline, channel_parcels, &rt)
                            {
                                self.scheme = Some(scheme);
                                self.present = Some(present);
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
                            .with_title("Goldy - Waveform Visualizer (Scheme + Present)")
                            .with_inner_size(winit::dpi::LogicalSize::new(1024, 600)),
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

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
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
    println!("Goldy Waveform Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
