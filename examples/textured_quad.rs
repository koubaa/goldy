//! Textured quad example - demonstrates texture sampling.
//!
//! Demonstrates retained scheme with offscreen render pass → copy-to-present.
//!
//! Run with: cargo run --example textured_quad

use goldy::{
    types::{AddressMode, FilterMode, SamplerDesc, TextureFlags, TextureFormat, TextureKind},
    BufferKind, Color, DeviceDescriptor, Grant, Instance, NodeAccess, Parcel, PresentGrant, RenderPipeline,
    RenderPipelineDesc, RenderTarget, RequestAdapterOptions, RetainedPool, Sampler, Scheme, ShaderModule,
    ShaderResourceSlot, SwapchainPool, Vertex2DUv,
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

const TEXTURED_SHADER: &str = r#"
import goldy_exp;

[goldy_vertex]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[goldy_fragment]
float4 fs_main(Interpolated<float4> tex, Filter smp, FullscreenVarying input) : SV_Target {
    return tex.Sample(smp, input.uv);
}
"#;

/// Generate a checkerboard texture in RGBA8 format
fn generate_checkerboard(width: u32, height: u32, checker_size: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        for x in 0..width {
            let checker_x = (x / checker_size) % 2;
            let checker_y = (y / checker_size) % 2;
            let is_white = (checker_x + checker_y).is_multiple_of(2);

            if is_white {
                data.extend_from_slice(&[255, 255, 255, 255]);
            } else {
                data.extend_from_slice(&[50, 100, 200, 255]);
            }
        }
    }

    data
}

const QUAD_VERTICES: [Vertex2DUv; 6] = [
    Vertex2DUv {
        position: [-1.0, -1.0],
        uv: [0.0, 1.0],
    },
    Vertex2DUv {
        position: [1.0, -1.0],
        uv: [1.0, 1.0],
    },
    Vertex2DUv {
        position: [1.0, 1.0],
        uv: [1.0, 0.0],
    },
    Vertex2DUv {
        position: [-1.0, -1.0],
        uv: [0.0, 1.0],
    },
    Vertex2DUv {
        position: [1.0, 1.0],
        uv: [1.0, 0.0],
    },
    Vertex2DUv {
        position: [-1.0, 1.0],
        uv: [0.0, 0.0],
    },
];

struct App {
    instance: Instance,
    ctx: Option<goldy::Context>,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    swapchain: Option<SwapchainPool>,
    screen: Option<goldy::PresentLease>,
    present: Option<PresentGrant>,
    scene_rt: Option<RenderTarget>,
    scheme: Option<Scheme>,
    _retained_pool: Option<RetainedPool>,
    vertex_buffer: Option<Parcel>,
    texture: Option<Parcel>,
    sampler: Option<Sampler>,
    start_time: Instant,
    frame_count: u64,
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
            swapchain: None,
            screen: None,
            present: None,
            scene_rt: None,
            scheme: None,
            _retained_pool: None,
            vertex_buffer: None,
            texture: None,
            sampler: None,
            start_time: Instant::now(),
            frame_count: 0,
        })
    }

    fn create_scene_rt(device: &goldy::Device, swapchain: &SwapchainPool) -> anyhow::Result<RenderTarget> {
        let (width, height) = swapchain.size();
        RenderTarget::new(device, width.max(1), height.max(1), swapchain.format())
    }

    fn create_pipeline(
        device: &goldy::Device,
        shader: &ShaderModule,
        swapchain: &SwapchainPool,
    ) -> anyhow::Result<RenderPipeline> {
        common::render_pipeline_for_swapchain(
            device,
            shader,
            swapchain,
            RenderPipelineDesc {
                vertex_layout: Vertex2DUv::layout(),
                ..Default::default()
            },
        )
    }

    fn record_scheme(
        scheme: &mut Scheme,
        pipeline: &RenderPipeline,
        vertex_buffer: &Parcel,
        texture: &Parcel,
        sampler: &Sampler,
        scene_rt: &RenderTarget,
        screen: &goldy::PresentLease,
    ) -> PresentGrant {
        let shader_resources = [
            ShaderResourceSlot::Parcel {
                parcel: texture,
                access: NodeAccess::Read,
            },
            ShaderResourceSlot::Sampler(sampler),
        ];

        let mut pass = scheme.render_pass("textured_quad", scene_rt);
        pass.bind_shader_resources(&shader_resources);
        pass.bind_parcel_mut(vertex_buffer, NodeAccess::Read);
        pass.clear(Color {
            r: 0.1,
            g: 0.1,
            b: 0.15,
            a: 1.0,
        });
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_buffer);
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

        let shader = ShaderModule::from_slang(&device, TEXTURED_SHADER)?;

        let tex_width = 256u32;
        let tex_height = 256u32;
        let checker_data = generate_checkerboard(tex_width, tex_height, 32);

        let mut retained_pool = RetainedPool::new(device.clone());
        let texture = retained_pool.acquire_texture(
            tex_width,
            tex_height,
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
            Some(&checker_data),
        )?;

        let sampler = Sampler::new(
            &device,
            &SamplerDesc {
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Nearest,
                address_mode_u: AddressMode::Repeat,
                address_mode_v: AddressMode::Repeat,
                address_mode_w: AddressMode::Repeat,
                max_anisotropy: 1.0,
                compare: None,
                lod_min_clamp: 0.0,
                lod_max_clamp: 32.0,
            },
        )?;

        let pipeline = Self::create_pipeline(&device, &shader, &swapchain)?;

        let vertex_buffer = retained_pool.acquire_buffer_with_data(&QUAD_VERTICES, BufferKind::Scattered)?;
        let scene_rt = Self::create_scene_rt(&device, &swapchain)?;

        let mut scheme = Scheme::new(&ctx);
        let present = Self::record_scheme(
            &mut scheme,
            &pipeline,
            &vertex_buffer,
            &texture,
            &sampler,
            &scene_rt,
            &screen,
        );

        self.ctx = Some(ctx);
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.swapchain = Some(swapchain);
        self.screen = Some(screen);
        self.present = Some(present);
        self.scene_rt = Some(scene_rt);
        self.scheme = Some(scheme);
        self._retained_pool = Some(retained_pool);
        self.vertex_buffer = Some(vertex_buffer);
        self.texture = Some(texture);
        self.sampler = Some(sampler);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

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
            if let (Some(device), Some(swapchain), Some(shader)) = (&self.device, &self.swapchain, &self.shader) {
                if let Ok(rt) = Self::create_scene_rt(device, swapchain) {
                    if let Ok(pipeline) = Self::create_pipeline(device, shader, swapchain) {
                        self.pipeline = Some(pipeline);
                        if let (
                            Some(scheme),
                            Some(pipeline),
                            Some(vertex_buffer),
                            Some(texture),
                            Some(sampler),
                            Some(screen),
                        ) = (
                            self.scheme.as_mut(),
                            self.pipeline.as_ref(),
                            self.vertex_buffer.as_ref(),
                            self.texture.as_ref(),
                            self.sampler.as_ref(),
                            self.screen.as_ref(),
                        ) {
                            scheme.begin_rerecord();
                            let present =
                                Self::record_scheme(scheme, pipeline, vertex_buffer, texture, sampler, &rt, screen);
                            self.present = Some(present);
                        }
                        self.scene_rt = Some(rt);
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
                            .with_title("Goldy - Textured Quad (Scheme + Present)")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 800)),
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
    println!("Goldy Textured Quad Example (Scheme + Present)");
    println!("Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
