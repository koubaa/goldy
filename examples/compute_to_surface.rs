//! Compute-to-Surface example — pure compute rendering without a graphics pipeline.
//!
//! This example demonstrates the new Surface API where a compute shader
//! writes directly to the swapchain texture via `frame.texture()`.
//! No `RenderPipeline`, no `CommandEncoder`, no vertex buffers.
//!
//! Run with: cargo run --example compute_to_surface

use anyhow::Result;
use goldy::{
    Buffer, ComputeEncoder, ComputePipeline, DataAccess, DeviceType, Instance, PresentMode,
    ShaderModule, Surface, SurfaceConfig,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    time: f32,
    _padding: f32,
}

const COMPUTE_SHADER: &str = r#"
import goldy_exp;

struct Uniforms {
    uint width;
    uint height;
    float time;
    float _padding;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    ReadOnlyBuffer<Uniforms> ub = goldy_dyn_buf_ro<Uniforms>(0);
    Uniforms u = ub[0];

    RWTexture2D<float4> output = goldy_dyn_direct_spatial<float4>(1);

    if (tid.x >= u.width || tid.y >= u.height)
        return;

    float2 uv = float2(float(tid.x) / float(u.width),
                       float(tid.y) / float(u.height));
    float2 p = uv * 2.0 - 1.0;
    p.x *= float(u.width) / float(u.height);

    float t = u.time;
    float v = 0.0;
    v += sin(p.x * 6.0 + t);
    v += sin(p.y * 6.0 + t * 1.3);
    v += sin((p.x + p.y) * 4.0 + t * 0.7);
    v += sin(length(p) * 8.0 - t * 2.0);
    v *= 0.25;

    float3 col = float3(0.5 + 0.5 * sin(v * 3.14159 + 0.0),
                        0.5 + 0.5 * sin(v * 3.14159 + 2.094),
                        0.5 + 0.5 * sin(v * 3.14159 + 4.188));
    output[tid.xy] = float4(col, 1.0);
}
"#;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("Goldy — Compute to Surface Example");
    println!("===================================");
    println!("Press V to toggle vsync, Escape to exit\n");

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
    surface: Surface,
    compute_pipeline: ComputePipeline,
    uniform_buffer: Buffer,
    start_time: std::time::Instant,
    vsync: bool,
}

impl App {
    fn init(&mut self, window: Arc<Window>) -> Result<()> {
        let instance = Instance::new()?;
        let device = Arc::new(instance.create_device(DeviceType::DiscreteGpu)?);

        let surface = Surface::new_with_config(
            &device,
            window.as_ref(),
            SurfaceConfig {
                present_mode: PresentMode::Fifo,
                depth_format: None,
            },
        )?;

        let shader = ShaderModule::from_slang(&device, COMPUTE_SHADER)?;
        let compute_pipeline = ComputePipeline::new(&device, &shader)?;

        // NOTE: pass a typed `&[Uniforms]`, NOT `bytemuck::bytes_of(&u)`.
        // `Buffer::with_data::<T>` uses `size_of::<T>()` as the structured-buffer
        // stride. Passing `&[u8]` gave stride=1, which on DX12 caused
        // `rawBufferLoad` of any field past offset 0 to return 0 (out-of-bounds
        // for a stride-1 structured buffer) — turning `u.height` into 0 and
        // triggering the `if (tid.y >= u.height) return;` guard for every
        // thread on the `compute_to_surface` path.
        let uniform_buffer = Buffer::with_data(
            &device,
            &[Uniforms {
                width: surface.width(),
                height: surface.height(),
                time: 0.0,
                _padding: 0.0,
            }],
            DataAccess::Scattered,
        )?;

        self.state = Some(RenderState {
            window,
            device,
            surface,
            compute_pipeline,
            uniform_buffer,
            start_time: std::time::Instant::now(),
            vsync: true,
        });

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Goldy — Compute to Surface")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        if let Err(e) = self.init(window.clone()) {
            tracing::error!("Failed to initialize: {}", e);
            event_loop.exit();
            return;
        }

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Character("v") => {
                        state.vsync = !state.vsync;
                        let mode = if state.vsync {
                            PresentMode::Fifo
                        } else {
                            PresentMode::Immediate
                        };
                        if let Err(e) = state.surface.set_present_mode(mode) {
                            eprintln!("Failed to set present mode: {e}");
                        } else {
                            println!(
                                "Vsync: {} (present mode: {:?})",
                                if state.vsync { "ON" } else { "OFF" },
                                mode
                            );
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    let _ = state.surface.resize(new_size.width, new_size.height);
                    state.window.request_redraw();
                }
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
    let (width, height) = state.surface.size();
    if width == 0 || height == 0 {
        return Ok(());
    }

    let elapsed = state.start_time.elapsed().as_secs_f32();

    // Update uniforms
    state.uniform_buffer.write(
        0,
        bytemuck::bytes_of(&Uniforms {
            width,
            height,
            time: elapsed,
            _padding: 0.0,
        }),
    )?;

    // Acquire the surface frame and get its texture
    let frame = state.surface.acquire()?;
    let texture = frame
        .texture()
        .expect("Backend does not support surface frame textures");

    // Dispatch the compute shader to write directly to the surface texture
    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);

    // The shader accesses the uniform buffer via `goldy_dyn_buf_ro<Uniforms>(0)` which
    // maps to `StructuredBuffer<Uniforms>` — an SRV on DX12. `bindless_srv_handle()`
    // returns the SRV index on DX12 (distinct from the UAV index) and falls back to
    // the unified storage-buffer index on Vulkan / Metal, so it's correct on every
    // backend.
    let uniform_handle = state
        .uniform_buffer
        .bindless_srv_handle()
        .expect("Uniform buffer has no bindless SRV handle");
    let texture_handle = texture
        .bindless_handle()
        .expect("Surface texture has no bindless handle");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&state.compute_pipeline);
        pass.set_push_constants_typed(&[uniform_handle, texture_handle]);
        pass.dispatch(wg_x, wg_y, 1);
    }
    encoder.submit(&state.device)?.wait()?;

    // Present — the compute shader already wrote the pixels
    frame.present()?;

    Ok(())
}
