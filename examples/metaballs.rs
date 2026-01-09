//! Metaballs example - organic blob simulation.
//!
//! Demonstrates raymarching-like effect in fragment shader.
//!
//! Run with: cargo run --example metaballs

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, FrameOutput,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat,
    VertexBufferLayout, VertexAttribute, VertexFormat,
};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const METABALLS_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) time: f32,
}

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) time: f32
) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.uv = uv;
    out.time = time;
    return out;
}

fn metaball(p: vec2<f32>, center: vec2<f32>, radius: f32) -> f32 {
    let d = distance(p, center);
    return radius / (d * d + 0.001);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = (in.uv - 0.5) * 2.0;
    let t = in.time;
    
    // Moving metaball centers
    let c1 = vec2<f32>(sin(t * 1.1) * 0.5, cos(t * 0.9) * 0.5);
    let c2 = vec2<f32>(sin(t * 0.8 + 1.0) * 0.6, cos(t * 1.2 + 2.0) * 0.4);
    let c3 = vec2<f32>(sin(t * 1.3 + 2.0) * 0.4, cos(t * 0.7 + 1.0) * 0.6);
    let c4 = vec2<f32>(cos(t * 0.9) * 0.5, sin(t * 1.1 + 3.0) * 0.5);
    let c5 = vec2<f32>(cos(t * 1.0 + 1.5) * 0.3, sin(t * 0.8 + 0.5) * 0.7);
    
    // Sum metaball influences
    var v = 0.0;
    v += metaball(uv, c1, 0.15);
    v += metaball(uv, c2, 0.12);
    v += metaball(uv, c3, 0.18);
    v += metaball(uv, c4, 0.10);
    v += metaball(uv, c5, 0.14);
    
    // Threshold and color
    let threshold = 1.0;
    if v > threshold {
        // Inside blob - gradient based on intensity
        let intensity = (v - threshold) / 2.0;
        let r = 0.2 + intensity * 0.3;
        let g = 0.5 + intensity * 0.4;
        let b = 0.8 + intensity * 0.2;
        return vec4<f32>(r, g, b, 1.0);
    } else {
        // Outside - dark background with glow
        let glow = v * 0.3;
        return vec4<f32>(glow * 0.2, glow * 0.3, glow * 0.5, 1.0);
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MetaVertex {
    position: [f32; 2],
    uv: [f32; 2],
    time: f32,
}

impl MetaVertex {
    fn layout() -> VertexBufferLayout {
        VertexBufferLayout {
            stride: std::mem::size_of::<Self>() as u32,
            attributes: vec![
                VertexAttribute { location: 0, format: VertexFormat::Float32x2, offset: 0 },
                VertexAttribute { location: 1, format: VertexFormat::Float32x2, offset: 8 },
                VertexAttribute { location: 2, format: VertexFormat::Float32, offset: 16 },
            ],
        }
    }
}

fn create_quad(time: f32) -> [MetaVertex; 6] {
    [
        MetaVertex { position: [-1.0, -1.0], uv: [0.0, 1.0], time },
        MetaVertex { position: [1.0, -1.0], uv: [1.0, 1.0], time },
        MetaVertex { position: [1.0, 1.0], uv: [1.0, 0.0], time },
        MetaVertex { position: [-1.0, -1.0], uv: [0.0, 1.0], time },
        MetaVertex { position: [1.0, 1.0], uv: [1.0, 0.0], time },
        MetaVertex { position: [-1.0, 1.0], uv: [0.0, 0.0], time },
    ]
}

struct App {
    instance: Instance,
    device: Option<rag::Device>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    start_time: Instant,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None, pipeline: None, shader: None,
            window: None, surface: None,
            start_time: Instant::now(),
        })
    }

    fn init_gpu(&mut self) -> anyhow::Result<()> {
        let device = self.instance.create_device(DeviceType::DiscreteGpu)?;
        let shader = ShaderModule::from_wgsl(&device, METABALLS_SHADER)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: MetaVertex::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        })?;
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        let (width, height) = (size.width, size.height);
        if width == 0 || height == 0 { return Ok(()); }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let time = self.start_time.elapsed().as_secs_f32();

        let vertices = create_quad(time);
        let vertex_buffer = Buffer::with_data(device, &vertices, BufferUsage::VERTEX)?;

        let frame = FrameOutput::new(device, width, height, TextureFormat::Rgba8Unorm);
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..6, 0..1);
        }

        let output = frame.render(encoder)?;
        let surface = self.surface.as_mut().unwrap();
        surface.resize(NonZeroU32::new(width).unwrap(), NonZeroU32::new(height).unwrap())
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut buffer = surface.buffer_mut()
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        for (i, pixel) in buffer.iter_mut().enumerate() {
            let o = i * 4;
            if o + 2 < output.len() {
                *pixel = ((output[o] as u32) << 16) | ((output[o + 1] as u32) << 8) | (output[o + 2] as u32);
            }
        }
        buffer.present().map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(event_loop.create_window(
                Window::default_attributes()
                    .with_title("RAG - Metaballs")
                    .with_inner_size(winit::dpi::LogicalSize::new(800, 800))
            ).unwrap());
            let ctx = softbuffer::Context::new(window.clone()).unwrap();
            self.surface = Some(softbuffer::Surface::new(&ctx, window.clone()).unwrap());
            self.window = Some(window);
            self.init_gpu().unwrap();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) { event_loop.exit(); }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame().ok();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    println!("RAG Metaballs Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}

