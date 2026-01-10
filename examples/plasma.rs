//! Plasma example - classic demoscene plasma effect.
//!
//! Demonstrates complex fragment shader math with time animation.
//!
//! Run with: cargo run --example plasma

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

const PLASMA_SHADER: &str = r#"
struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
    float time : TEXCOORD1;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float time : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    output.time = input.time;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = input.uv * 4.0;
    float t = input.time;
    
    // Classic plasma formula
    float v = sin(uv.x + t);
    v += sin(uv.y + t);
    v += sin(uv.x + uv.y + t);
    
    float cx = uv.x + 0.5 * sin(t / 3.0);
    float cy = uv.y + 0.5 * cos(t / 2.0);
    v += sin(sqrt(cx * cx + cy * cy + 1.0) + t);
    
    v = v / 2.0;
    
    // Color palette
    float r = sin(v * 3.14159);
    float g = sin(v * 3.14159 + 2.094);
    float b = sin(v * 3.14159 + 4.188);
    
    return float4(r * 0.5 + 0.5, g * 0.5 + 0.5, b * 0.5 + 0.5, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PlasmaVertex {
    position: [f32; 2],
    uv: [f32; 2],
    time: f32,
}

impl PlasmaVertex {
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

fn create_quad(time: f32) -> [PlasmaVertex; 6] {
    [
        PlasmaVertex { position: [-1.0, -1.0], uv: [0.0, 1.0], time },
        PlasmaVertex { position: [1.0, -1.0], uv: [1.0, 1.0], time },
        PlasmaVertex { position: [1.0, 1.0], uv: [1.0, 0.0], time },
        PlasmaVertex { position: [-1.0, -1.0], uv: [0.0, 1.0], time },
        PlasmaVertex { position: [1.0, 1.0], uv: [1.0, 0.0], time },
        PlasmaVertex { position: [-1.0, 1.0], uv: [0.0, 0.0], time },
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
        let shader = ShaderModule::from_slang(&device, PLASMA_SHADER)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: PlasmaVertex::layout(),
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
                    .with_title("RAG - Plasma Effect")
                    .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
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
    println!("RAG Plasma Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}

