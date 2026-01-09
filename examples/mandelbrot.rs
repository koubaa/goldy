//! Mandelbrot example - interactive fractal explorer.
//!
//! Demonstrates complex math in fragment shader with zoom/pan.
//!
//! Run with: cargo run --example mandelbrot

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, FrameOutput,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat,
    VertexBufferLayout, VertexAttribute, VertexFormat,
};
use std::num::NonZeroU32;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const MANDELBROT_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) complex_coord: vec2<f32>,
}

struct Params {
    center_x: f32,
    center_y: f32,
    zoom: f32,
    max_iter: f32,
}

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) center: vec2<f32>,
    @location(2) zoom: f32,
    @location(3) uv: vec2<f32>
) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    // Map UV to complex plane around center with zoom
    out.complex_coord = center + (uv - 0.5) * 4.0 / zoom;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = in.complex_coord;
    var z = vec2<f32>(0.0, 0.0);
    var i: u32 = 0u;
    let max_iter: u32 = 256u;
    
    loop {
        if i >= max_iter { break; }
        if dot(z, z) > 4.0 { break; }
        
        // z = z^2 + c
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        i = i + 1u;
    }
    
    if i >= max_iter {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    // Smooth coloring
    let t = f32(i) / f32(max_iter);
    let r = sin(t * 5.0) * 0.5 + 0.5;
    let g = sin(t * 7.0 + 1.0) * 0.5 + 0.5;
    let b = sin(t * 11.0 + 2.0) * 0.5 + 0.5;
    
    return vec4<f32>(r, g, b, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MandelbrotVertex {
    position: [f32; 2],
    center: [f32; 2],
    zoom: f32,
    uv: [f32; 2],
}

impl MandelbrotVertex {
    fn layout() -> VertexBufferLayout {
        VertexBufferLayout {
            stride: std::mem::size_of::<Self>() as u32,
            attributes: vec![
                VertexAttribute { location: 0, format: VertexFormat::Float32x2, offset: 0 },
                VertexAttribute { location: 1, format: VertexFormat::Float32x2, offset: 8 },
                VertexAttribute { location: 2, format: VertexFormat::Float32, offset: 16 },
                VertexAttribute { location: 3, format: VertexFormat::Float32x2, offset: 20 },
            ],
        }
    }
}

fn create_quad(center: [f32; 2], zoom: f32) -> [MandelbrotVertex; 6] {
    [
        MandelbrotVertex { position: [-1.0, -1.0], center, zoom, uv: [0.0, 1.0] },
        MandelbrotVertex { position: [1.0, -1.0], center, zoom, uv: [1.0, 1.0] },
        MandelbrotVertex { position: [1.0, 1.0], center, zoom, uv: [1.0, 0.0] },
        MandelbrotVertex { position: [-1.0, -1.0], center, zoom, uv: [0.0, 1.0] },
        MandelbrotVertex { position: [1.0, 1.0], center, zoom, uv: [1.0, 0.0] },
        MandelbrotVertex { position: [-1.0, 1.0], center, zoom, uv: [0.0, 0.0] },
    ]
}

struct App {
    instance: Instance,
    device: Option<rag::Device>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    center: [f32; 2],
    zoom: f32,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None, pipeline: None, shader: None,
            window: None, surface: None,
            center: [-0.5, 0.0],
            zoom: 1.0,
        })
    }

    fn init_gpu(&mut self) -> anyhow::Result<()> {
        let device = self.instance.create_device(DeviceType::DiscreteGpu)?;
        let shader = ShaderModule::from_wgsl(&device, MANDELBROT_SHADER)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: MandelbrotVertex::layout(),
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

        let vertices = create_quad(self.center, self.zoom);
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
                    .with_title("RAG - Mandelbrot (Arrows=pan, +/-=zoom, R=reset)")
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
                let pan = 0.1 / self.zoom;
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::ArrowUp) => self.center[1] += pan,
                    Key::Named(NamedKey::ArrowDown) => self.center[1] -= pan,
                    Key::Named(NamedKey::ArrowLeft) => self.center[0] -= pan,
                    Key::Named(NamedKey::ArrowRight) => self.center[0] += pan,
                    Key::Character(ref c) if c == "=" || c == "+" => self.zoom *= 1.5,
                    Key::Character(ref c) if c == "-" => self.zoom /= 1.5,
                    Key::Character(ref c) if c == "r" || c == "R" => {
                        self.center = [-0.5, 0.0];
                        self.zoom = 1.0;
                    }
                    _ => {}
                }
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame().ok();
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    println!("RAG Mandelbrot Example");
    println!("  Arrows - Pan");
    println!("  +/- - Zoom in/out");
    println!("  R - Reset view");
    println!("  Escape - Exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}

