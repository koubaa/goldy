//! Instancing example - render many objects efficiently.
//!
//! Demonstrates instanced rendering with many rotating quads.
//!
//! Run with: cargo run --example instancing

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, FrameOutput,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat,
    Vertex2D,
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

const GRID_SIZE: i32 = 20;
const QUAD_SIZE: f32 = 0.03;

fn rotate_point(x: f32, y: f32, angle: f32, cx: f32, cy: f32) -> (f32, f32) {
    let dx = x - cx;
    let dy = y - cy;
    let (s, c) = (angle.sin(), angle.cos());
    (cx + dx * c - dy * s, cy + dx * s + dy * c)
}

fn create_rotating_quad(cx: f32, cy: f32, size: f32, angle: f32, color: Color) -> [Vertex2D; 6] {
    let half = size / 2.0;
    let corners = [
        rotate_point(cx - half, cy - half, angle, cx, cy),
        rotate_point(cx + half, cy - half, angle, cx, cy),
        rotate_point(cx + half, cy + half, angle, cx, cy),
        rotate_point(cx - half, cy + half, angle, cx, cy),
    ];
    
    [
        Vertex2D::new(corners[0].0, corners[0].1, color),
        Vertex2D::new(corners[1].0, corners[1].1, color),
        Vertex2D::new(corners[2].0, corners[2].1, color),
        Vertex2D::new(corners[0].0, corners[0].1, color),
        Vertex2D::new(corners[2].0, corners[2].1, color),
        Vertex2D::new(corners[3].0, corners[3].1, color),
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
        let shader = ShaderModule::from_wgsl(&device, rag::shader::builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
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

        let time = self.start_time.elapsed().as_secs_f32();

        // Generate all quads
        let mut vertices: Vec<Vertex2D> = Vec::new();
        let total = GRID_SIZE * GRID_SIZE;
        
        for i in 0..GRID_SIZE {
            for j in 0..GRID_SIZE {
                let idx = i * GRID_SIZE + j;
                
                // Position in grid
                let nx = (i as f32 / (GRID_SIZE - 1) as f32) * 2.0 - 1.0;
                let ny = (j as f32 / (GRID_SIZE - 1) as f32) * 2.0 - 1.0;
                let cx = nx * 0.85;
                let cy = ny * 0.85;
                
                // Individual rotation based on position and time
                let phase = (idx as f32 / total as f32) * std::f32::consts::PI * 2.0;
                let angle = time * 2.0 + phase;
                
                // Color based on position
                let hue = (idx as f32 / total as f32 + time * 0.1) % 1.0;
                let color = hsv_to_rgb(hue, 0.8, 0.9);
                
                vertices.extend_from_slice(&create_rotating_quad(cx, cy, QUAD_SIZE, angle, color));
            }
        }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let vertex_buffer = Buffer::with_data(device, &vertices, BufferUsage::VERTEX)?;

        let frame = FrameOutput::new(device, width, height, TextureFormat::Rgba8Unorm);
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 });
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..vertices.len() as u32, 0..1);
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

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = match (h * 6.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    
    Color { r: r + m, g: g + m, b: b + m, a: 1.0 }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(event_loop.create_window(
                Window::default_attributes()
                    .with_title(&format!("RAG - Instancing ({} quads)", GRID_SIZE * GRID_SIZE))
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
    println!("RAG Instancing Example - {} quads", GRID_SIZE * GRID_SIZE);
    println!("Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}

