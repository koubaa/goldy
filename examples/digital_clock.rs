//! Digital Clock example - render an animated 7-segment clock in a window.
//!
//! Run with: cargo run --example digital_clock

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, FrameOutput,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat, Vertex2D,
    shader::builtins,
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

/// Seven-segment display patterns
const SEGMENT_PATTERNS: [[bool; 7]; 11] = [
    [true, true, true, false, true, true, true],     // 0
    [false, false, true, false, false, true, false], // 1
    [true, false, true, true, true, false, true],    // 2
    [true, false, true, true, false, true, true],    // 3
    [false, true, true, true, false, true, false],   // 4
    [true, true, false, true, false, true, true],    // 5
    [true, true, false, true, true, true, true],     // 6
    [true, false, true, false, false, true, false],  // 7
    [true, true, true, true, true, true, true],      // 8
    [true, true, true, true, false, true, true],     // 9
    [false, false, false, false, false, false, false], // 10 = blank for colon
];

/// Color palette
const COLORS: [Color; 8] = [
    Color { r: 1.0, g: 0.1, b: 0.1, a: 1.0 },    // Red
    Color { r: 1.0, g: 0.65, b: 0.0, a: 1.0 },   // Orange
    Color { r: 1.0, g: 1.0, b: 0.0, a: 1.0 },    // Yellow
    Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 },    // Green
    Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 },    // Cyan
    Color { r: 0.0, g: 0.5, b: 1.0, a: 1.0 },    // Blue
    Color { r: 0.5, g: 0.0, b: 1.0, a: 1.0 },    // Purple
    Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 },    // Magenta
];

fn quad_vertices(x: f32, y: f32, w: f32, h: f32, color: Color) -> [Vertex2D; 6] {
    [
        Vertex2D::new(x, y, color),
        Vertex2D::new(x + w, y, color),
        Vertex2D::new(x + w, y + h, color),
        Vertex2D::new(x, y, color),
        Vertex2D::new(x + w, y + h, color),
        Vertex2D::new(x, y + h, color),
    ]
}

fn pixel_to_ndc(px: f32, py: f32, width: f32, height: f32) -> (f32, f32) {
    let x = (px / width) * 2.0 - 1.0;
    let y = 1.0 - (py / height) * 2.0;
    (x, y)
}

fn digit_vertices(
    digit: u8,
    cx: f32,
    cy: f32,
    scale: f32,
    color: Color,
    width: f32,
    height: f32,
) -> Vec<Vertex2D> {
    let mut vertices = Vec::new();

    let seg_w = 60.0 * scale;
    let seg_h = 12.0 * scale;
    let dig_h = 120.0 * scale;
    let gap = 4.0 * scale;

    // Colon
    if digit == 10 {
        let dot_size = seg_h * 1.5;
        let dot_spacing = dig_h * 0.5;

        let (x, y) = pixel_to_ndc(cx - dot_size / 2.0, cy - dot_spacing - dot_size / 2.0, width, height);
        let (w, h) = (dot_size / width * 2.0, dot_size / height * 2.0);
        vertices.extend_from_slice(&quad_vertices(x, y, w, -h, color));

        let (x, y) = pixel_to_ndc(cx - dot_size / 2.0, cy + dot_spacing - dot_size / 2.0, width, height);
        vertices.extend_from_slice(&quad_vertices(x, y, w, -h, color));

        return vertices;
    }

    let pattern = SEGMENT_PATTERNS[digit as usize];

    let mut add_segment = |px: f32, py: f32, pw: f32, ph: f32| {
        let (x, y) = pixel_to_ndc(px, py, width, height);
        let (w, h) = (pw / width * 2.0, ph / height * 2.0);
        vertices.extend_from_slice(&quad_vertices(x, y, w, -h, color));
    };

    if pattern[0] { add_segment(cx - seg_w / 2.0, cy - dig_h, seg_w, seg_h); }
    if pattern[1] { add_segment(cx - seg_w / 2.0 - seg_h, cy - dig_h + seg_h + gap, seg_h, dig_h - seg_h - gap * 2.0); }
    if pattern[2] { add_segment(cx + seg_w / 2.0, cy - dig_h + seg_h + gap, seg_h, dig_h - seg_h - gap * 2.0); }
    if pattern[3] { add_segment(cx - seg_w / 2.0, cy - seg_h / 2.0, seg_w, seg_h); }
    if pattern[4] { add_segment(cx - seg_w / 2.0 - seg_h, cy + gap, seg_h, dig_h - seg_h - gap * 2.0); }
    if pattern[5] { add_segment(cx + seg_w / 2.0, cy + gap, seg_h, dig_h - seg_h - gap * 2.0); }
    if pattern[6] { add_segment(cx - seg_w / 2.0, cy + dig_h - seg_h, seg_w, seg_h); }

    vertices
}

fn generate_clock_vertices(elapsed_secs: u64, color: Color, width: u32, height: u32) -> Vec<Vertex2D> {
    let hours = ((elapsed_secs / 3600) % 100) as u8;
    let minutes = ((elapsed_secs % 3600) / 60) as u8;
    let seconds = (elapsed_secs % 60) as u8;

    let digits: [u8; 8] = [
        hours / 10, hours % 10,
        10, // colon
        minutes / 10, minutes % 10,
        10, // colon
        seconds / 10, seconds % 10,
    ];

    let scale = height as f32 / 720.0;
    let digit_width = 80.0 * scale;
    let colon_width = 40.0 * scale;
    let spacing = 20.0 * scale;

    let total_width = digit_width * 6.0 + colon_width * 2.0 + spacing * 7.0;

    let cy = height as f32 / 2.0;
    let mut cx = (width as f32 - total_width) / 2.0 + digit_width / 2.0;

    let mut all_vertices = Vec::new();

    for &digit in digits.iter() {
        let w = if digit == 10 { colon_width } else { digit_width };
        let verts = digit_vertices(digit, cx, cy, scale, color, width as f32, height as f32);
        all_vertices.extend_from_slice(&verts);
        cx += w + spacing;
    }

    all_vertices
}

struct App {
    instance: Instance,
    device: Option<rag::Device>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    
    start_time: Instant,
    color_index: usize,
    paused: bool,
    pause_time: u64,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let instance = Instance::new()?;
        Ok(Self {
            instance,
            device: None,
            pipeline: None,
            shader: None,
            window: None,
            surface: None,
            start_time: Instant::now(),
            color_index: 0,
            paused: false,
            pause_time: 0,
        })
    }

    fn init_gpu(&mut self) -> anyhow::Result<()> {
        let device = self.instance.create_device(DeviceType::DiscreteGpu)?;
        
        let shader = ShaderModule::from_wgsl(&device, builtins::VERTEX_COLOR_2D)?;
        let pipeline_desc = RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        };
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &pipeline_desc)?;
        
        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        
        Ok(())
    }

    fn elapsed_secs(&self) -> u64 {
        if self.paused {
            self.pause_time
        } else {
            self.start_time.elapsed().as_secs() + self.pause_time
        }
    }

    fn toggle_pause(&mut self) {
        if self.paused {
            self.start_time = Instant::now();
            self.paused = false;
        } else {
            self.pause_time = self.elapsed_secs();
            self.paused = true;
        }
    }

    fn next_color(&mut self) {
        self.color_index = (self.color_index + 1) % COLORS.len();
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        let width = size.width;
        let height = size.height;

        if width == 0 || height == 0 {
            return Ok(());
        }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();

        let elapsed = self.elapsed_secs();
        let color = COLORS[self.color_index];
        
        // Dim color when paused
        let color = if self.paused {
            Color {
                r: color.r * 0.5,
                g: color.g * 0.5,
                b: color.b * 0.5,
                a: color.a,
            }
        } else {
            color
        };

        // Generate clock vertices
        let vertices = generate_clock_vertices(elapsed, color, width, height);
        let vertex_buffer = Buffer::with_data(device, &vertices, BufferUsage::VERTEX)?;

        // Background color (slightly lighter when paused)
        let bg = if self.paused { 0.06 } else { 0.02 };
        let bg_color = Color { r: bg, g: bg, b: bg, a: 1.0 };

        // Render
        let frame = FrameOutput::new(device, width, height, TextureFormat::Rgba8Unorm);
        
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(bg_color);
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..vertices.len() as u32, 0..1);
        }

        let output = frame.render(encoder)?;

        // Display in window
        let surface = self.surface.as_mut().unwrap();
        surface.resize(
            NonZeroU32::new(width).unwrap(),
            NonZeroU32::new(height).unwrap(),
        ).map_err(|e| anyhow::anyhow!("Failed to resize surface: {}", e))?;

        let mut buffer = surface.buffer_mut()
            .map_err(|e| anyhow::anyhow!("Failed to get buffer: {}", e))?;

        for (i, pixel) in buffer.iter_mut().enumerate() {
            let offset = i * 4;
            if offset + 3 < output.len() {
                let r = output[offset] as u32;
                let g = output[offset + 1] as u32;
                let b = output[offset + 2] as u32;
                *pixel = (r << 16) | (g << 8) | b;
            }
        }

        buffer.present()
            .map_err(|e| anyhow::anyhow!("Failed to present: {}", e))?;

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("RAG - Clock (Space: pause, Click: color)")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            
            let context = softbuffer::Context::new(window.clone()).unwrap();
            let surface = softbuffer::Surface::new(&context, window.clone()).unwrap();
            
            self.window = Some(window);
            self.surface = Some(surface);
            
            if let Err(e) = self.init_gpu() {
                eprintln!("Failed to initialize GPU: {}", e);
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Named(NamedKey::Space) => self.toggle_pause(),
                        Key::Character(ref c) if c == "c" || c == "C" => self.next_color(),
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, .. } if state.is_pressed() => {
                self.next_color();
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    eprintln!("Render error: {}", e);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(_) => {
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
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("RAG Clock Window Example");
    println!("========================");
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

