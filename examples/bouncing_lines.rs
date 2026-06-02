//! Bouncing lines example - animated lines bouncing off walls.
//!
//! Demonstrates GPU compute + graphics integration with line primitives.
//! The compute shader updates line endpoint positions, the graphics shader renders them.
//!
//! Run with: cargo run --example bouncing_lines

use anyhow::Result;
use goldy::{
    Buffer, Color, CommandEncoder, ComputeEncoder, ComputePipeline, DataAccess, DeviceDescriptor,
    Instance, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RequestAdapterOptions,
    ShaderModule, Surface, VertexBufferLayout,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const NUM_LINES: u32 = 20;

/// Line structure matching the shader layout
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Line {
    p1: [f32; 2],     // First endpoint position
    v1: [f32; 2],     // First endpoint velocity
    p2: [f32; 2],     // Second endpoint position
    v2: [f32; 2],     // Second endpoint velocity
    color_index: u32, // Color lookup index
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
impl goldy::StructuredBufferElement for Line {}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    println!("Goldy Bouncing Lines Example");
    println!("  Escape - Exit");

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
    context: goldy::Context,
    surface: Surface,
    // Compute resources
    compute_pipeline: ComputePipeline,
    // Buffer
    line_buffer: Buffer,
    // Graphics resources
    render_pipeline: RenderPipeline,
    // Frame counter
    frame_count: u32,
    start_time: std::time::Instant,
}

impl RenderState {
    fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(
            instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let surface = Surface::new(&ctx, window.as_ref())?;

        // Compute shader for line physics
        let compute_shader = ShaderModule::from_slang(
            &device,
            include_str!("../shaders/bouncing_lines_update.slang"),
        )?;

        // Render shader for visualization
        let render_shader = ShaderModule::from_slang(
            &device,
            include_str!("../shaders/bouncing_lines_render.slang"),
        )?;

        // Create line buffer with initial positions matching the original example
        let mut lines = Vec::with_capacity(NUM_LINES as usize);
        for idx in 0..NUM_LINES {
            let angle = (idx as f32 / NUM_LINES as f32) * std::f32::consts::PI * 2.0;
            lines.push(Line {
                p1: [angle.cos() * 0.3, angle.sin() * 0.3],
                v1: [
                    0.01 * (idx as f32 * 0.7).cos(),
                    0.012 * (idx as f32 * 0.9).sin(),
                ],
                p2: [-angle.cos() * 0.3, -angle.sin() * 0.3],
                v2: [
                    -0.011 * (idx as f32 * 1.1).cos(),
                    0.009 * (idx as f32 * 1.3).sin(),
                ],
                color_index: idx,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            });
        }

        let line_buffer = Buffer::with_data(&device, &lines, DataAccess::Scattered)?;

        // Create compute pipeline
        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;

        // Create render pipeline
        let render_pipeline = RenderPipeline::new(
            &device,
            &render_shader,
            &render_shader,
            &RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::LineList,
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        println!("Created bouncing lines with {} lines", NUM_LINES);

        Ok(Self {
            window,
            context: ctx,
            surface,
            compute_pipeline,
            line_buffer,
            render_pipeline,
            frame_count: 0,
            start_time: std::time::Instant::now(),
        })
    }

    fn render(&mut self) -> Result<()> {
        self.frame_count += 1;

        // Run compute pass to update line positions
        let mut compute_encoder = ComputeEncoder::new();
        {
            let mut pass = compute_encoder.begin_compute_pass();
            pass.set_pipeline(&self.compute_pipeline);
            // Pass buffer indices via push constants
            pass.bind_resources(&[&self.line_buffer]);
            // Only 20 lines, but dispatch at least 1 workgroup
            let workgroups = NUM_LINES.div_ceil(64);
            pass.dispatch(workgroups.max(1), 1, 1);
        }
        compute_encoder.dispatch(&self.context)?;

        // Render lines
        let frame = self.surface.begin()?;

        let bg_color = Color {
            r: 0.05,
            g: 0.05,
            b: 0.1,
            a: 1.0,
        };

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(bg_color);
            pass.set_pipeline(&self.render_pipeline);
            // Pass buffer indices via push constants
            pass.bind_resources(&[&self.line_buffer]);
            // Draw 2 vertices (line) per instance
            pass.draw(0..2, 0..NUM_LINES);
        }

        frame.render(encoder)?;
        frame.present()?;

        self.window.request_redraw();
        Ok(())
    }
}

impl Drop for RenderState {
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
        if self.state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Bouncing Lines")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
                    )
                    .expect("Failed to create window"),
            );

            match RenderState::new(window.clone()) {
                Ok(state) => {
                    self.state = Some(state);
                    window.request_redraw();
                }
                Err(e) => {
                    tracing::error!("Failed to create render state: {}", e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    if size.width > 0 && size.height > 0 {
                        state.surface.resize(size.width, size.height).ok();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    if let Err(e) = state.render() {
                        tracing::error!("Render error: {}", e);
                    }
                }
            }
            _ => {}
        }
    }
}
