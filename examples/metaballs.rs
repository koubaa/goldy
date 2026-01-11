//! Metaballs example - organic blob simulation.
//!
//! Demonstrates uniform buffers with bind groups for time animation.
//!
//! Run with: cargo run --example metaballs

use goldy::{
    shaders, BindGroup, BindGroupLayout, BindGroupLayoutBinding, Buffer, BufferBinding,
    BufferUsage, Color, CommandEncoder, DeviceType, Instance, RenderPipeline, RenderPipelineDesc,
    ShaderModule, Surface, Vertex2DUv, FULLSCREEN_QUAD,
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

/// Uniform buffer data (must match shader cbuffer layout)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    time: f32,
}

struct App {
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    bind_group_layout: Option<BindGroupLayout>,
    bind_group: Option<BindGroup>,
    uniform_buffer: Option<Buffer>,
    vertex_buffer: Option<Buffer>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    start_time: Instant,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None,
            pipeline: None,
            shader: None,
            bind_group_layout: None,
            bind_group: None,
            uniform_buffer: None,
            vertex_buffer: None,
            window: None,
            surface: None,
            start_time: Instant::now(),
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);

        // Create surface first to get the correct format
        let surface = Surface::new(&device, window.as_ref())?;

        // Create shader
        let shader = ShaderModule::from_slang(&device, shaders::METABALLS)?;

        // Create bind group layout for uniforms (binding 0)
        let bind_group_layout =
            BindGroupLayout::new(&device, &[BindGroupLayoutBinding::uniform_fragment(0)])?;

        // Create pipeline with bind group layout
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: Vertex2DUv::layout(),
                target_format: surface.format(),
                bind_group_layouts: &[&bind_group_layout],
                ..Default::default()
            },
        )?;

        // Create static vertex buffer (fullscreen quad)
        let vertex_buffer =
            Buffer::with_data(device.as_ref(), &FULLSCREEN_QUAD, BufferUsage::VERTEX)?;

        // Create uniform buffer
        let uniform_buffer = Buffer::new(
            device.as_ref(),
            std::mem::size_of::<Uniforms>() as u64,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        )?;

        // Create bind group
        let bind_group = BindGroup::new(
            &device,
            &bind_group_layout,
            &[BufferBinding::new(0, &uniform_buffer)],
        )?;

        self.device = Some(device);
        self.shader = Some(shader);
        self.bind_group_layout = Some(bind_group_layout);
        self.pipeline = Some(pipeline);
        self.vertex_buffer = Some(vertex_buffer);
        self.uniform_buffer = Some(uniform_buffer);
        self.bind_group = Some(bind_group);
        self.surface = Some(surface);

        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let uniform_buffer = self.uniform_buffer.as_ref().unwrap();
        let bind_group = self.bind_group.as_ref().unwrap();

        // Update uniform buffer with current time
        let time = self.start_time.elapsed().as_secs_f32();
        let uniforms = Uniforms { time };
        uniform_buffer.write_data(0, &[uniforms])?;

        // Acquire frame
        let frame = surface.acquire()?;

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group);
            pass.set_vertex_buffer(0, vertex_buffer);
            pass.draw(0..6, 0..1);
        }

        frame.render(encoder)?;
        surface.present(frame)?;

        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &mut self.surface {
                let _ = surface.resize(new_size.width, new_size.height);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Metaballs (Uniform Buffers)")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 800)),
                    )
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
            window.request_redraw();
        }
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
                    eprintln!("Render error: {}", e);
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
    tracing_subscriber::fmt().with_env_filter("info").init();
    println!("Goldy Metaballs Example (Uniform Buffers) - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
