//! Gradient example - animated color gradient.
//!
//! Demonstrates fragment shader with time-based animation using the Surface API.
//! Uses vertex-less fullscreen triangle (Goldy-native pattern).
//!
//! Run with: `cargo run --example gradient`
//!
//! Optional: validate the Rust `TimeUniforms` layout against Slang on the shader compile path:
//! `GOLDY_VALIDATE_LAYOUTS=1 cargo run --example gradient`

use goldy::{
    shaders, Buffer, BufferKind, Color, DeviceDescriptor, Instance, LayoutCheckable, NodeAccess, RenderPipeline,
    RenderPipelineDesc, RenderTarget, RequestAdapterOptions, ShaderModule, Surface, TaskGraph, VertexBufferLayout,
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

/// Uniform buffer data — fields must match `struct TimeUniforms` in `shaders/gradient.slang`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, LayoutCheckable)]
struct TimeUniforms {
    time: f32,
}

struct App {
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    shader: Option<ShaderModule>,
    uniform_buffer: Option<Buffer>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    scene_rt: Option<RenderTarget>,
    frame_graph: TaskGraph,
    start_time: Instant,
    frame_count: u32,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None,
            pipeline: None,
            shader: None,
            uniform_buffer: None,
            window: None,
            surface: None,
            scene_rt: None,
            frame_graph: TaskGraph::new(),
            start_time: Instant::now(),
            frame_count: 0,
        })
    }

    fn create_scene_rt(device: &goldy::Device, surface: &Surface) -> anyhow::Result<RenderTarget> {
        let (width, height) = surface.size();
        RenderTarget::new(device, width.max(1), height.max(1), surface.format())
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;
        let surface = Surface::new(&ctx, window.as_ref())?;
        let shader = ShaderModule::from_slang_with_options(
            &device,
            shaders::GRADIENT,
            &[],
            &[],
            Default::default(),
            &[TimeUniforms::LAYOUT_CHECK],
        )?;

        // Create pipeline - no vertex buffer needed, shader uses SV_VertexID
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        // Create uniform buffer for time
        let uniform_buffer = device.as_ref().alloc_buffer(
            std::mem::size_of::<TimeUniforms>() as u64,
            BufferKind::Broadcast,
            None,
            goldy::BufferFlags::empty(),
        )?;

        let scene_rt = Self::create_scene_rt(&device, &surface)?;

        self.device = Some(device);
        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        self.uniform_buffer = Some(uniform_buffer);
        self.surface = Some(surface);
        self.scene_rt = Some(scene_rt);
        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        self.frame_count += 1;

        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let scene_rt = self.scene_rt.as_ref().unwrap();
        let uniform_buffer = self.uniform_buffer.as_ref().unwrap();

        // Update uniform buffer with current time
        let time = self.start_time.elapsed().as_secs_f32();
        let uniforms = TimeUniforms { time };
        uniform_buffer.write_data(0, &[uniforms])?;

        self.frame_graph.clear();

        let mut pass = self.frame_graph.render_pass("gradient", scene_rt);
        pass.bind_buffer_mut(uniform_buffer, NodeAccess::Read);
        pass.clear(Color::BLACK);
        pass.set_pipeline(pipeline);
        pass.bind_resources(&[uniform_buffer]);
        pass.draw_fullscreen();
        pass.finish_recorded();

        let swapchain = self.frame_graph.declare_swapchain_output();
        self.frame_graph.copy_render_target_to_swapchain(scene_rt, swapchain);

        let frame = surface.begin()?;
        let frame = surface.submit_graph_to_frame(&mut self.frame_graph, frame)?;
        frame.present()?;

        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &mut self.surface {
                let _ = surface.resize(new_size.width, new_size.height);
            }
            if let (Some(device), Some(surface)) = (&self.device, &self.surface) {
                if let Ok(rt) = Self::create_scene_rt(device, surface) {
                    self.scene_rt = Some(rt);
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
                            .with_title("Goldy - Animated Gradient (Surface API)")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
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
    println!("Goldy Gradient Example - Press Escape to exit");
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
