//! Instancing example - render many objects efficiently.
//!
//! Demonstrates the Goldy-native compute+graphics pattern:
//! 1. Compute shader updates instance transforms/colors each frame
//! 2. Graphics shader renders quads from the instance buffer
//! 3. No CPU vertex generation, no vertex buffer - all GPU-driven
//!
//! Run with: cargo run --example instancing

use anyhow::Result;
use goldy::{
    Buffer, Color, CommandEncoder, ComputeEncoder, ComputePipeline, DataAccess, DeviceType,
    Instance, Instance2D, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, ShaderModule,
    Surface, VertexBufferLayout,
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

const GRID_SIZE: u32 = 20;
const QUAD_SIZE: f32 = 0.03;
const NUM_QUADS: u32 = GRID_SIZE * GRID_SIZE;

/// Animation parameters for compute shader
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AnimParams {
    time: f32,
    delta_time: f32,
    total_instances: u32,
    _pad: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    println!(
        "Goldy Instancing Example - {} quads (GPU-driven)",
        NUM_QUADS
    );
    println!("Press Escape to exit");

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
    // Compute resources
    compute_pipeline: ComputePipeline,
    // Graphics resources
    render_pipeline: RenderPipeline,
    // Buffers
    instance_buffer: Buffer,
    params_buffer: Buffer,
    // State
    start_time: Instant,
    last_time: f32,
}

impl RenderState {
    fn new(window: Arc<Window>) -> Result<Self> {
        let instance = Instance::new()?;
        let device = Arc::new(instance.create_device(DeviceType::DiscreteGpu)?);
        let surface = Surface::new(&device, window.as_ref())?;

        // Load shaders
        let compute_shader =
            ShaderModule::from_slang(&device, include_str!("../shaders/instancing_update.slang"))?;
        let render_shader =
            ShaderModule::from_slang(&device, include_str!("../shaders/instancing_render.slang"))?;

        // Create instance buffer with initial positions
        // Positions are static, compute shader updates rotation and color
        let mut instances = Vec::with_capacity(NUM_QUADS as usize);
        for i in 0..GRID_SIZE {
            for j in 0..GRID_SIZE {
                // Position in grid (static)
                let nx = (i as f32 / (GRID_SIZE - 1) as f32) * 2.0 - 1.0;
                let ny = (j as f32 / (GRID_SIZE - 1) as f32) * 2.0 - 1.0;
                let cx = nx * 0.85;
                let cy = ny * 0.85;

                instances.push(Instance2D::new(
                    cx,
                    cy,
                    0.0, // rotation - will be updated by compute
                    QUAD_SIZE,
                    [1.0, 1.0, 1.0, 1.0], // color - will be updated by compute
                ));
            }
        }

        let instance_buffer = Buffer::with_data(
            &device,
            &instances,
            DataAccess::Scattered,
        )?;

        // Create params buffer
        let params = AnimParams {
            time: 0.0,
            delta_time: 0.016,
            total_instances: NUM_QUADS,
            _pad: 0,
        };
        let params_buffer = Buffer::with_data(&device, &[params], DataAccess::Broadcast)?;

        // Create compute pipeline
        let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;

        // Create render pipeline - no vertex buffer needed
        let render_pipeline = RenderPipeline::new(
            &device,
            &render_shader,
            &render_shader,
            &RenderPipelineDesc {
                vertex_layout: VertexBufferLayout::empty(),
                topology: PrimitiveTopology::TriangleList,
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        println!(
            "Created instancing example with {} quads (GPU compute + graphics)",
            NUM_QUADS
        );

        Ok(Self {
            window,
            device,
            surface,
            compute_pipeline,
            render_pipeline,
            instance_buffer,
            params_buffer,
            start_time: Instant::now(),
            last_time: 0.0,
        })
    }

    fn render(&mut self) -> Result<()> {
        let time = self.start_time.elapsed().as_secs_f32();
        let delta_time = time - self.last_time;
        self.last_time = time;

        // Update params buffer
        let params = AnimParams {
            time,
            delta_time,
            total_instances: NUM_QUADS,
            _pad: 0,
        };
        self.params_buffer.write_data(0, &[params])?;

        // Run compute pass to update instance transforms and colors
        let mut compute_encoder = ComputeEncoder::new();
        {
            let mut pass = compute_encoder.begin_compute_pass();
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_push_constants(&[&self.instance_buffer, &self.params_buffer]);
            // Dispatch enough workgroups for all instances (64 threads per group)
            let workgroups = (NUM_QUADS + 63) / 64;
            pass.dispatch(workgroups, 1, 1);
        }
        compute_encoder.dispatch(&self.device)?;

        // Render quads from instance buffer
        let frame = self.surface.acquire()?;

        let bg_color = Color {
            r: 0.02,
            g: 0.02,
            b: 0.04,
            a: 1.0,
        };

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(bg_color);
            pass.set_pipeline(&self.render_pipeline);
            pass.set_push_constants(&[&self.instance_buffer]);
            // Draw 6 vertices (quad) per instance - no vertex buffer!
            pass.draw_quads(NUM_QUADS);
        }

        frame.render(encoder)?;
        self.surface.present(frame)?;

        self.window.request_redraw();
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title(&format!(
                                "Goldy - Instancing ({} quads, GPU-driven)",
                                NUM_QUADS
                            ))
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 800)),
                    )
                    .expect("Failed to create window"),
            );

            match RenderState::new(window.clone()) {
                Ok(state) => {
                    self.state = Some(state);
                    window.request_redraw();
                }
                Err(e) => {
                    eprintln!("Failed to create render state: {}", e);
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
                        eprintln!("Render error: {}", e);
                    }
                }
            }
            _ => {}
        }
    }
}
