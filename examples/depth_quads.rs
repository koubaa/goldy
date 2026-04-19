//! Depth quads example - two fullscreen quads whose depths cross periodically.
//!
//! This example demonstrates `Surface::new_with_depth` and depth-tested rendering.
//! A warm (red/orange) quad and a cool (teal/blue) quad both cover the entire
//! screen.  They are *always drawn in the same order* (warm first, cool second),
//! so without a depth buffer cool would always win.  With depth testing the quad
//! with the smaller z value wins regardless of draw order, so the screen flips
//! colour every time the two z-curves cross.
//!
//! Run with: cargo run --example depth_quads

use bytemuck::{Pod, Zeroable};
use goldy::{
    Buffer, Color, CommandEncoder, CompareFunction, DataAccess, DepthFormat, DepthStencilState,
    DeviceType, Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, Surface,
    VertexAttribute, VertexBufferLayout, VertexFormat,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

// ============================================================================
// Vertex type: (x, y, z) NDC position + RGBA color
// ============================================================================

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct DepthVertex {
    position: [f32; 3],
    color: [f32; 4],
}
impl goldy::StructuredBufferElement for DepthVertex {}

impl DepthVertex {
    const fn new(x: f32, y: f32, z: f32, r: f32, g: f32, b: f32) -> Self {
        Self {
            position: [x, y, z],
            color: [r, g, b, 1.0],
        }
    }
}

fn depth_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout {
        stride: std::mem::size_of::<DepthVertex>() as u32,
        attributes: vec![
            VertexAttribute {
                location: 0,
                format: VertexFormat::Float32x3,
                offset: 0,
            },
            VertexAttribute {
                location: 1,
                format: VertexFormat::Float32x4,
                offset: 12,
            },
        ],
    }
}

/// Build 6 vertices for a screen-space quad (two triangles, CCW).
#[allow(clippy::too_many_arguments)]
fn quad_verts(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    z: f32,
    r: f32,
    g: f32,
    b: f32,
) -> [DepthVertex; 6] {
    let tl = DepthVertex::new(x0, y1, z, r, g, b);
    let bl = DepthVertex::new(x0, y0, z, r, g, b);
    let br = DepthVertex::new(x1, y0, z, r, g, b);
    let tr = DepthVertex::new(x1, y1, z, r, g, b);
    [tl, bl, br, tl, br, tr]
}

// ============================================================================
// App state
// ============================================================================

struct App {
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    pipeline: Option<RenderPipeline>,
    surface: Option<Surface>,
    window: Option<Arc<Window>>,
    frame_count: u64,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let instance = Instance::new()?;
        Ok(Self {
            instance,
            device: None,
            pipeline: None,
            surface: None,
            window: None,
            frame_count: 0,
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);

        // Surface with a depth buffer for 3D depth testing
        let surface =
            Surface::new_with_depth(&device, window.as_ref(), Some(DepthFormat::Depth32Float))?;

        let shader =
            ShaderModule::from_slang(&device, include_str!("../shaders/depth_test.slang"))?;

        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: depth_vertex_layout(),
                target_format: surface.format(),
                depth_stencil: Some(DepthStencilState {
                    format: DepthFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::Less,
                }),
                ..Default::default()
            },
        )?;

        self.device = Some(device);
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);

        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let size = self.window.as_ref().unwrap().inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();

        // Two independent sine-wave oscillations chosen so they cross ~twice per
        // second at 60 fps.  Both stay in (0.1, 0.9) so neither is ever clipped.
        let t = self.frame_count as f32 * 0.04;
        let warm_z = t.sin() * 0.4 + 0.5; // ~1 Hz
        let cool_z = (t * 1.3 + 1.0).sin() * 0.4 + 0.5; // ~1.3 Hz

        // Both quads are FULLSCREEN.  Draw order is always: warm first, cool second.
        // Without depth testing cool would always overwrite warm.
        // With depth testing (`Less`): whichever has the smaller z wins every pixel.
        let warm_verts = quad_verts(-1.0, -1.0, 1.0, 1.0, warm_z, 0.95, 0.35, 0.1);
        let cool_verts = quad_verts(-1.0, -1.0, 1.0, 1.0, cool_z, 0.1, 0.6, 0.95);

        let warm_vb = Buffer::with_data(device, &warm_verts, DataAccess::Scattered)?;
        let cool_vb = Buffer::with_data(device, &cool_verts, DataAccess::Scattered)?;

        // Update window title with live depth values so it is easy to reason
        // about what is happening.
        let winner = if warm_z < cool_z {
            "WARM wins"
        } else {
            "COOL wins"
        };
        if let Some(window) = &self.window {
            window.set_title(&format!(
                "Depth Quads  |  warm z={:.3}  cool z={:.3}  →  {}",
                warm_z, cool_z, winner
            ));
        }

        let frame = surface.acquire()?;

        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.clear_depth(1.0);
            pass.set_pipeline(pipeline);

            pass.set_vertex_buffer(0, &warm_vb);
            pass.draw(0..6, 0..1);

            pass.set_vertex_buffer(0, &cool_vb);
            pass.draw(0..6, 0..1);
        }

        frame.render(encoder)?;
        frame.present()?;

        self.frame_count += 1;
        Ok(())
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            if let Some(surface) = &mut self.surface {
                if let Err(e) = surface.resize(new_size.width, new_size.height) {
                    tracing::error!("Failed to resize surface: {}", e);
                }
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Goldy - Depth Quads (depth buffer demo)")
                .with_inner_size(winit::dpi::LogicalSize::new(900, 600));

            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.window = Some(window.clone());

            if let Err(e) = self.init_gpu(&window) {
                tracing::error!("Failed to initialize GPU: {}", e);
            }
            window.request_redraw();
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
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render_frame() {
                    tracing::error!("Render error: {}", e);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
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
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("Goldy Depth Quads Example");
    println!("=========================");
    println!("Two overlapping quads whose depths oscillate independently.");
    println!("The depth buffer ensures the nearer quad always wins the overlap region.");
    println!("Press Escape or close window to exit.\n");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
