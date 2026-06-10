//! Depth quads example - two fullscreen quads whose depths cross periodically.
//!
//! Depth-tested rendering via an offscreen [`RenderTarget`] with a depth attachment
//! (`RenderTarget::new_with_depth`), then blit to the swapchain through the task graph.
//! A warm (red/orange) quad and a cool (teal/blue) quad both cover the entire
//! screen.  They are *always drawn in the same order* (warm first, cool second),
//! so without a depth buffer cool would always win.  With depth testing the quad
//! with the smaller z value wins regardless of draw order, so the screen flips
//! colour every time the two z-curves cross.
//!
//! Run with: cargo run --example depth_quads

use bytemuck::{Pod, Zeroable};
use goldy::{
    BufferFlags, BufferKind, Color, CompareFunction, DepthFormat, DepthStencilState, DeviceDescriptor, Instance,
    NodeAccess, Parcel, RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions, RetainedPool,
    ShaderModule, Surface, TaskGraph, VertexAttribute, VertexBufferLayout, VertexFormat,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};
mod common;

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
fn quad_verts(x0: f32, y0: f32, x1: f32, y1: f32, z: f32, r: f32, g: f32, b: f32) -> [DepthVertex; 6] {
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
    _retained_pool: Option<RetainedPool>,
    warm_parcel: Option<Parcel>,
    cool_parcel: Option<Parcel>,
    surface: Option<Surface>,
    scene_rt: Option<RenderTarget>,
    frame_graph: TaskGraph,
    window: Option<Arc<Window>>,
    frame_count: u64,
    start_time: std::time::Instant,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let instance = Instance::new()?;
        Ok(Self {
            instance,
            device: None,
            pipeline: None,
            _retained_pool: None,
            warm_parcel: None,
            cool_parcel: None,
            surface: None,
            scene_rt: None,
            frame_graph: TaskGraph::new(),
            window: None,
            frame_count: 0,
            start_time: std::time::Instant::now(),
        })
    }

    fn create_scene_rt(device: &goldy::Device, surface: &Surface) -> anyhow::Result<RenderTarget> {
        let (width, height) = surface.size();
        RenderTarget::new_with_depth(
            device,
            width.max(1),
            height.max(1),
            surface.format(),
            Some(DepthFormat::Depth32Float),
        )
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(
            self.instance
                .request_adapter(&RequestAdapterOptions::default())?
                .request_device(&DeviceDescriptor::default())?,
        );
        let ctx = device.create_context()?;

        let surface = Surface::new(&ctx, window.as_ref())?;

        let shader = ShaderModule::from_slang(&device, include_str!("../shaders/depth_test.slang"))?;

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

        let scene_rt = Self::create_scene_rt(&device, &surface)?;

        let mut retained_pool = RetainedPool::new(device.clone());
        let stride = std::mem::size_of::<DepthVertex>() as u32;
        let quad_bytes = 6 * stride as usize;
        let warm_parcel = retained_pool.acquire_buffer(
            quad_bytes as u64,
            BufferKind::Scattered,
            Some(stride),
            BufferFlags::empty(),
            None,
        )?;
        let cool_parcel = retained_pool.acquire_buffer(
            quad_bytes as u64,
            BufferKind::Scattered,
            Some(stride),
            BufferFlags::empty(),
            None,
        )?;

        self.device = Some(device);
        self.pipeline = Some(pipeline);
        self._retained_pool = Some(retained_pool);
        self.warm_parcel = Some(warm_parcel);
        self.cool_parcel = Some(cool_parcel);
        self.surface = Some(surface);
        self.scene_rt = Some(scene_rt);

        Ok(())
    }

    fn render_frame(&mut self) -> anyhow::Result<()> {
        let size = self.window.as_ref().unwrap().inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let scene_rt = self.scene_rt.as_ref().unwrap();
        let warm_parcel = self.warm_parcel.as_ref().unwrap();
        let cool_parcel = self.cool_parcel.as_ref().unwrap();

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

        // Update window title with live depth values so it is easy to reason
        // about what is happening.
        let winner = if warm_z < cool_z { "WARM wins" } else { "COOL wins" };
        if let Some(window) = &self.window {
            window.set_title(&format!(
                "Depth Quads  |  warm z={:.3}  cool z={:.3}  →  {}",
                warm_z, cool_z, winner
            ));
        }

        self.frame_graph.clear();
        self.frame_graph.write_parcel(
            warm_parcel,
            0,
            bytemuck::cast_slice(&warm_verts).to_vec(),
        )?;
        self.frame_graph.write_parcel(
            cool_parcel,
            0,
            bytemuck::cast_slice(&cool_verts).to_vec(),
        )?;

        let mut pass = self.frame_graph.render_pass("depth_quads", scene_rt);
        pass.bind_parcel_mut(warm_parcel, NodeAccess::Read);
        pass.bind_parcel_mut(cool_parcel, NodeAccess::Read);
        pass.clear(Color::BLACK);
        pass.clear_depth(1.0);
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, warm_parcel);
        pass.draw(0..6, 0..1);
        pass.set_vertex_buffer(0, cool_parcel);
        pass.draw(0..6, 0..1);
        pass.finish_recorded();

        let swapchain = self.frame_graph.declare_swapchain_output();
        self.frame_graph.copy_render_target_to_swapchain(scene_rt, swapchain);

        let frame = surface.begin()?;
        let frame = surface.submit_graph_to_frame(&mut self.frame_graph, frame)?;
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
            if let (Some(device), Some(surface)) = (&self.device, &self.surface) {
                match Self::create_scene_rt(device, surface) {
                    Ok(rt) => self.scene_rt = Some(rt),
                    Err(e) => tracing::error!("Failed to recreate scene render target: {}", e),
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

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        common::exit_if_timed_out(event_loop, self.start_time);
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
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
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
