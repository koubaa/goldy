# Your First Triangle

This tutorial draws a colored triangle in a window using Goldy's render pipeline and Surface API.

## Complete Code

```rust
use goldy::{
    shader::builtins, BufferKind, Color, DeviceType, Instance, NodeAccess, Parcel,
    RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions, RetainedPool,
    ShaderModule, Surface, TaskGraph, Vertex2D,
};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct App {
    instance: Instance,
    device: Option<Arc<goldy::Device>>,
    _retained_pool: Option<RetainedPool>,
    vertex_buffer: Option<Parcel>,
    pipeline: Option<RenderPipeline>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
    scene_rt: Option<RenderTarget>,
    frame_graph: TaskGraph,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None,
            _retained_pool: None,
            vertex_buffer: None,
            pipeline: None,
            window: None,
            surface: None,
            scene_rt: None,
            frame_graph: TaskGraph::new(),
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);

        let vertices = [
            Vertex2D::new(0.0, -0.5, Color::RED),
            Vertex2D::new(-0.5, 0.5, Color::GREEN),
            Vertex2D::new(0.5, 0.5, Color::BLUE),
        ];
        let mut retained_pool = RetainedPool::new(device.clone());
        let vertex_buffer = retained_pool.acquire_buffer_with_data(&vertices, BufferKind::Scattered)?;

        let surface = Surface::new(&device, window.as_ref())?;

        let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: Vertex2D::layout(),
                target_format: surface.format(),
                ..Default::default()
            },
        )?;

        let scene_rt = RenderTarget::new(&device, surface.width(), surface.height(), surface.format())?;

        self.device = Some(device);
        self._retained_pool = Some(retained_pool);
        self.vertex_buffer = Some(vertex_buffer);
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);
        self.scene_rt = Some(scene_rt);
        Ok(())
    }

    fn render(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let pipeline = self.pipeline.as_ref().unwrap();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let scene_rt = self.scene_rt.as_ref().unwrap();

        self.frame_graph.clear();
        let mut pass = self.frame_graph.render_pass("triangle", scene_rt);
        pass.bind_parcel_mut(vertex_buffer, NodeAccess::Read);
        pass.clear(Color { r: 0.1, g: 0.1, b: 0.2, a: 1.0 });
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, vertex_buffer);
        pass.draw(0..3, 0..1);
        pass.finish_recorded();

        let swapchain = self.frame_graph.declare_swapchain_output();
        self.frame_graph.copy_render_target_to_swapchain(scene_rt, swapchain);

        let frame = surface.begin()?;
        let frame = surface.submit_graph_to_frame(&mut self.frame_graph, frame)?;
        frame.present()?;
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("Goldy - Triangle")
                            .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
                    )
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.init_gpu(&window).unwrap();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.render().ok();
                self.window.as_ref().unwrap().request_redraw();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    if let Some(surface) = &mut self.surface {
                        let _ = surface.resize(new_size.width, new_size.height);
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
```

## Walkthrough

### Instance and Device

```rust
let instance = Instance::new()?;
let device = Arc::new(instance.create_device(DeviceType::DiscreteGpu)?);
```

`Instance` discovers available GPUs. `create_device` opens a connection to one. The `Arc` wrapper is required for `Surface` lifetime management.

### Vertex Buffer

```rust
let vertices = [
    Vertex2D::new(0.0, -0.5, Color::RED),
    Vertex2D::new(-0.5, 0.5, Color::GREEN),
    Vertex2D::new(0.5, 0.5, Color::BLUE),
];
let vertex_buffer = device.alloc_buffer_with_data( &vertices, BufferKind::Scattered)?;
```

`Vertex2D` is a built-in vertex type with position and color. `RetainedPool::acquire_buffer_with_data` allocates a retained parcel and uploads the data. `BufferKind::Scattered` marks it as a bindless storage buffer. Keep the pool alive for the parcel's lifetime (see `examples/triangle.rs`).

### Shader and Pipeline

```rust
let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
    vertex_layout: Vertex2D::layout(),
    target_format: surface.format(),
    ..Default::default(),
})?;
```

`builtins::VERTEX_COLOR_2D` is a built-in Slang shader from the `goldy_exp` library that uses `[goldy_vertex]` and `[goldy_fragment]` virtual entry points to render vertex-colored geometry. `ShaderModule::from_slang` compiles Slang source to the active backend's IR at runtime.

The pipeline takes the same shader module for both vertex and fragment stages — `goldy_exp` virtual entry points let a single source file define both.

### Surface and Presentation

```rust
let surface = Surface::new(&device, window.as_ref())?;
let scene_rt = RenderTarget::new(&device, surface.width(), surface.height(), surface.format())?;

// Per frame: render_pass on scene_rt → blit → submit_graph_to_frame → present
// See examples/triangle.rs for resize handling and animation.
```

`Surface` manages the swapchain. Scene color is rendered to an offscreen `RenderTarget`, copied to the swapchain via the task graph, then presented. Rendering stays on the GPU — no CPU readback.

## Run It

```bash
cargo run --example triangle
```

You should see a window with a colored triangle on a dark blue background.

## Next Steps

- [Your First Compute Shader](./first-compute.md) — bypass the graphics pipeline entirely
- [Examples](../examples/overview.md) — more complex demos
