# Your First Triangle

Let's draw a colored triangle in a window using RAG.

## Complete Code

```rust
use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, Surface,
    Instance, RenderPipeline, RenderPipelineDesc, ShaderModule, TextureFormat,
    Vertex2D, shader::builtins,
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
    device: Option<Arc<rag::Device>>,
    vertex_buffer: Option<Buffer>,
    pipeline: Option<RenderPipeline>,
    window: Option<Arc<Window>>,
    surface: Option<Surface>,
}

impl App {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            instance: Instance::new()?,
            device: None,
            vertex_buffer: None,
            pipeline: None,
            window: None,
            surface: None,
        })
    }

    fn init_gpu(&mut self, window: &Arc<Window>) -> anyhow::Result<()> {
        let device = Arc::new(self.instance.create_device(DeviceType::DiscreteGpu)?);
        
        // Triangle vertices with colors
        let vertices = [
            Vertex2D::new(0.0, -0.5, Color::RED),
            Vertex2D::new(-0.5, 0.5, Color::GREEN),
            Vertex2D::new(0.5, 0.5, Color::BLUE),
        ];
        let vertex_buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;
        
        // Load built-in vertex color shader
        let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Bgra8UnormSrgb, // Swapchain format
            ..Default::default()
        })?;
        
        // Create Surface for zero-copy presentation
        let surface = Surface::new(&device, window.as_ref())?;
        
        self.device = Some(device);
        self.vertex_buffer = Some(vertex_buffer);
        self.pipeline = Some(pipeline);
        self.surface = Some(surface);
        Ok(())
    }

    fn render(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 { return Ok(()); }

        let pipeline = self.pipeline.as_ref().unwrap();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();

        // Acquire next frame from swapchain
        let frame = surface.acquire()?;
        
        // Record render commands
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color { r: 0.1, g: 0.1, b: 0.2, a: 1.0 });
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, vertex_buffer);
            pass.draw(0..3, 0..1);
        }

        // Render to swapchain (zero-copy - no CPU readback!)
        frame.render(encoder)?;
        
        // Present to screen
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
            let window = Arc::new(event_loop.create_window(
                Window::default_attributes()
                    .with_title("RAG - Triangle")
                    .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            ).unwrap());
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
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut App::new()?)?;
    Ok(())
}
```

## Breaking It Down

### 1. Create Instance and Device

```rust
let instance = Instance::new()?;
let device = Arc::new(instance.create_device(DeviceType::DiscreteGpu)?);
```

The `Instance` discovers available GPUs. `create_device` opens a connection to one. We wrap in `Arc` for Surface lifetime management.

### 2. Create Vertex Buffer

```rust
let vertices = [
    Vertex2D::new(0.0, -0.5, Color::RED),
    Vertex2D::new(-0.5, 0.5, Color::GREEN),
    Vertex2D::new(0.5, 0.5, Color::BLUE),
];
let vertex_buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;
```

`Vertex2D` is a built-in vertex type with position and color. `Buffer::with_data` creates a GPU buffer and uploads the data.

### 3. Create Pipeline

```rust
let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
let pipeline = RenderPipeline::new(&device, &shader, &shader, &desc)?;
```

RAG uses Slang shaders compiled at runtime. The pipeline combines vertex and fragment shaders with rendering state.

### 4. Create Surface

```rust
let surface = Surface::new(&device, window.as_ref())?;
```

`Surface` manages the swapchain for zero-copy GPU presentation. Unlike CPU-readback approaches, rendering happens directly to the window's framebuffer.

### 5. Render and Present

```rust
let frame = surface.acquire()?;  // Get next swapchain image

let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    pass.clear(Color { r: 0.1, g: 0.1, b: 0.2, a: 1.0 });
    pass.set_pipeline(pipeline);
    pass.set_vertex_buffer(0, vertex_buffer);
    pass.draw(0..3, 0..1);  // 3 vertices, 1 instance
}

frame.render(encoder)?;    // Render to swapchain
surface.present(frame)?;   // Present to screen
```

Commands are recorded into an encoder, then rendered directly to the swapchain. No CPU readback needed!

## Run It

```bash
cargo run --example triangle --release
```

You should see a window with a colored triangle on a dark blue background.

## Next Steps

- [Understanding the API](./understanding-api.md) - Deeper dive into concepts
- [Digital Clock Example](../examples/digital-clock.md) - More complex rendering
- [Shaders](../reference/shaders.md) - Write your own Slang shaders
