# Your First Triangle

Let's draw a colored triangle in a window using RAG.

## Complete Code

```rust
use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, FrameOutput,
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
    device: Option<rag::Device>,
    vertex_buffer: Option<Buffer>,
    pipeline: Option<RenderPipeline>,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
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

    fn init_gpu(&mut self) -> anyhow::Result<()> {
        let device = self.instance.create_device(DeviceType::DiscreteGpu)?;
        
        // Triangle vertices with colors
        let vertices = [
            Vertex2D::new(0.0, -0.5, Color::RED),
            Vertex2D::new(-0.5, 0.5, Color::GREEN),
            Vertex2D::new(0.5, 0.5, Color::BLUE),
        ];
        let vertex_buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;
        
        // Load built-in vertex color shader
        let shader = ShaderModule::from_wgsl(&device, builtins::VERTEX_COLOR_2D)?;
        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        })?;
        
        self.device = Some(device);
        self.vertex_buffer = Some(vertex_buffer);
        self.pipeline = Some(pipeline);
        Ok(())
    }

    fn render(&mut self) -> anyhow::Result<()> {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 { return Ok(()); }

        let device = self.device.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();

        // Create frame and encode commands
        let frame = FrameOutput::new(device, size.width, size.height, TextureFormat::Rgba8Unorm);
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color { r: 0.1, g: 0.1, b: 0.2, a: 1.0 });
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, vertex_buffer);
            pass.draw(0..3, 0..1);
        }

        // Render and display
        let output = frame.render(encoder)?;
        self.display_frame(&output, size.width, size.height)?;
        Ok(())
    }

    fn display_frame(&mut self, pixels: &[u8], width: u32, height: u32) -> anyhow::Result<()> {
        use std::num::NonZeroU32;
        let surface = self.surface.as_mut().unwrap();
        surface.resize(NonZeroU32::new(width).unwrap(), NonZeroU32::new(height).unwrap())
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let mut buffer = surface.buffer_mut().map_err(|e| anyhow::anyhow!("{}", e))?;
        
        for (i, pixel) in buffer.iter_mut().enumerate() {
            let o = i * 4;
            if o + 2 < pixels.len() {
                *pixel = ((pixels[o] as u32) << 16) | ((pixels[o+1] as u32) << 8) | pixels[o+2] as u32;
            }
        }
        buffer.present().map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(())
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
            let ctx = softbuffer::Context::new(window.clone()).unwrap();
            self.surface = Some(softbuffer::Surface::new(&ctx, window.clone()).unwrap());
            self.window = Some(window);
            self.init_gpu().unwrap();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.render().ok();
                self.window.as_ref().unwrap().request_redraw();
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
let device = instance.create_device(DeviceType::DiscreteGpu)?;
```

The `Instance` discovers available GPUs. `create_device` opens a connection to one.

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
let shader = ShaderModule::from_wgsl(&device, builtins::VERTEX_COLOR_2D)?;
let pipeline = RenderPipeline::new(&device, &shader, &shader, &desc)?;
```

RAG includes built-in shaders for common cases. The pipeline combines vertex and fragment shaders with rendering state.

### 4. Record Commands

```rust
let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    pass.clear(Color { r: 0.1, g: 0.1, b: 0.2, a: 1.0 });
    pass.set_pipeline(pipeline);
    pass.set_vertex_buffer(0, vertex_buffer);
    pass.draw(0..3, 0..1);  // 3 vertices, 1 instance
}
```

Commands are recorded into an encoder, then executed by `frame.render()`.

### 5. Render and Display

```rust
let frame = FrameOutput::new(device, width, height, format);
let output = frame.render(encoder)?;
// output is Vec<u8> of RGBA pixels
```

`FrameOutput` manages the render target. After rendering, you get raw pixels that can be displayed with `softbuffer` or saved to an image.

## Run It

```bash
cargo run --example triangle --release
```

You should see a window with a colored triangle on a dark blue background.

## Next Steps

- [Understanding the API](./understanding-api.md) - Deeper dive into concepts
- [Digital Clock Example](../examples/digital-clock.md) - More complex rendering
- [Shaders](../reference/shaders.md) - Write your own WGSL shaders

