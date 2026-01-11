# Rendering Outputs

Goldy provides two complementary APIs for rendering output: **Surface** for zero-copy window presentation, and **RenderTarget** for headless rendering with optional CPU readback.

## Surface (Window Display)

`Surface` manages a swapchain for direct GPU-to-display rendering without CPU involvement.

### Creating a Surface

```rust
use goldy::{Surface, Device, DeviceType, Instance};
use std::sync::Arc;

let instance = Instance::new()?;
let device = Arc::new(instance.create_device(DeviceType::DiscreteGpu)?);

// Create surface for a winit window
let surface = Surface::new(&device, &window)?;
```

### Render Loop

```rust
loop {
    // Acquire next swapchain image
    let frame = surface.acquire()?;
    
    // Record commands
    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::CORNFLOWER_BLUE);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertices);
        pass.draw(0..3, 0..1);
    }
    
    // Render directly to swapchain (zero-copy!)
    frame.render(encoder)?;
    
    // Present to screen
    surface.present(frame)?;
}
```

### Resize Handling

```rust
fn handle_resize(&mut self, new_size: PhysicalSize<u32>) {
    if new_size.width > 0 && new_size.height > 0 {
        self.surface.resize(new_size.width, new_size.height)?;
    }
}
```

## RenderTarget (Headless/Streaming)

`RenderTarget` renders to a GPU texture with optional CPU readback. Use this for:

- Headless rendering (servers, testing)
- Video encoding/streaming
- Image generation

### Creating a RenderTarget

```rust
use goldy::{RenderTarget, TextureFormat};

let target = RenderTarget::new(&device, 1920, 1080, TextureFormat::Rgba8Unorm)?;
```

### Rendering

```rust
let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    pass.clear(Color::BLACK);
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &vertices);
    pass.draw(0..vertex_count, 0..1);
}

// Render to GPU texture (stays on GPU)
target.render(encoder)?;
```

### CPU Readback (Optional)

```rust
// Only call when you need pixels on CPU
let pixels: Vec<u8> = target.read_to_cpu()?;

// Or read into existing buffer
let mut buffer = vec![0u8; target.buffer_size()];
target.read_to_buffer(&mut buffer)?;
```

### Save to Image

```rust
use image::{ImageBuffer, Rgba};

let pixels = target.read_to_cpu()?;
let img = ImageBuffer::<Rgba<u8>, _>::from_raw(
    target.width(), target.height(), pixels
).unwrap();

img.save("output.png")?;
```

## Surface vs RenderTarget

| Use Case | API | CPU Copy |
|----------|-----|----------|
| Window display | `Surface` | No (zero-copy) |
| Headless testing | `RenderTarget` | Yes (via `read_to_cpu()`) |
| Video streaming | `RenderTarget` | When needed |
| Image generation | `RenderTarget` | Yes |

## Pixel Formats

```rust
TextureFormat::Rgba8Unorm      // Standard 8-bit per channel
TextureFormat::Bgra8UnormSrgb  // Swapchain format (use for Surface)
TextureFormat::Rgba16Float     // HDR (16-bit float)
TextureFormat::Rgba32Float     // Full precision (32-bit float)
```

## Performance

### Surface (Zero-Copy)

- No CPU memory allocation per frame
- Renders directly to swapchain images
- Optimal for real-time display

### RenderTarget

- GPU texture allocation at creation
- Staging buffer created lazily on first `read_to_cpu()`
- Subsequent readbacks reuse staging buffer

```
First read_to_cpu():
  ├── Allocate staging buffer (one-time)
  ├── Copy GPU → staging
  └── Map and read

Subsequent read_to_cpu():
  ├── (staging buffer reused)
  ├── Copy GPU → staging
  └── Map and read
```

## Synchronization

Both `frame.render()` and `target.render()` are synchronous—they wait for GPU completion. Surface uses proper frame pipelining with semaphores internally.

## Error Handling

```rust
// Surface errors
let frame = surface.acquire()?;  // May fail if swapchain outdated
surface.present(frame)?;         // May need resize

// RenderTarget errors
target.render(encoder)?;         // GPU command failure
let pixels = target.read_to_cpu()?;  // Memory transfer failure
```
