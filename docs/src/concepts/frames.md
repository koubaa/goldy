# Rendering Outputs

Goldy provides two complementary APIs for rendering output: **Surface** for zero-copy window presentation, and **RenderTarget** for headless rendering with optional CPU readback.

## Surface (Window Display)

`Surface` manages a swapchain for direct GPU-to-display rendering without CPU involvement.

### Creating a Surface

```rust
use goldy::{Surface, Device, DeviceType, Instance, PresentMode, SurfaceConfig};

let instance = Instance::new()?;
let device = instance.create_device(DeviceType::DiscreteGpu)?;

// Simple — defaults to Auto present mode, no depth buffer
let surface = Surface::new(&device, &window)?;

// With vsync control and depth buffer
let surface = Surface::new_with_config(&device, &window, SurfaceConfig {
    present_mode: PresentMode::Fifo,  // vsync on
    depth_format: Some(DepthFormat::Depth32Float),
})?;
```

### Render Loop (Graphics Pipeline)

The traditional path using render commands:

```rust
loop {
    let frame = surface.acquire()?;
    
    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::CORNFLOWER_BLUE);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertices);
        pass.draw(0..3, 0..1);
    }
    
    frame.render(encoder)?;
    frame.present()?;  // Consumes the frame
}
```

### Render Loop (Compute Path)

The compute path exposes the frame's texture for direct compute shader access. This is the path used by Ekrano:

```rust
loop {
    let frame = surface.acquire()?;
    let texture = frame.texture().expect("frame texture");
    
    let tex_idx = texture.bindless_index().unwrap();
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&compute_pipeline);
        pass.bind_resources_raw(&[uniform_idx, tex_idx]);
        pass.dispatch(width / 8, height / 8, 1);
    }
    encoder.submit(&device)?.wait()?;
    
    frame.present()?;
}
```

### Present Modes

Control vsync behavior at creation time or dynamically:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `Fifo` | Vsync: wait for display refresh | Smooth display, capped at monitor Hz |
| `Mailbox` | Triple-buffered: latest frame queued | Low latency + no tearing |
| `Immediate` | No sync, may tear | Benchmarks, maximum throughput |
| `Auto` | Goldy chooses (Mailbox → Fifo) | Default, good for most apps |

```rust
// Toggle vsync at runtime
surface.set_present_mode(PresentMode::Immediate)?;
println!("Current mode: {:?}", surface.present_mode());
```

**Backend mapping:**
- **Metal:** `CAMetalLayer.displaySyncEnabled` (on/off only — no mailbox)
- **Vulkan:** `VK_PRESENT_MODE_{FIFO,MAILBOX,IMMEDIATE}_KHR`
- **DX12:** `IDXGISwapChain::Present` sync interval

### Resize Handling

```rust
fn handle_resize(&mut self, new_size: PhysicalSize<u32>) {
    if new_size.width > 0 && new_size.height > 0 {
        self.surface.resize(new_size.width, new_size.height)?;
    }
}
```

### Frame Lifetime

`SurfaceFrame` follows Rust ownership semantics:

- `acquire()` returns a `SurfaceFrame` that borrows the swapchain image
- `texture()` returns a reference valid until `present()` is called
- `present()` consumes `self` — the borrow checker prevents use-after-present
- Dropping a frame without presenting cleans up the drawable (but wastes a frame)

```rust
let frame = surface.acquire()?;
let tex = frame.texture().unwrap();
// tex is valid here...
frame.present()?;
// tex is now invalid — Rust prevents accessing it after move
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

| Use Case | API | CPU Copy | Vsync |
|----------|-----|----------|-------|
| Window display | `Surface` | No (zero-copy) | Configurable |
| Compute → display | `Surface` + `frame.texture()` | No | Configurable |
| Headless testing | `RenderTarget` | Yes (via `read_to_cpu()`) | N/A |
| Video streaming | `RenderTarget` | When needed | N/A |
| Image generation | `RenderTarget` | Yes | N/A |

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
- Frame pacing via `PresentMode` prevents GPU from running ahead
- `frame.texture()` enables compute → display without a blit

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

When using `frame.texture()` with compute shaders, ensure your compute work completes before calling `frame.present()`. The `ComputeEncoder::submit().wait()` pattern handles this.

## Error Handling

```rust
// Surface errors
let frame = surface.acquire()?;  // May fail if swapchain outdated
frame.present()?;                // May need resize

// RenderTarget errors
target.render(encoder)?;              // GPU command failure
let pixels = target.read_to_cpu()?;   // Memory transfer failure
```
