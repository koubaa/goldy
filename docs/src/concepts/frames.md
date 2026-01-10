# Frame Output

`FrameOutput` manages rendering to an offscreen buffer with CPU readback.

## Creating a Frame

```rust
use rag::{FrameOutput, TextureFormat};

let frame = FrameOutput::new(&device, width, height, TextureFormat::Rgba8Unorm);
```

## Rendering

Execute commands and get pixel data:

```rust
let frame = FrameOutput::new(&device, 800, 600, TextureFormat::Rgba8Unorm);

let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    pass.clear(Color::CORNFLOWER_BLUE);
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &vertices);
    pass.draw(0..3, 0..1);
}

let pixels: Vec<u8> = frame.render(encoder)?;
```

## Pixel Format

The output is a flat byte array in RGBA order:

```
pixels = [R0, G0, B0, A0, R1, G1, B1, A1, R2, ...]
```

For an 800x600 image:
- Length: 800 × 600 × 4 = 1,920,000 bytes
- Pixels are row-major (left to right, top to bottom)

## Using the Output

### Display with softbuffer

```rust
use std::num::NonZeroU32;

let surface = /* softbuffer surface */;
surface.resize(
    NonZeroU32::new(width).unwrap(),
    NonZeroU32::new(height).unwrap(),
)?;

let mut buffer = surface.buffer_mut()?;

// Convert RGBA to softbuffer format (0xRRGGBB)
for (i, pixel) in buffer.iter_mut().enumerate() {
    let offset = i * 4;
    let r = pixels[offset] as u32;
    let g = pixels[offset + 1] as u32;
    let b = pixels[offset + 2] as u32;
    *pixel = (r << 16) | (g << 8) | b;
}

buffer.present()?;
```

### Save to Image

```rust
use image::{ImageBuffer, Rgba};

let img = ImageBuffer::<Rgba<u8>, _>::from_raw(
    width, height, pixels
).unwrap();

img.save("output.png")?;
```

### Process Pixels

```rust
// Count red pixels
let red_count = pixels
    .chunks(4)
    .filter(|rgba| rgba[0] > 200 && rgba[1] < 50 && rgba[2] < 50)
    .count();
```

## Frame Size

The frame size is fixed at creation. For window resize, create a new frame:

```rust
fn render(&mut self, width: u32, height: u32) -> anyhow::Result<Vec<u8>> {
    // New frame for current size
    let frame = FrameOutput::new(&self.device, width, height, TextureFormat::Rgba8Unorm);
    
    let mut encoder = CommandEncoder::new();
    // ... record commands ...
    
    frame.render(encoder)
}
```

## Texture Formats

```rust
TextureFormat::Rgba8Unorm   // Standard 8-bit per channel
TextureFormat::Rgba8Srgb    // sRGB color space
TextureFormat::Rgba16Float  // HDR (16-bit float)
TextureFormat::Rgba32Float  // Full precision (32-bit float)
```

For most cases, `Rgba8Unorm` is appropriate.

## Performance

### Frame Creation

Creating a frame allocates GPU memory. For real-time rendering, consider frame pooling:

```rust
struct FramePool {
    frames: Vec<FrameOutput>,
    current: usize,
}
```

### CPU Readback

`frame.render()` copies pixels from GPU to CPU. This is relatively slow compared to GPU-only rendering. For display purposes, this is fine. For high-performance rendering without readback, direct swapchain presentation is planned.

## Synchronization

`frame.render()` is synchronous - it waits for GPU completion before returning. This simplifies the API but limits parallelism. Future versions may offer async options.

## Error Handling

```rust
let pixels = frame.render(encoder)?;
```

Errors can occur if:
- GPU command execution fails
- Memory transfer fails
- Device is lost

## Example: Animated Rendering

```rust
fn render_loop(device: &Device, pipeline: &RenderPipeline) {
    let mut time = 0.0f32;
    
    loop {
        let frame = FrameOutput::new(device, 800, 600, TextureFormat::Rgba8Unorm);
        
        // Generate animated vertices
        let vertices = generate_vertices(time);
        let buffer = Buffer::with_data(device, &vertices, BufferUsage::VERTEX)?;
        
        let mut encoder = CommandEncoder::new();
        {
            let mut pass = encoder.begin_render_pass();
            pass.clear(Color::BLACK);
            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, &buffer);
            pass.draw(0..vertices.len() as u32, 0..1);
        }
        
        let pixels = frame.render(encoder)?;
        display_pixels(&pixels);
        
        time += 0.016;  // ~60fps
    }
}
```

