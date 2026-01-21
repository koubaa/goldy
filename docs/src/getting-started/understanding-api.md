# Understanding the API

Goldy's API is designed to be minimal and predictable. This page covers the core concepts.

## Resource Ownership

All GPU resources are owned values. When dropped, resources are destroyed:

```rust
{
    let buffer = Buffer::with_data(&device, &data, DataAccess::Scattered)?;
    // buffer is valid here
} // buffer is destroyed here
```

There's no hidden reference counting. If you need shared ownership, use `Arc<Buffer>`.

## The Rendering Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Vertex    │────▶│   Shader    │────▶│  Surface /  │
│   Buffer    │     │  Pipeline   │     │RenderTarget │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       │            ┌──────┴──────┐            │
       │            │             │            │
       ▼            ▼             ▼            ▼
    Vertices    Vertex Shader  Fragment    Display /
                               Shader      Pixels
```

### 1. Buffers Hold Data

```rust
// Vertex data (scattered access - any thread, any index)
let vertices = Buffer::with_data(&device, &vertex_array, DataAccess::Scattered)?;

// Index data (scattered access)
let indices = Buffer::with_data(&device, &index_array, DataAccess::Scattered)?;

// Uniform data (broadcast access - all threads read same values)
let uniforms = Buffer::new(&device, size, DataAccess::Broadcast)?;
uniforms.write(0, &data)?;
```

### 2. Shaders Process Data

```rust
// From Slang source
let shader = ShaderModule::from_slang(&device, slang_source)?;

// Built-in shaders
let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
```

### 3. Pipelines Configure Rendering

```rust
let pipeline = RenderPipeline::new(&device, &vertex_shader, &fragment_shader, &RenderPipelineDesc {
    vertex_layout: Vertex2D::layout(),
    target_format: TextureFormat::Rgba8Unorm,
    topology: PrimitiveTopology::TriangleList,
})?;
```

### 4. Commands Record Work

```rust
let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    pass.clear(Color::BLACK);
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &vertices);
    pass.draw(0..vertex_count, 0..1);
}
// encoder now contains recorded commands
```

### 5. Surfaces Present to Windows

```rust
// For window display (zero-copy)
let surface = Surface::new(&device, &window)?;
let frame = surface.acquire()?;
frame.render(encoder)?;
surface.present(frame)?;

// For headless/streaming (with optional CPU readback)
let target = RenderTarget::new(&device, width, height, TextureFormat::Rgba8Unorm)?;
target.render(encoder)?;
let pixels = target.read_to_cpu()?;  // Only when needed
```

## Vertex Types

Goldy provides `Vertex2D` for simple cases:

```rust
#[repr(C)]
pub struct Vertex2D {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex2D {
    pub fn new(x: f32, y: f32, color: Color) -> Self;
    pub fn layout() -> VertexBufferLayout;
}
```

For custom vertices, implement the layout:

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MyVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

fn my_layout() -> VertexBufferLayout {
    VertexBufferLayout {
        stride: std::mem::size_of::<MyVertex>() as u32,
        attributes: vec![
            VertexAttribute { location: 0, format: VertexFormat::Float32x3, offset: 0 },
            VertexAttribute { location: 1, format: VertexFormat::Float32x3, offset: 12 },
            VertexAttribute { location: 2, format: VertexFormat::Float32x2, offset: 24 },
        ],
    }
}
```

## Colors

```rust
// Named colors
Color::RED
Color::GREEN
Color::BLUE
Color::BLACK
Color::WHITE
Color::CORNFLOWER_BLUE

// Custom RGBA (0.0 to 1.0)
Color { r: 0.5, g: 0.2, b: 0.8, a: 1.0 }
```

## Error Handling

Goldy uses `anyhow::Result` for most operations:

```rust
fn setup() -> anyhow::Result<()> {
    let instance = Instance::new()?;  // May fail
    let device = instance.create_device(DeviceType::DiscreteGpu)?;  // May fail
    Ok(())
}
```

Common error cases:
- No compatible GPU found
- Invalid shader code
- Out of GPU memory
- Invalid buffer/pipeline usage

## Coordinate System

Goldy uses normalized device coordinates (NDC):

```
        +Y (1.0)
           │
           │
-X (-1.0) ─┼─ +X (1.0)
           │
           │
        -Y (-1.0)
```

- Center is (0, 0)
- Top is +Y, bottom is -Y
- Right is +X, left is -X
- Z ranges from 0.0 (near) to 1.0 (far)

## Primitive Topologies

```rust
PrimitiveTopology::PointList      // Individual points
PrimitiveTopology::LineList       // Pairs of vertices form lines
PrimitiveTopology::LineStrip      // Connected line segments
PrimitiveTopology::TriangleList   // Every 3 vertices form a triangle
PrimitiveTopology::TriangleStrip  // Connected triangles
```

## Next Steps

- [Buffers](../concepts/buffers.md) - Deep dive into buffer management
- [Shaders](../concepts/shaders.md) - Writing Slang shaders
- [Examples](../examples/overview.md) - See these concepts in action

