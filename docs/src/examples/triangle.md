# Triangle Example

The simplest RAG example - a colored triangle.

<div class="rag-demo" data-canvas="triangle-canvas" data-demo="TriangleDemo">
    <canvas id="triangle-canvas"></canvas>
</div>

*Interactive demo running via WebGPU. Requires Chrome 113+, Edge 113+, or Firefox with WebGPU enabled.*

## Run It

```bash
cargo run --example triangle --release
```

## What It Demonstrates

- GPU device initialization
- Vertex buffer creation
- Using built-in shaders
- Basic render pipeline
- Command encoding

## Key Code

### Vertices

```rust
let vertices = [
    Vertex2D::new(0.0, -0.5, Color::RED),    // Top (red)
    Vertex2D::new(-0.5, 0.5, Color::GREEN),  // Bottom-left (green)
    Vertex2D::new(0.5, 0.5, Color::BLUE),    // Bottom-right (blue)
];
let buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;
```

### Built-in Shader

RAG includes `builtins::VERTEX_COLOR_2D`, a simple shader that:
- Passes through 2D positions
- Interpolates vertex colors across the triangle

```rust
let shader = ShaderModule::from_wgsl(&device, builtins::VERTEX_COLOR_2D)?;
```

### Rendering

```rust
let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    pass.clear(Color { r: 0.1, g: 0.1, b: 0.2, a: 1.0 });
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &buffer);
    pass.draw(0..3, 0..1);  // 3 vertices, 1 instance
}
```

## Variations to Try

### Different Colors

```rust
let vertices = [
    Vertex2D::new(0.0, -0.5, Color { r: 1.0, g: 0.5, b: 0.0, a: 1.0 }),
    Vertex2D::new(-0.5, 0.5, Color { r: 0.0, g: 0.5, b: 1.0, a: 1.0 }),
    Vertex2D::new(0.5, 0.5, Color { r: 0.5, g: 1.0, b: 0.0, a: 1.0 }),
];
```

### Animated Position

```rust
let time = start_time.elapsed().as_secs_f32();
let vertices = [
    Vertex2D::new(time.sin() * 0.3, -0.5, Color::RED),
    Vertex2D::new(-0.5, 0.5 + time.cos() * 0.1, Color::GREEN),
    Vertex2D::new(0.5, 0.5, Color::BLUE),
];
// Recreate buffer each frame for animation
let buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;
```

### Multiple Triangles

```rust
let vertices = [
    // First triangle
    Vertex2D::new(-0.5, -0.5, Color::RED),
    Vertex2D::new(-0.7, 0.3, Color::GREEN),
    Vertex2D::new(-0.3, 0.3, Color::BLUE),
    // Second triangle
    Vertex2D::new(0.5, -0.5, Color::YELLOW),
    Vertex2D::new(0.3, 0.3, Color::CYAN),
    Vertex2D::new(0.7, 0.3, Color::MAGENTA),
];
// ...
pass.draw(0..6, 0..1);  // 6 vertices
```

## Full Source

See `rag/examples/triangle.rs` for the complete code.

