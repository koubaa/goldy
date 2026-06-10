# Vertex Types and Layouts

Goldy provides built-in vertex types for common 2D rendering and a layout builder for custom vertex formats. Vertex data is described by a `VertexBufferLayout` that tells the pipeline how to interpret buffer memory.

## Built-in Vertex Types

### Vertex2D

Position + color. Use for colored primitives, particles, and debug visualization.

```rust
use goldy::{Vertex2D, Color};

let vertices = vec![
    Vertex2D::new(-0.5, -0.5, Color::RED),
    Vertex2D::new( 0.5, -0.5, Color::GREEN),
    Vertex2D::new( 0.0,  0.5, Color::BLUE),
];
```

Memory layout (24 bytes per vertex):

| Location | Field | Format | Offset |
|----------|-------|--------|--------|
| 0 | `position` | `Float32x2` | 0 |
| 1 | `color` | `Float32x4` | 8 |

Get the pipeline layout with `Vertex2D::layout()`.

### Vertex2DUv

Position + texture coordinates. Use for textured quads, sprites, and shader effects.

```rust
use goldy::Vertex2DUv;

let vertices = vec![
    Vertex2DUv::new(-1.0, -1.0, 0.0, 1.0),
    Vertex2DUv::new( 1.0, -1.0, 1.0, 1.0),
    Vertex2DUv::new( 0.0,  1.0, 0.5, 0.0),
];
```

Memory layout (16 bytes per vertex):

| Location | Field | Format | Offset |
|----------|-------|--------|--------|
| 0 | `position` | `Float32x2` | 0 |
| 1 | `uv` | `Float32x2` | 8 |

Get the pipeline layout with `Vertex2DUv::layout()`.

### Using Built-in Types in Pipelines

Both types provide a `layout()` method that returns the correct `VertexBufferLayout`:

```rust
let pipeline = RenderPipeline::new(&device, &vs, &fs, &RenderPipelineDesc {
    vertex_layout: Vertex2D::layout(),
    target_format: surface.format(),
    ..Default::default()
})?;
```

Both types implement `StructuredBufferElement`, so they can also be stored via `RetainedPool::acquire_buffer_with_data` and `BufferPool::alloc_with_data`.

## Custom Vertex Layouts

### Defining a Custom Vertex

Custom vertex types must be `#[repr(C)]` and derive `bytemuck::Pod` and `bytemuck::Zeroable`:

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MyVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: u32,
}
```

### Building a Layout with from_formats

`VertexBufferLayout::from_formats::<T>` infers locations (sequential from 0) and offsets (accumulated from format sizes), then validates that the total matches `size_of::<T>()`:

```rust
use goldy::types::{VertexBufferLayout, VertexFormat};

let layout = VertexBufferLayout::from_formats::<MyVertex>(&[
    VertexFormat::Float32x3, // position (12 bytes)
    VertexFormat::Float32x3, // normal   (12 bytes)
    VertexFormat::Float32x2, // uv       (8 bytes)
    VertexFormat::Uint32,    // color    (4 bytes)
]);
// stride = 36, 4 attributes
```

The builder panics if the summed format sizes don't equal `size_of::<T>()`, catching field-list mismatches at pipeline creation rather than producing silent GPU corruption.

### Manual Layout

For full control, construct the layout directly:

```rust
use goldy::types::{VertexBufferLayout, VertexAttribute, VertexFormat};

let layout = VertexBufferLayout {
    stride: 32,
    attributes: vec![
        VertexAttribute { location: 0, format: VertexFormat::Float32x3, offset: 0 },
        VertexAttribute { location: 1, format: VertexFormat::Float32x3, offset: 12 },
        VertexAttribute { location: 2, format: VertexFormat::Float32x2, offset: 24 },
    ],
};
```

### Empty Layout

When the vertex shader generates geometry from `SV_VertexID` (fullscreen triangles, instanced quads), use the default empty layout:

```rust
let pipeline = RenderPipeline::new(&device, &vs, &fs, &RenderPipelineDesc {
    vertex_layout: VertexBufferLayout::empty(),
    ..Default::default()
})?;
```

`VertexBufferLayout::default()` also returns an empty layout.

## Vertex Formats

Available formats for vertex attributes:

| Format | Rust Type | Size |
|--------|-----------|------|
| `Float32` | `f32` | 4 |
| `Float32x2` | `[f32; 2]` | 8 |
| `Float32x3` | `[f32; 3]` | 12 |
| `Float32x4` | `[f32; 4]` | 16 |
| `Uint32` | `u32` | 4 |
| `Sint32` | `i32` | 4 |
| `Uint8x4` | `[u8; 4]` (packed) | 4 |
| `Unorm8x4` | `[u8; 4]` (normalized) | 4 |

## Vertex Data Flow

In Slang shaders, vertex attributes arrive through the `[goldy_vertex]` virtual entry point. The pipeline's `VertexBufferLayout` determines which attributes the hardware feeds into the shader's input struct. Attribute locations in the layout must match the shader's declared input locations.

For passes that bypass vertex buffers entirely, Slang helpers like `vs_fullscreen_triangle()` and `quad_position()` in `goldy_exp.primitives` generate geometry from `SV_VertexID` and `SV_InstanceID`.
