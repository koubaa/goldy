# Pipelines

A `RenderPipeline` combines shaders with rendering configuration.

## Creating a Pipeline

```rust
use rag::{RenderPipeline, RenderPipelineDesc, TextureFormat, PrimitiveTopology};

let pipeline = RenderPipeline::new(
    &device,
    &vertex_shader,
    &fragment_shader,
    &RenderPipelineDesc {
        vertex_layout: Vertex2D::layout(),
        target_format: TextureFormat::Rgba8Unorm,
        topology: PrimitiveTopology::TriangleList,
    }
)?;
```

## Pipeline Description

```rust
pub struct RenderPipelineDesc {
    /// Vertex buffer layout
    pub vertex_layout: VertexBufferLayout,
    
    /// Output texture format
    pub target_format: TextureFormat,
    
    /// How vertices form primitives
    pub topology: PrimitiveTopology,
}
```

## Vertex Layout

Describes how vertex data is structured:

```rust
pub struct VertexBufferLayout {
    /// Bytes between consecutive vertices
    pub stride: u32,
    
    /// Attribute descriptions
    pub attributes: Vec<VertexAttribute>,
}

pub struct VertexAttribute {
    /// Shader location (@location(n))
    pub location: u32,
    
    /// Data type
    pub format: VertexFormat,
    
    /// Byte offset within vertex
    pub offset: u32,
}
```

### Example Layout

```rust
#[repr(C)]
struct MyVertex {
    position: [f32; 3],  // 12 bytes at offset 0
    normal: [f32; 3],    // 12 bytes at offset 12
    uv: [f32; 2],        // 8 bytes at offset 24
}
// Total stride: 32 bytes

let layout = VertexBufferLayout {
    stride: 32,
    attributes: vec![
        VertexAttribute { location: 0, format: VertexFormat::Float32x3, offset: 0 },
        VertexAttribute { location: 1, format: VertexFormat::Float32x3, offset: 12 },
        VertexAttribute { location: 2, format: VertexFormat::Float32x2, offset: 24 },
    ],
};
```

## Vertex Formats

```rust
pub enum VertexFormat {
    Float32,        // f32
    Float32x2,      // vec2<f32>
    Float32x3,      // vec3<f32>
    Float32x4,      // vec4<f32>
    Uint32,         // u32
    Sint32,         // i32
    Uint8x4,        // vec4<u32> (packed)
    // ... more
}
```

## Texture Formats

```rust
pub enum TextureFormat {
    Rgba8Unorm,     // Standard 8-bit RGBA
    Rgba8Srgb,      // sRGB color space
    Bgra8Unorm,     // Swapped channels
    Rgba16Float,    // HDR
    Rgba32Float,    // Full precision
    Depth32Float,   // Depth buffer
    // ... more
}
```

## Primitive Topology

How vertices are assembled into primitives:

```rust
pub enum PrimitiveTopology {
    /// Each vertex is a point
    PointList,
    
    /// Every 2 vertices form a line
    LineList,
    
    /// Vertices form connected line segments
    LineStrip,
    
    /// Every 3 vertices form a triangle
    TriangleList,
    
    /// Vertices form connected triangles
    TriangleStrip,
}
```

### Visual Reference

```
PointList:     •  •  •  •

LineList:      •——•  •——•

LineStrip:     •——•——•——•

TriangleList:  △  △

TriangleStrip: △▽△▽
```

## Using Pipelines

```rust
let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    pass.set_pipeline(&pipeline);  // Bind the pipeline
    pass.set_vertex_buffer(0, &vertices);
    pass.draw(0..vertex_count, 0..1);
}
```

## Multiple Pipelines

You can switch pipelines within a render pass:

```rust
{
    let mut pass = encoder.begin_render_pass();
    pass.clear(Color::BLACK);
    
    // Draw triangles
    pass.set_pipeline(&triangle_pipeline);
    pass.set_vertex_buffer(0, &triangle_vertices);
    pass.draw(0..triangle_count, 0..1);
    
    // Draw lines
    pass.set_pipeline(&line_pipeline);
    pass.set_vertex_buffer(0, &line_vertices);
    pass.draw(0..line_count, 0..1);
}
```

## Pipeline State

In traditional APIs, pipeline state includes:
- Blend mode
- Depth/stencil testing
- Culling mode
- Polygon mode (fill/wireframe)

RAG currently uses sensible defaults:
- Alpha blending disabled
- No depth testing
- No culling
- Fill mode

Future versions will expose these options.

## Default Pipeline

```rust
impl Default for RenderPipelineDesc {
    fn default() -> Self {
        Self {
            vertex_layout: VertexBufferLayout::default(),
            target_format: TextureFormat::Rgba8Unorm,
            topology: PrimitiveTopology::TriangleList,
        }
    }
}
```

## Performance

Pipelines are expensive to create but cheap to use. Create them once at startup:

```rust
struct Renderer {
    triangle_pipeline: RenderPipeline,
    line_pipeline: RenderPipeline,
    point_pipeline: RenderPipeline,
}

impl Renderer {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            triangle_pipeline: create_triangle_pipeline(device)?,
            line_pipeline: create_line_pipeline(device)?,
            point_pipeline: create_point_pipeline(device)?,
        })
    }
}
```

