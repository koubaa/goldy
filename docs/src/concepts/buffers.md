# Buffers

Buffers store data on the GPU - vertices, indices, uniform data, etc.

## Creating Buffers

### With Initial Data

```rust
use rag::{Buffer, BufferUsage};

let vertices = [
    Vertex2D::new(0.0, -0.5, Color::RED),
    Vertex2D::new(-0.5, 0.5, Color::GREEN),
    Vertex2D::new(0.5, 0.5, Color::BLUE),
];

let buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;
```

### Empty Buffer

```rust
let buffer = Buffer::new(&device, size_in_bytes, BufferUsage::VERTEX)?;
```

## Buffer Usage

```rust
bitflags! {
    pub struct BufferUsage: u32 {
        const VERTEX = 0x01;    // Vertex data
        const INDEX = 0x02;     // Index data
        const UNIFORM = 0x04;   // Uniform/constant data
        const STORAGE = 0x08;   // Shader storage
        const COPY_SRC = 0x10;  // Copy source
        const COPY_DST = 0x20;  // Copy destination
    }
}
```

Usages can be combined:

```rust
let buffer = Buffer::new(
    &device, 
    size, 
    BufferUsage::VERTEX | BufferUsage::COPY_DST
)?;
```

## Writing Data

### Full Replace

```rust
let buffer = Buffer::new(&device, size, BufferUsage::VERTEX)?;
buffer.write(&data)?;
```

### With Offset (future)

```rust
// Coming soon
buffer.write_at(offset, &data)?;
```

## Buffer Size

```rust
let size = buffer.size();  // Size in bytes
```

## Memory Layout

Data must be `Pod` (plain old data) and correctly aligned:

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MyVertex {
    position: [f32; 3],  // 12 bytes
    color: [f32; 4],     // 16 bytes
}
// Total: 28 bytes per vertex
```

Use `bytemuck` for safe casting:

```rust
let bytes: &[u8] = bytemuck::cast_slice(&vertices);
```

## Vertex Buffers

For vertex buffers, you need to describe the layout:

```rust
let layout = VertexBufferLayout {
    stride: std::mem::size_of::<MyVertex>() as u32,
    attributes: vec![
        VertexAttribute { 
            location: 0, 
            format: VertexFormat::Float32x3, 
            offset: 0 
        },
        VertexAttribute { 
            location: 1, 
            format: VertexFormat::Float32x4, 
            offset: 12 
        },
    ],
};
```

### Built-in Vertex2D

RAG provides `Vertex2D` for common 2D rendering:

```rust
pub struct Vertex2D {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex2D {
    pub fn new(x: f32, y: f32, color: Color) -> Self;
    pub fn layout() -> VertexBufferLayout;
}
```

## Index Buffers

```rust
let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];  // Two triangles
let index_buffer = Buffer::with_data(&device, &indices, BufferUsage::INDEX)?;

// In render pass
pass.set_index_buffer(&index_buffer, IndexFormat::Uint16);
pass.draw_indexed(0..6, 0, 0..1);
```

## Uniform Buffers (future)

```rust
#[repr(C)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    time: f32,
}

let uniforms = Buffer::with_data(&device, &[data], BufferUsage::UNIFORM)?;
```

## Performance Tips

### Batch Updates

Create buffers with all data at once when possible:

```rust
// Good: single allocation
let vertices = generate_all_vertices();
let buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;

// Less efficient: multiple writes
let buffer = Buffer::new(&device, size, BufferUsage::VERTEX)?;
buffer.write(&part1)?;
buffer.write_at(offset, &part2)?;  // Multiple writes
```

### Reuse Buffers

For dynamic data, consider buffer pools:

```rust
struct BufferPool {
    buffers: Vec<Buffer>,
    index: usize,
}

impl BufferPool {
    fn get_or_create(&mut self, device: &Device, size: usize) -> &Buffer {
        // Reuse existing buffer if large enough
        // Or create new one
    }
}
```

### Buffer Size

GPU memory is precious. Size buffers appropriately:

```rust
// Know your data size
let vertex_size = std::mem::size_of::<MyVertex>();
let total_size = vertex_size * num_vertices;
let buffer = Buffer::new(&device, total_size, BufferUsage::VERTEX)?;
```

## Ownership

Buffers are owned resources. When dropped, GPU memory is freed:

```rust
{
    let buffer = Buffer::new(&device, size, usage)?;
    // buffer is valid
} // buffer destroyed, GPU memory freed
```

For shared ownership, use `Arc<Buffer>`:

```rust
let buffer = Arc::new(Buffer::new(&device, size, usage)?);
let buffer2 = buffer.clone();  // Same buffer, reference counted
```

