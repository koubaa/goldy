# Buffers

Buffers store data on the GPU - uniform data, shader storage, etc.

## Creating Buffers

### With Initial Data

```rust
use goldy::{Buffer, DataAccess};

let data = [1.0f32, 2.0, 3.0, 4.0];
let buffer = Buffer::with_data(&device, &data, DataAccess::Broadcast)?;
```

### Empty Buffer

```rust
let buffer = Buffer::new(&device, size_in_bytes, DataAccess::Scattered)?;
```

## Data Access Patterns

Goldy uses access-pattern-based resource binding instead of traditional graphics categories. This describes **how** threads access data:

```rust
pub enum DataAccess {
    /// Any thread, any address, read/write. No coherence assumptions.
    /// Maps to StructuredBuffer, RWStructuredBuffer in shaders.
    Scattered,
    
    /// All threads read same address. Hardware broadcast optimization.
    /// Maps to ConstantBuffer in shaders.
    Broadcast,
}
```

Choose based on access pattern:
- **Scattered**: General-purpose storage (particles, compute data)
- **Broadcast**: Uniform data (transforms, time, settings)

## Writing Data

### Full Replace

```rust
let buffer = Buffer::new(&device, size, DataAccess::Scattered)?;
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

## Vertex Data (Bindless)

Goldy uses bindless vertex access - vertex data is stored in `Scattered` buffers 
and accessed directly in shaders via bindless descriptors:

```rust
// Store vertex data in a scattered buffer
let vertices = Buffer::with_data(&device, &vertex_data, DataAccess::Scattered)?;

// In shader: access via bindless index
// StructuredBuffer<Vertex> vertices = getBuffer(push_constants.vertex_buffer_index);
```

### Built-in Vertex2D

Goldy provides `Vertex2D` for common 2D rendering:

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

## Index Data (Bindless)

Index data is also stored as `Scattered` buffers:

```rust
let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];  // Two triangles
let index_buffer = Buffer::with_data(&device, &indices, DataAccess::Scattered)?;

// In render pass
pass.set_index_buffer(&index_buffer, IndexFormat::Uint16);
pass.draw_indexed(0..6, 0, 0..1);
```

## Uniform/Constant Buffers

For data that all threads read from the same address (broadcast pattern):

```rust
#[repr(C)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    time: f32,
}

let uniforms = Buffer::with_data(&device, &[data], DataAccess::Broadcast)?;
```

## Performance Tips

### Batch Updates

Create buffers with all data at once when possible:

```rust
// Good: single allocation
let data = generate_all_data();
let buffer = Buffer::with_data(&device, &data, DataAccess::Scattered)?;

// Less efficient: multiple writes
let buffer = Buffer::new(&device, size, DataAccess::Scattered)?;
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
let element_size = std::mem::size_of::<MyData>();
let total_size = element_size * num_elements;
let buffer = Buffer::new(&device, total_size, DataAccess::Scattered)?;
```

## Ownership

Buffers are owned resources. When dropped, GPU memory is freed:

```rust
{
    let buffer = Buffer::new(&device, size, DataAccess::Scattered)?;
    // buffer is valid
} // buffer destroyed, GPU memory freed
```

For shared ownership, use `Arc<Buffer>`:

```rust
let buffer = Arc::new(Buffer::new(&device, size, DataAccess::Scattered)?);
let buffer2 = buffer.clone();  // Same buffer, reference counted
```

