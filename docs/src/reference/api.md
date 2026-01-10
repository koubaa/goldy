# API Reference

Complete API documentation for RAG.

## Core Types

### Instance

```rust
pub struct Instance { /* ... */ }

impl Instance {
    /// Create a new RAG instance
    pub fn new() -> Result<Self>;
    
    /// List available GPU adapters
    pub fn enumerate_adapters(&self) -> Vec<AdapterInfo>;
    
    /// Create a device for the specified type
    pub fn create_device(&self, device_type: DeviceType) -> Result<Device>;
}
```

### Device

```rust
pub struct Device { /* ... */ }

impl Device {
    /// Get adapter information
    pub fn adapter_info(&self) -> AdapterInfo;
}
```

### AdapterInfo

```rust
pub struct AdapterInfo {
    pub name: String,
    pub device_type: DeviceType,
    pub backend: BackendType,
}
```

### DeviceType

```rust
pub enum DeviceType {
    DiscreteGpu,
    IntegratedGpu,
    Cpu,
    Other,
}
```

### BackendType

```rust
pub enum BackendType {
    Vulkan,
    Metal,
    Dx12,
}
```

## Buffers

### Buffer

```rust
pub struct Buffer { /* ... */ }

impl Buffer {
    /// Create an empty buffer
    pub fn new(device: &Device, size: usize, usage: BufferUsage) -> Result<Self>;
    
    /// Create a buffer with initial data
    pub fn with_data<T: Pod>(device: &Device, data: &[T], usage: BufferUsage) -> Result<Self>;
    
    /// Write data to the buffer
    pub fn write<T: Pod>(&self, data: &[T]) -> Result<()>;
    
    /// Get buffer size in bytes
    pub fn size(&self) -> usize;
}
```

### BufferUsage

```rust
bitflags! {
    pub struct BufferUsage: u32 {
        const VERTEX = 0x01;
        const INDEX = 0x02;
        const UNIFORM = 0x04;
        const STORAGE = 0x08;
        const COPY_SRC = 0x10;
        const COPY_DST = 0x20;
    }
}
```

## Shaders

### ShaderModule

```rust
pub struct ShaderModule { /* ... */ }

impl ShaderModule {
    /// Create shader from WGSL source
    pub fn from_wgsl(device: &Device, source: &str) -> Result<Self>;
}
```

### Built-in Shaders

```rust
pub mod shader {
    pub mod builtins {
        /// Vertex shader for 2D colored vertices
        pub const VERTEX_COLOR_2D: &str = /* ... */;
    }
}
```

## Pipelines

### RenderPipeline

```rust
pub struct RenderPipeline { /* ... */ }

impl RenderPipeline {
    /// Create a new render pipeline
    pub fn new(
        device: &Device,
        vertex_shader: &ShaderModule,
        fragment_shader: &ShaderModule,
        desc: &RenderPipelineDesc,
    ) -> Result<Self>;
}
```

### RenderPipelineDesc

```rust
pub struct RenderPipelineDesc {
    pub vertex_layout: VertexBufferLayout,
    pub target_format: TextureFormat,
    pub topology: PrimitiveTopology,
}

impl Default for RenderPipelineDesc {
    fn default() -> Self;
}
```

### VertexBufferLayout

```rust
pub struct VertexBufferLayout {
    pub stride: u32,
    pub attributes: Vec<VertexAttribute>,
}
```

### VertexAttribute

```rust
pub struct VertexAttribute {
    pub location: u32,
    pub format: VertexFormat,
    pub offset: u32,
}
```

### VertexFormat

```rust
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Sint32,
    Uint8x4,
}
```

### PrimitiveTopology

```rust
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}
```

### TextureFormat

```rust
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8Srgb,
    Bgra8Unorm,
    Rgba16Float,
    Rgba32Float,
    Depth32Float,
}
```

## Commands

### CommandEncoder

```rust
pub struct CommandEncoder { /* ... */ }

impl CommandEncoder {
    /// Create a new command encoder
    pub fn new() -> Self;
    
    /// Begin a render pass
    pub fn begin_render_pass(&mut self) -> RenderPass<'_>;
}
```

### RenderPass

```rust
pub struct RenderPass<'a> { /* ... */ }

impl<'a> RenderPass<'a> {
    /// Clear the render target
    pub fn clear(&mut self, color: Color);
    
    /// Set the active pipeline
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline);
    
    /// Bind a vertex buffer to a slot
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &Buffer);
    
    /// Draw primitives
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
}
```

## Frame Output

### FrameOutput

```rust
pub struct FrameOutput { /* ... */ }

impl FrameOutput {
    /// Create a new frame output
    pub fn new(device: &Device, width: u32, height: u32, format: TextureFormat) -> Self;
    
    /// Execute commands and get pixel data
    pub fn render(self, encoder: CommandEncoder) -> Result<Vec<u8>>;
}
```

## Vertex Types

### Vertex2D

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex2D {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex2D {
    pub fn new(x: f32, y: f32, color: Color) -> Self;
    pub fn layout() -> VertexBufferLayout;
}
```

### Color

```rust
#[derive(Clone, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const RED: Color;
    pub const GREEN: Color;
    pub const BLUE: Color;
    pub const BLACK: Color;
    pub const WHITE: Color;
    pub const CORNFLOWER_BLUE: Color;
}
```

## Re-exports

```rust
pub use types::*;
pub use device::*;
pub use buffer::*;
pub use shader::*;
pub use pipeline::*;
pub use encoder::*;
pub use frame::*;
```

