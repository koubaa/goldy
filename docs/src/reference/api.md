# API Reference

Complete API documentation for Goldy.

## Core Types

### Instance

```rust
pub struct Instance { /* ... */ }

impl Instance {
    /// Create a new Goldy instance
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
    /// Create an empty buffer with the specified access pattern
    pub fn new(device: &Device, size: u64, access: DataAccess) -> Result<Self>;
    
    /// Create a buffer with initial data
    pub fn with_data<T: Pod>(device: &Device, data: &[T], access: DataAccess) -> Result<Self>;
    
    /// Write data to the buffer
    pub fn write<T: Pod>(&self, offset: u64, data: &[T]) -> Result<()>;
    
    /// Get buffer size in bytes
    pub fn size(&self) -> u64;
    
    /// Get the buffer's access pattern
    pub fn access(&self) -> DataAccess;
}
```

### DataAccess

```rust
/// Data access pattern for buffers.
/// Describes how threads will access the buffer, determining hardware optimizations.
pub enum DataAccess {
    /// Any thread, any address, read/write. No coherence assumptions.
    /// Maps to StructuredBuffer, RWStructuredBuffer in shaders.
    Scattered,
    
    /// All threads read same address. Hardware broadcast optimization.
    /// Maps to ConstantBuffer in shaders.
    Broadcast,
}
```

## Shaders

### ShaderModule

```rust
pub struct ShaderModule { /* ... */ }

impl ShaderModule {
    /// Create shader from Slang source
    pub fn from_slang(device: &Device, source: &str) -> Result<Self>;
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

## Surface (Window Display)

### Surface

```rust
pub struct Surface { /* ... */ }

impl Surface {
    /// Create a surface for a window
    pub fn new(device: Arc<Device>, window: &impl HasWindowHandle) -> Result<Self>;
    
    /// Acquire next swapchain image
    pub fn acquire(&self) -> Result<SurfaceFrame>;
    
    /// Present a rendered frame
    pub fn present(&self, frame: SurfaceFrame) -> Result<()>;
    
    /// Resize the swapchain
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()>;
    
    /// Current dimensions
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
}
```

### SurfaceFrame

```rust
pub struct SurfaceFrame { /* ... */ }

impl SurfaceFrame {
    /// Render commands to this frame
    pub fn render(&self, encoder: CommandEncoder) -> Result<()>;
}
```

## RenderTarget (Headless/Streaming)

### RenderTarget

```rust
pub struct RenderTarget { /* ... */ }

impl RenderTarget {
    /// Create a render target
    pub fn new(device: &Device, width: u32, height: u32, format: TextureFormat) -> Result<Self>;
    
    /// Render commands to GPU texture (stays on GPU)
    pub fn render(&self, encoder: CommandEncoder) -> Result<()>;
    
    /// Explicit CPU readback (lazy staging buffer allocation)
    pub fn read_to_cpu(&self) -> Result<Vec<u8>>;
    pub fn read_to_buffer(&self, output: &mut [u8]) -> Result<()>;
    
    /// Dimensions
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn format(&self) -> TextureFormat;
    pub fn buffer_size(&self) -> usize;
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

## Texture

```rust
pub struct Texture { /* ... */ }

impl Texture {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        access: SpatialAccess,
        flags: TextureFlags,
    ) -> Result<Self>;

    pub fn write(&self, data: &[u8]) -> Result<()>;
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn format(&self) -> TextureFormat;

    /// Bindless descriptor index for use in push constants.
    pub fn bindless_index(&self) -> Option<u32>;
}
```

## Sampler

```rust
pub struct Sampler { /* ... */ }

impl Sampler {
    pub fn new(device: &Device, desc: &SamplerDesc) -> Result<Self>;
    /// Bindless descriptor index for use in push constants.
    pub fn bindless_index(&self) -> Option<u32>;
}
```

### SamplerDesc

```rust
pub struct SamplerDesc {
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
}

impl Default for SamplerDesc { /* linear + repeat */ }

pub enum FilterMode  { Nearest, Linear }
pub enum AddressMode { Repeat, MirrorRepeat, ClampToEdge, ClampToBorder }
```

### SpatialAccess

```rust
pub enum SpatialAccess {
    /// Hardware-filtered texture sampling (bilinear etc.). Maps to Texture2D + sampler.
    Interpolated,
    /// Direct pixel read/write, no filtering. Maps to RWTexture2D.
    Direct,
}
```

### TextureFlags

```rust
bitflags! {
    pub struct TextureFlags: u32 {
        const COPY_SRC = 1 << 0;
        const COPY_DST = 1 << 1;
    }
}
```

## ShaderLibrary

```rust
pub struct ShaderLibrary { /* ... */ }

impl ShaderLibrary {
    /// Create a single-module library from inline Slang source.
    pub fn from_source(name: &str, source: &str) -> Self;

    /// Load a multi-file library from a directory of .slang files.
    pub fn from_directory(name: &str, path: &Path) -> Result<Self>;
}

// Registration:
impl Device {
    pub fn register_library(&self, library: ShaderLibrary) -> Result<()>;
    pub fn has_library(&self, name: &str) -> bool;
}
```

## Compute

```rust
pub struct ComputePipeline { /* ... */ }

impl ComputePipeline {
    pub fn new(device: &Device, compute_shader: &ShaderModule) -> Result<Self>;
}

pub struct ComputeEncoder { /* ... */ }

impl ComputeEncoder {
    pub fn new() -> Self;
    pub fn begin_compute_pass(&mut self) -> ComputePass<'_>;
    /// Submit and block until complete.
    pub fn dispatch(&self, device: &Device) -> Result<()>;
    /// Submit without blocking; returns a GpuFuture.
    pub fn submit(&self, device: &Device) -> Result<GpuFuture>;
}

pub struct ComputePass<'a> { /* ... */ }

impl<'a> ComputePass<'a> {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline);
    /// Bind buffers via bindless indices (push constants).
    pub fn set_push_constants(&mut self, buffers: &[&Buffer]);
    /// Pass raw u32 indices (for textures/samplers or mixed resources).
    pub fn set_push_constants_raw(&mut self, indices: &[u32]);
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32);
    /// Workgroup counts read from buffer at offset (3 × u32).
    pub fn dispatch_indirect(&mut self, buffer: &Buffer, offset: u64);
    /// Record a buffer clear into the command stream.
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: u64);
}
```

## GpuFuture

```rust
pub struct GpuFuture { /* ... */ }

impl GpuFuture {
    /// Non-blocking poll.
    pub fn is_complete(&self) -> bool;
    /// Block until done.
    pub fn wait(&self) -> Result<()>;
    /// Block with timeout. Returns Ok(true) = done, Ok(false) = timed out.
    pub fn wait_timeout(&self, timeout_ms: u32) -> Result<bool>;
}
```

## Additional Types

### Vertex2DUv

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex2DUv {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

impl Vertex2DUv {
    pub fn new(x: f32, y: f32, u: f32, v: f32) -> Self;
    pub fn layout() -> VertexBufferLayout;
}
```

### Common 2D/3D Structs

```rust
/// Per-frame data uploaded to a Broadcast buffer each frame.
pub struct FrameUniforms {
    pub time: f32,
    pub delta_time: f32,
    pub resolution: [f32; 2],
}

/// 2D instance data for instanced rendering.
pub struct Instance2D {
    pub position: [f32; 2],
    pub scale: f32,
    pub rotation: f32,
    pub color: [f32; 4],
}

/// 2D transform matrix (column-major).
pub struct Transform2D {
    pub matrix: [[f32; 4]; 4],
}

/// 2D particle state.
pub struct Particle2D {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub life: f32,
    pub size: f32,
}

/// 3D particle state.
pub struct Particle3D {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub life: f32,
    pub size: f32,
}
```

### Depth State

```rust
pub enum DepthFormat {
    Depth32Float,
    Depth24Stencil8,
}

pub enum CompareFunction {
    Never, Less, Equal, LessEqual,
    Greater, NotEqual, GreaterEqual, Always,
}

pub struct DepthStencilState {
    pub depth_write_enabled: bool,
    pub depth_compare: CompareFunction,
}
```

### BufferFlags

```rust
bitflags! {
    pub struct BufferFlags: u32 {
        const COPY_SRC = 1 << 0;
        const COPY_DST = 1 << 1;
    }
}
```

## Re-exports

```rust
pub use types::*;
pub use device::*;
pub use buffer::*;
pub use shader::*;
pub use shader_library::*;
pub use pipeline::*;
pub use encoder::*;
pub use surface::*;
pub use render_target::*;
pub use texture::*;
pub use sampler::*;
pub use compute::*;
pub use gpu_future::*;
pub use common_types::*;
```

