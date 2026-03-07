# What Goldy Sheds

Because Goldy targets **modern hardware only**, it can drop significant complexity that other libraries must maintain.

## From Vulkan Legacy

| Dropped | Replacement | Why |
|---------|-------------|-----|
| Render pass objects | Dynamic rendering (1.4+) | Modern GPUs don't need ahead-of-time render pass specification |
| Descriptor set layouts | Bindless descriptors (1.2+) | Index into descriptor heaps directly |
| Image layout transitions | Unified layouts (extension) | Hardware handles hazards automatically |
| Pipeline layouts as objects | Implicit from shader reflection | Redundant with shader metadata |
| VkBuffer + VkDeviceMemory split | Unified allocation | Modern allocators handle this |
| Separate transfer queues | Simplified queue model | Async compute is the common case |

## From OpenGL Heritage

| Dropped | Notes |
|---------|-------|
| Fixed-function pipeline | Shaders only - no legacy transform/lighting |
| Immediate mode | Command buffers only |
| Binding points (GL_TEXTURE0, etc.) | Bindless access by index |
| Client-side vertex arrays | GPU buffers only |
| glGet* state queries | Explicit state tracking |
| Per-shader uniforms | Shared uniform model |

## From OpenCL Complexity

| Dropped | Notes |
|---------|-------|
| Separate compute API | Unified with graphics pipeline |
| Platform enumeration | Simplified device model |
| NDRange complexity | Simple dispatch(x, y, z) |
| Kernel argument binding | Bindless buffers |

## What This Enables

By dropping legacy support, Goldy can assume:

### 1. Coherent Caches
No complex flush/invalidate patterns. Modern GPUs have coherent L2 caches that handle synchronization automatically.

### 2. Bindless Everything
Descriptors live in GPU memory. Shaders access resources by index:

```rust
// Goldy - resources are bindless
let buffer = device.create_buffer(&desc)?;
// Access in shader by index, no explicit binding
```

### 3. Unified Queues
Graphics and compute on the same queue. No complex multi-queue synchronization for common cases.

### 4. 64-bit Pointers
Buffer device address in shaders. Pointers just work:

```wgsl
// In shader - direct memory access
let data = buffer_ptr[index];
```

### 5. Collapsed Pipeline Permutations
The industry's PSO explosion comes from the *product* of many baked-in dimensions: render pass × descriptor layout × pipeline layout × viewport state × blend mode × .... Goldy collapses most of these:

- **Dynamic rendering** eliminates render pass compatibility as a pipeline dimension.
- **One global bindless layout** eliminates descriptor set layout and pipeline layout permutations.
- **Dynamic state** (viewport, scissor) removes those from the baked PSO.

What remains — shader × vertex format × target format × depth config — is a small, manageable space. Goldy addresses PSO churn by having fewer pipelines, not by building infrastructure to manage many variants.

## The Simplicity Dividend

A typical Vulkan "hello triangle" requires:
- Instance creation with extensions
- Physical device selection
- Logical device and queue creation
- Swapchain setup
- Render pass creation
- Framebuffer creation
- Pipeline layout
- Graphics pipeline with all state
- Command pool and buffers
- Synchronization primitives
- Drawing loop with acquire/present

Goldy's equivalent:

```rust
let instance = Instance::new()?;
let device = Arc::new(instance.create_device(DeviceType::DiscreteGpu)?);
let shader = ShaderModule::from_slang(&device, SHADER)?;
let pipeline = RenderPipeline::new(&device, &shader, &shader, &desc)?;

// Create surface for zero-copy window presentation
let surface = Surface::new(&device, &window)?;

// Render loop
let frame = surface.acquire()?;
let mut encoder = CommandEncoder::new();
{
    let mut pass = encoder.begin_render_pass();
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &vertices);
    pass.draw(0..3, 0..1);
}
frame.render(encoder)?;
surface.present(frame)?;
```

That's it. No render passes, no framebuffers, no command pools, no explicit synchronization. Goldy handles the complexity internally using modern GPU features.


