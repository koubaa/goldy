# Compute to Surface

Compute-to-surface lets a compute shader write directly to the swapchain texture, bypassing the rasterization pipeline entirely. There is no `RenderPipeline`, no vertex buffers, no `CommandEncoder` — just a compute dispatch that fills pixels.

## When to use compute-to-surface

Use compute-to-surface when your rendering is naturally a per-pixel computation rather than geometry rasterization:

- Fullscreen image effects (plasma, fractals, ray marching)
- GPU-driven 2D renderers where the compute shader owns the output layout
- Post-processing that doesn't need triangle rasterization
- Prototyping visual effects without setting up a render pipeline

Use traditional rendering when you need the rasterization pipeline's features: triangle assembly, depth testing, MSAA, alpha blending, or vertex/fragment shader stages.

## Getting the swapchain texture

Acquire a frame from the surface and call `frame.texture()` to get a writable `Texture` handle to the current swapchain image:

```rust
let frame = surface.begin()?;
let texture = frame.texture();
```

This texture is valid until the frame is presented. You can obtain its bindless handle and pass it to a compute shader like any other texture:

```rust
let texture_handle = texture
    .bindless_handle()
    .expect("Surface texture has no bindless handle");
```

## Building the task graph

Create a `TaskGraph` with a compute node that writes to the swapchain texture. The task graph handles barrier insertion between compute writes and the presentation engine:

```rust
let wg_x = width.div_ceil(8);
let wg_y = height.div_ceil(8);

let mut graph = TaskGraph::new();
graph.node("compute", &compute_pipeline)
    .bind_buffer(&uniform_buffer, NodeAccess::Read)
    .bind_resources_raw(&[uniform_handle.index(), texture_handle.index()])
    .dispatch(wg_x, wg_y, 1);
```

## Submitting and presenting

Use `frame.submit_compute(graph)` to record the compute work into the frame, then present:

```rust
frame.submit_compute(&graph)?;
frame.present()?;
```

`submit_compute` compiles the task graph into a command stream and records it into the frame's command buffer. Presentation happens when you call `present()` — the compute shader has already written the pixels.

## The compute shader

The shader receives the output texture as a `DirectSpatial<float4>` — a read-write 2D texture accessed by integer coordinates:

```slang
import goldy_exp;

struct Uniforms {
    uint width;
    uint height;
    float time;
    float _padding;
};

[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(BufRO<Uniforms> uniforms_buf, DirectSpatial<float4> output, ThreadId tid) {
    Uniforms u = uniforms_buf[0];

    if (tid.x >= u.width || tid.y >= u.height)
        return;

    float2 uv = float2(float(tid.x) / float(u.width),
                       float(tid.y) / float(u.height));

    // Compute pixel color...
    float3 col = my_color_function(uv, u.time);
    output[tid.xy] = float4(col, 1.0);
}
```

The `[numthreads(8, 8, 1)]` workgroup size maps naturally to 2D image tiles. Dispatch enough workgroups to cover the full resolution:

```rust
let wg_x = width.div_ceil(8);
let wg_y = height.div_ceil(8);
```

Guard against out-of-bounds writes in the shader when the resolution isn't a multiple of the workgroup size.

## Full example

A complete compute-to-surface application rendering an animated plasma effect:

```rust
use goldy::{
    Buffer, ComputePipeline, DataAccess, DeviceType, Instance,
    NodeAccess, PresentMode, ShaderModule, Surface, SurfaceConfig, TaskGraph,
};

// Create device and surface
let instance = Instance::new()?;
let device = instance.create_device(DeviceType::DiscreteGpu)?;

let surface = Surface::new_with_config(
    &device,
    &window,
    SurfaceConfig {
        present_mode: PresentMode::Fifo,
        depth_format: None,
    },
)?;

// Compile compute shader and create pipeline
let shader = ShaderModule::from_slang(&device, COMPUTE_SHADER)?;
let compute_pipeline = ComputePipeline::new(&device, &shader)?;

// Create uniform buffer
let uniform_buffer = Buffer::with_data(
    &device,
    &[Uniforms {
        width: surface.width(),
        height: surface.height(),
        time: 0.0,
        _padding: 0.0,
    }],
    DataAccess::Scattered,
)?;

// --- Render loop ---

// Update uniforms
uniform_buffer.write(0, bytemuck::bytes_of(&Uniforms {
    width, height, time: elapsed, _padding: 0.0,
}))?;

// Acquire frame and get swapchain texture
let frame = surface.begin()?;
let texture = frame.texture();

let uniform_handle = uniform_buffer
    .bindless_srv_handle()
    .expect("Uniform buffer has no bindless SRV handle");
let texture_handle = texture
    .bindless_handle()
    .expect("Surface texture has no bindless handle");

// Build and submit compute graph
let wg_x = width.div_ceil(8);
let wg_y = height.div_ceil(8);

let mut graph = TaskGraph::new();
graph.node("compute", &compute_pipeline)
    .bind_buffer(&uniform_buffer, NodeAccess::Read)
    .bind_resources_raw(&[uniform_handle.index(), texture_handle.index()])
    .dispatch(wg_x, wg_y, 1);

frame.submit_compute(&graph)?;
frame.present()?;
```

The uniform buffer uses `bindless_srv_handle()` because the shader accesses it through `BufRO<Uniforms>`, which maps to a read-only SRV on DX12. On Vulkan and Metal this falls back to the unified storage-buffer index.
