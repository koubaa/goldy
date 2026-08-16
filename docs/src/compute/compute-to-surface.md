# Compute to Surface

Compute-to-surface lets a compute shader write directly to a swapchain drawable, bypassing the rasterization pipeline entirely. There is no `RenderPipeline`, no vertex buffers, and no raster pass — just a compute dispatch that fills pixels.

## When to use compute-to-surface

Use compute-to-surface when your rendering is naturally a per-pixel computation rather than geometry rasterization:

- Fullscreen image effects (plasma, fractals, ray marching)
- GPU-driven 2D renderers where the compute shader owns the output layout
- Post-processing that doesn't need triangle rasterization
- Prototyping visual effects without setting up a render pipeline

Use traditional rendering when you need the rasterization pipeline's features: triangle assembly, depth testing, MSAA, alpha blending, or vertex/fragment shader stages.

## Surface exchange

Create a [`SurfaceExchange`](../surfaces/overview.md) and call `bind_destination` to register direct compute-to-present in the scheme:

```rust
let surface = SurfaceExchange::new_with_depth(&ctx, &window, 3, SurfaceConfig::default())?;
let (lease, present) = surface.bind_destination(&mut scheme)?;
```

Bind the returned lease in a compute node with `with_present(&lease)`. Goldy handles barrier insertion between compute writes and the presentation engine.

On CUDA+DX12, present scratch is a depth-3 imported staging ring (not the DXGI
backbuffer). Compute does not reuse frame N's scratch in frame N+1; that extra
memory is the interop tradeoff until CUDA/DX12 synchronization APIs improve.

## Building the scheme

Record a retained scheme with a compute node that writes to the present lease:

```rust
let wg_x = width.div_ceil(8);
let wg_y = height.div_ceil(8);

let mut scheme = Scheme::new(&ctx);
let (lease, present) = surface.bind_destination(&mut scheme)?;
scheme
    .node("compute", &compute_pipeline)
    .with_parcel(&uniform_buffer, NodeAccess::Read)
    .with_present(&lease)
    .dispatch(wg_x, wg_y, 1);
```

## Submitting and presenting

Each frame, submit the scheme and consume the surface claim:

```rust
let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

`submit` resolves transient resources, compiles the scheme into a command stream, and submits to the GPU. Presentation happens when you call `claim(...).consume()` — the compute shader has already written the pixels.

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

## Full example sketch

```rust
use goldy::{
    BufferKind, ComputePipeline, DeviceDescriptor, Instance, MemoryExchange, NodeAccess, PresentMode,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, SurfaceConfig, SurfaceExchange,
};

let instance = Instance::new()?;
let device = instance
    .request_adapter(&RequestAdapterOptions::default())?
    .request_device(&DeviceDescriptor::default())?;
let ctx = device.create_context()?;

let surface = SurfaceExchange::new_with_config(
    &ctx,
    &window,
    SurfaceConfig {
        present_mode: PresentMode::Fifo,
        depth_format: None,
    },
)?;

let shader = ShaderModule::from_slang(&device, COMPUTE_SHADER)?;
let compute_pipeline = ComputePipeline::new(&device, &shader)?;

let mut retained_pool = RetainedPool::new(device.clone());
let uniform_buffer = retained_pool.acquire_buffer_with_data(
    &[Uniforms { width, height, time: 0.0, _padding: 0.0 }],
    BufferKind::Scattered,
)?;

let mut scheme = Scheme::new(&ctx);
let (lease, present) = surface.bind_destination(&mut scheme)?;
scheme
    .node("compute", &compute_pipeline)
    .with_parcel(&uniform_buffer, NodeAccess::Read)
    .with_present(&lease)
    .dispatch(width.div_ceil(8), height.div_ceil(8), 1);

// --- Render loop ---
let mut upload = Scheme::new(&ctx);
let uniform_deposit = MemoryExchange::new(&ctx).bind_deposit_buffer(
    &mut upload,
    &uniform_buffer,
    std::mem::size_of::<Uniforms>() as u64,
)?;
uniform_deposit.write(
    &mut upload,
    0,
    bytemuck::bytes_of(&Uniforms { width, height, time: elapsed, _padding: 0.0 }),
)?;
upload.submit()?;

let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

See [`examples/compute_to_surface.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/compute_to_surface.rs) for the complete winit application.
