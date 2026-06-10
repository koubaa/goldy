# Your First Compute Shader

This tutorial renders an animated plasma effect by dispatching a compute shader directly to the swapchain texture — no graphics pipeline, no vertex buffers, no render passes.

## The Shader

The compute shader uses `goldy_exp` virtual entry points. It reads uniforms via `BufRO<Uniforms>` and writes pixels to the swapchain texture via `DirectSpatial<float4>`:

```c
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
    float2 p = uv * 2.0 - 1.0;
    p.x *= float(u.width) / float(u.height);

    float t = u.time;
    float v = 0.0;
    v += sin(p.x * 6.0 + t);
    v += sin(p.y * 6.0 + t * 1.3);
    v += sin((p.x + p.y) * 4.0 + t * 0.7);
    v += sin(length(p) * 8.0 - t * 2.0);
    v *= 0.25;

    float3 col = float3(0.5 + 0.5 * sin(v * 3.14159 + 0.0),
                        0.5 + 0.5 * sin(v * 3.14159 + 2.094),
                        0.5 + 0.5 * sin(v * 3.14159 + 4.188));
    output[tid.xy] = float4(col, 1.0);
}
```

Key points:

- `BufRO<Uniforms>` is a read-only structured buffer. Index with `[0]` to load the single element.
- `DirectSpatial<float4>` is an `RWTexture2D<float4>` — write to it with `output[tid.xy]`.
- `ThreadId` maps to `SV_DispatchThreadID`. Each thread handles one pixel.
- The `[goldy_compute]` attribute tells the Goldy compiler to wire up bindless slots automatically.

## Rust Side

### Uniform Buffer

Define the uniform struct on the Rust side with matching layout:

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    time: f32,
    _padding: f32,
}
impl goldy::StructuredBufferElement for Uniforms {}
```

Create the buffer with `BufferKind::Scattered` so it gets a bindless descriptor:

```rust
let mut retained_pool = RetainedPool::new(device.clone());
let uniform_buffer = retained_pool.acquire_buffer_with_data(
    &[Uniforms { width, height, time: 0.0, _padding: 0.0 }],
    BufferKind::Scattered,
)?;
```

Pass a typed `&[Uniforms]` slice, not raw bytes. `acquire_buffer_with_data` uses `size_of::<T>()` as the structured-buffer stride, which backends rely on for correct addressing.

### Compute Pipeline

Compile the Slang source and create a `ComputePipeline`:

```rust
let shader = ShaderModule::from_slang(&device, COMPUTE_SHADER)?;
let compute_pipeline = ComputePipeline::new(&device, &shader)?;
```

### Rendering a Frame

Each frame follows this pattern: update uniforms, acquire the swapchain texture, build a `TaskGraph`, submit, present.

```rust
fn render_frame(state: &mut RenderState) -> Result<()> {
    let (width, height) = state.surface.size();
    let elapsed = state.start_time.elapsed().as_secs_f32();

    state.uniform_buffer.write(
        0,
        bytemuck::bytes_of(&Uniforms {
            width,
            height,
            time: elapsed,
            _padding: 0.0,
        }),
    )?;

    let frame = state.surface.begin()?;
    let texture = frame.texture();

    let wg_x = width.div_ceil(8);
    let wg_y = height.div_ceil(8);

    let uniform_handle = state.uniform_buffer
        .handle(ResourceAccess::Read)
        .expect("Uniform buffer has no bindless SRV handle");
    let texture_handle = texture
        .handle(ResourceAccess::Read)
        .expect("Surface texture has no bindless handle");

    let mut graph = TaskGraph::new();
    graph
        .node("compute", &state.compute_pipeline)
        .bind_buffer(&state.uniform_buffer, NodeAccess::Read)
        .bind_resources_raw(&[uniform_handle.index(), texture_handle.index()])
        .dispatch(wg_x, wg_y, 1);

    frame.submit_compute(&graph)?;
    frame.present()?;
    Ok(())
}
```

### Step by Step

**Update uniforms** — `Buffer::write` uploads new time/size values each frame.

**Acquire the frame** — `surface.begin()` returns a `Frame`. `frame.texture()` gives you the swapchain `Texture` for this frame.

**Get bindless handles** — `bindless_srv_handle()` returns the read-only descriptor index for the uniform buffer. `bindless_handle()` returns the storage-image descriptor index for the swapchain texture. These indices are passed to the shader as the `BufRO<Uniforms>` and `DirectSpatial<float4>` slots respectively.

**Build the TaskGraph** — `graph.node()` creates a compute node bound to a pipeline. `bind_buffer()` declares the dependency (the uniform buffer is read). `bind_resources_raw()` passes the bindless descriptor indices as push-constant slots. `dispatch()` sets the workgroup count.

**Submit and present** — `frame.submit_compute(&graph)` records and submits the compute work to the GPU. `frame.present()` presents the swapchain image. The compute shader already wrote the pixels — there is no blit or copy step.

## Run It

```bash
cargo run --example compute_to_surface
```

You should see an animated plasma pattern filling the window, rendered entirely from compute.

## Next Steps

- [Task Graph](../compute/task-graph.md) — multi-node graphs, transient resources, indirect dispatch
- [Examples](../examples/overview.md) — particles, game of life, and more compute examples
