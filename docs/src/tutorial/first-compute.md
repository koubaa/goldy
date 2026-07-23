# Your First Compute Shader

This tutorial renders an animated plasma effect by dispatching a compute shader directly to a swapchain drawable — no graphics pipeline, no vertex buffers, no render passes.

See [`examples/compute_to_surface.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/compute_to_surface.rs) for the full source.

## The Shader

The compute shader uses `goldy_exp` virtual entry points. It reads uniforms via `BufRO<Uniforms>` and writes pixels via `DirectSpatial<float4>`:

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

### Compute Pipeline and Scheme

Compile the Slang source, create a `ComputePipeline`, and record a retained scheme once via [`SurfaceExchange::bind_destination`](../surfaces/overview.md):

```rust
let shader = ShaderModule::from_slang(&device, COMPUTE_SHADER)?;
let compute_pipeline = ComputePipeline::new(&device, &shader)?;

let surface = SurfaceExchange::new_with_depth(&ctx, window.as_ref(), 3, SurfaceConfig::default())?;

let mut scheme = Scheme::new(&ctx);
let wg_x = width.div_ceil(8);
let wg_y = height.div_ceil(8);
let (lease, present) = surface.bind_destination(&mut scheme)?;
scheme
    .node("compute", &compute_pipeline)
    .with_parcel(&uniform_buffer, NodeAccess::Read)
    .with_present(&lease)
    .dispatch(wg_x, wg_y, 1);
```

### Rendering a Frame

Each frame: upload new uniform values via a small upload scheme with a bound deposit, submit the main scheme, claim and consume the surface transaction.

```rust
fn render_frame(state: &mut RenderState) -> Result<()> {
    let (width, height) = state.surface.size();
    let elapsed = state.start_time.elapsed().as_secs_f32();

    let uniforms = Uniforms {
        width,
        height,
        time: elapsed,
        _padding: 0.0,
    };

    state.uniform_deposit.write(
        &mut state.upload_scheme,
        0,
        bytemuck::bytes_of(&uniforms),
    )?;
    state.upload_scheme.submit()?;

    let mut submission = state.scheme.submit()?;
    state.present.claim(&mut submission)?.consume()?;
    Ok(())
}
```

At init, bind the deposit once on a retained upload scheme:

```rust
let mut upload_scheme = Scheme::new(&ctx);
let uniform_deposit = MemoryExchange::new(&ctx).bind_deposit_buffer(
    &mut upload_scheme,
    &uniform_buffer,
    std::mem::size_of::<Uniforms>() as u64,
)?;
```

### Step by Step

**Update uniforms** — `MemoryExchange::bind_deposit_buffer` records the upload topology once; each frame call `deposit.write` on the upload scheme before the main submit.

**Record the scheme once** — `SurfaceExchange::bind_destination` registers the present exchange and returns a [`PresentLease`](https://docs.rs/goldy/latest/goldy/struct.PresentLease.html) plus a [`Transaction`](https://docs.rs/goldy/latest/goldy/struct.Transaction.html). `scheme.node()` creates a compute node bound to a pipeline. `with_parcel()` declares the uniform buffer dependency. `with_present()` binds the drawable lease. `dispatch()` sets the workgroup count.

**Submit and present** — `scheme.submit()` records and submits GPU work. `present.claim(&mut submission)?.consume()` presents the swapchain image. The compute shader already wrote the pixels — there is no blit or copy step.

## Run It

```bash
cargo run --example compute_to_surface
```

You should see an animated plasma pattern filling the window, rendered entirely from compute.

## Next Steps

- [Compute to Surface](../compute/compute-to-surface.md) — present-on-scheme details
- [Examples](../examples/overview.md) — particles, game of life, and more compute examples
