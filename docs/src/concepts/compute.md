# Compute

Goldy's compute API lets you run general-purpose GPU programs (GPGPU) alongside graphics rendering. Compute shaders have access to the same bindless resource model as graphics shaders.

## Key Types

| Type | Purpose |
|------|---------|
| `ComputePipeline` | Compiled compute shader + pipeline state |
| `ComputeEncoder` | Records compute commands |
| `ComputePass` | Scoped recording context inside an encoder |
| `GpuFuture` | Handle to non-blocking submitted work |

## Creating a Compute Pipeline

```rust
use goldy::{ComputePipeline, ShaderModule};

let shader = ShaderModule::from_slang(&device, r#"
    #include "goldy_exp.slang"

    struct PushConstants { uint buffer_idx; };
    [[vk::push_constant]] PushConstants pc;

    [shader("compute")]
    [numthreads(64, 1, 1)]
    void cs_main(uint3 id : SV_DispatchThreadID) {
        float val = asfloat(g_StorageBuffers[pc.buffer_idx].Load(id.x * 4));
        g_StorageBuffers[pc.buffer_idx].Store(id.x * 4, asuint(val * 2.0));
    }
"#)?;

let pipeline = ComputePipeline::new(&device, &shader)?;
```

## Recording and Dispatching

```rust
use goldy::{ComputeEncoder, Buffer, DataAccess};

let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
let buffer = Buffer::with_data(&device, &data, DataAccess::Scattered)?;

let mut encoder = ComputeEncoder::new();
{
    let mut pass = encoder.begin_compute_pass();
    pass.set_pipeline(&pipeline);
    pass.set_push_constants(&[&buffer]);
    pass.dispatch(16, 1, 1); // 16 workgroups × 64 threads = 1024 threads
}

// Blocking: wait for the GPU to finish
encoder.dispatch(&device)?;
```

## Numthreads and Dispatch Size

The `[numthreads(x, y, z)]` attribute in the shader and the `dispatch(x, y, z)` call multiply together:

```
total_threads = dispatch_x * numthreads_x
              × dispatch_y * numthreads_y
              × dispatch_z * numthreads_z
```

For a 1024-element buffer with `[numthreads(64, 1, 1)]`:

```rust
let elements = 1024u32;
let threads_per_group = 64u32;
let groups = elements.div_ceil(threads_per_group); // = 16
pass.dispatch(groups, 1, 1);
```

## Indirect Dispatch

Let a prior compute pass write the workgroup counts into a buffer:

```rust
// buffer contains [groups_x, groups_y, groups_z] as u32s at offset 0
pass.dispatch_indirect(&count_buffer, 0);
```

## Clearing Buffers in a Pass

Unlike `Buffer::clear()` which submits immediately, `ComputePass::clear_buffer` records the clear into the same submission batch:

```rust
let mut pass = encoder.begin_compute_pass();
pass.clear_buffer(&buffer, 0, 0); // size=0 means "to end of buffer"
pass.set_pipeline(&pipeline);
pass.set_push_constants(&[&buffer]);
pass.dispatch(groups, 1, 1);
```

## Non-blocking Submission

Use `submit` to overlap CPU and GPU work:

```rust
let future = encoder.submit(&device)?;

// CPU work here while GPU is busy
prepare_next_frame();

// Wait when you need the result
future.wait()?;
```

See [GpuFuture](./gpu-future.md) for polling and timeout details.

## Resource Access in Shaders

Buffers are accessed via **bindless indices** passed through push constants:

```rust
// Pass multiple buffers
pass.set_push_constants(&[&input_buf, &output_buf]);
```

```slang
struct PC { uint in_idx; uint out_idx; };
[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    float val = asfloat(g_StorageBuffers[pc.in_idx].Load(id.x * 4));
    g_StorageBuffers[pc.out_idx].Store(id.x * 4, asuint(val * val));
}
```

For raw u32 indices (e.g., to mix textures and buffers):

```rust
let buf_idx = buffer.bindless_index().unwrap();
let tex_idx = texture.bindless_index().unwrap();
pass.set_push_constants_raw(&[buf_idx, tex_idx]);
```

## Examples

See [`compute_particles`](../examples/overview.md) and [`game_of_life`](../examples/overview.md) for full working examples.
