# Compute

Goldy's compute API lets you run general-purpose GPU programs (GPGPU) alongside graphics rendering. Compute shaders have access to the same bindless resource model as graphics shaders.

## Key Types

| Type | Purpose |
|------|---------|
| `ComputePipeline` | Compiled compute shader + pipeline state |
| `ComputeEncoder` | Records compute commands |
| `ComputePass` | Scoped recording context inside an encoder |
| `TimelineValue` | Opaque `u64` device timeline counter from non-blocking `submit` |

## Creating a Compute Pipeline

```rust
use goldy::{ComputePipeline, ShaderModule};

let shader = ShaderModule::from_slang(&device, r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<float> data, ThreadId id) {
        data[id.x] = data[id.x] * 2.0;
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
    pass.bind_resources_raw(&[buffer.bindless_index().unwrap()]);
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
pass.bind_resources_raw(&[buffer.bindless_index().unwrap()]);
pass.dispatch(groups, 1, 1);
```

## Non-blocking Submission

Use `submit` to overlap CPU and GPU work:

```rust
let tv = encoder.submit(&device)?;

// CPU work here while GPU is busy
prepare_next_frame();

// Wait when you need the result
device.wait_until(tv)?;
```

See [Device timeline](./gpu-future.md) for polling with `gpu_progress` and timeouts.

## Resource Access in Shaders

Buffers are accessed via **resource bindings** — shader entry-point parameters that
Goldy resolves to resource handles. The `[goldy_compute]` virtual entry point generates
the underlying `uniform uint` resource slots automatically.

```slang
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> inp, Scattered<uint> out, ThreadId id) {
    out[id.x] = inp[id.x] * inp[id.x];
}
```

```rust
// Rust side: pass the heap indices in declaration order
pass.bind_resources_raw(&[
    input_buf.bindless_index().unwrap(),
    output_buf.bindless_index().unwrap(),
]);
pass.dispatch(16, 1, 1);
```

### Per-dispatch Scalar Parameters

Parameters that are not heap indices — thread base offsets, element counts, flags —
are declared as typed entry-point parameters. No helper function is needed; the raw
resource slot *is* the value:

```slang
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, uint offset, uint stride, ThreadId id) {
    data[id.x * stride + offset] += 1;
}
```

```rust
// Rust: [heap_index, offset, stride]
pass.bind_resources_raw(&[data_buf.bindless_index().unwrap(), offset, stride]);
```

Scalar parameters support any type ≤ 4 bytes (`uint`, `int`, `float`, `bool`).

> **Tip:** The total resource slot budget is 64 bytes (16 `u32` slots) on all
> supported backends. Goldy's `ResourceSlots` layout uses this full budget.

### Advanced: Manual Entry Points

For fine-grained control, you can write `[shader("compute")]` entry points with
explicit `uniform uint` slots instead of using `[goldy_compute]`:

```slang
import goldy_exp;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uniform uint data_slot, uniform uint base, uniform uint stride,
             uint3 id : SV_DispatchThreadID) {
    StorageBuffer<uint> data = goldy_scattered<uint>(data_slot);
    data[id.x * stride + base] += 1;
}
```

This is equivalent to what `[goldy_compute]` generates, but requires manual resource
resolution. Prefer `[goldy_compute]` for most use cases.

## Compute Graph

For multi-dispatch pipelines with data dependencies between shaders, see the [Compute Graph](./compute-graph.md) API. It analyzes declared resource access patterns and inserts optimal barriers automatically, enabling SWMR parallelism across all backends.

## Examples

See [`compute_particles`](../examples/overview.md) and [`game_of_life`](../examples/overview.md) for full working examples.
