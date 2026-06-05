# ComputeEncoder

`ComputeEncoder` records compute commands into a flat command list. It is lock-free and can be used from any thread — no GPU interaction happens until you submit.

For multi-dispatch workloads with data dependencies between passes, prefer the [Task Graph](./task-graph.md), which analyzes dependencies and inserts barriers automatically. `ComputeEncoder` is best for simple, single-dispatch workloads or cases where you manage barriers yourself.

## Creating an encoder

```rust
let mut encoder = ComputeEncoder::new();
```

## Recording a compute pass

Open a `ComputePass`, set a pipeline, bind resources, and dispatch:

```rust
let mut pass = encoder.begin_compute_pass();
pass.set_pipeline(&pipeline);
pass.bind_resources_raw(&[buffer.resource_index(ResourceAccess::Read).unwrap()]);
pass.dispatch(16, 1, 1);
```

The pass borrows the encoder mutably. Drop it (or let it go out of scope) before opening another pass or finishing the encoder.

## Binding resources

There are three ways to pass resource handles to a compute shader:

**`bind_resources`** — pass `Buffer` references directly. Indices are bound in declaration order:

```rust
pass.bind_resources(&[&particle_buffer, &params_buffer]);
```

**`bind_resources_raw`** — pass raw `u32` slot indices. Use this when you need to mix buffer, texture, and sampler indices:

```rust
let tex_idx = texture.resource_index(ResourceAccess::Read).unwrap();
let buf_idx = buffer.resource_index(ResourceAccess::Read).unwrap();
pass.bind_resources_raw(&[buf_idx, tex_idx]);
```

**`bind_resources_typed`** — pass typed `ResourceHandle`s that carry both the index and the resource category:

```rust
let uniforms = uniform_buf.handle(ResourceAccess::Read).unwrap();
let output = output_tex.handle(ResourceAccess::Write).unwrap();
pass.bind_resources_typed(&[uniforms, output]);
```

### Per-dispatch scalar parameters

Parameters that aren't heap indices — offsets, counts, flags — are declared as typed entry-point parameters in the shader and passed alongside resource indices:

```slang
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, uint offset, uint stride, ThreadId id) {
    data[id.x * stride + offset] += 1;
}
```

```rust
pass.bind_resources_raw(&[data_buf.resource_index(ResourceAccess::Read).unwrap(), offset, stride]);
```

Or use the two-region form to separate resource indices (region A) from user scalars (region B):

```rust
pass.bind_resources_raw_with_user(
    &[data_buf.resource_index(ResourceAccess::Read).unwrap()],
    &[offset, stride],
);
```

## Dispatching workgroups

The total thread count is the product of `dispatch(x, y, z)` and the shader's `[numthreads(x, y, z)]`:

```rust
let elements = 1024u32;
let threads_per_group = 64u32;
let groups = elements.div_ceil(threads_per_group);
pass.dispatch(groups, 1, 1); // 16 groups × 64 threads = 1024
```

### Indirect dispatch

Let a prior pass write the workgroup counts into a buffer, then read them at dispatch time:

```rust
pass.dispatch_indirect(&count_buffer, 0);
```

The buffer must contain three consecutive `u32` values `(x, y, z)` at the given byte offset.

## Barriers and buffer clears

Insert a global memory barrier between dispatches within the same encoder:

```rust
pass.barrier();
```

Clear a buffer region to zero, batched into the same submission:

```rust
pass.clear_buffer(&buffer, 0, 0); // size=0 → clear to end of buffer
```

## Submitting

**Blocking** — submit and wait for the GPU to finish:

```rust
encoder.dispatch(&device)?;
```

**Non-blocking** — submit and get a `TimelineValue` for later synchronization:

```rust
let ctx = device.create_context();
let tv = encoder.submit(&device)?;

// CPU work while GPU is busy...

ctx.wait_until(tv)?;
```

See [Device Timeline](./timeline.md) for more on `TimelineValue` and `gpu_progress`.

## Recording into a task graph

For multi-pass workloads, record each dispatch as a task graph node instead of using `ComputeEncoder` directly. The task graph handles barriers for you:

```rust
let mut graph = TaskGraph::new();

graph.node("my_pass", &pipeline)
    .bind_buffer(&buf, NodeAccess::ReadWrite)
    .bind_resources_raw(&[buf.resource_index(ResourceAccess::Read).unwrap()])
    .dispatch(16, 1, 1);

graph.dispatch(&device)?;
```

See [Task Graph](./task-graph.md) for the full API.
