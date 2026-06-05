# Task Graph

The task graph is one of Goldy's core abstractions. It pairs the bindless resource model with explicit dependency declarations so the runtime can insert optimal barriers and maximize GPU parallelism — all within a single command buffer.

## Why the task graph exists

Goldy uses a bindless resource model: shaders access buffers and textures through heap-backed argument buffers indexed by slot numbers. This gives shaders flexible, low-overhead access to any resource, but it makes the GPU's automatic dependency tracking blind. Metal, for example, cannot see through argument buffer indirection to know which resources a dispatch reads or writes, so it cannot insert barriers automatically.

Without the task graph, the only correct approach is to submit each dispatch as a separate command buffer. This works, but it serializes everything and adds per-command-buffer scheduling overhead — worse than APIs like wgpu that use bind groups to infer hazards.

The task graph solves this: you declare what each node reads and writes, and Goldy does the rest.

1. Builds a dependency DAG from declared resource access patterns
2. Groups independent dispatches into **waves** that execute concurrently
3. Inserts **per-resource barriers** only at true dependency edges (RAW, WAR, WAW)
4. Submits everything in a single command buffer

## Building a task graph

Create a `TaskGraph`, add nodes with resource access declarations, and submit:

```rust
use goldy::{TaskGraph, NodeAccess};

let mut graph = TaskGraph::new();

graph.node("write_data", &pipeline_a)
    .bind_buffer(&buf, NodeAccess::Write)
    .bind_resources_raw(&[buf_idx])
    .dispatch(64, 1, 1);

graph.node("read_data", &pipeline_b)
    .bind_buffer(&buf, NodeAccess::Read)
    .bind_resources_raw(&[buf_idx])
    .dispatch(64, 1, 1);

let ctx = device.create_context();
let tv = graph.submit(&device)?;
ctx.wait_until(tv)?;
```

The analyzer sees that `read_data` depends on `write_data` (RAW hazard on `buf`) and inserts a barrier between them. If two nodes touch completely different resources, they execute in the same wave with no barrier.

### Node types

| Builder method | GPU operation |
|---|---|
| `graph.node(label, &pipeline)` | Compute dispatch (direct or indirect) |
| `graph.clear_buffer(&buf, offset, size)` | GPU-side buffer zero-fill |
| `graph.clear_buffer_view(&view, offset, size)` | GPU-side zero-fill of a pool view region |
| `graph.write_buffer(&buf, offset, data)` | CPU→GPU buffer upload |
| `graph.write_texture(&tex, data)` | CPU→GPU texture upload |
| `graph.render_pass(label, &target)` | Offscreen render pass |

All node types participate in the same dependency analysis.

### Declaring resource access

Each node declares its resource access via `bind_buffer`, `bind_buffer_view`, or `bind_texture`:

```rust
graph.node("reduce", &pipeline)
    .bind_buffer(&input, NodeAccess::Read)
    .bind_buffer(&output, NodeAccess::Write)
    .bind_resources_raw(&[input_idx, output_idx])
    .dispatch(64, 1, 1);
```

`bind_resources_raw` sets the actual shader slot indices. The `bind_buffer` / `bind_texture` calls are purely for dependency analysis — they tell the scheduler *what* this node touches, not *how* to bind it.

### Finalizing nodes

Compute nodes must be finalized with `dispatch(x, y, z)` or `dispatch_indirect(&buf, offset)`. Render pass nodes are finalized with `finish(commands)` or `finish_encoder(encoder)`.

## NodeAccess and SWMR scheduling

`NodeAccess` is the per-node logical access, orthogonal to a buffer's physical `BufferKind`:

```rust
pub enum NodeAccess {
    Read,      // can overlap with other Reads
    Write,     // exclusive access
    ReadWrite, // exclusive access
}
```

The scheduler implements single-writer/multiple-reader (SWMR) parallelism:

- Multiple `Read` nodes on the same resource run concurrently in the same wave.
- A `Write` or `ReadWrite` node serializes against all prior accessors of that resource.
- Barriers are inserted only at true RAW/WAR/WAW edges.

### Diamond example

```rust
let mut graph = TaskGraph::new();

// Wave 0: A writes buf_x
graph.node("A", &p1)
    .bind_buffer(&buf_x, NodeAccess::Write)
    .dispatch(1, 1, 1);

// Wave 1: B and C both read buf_x (SWMR — they run concurrently)
graph.node("B", &p2)
    .bind_buffer(&buf_x, NodeAccess::Read)
    .bind_buffer(&buf_y, NodeAccess::Write)
    .dispatch(1, 1, 1);

graph.node("C", &p3)
    .bind_buffer(&buf_x, NodeAccess::Read)
    .bind_buffer(&buf_z, NodeAccess::Write)
    .dispatch(1, 1, 1);

// Wave 2: D reads both outputs
graph.node("D", &p4)
    .bind_buffer(&buf_y, NodeAccess::Read)
    .bind_buffer(&buf_z, NodeAccess::Read)
    .dispatch(1, 1, 1);

graph.dispatch(&device)?;
```

This produces three waves with two barriers — the minimum possible for this dependency pattern.

## Buffer views and pool tracking

When using `BufferPool`, you can declare access at view granularity. Non-overlapping views of the same pool produce no dependency edge and execute in the same wave:

```rust
let view_a = pool.alloc::<u32>(64)?;
let view_b = pool.alloc::<u32>(64)?;

let mut graph = TaskGraph::new();

graph.node("write_a", &pipeline)
    .bind_buffer_view(&view_a, NodeAccess::Write)
    .dispatch(1, 1, 1);

graph.node("write_b", &pipeline)
    .bind_buffer_view(&view_b, NodeAccess::Write)
    .dispatch(1, 1, 1);

// No barrier — view_a and view_b occupy disjoint byte ranges
graph.dispatch(&device)?;
```

Barriers are emitted against the parent buffer handle, so backends require no changes. The scheduler tracks byte ranges internally to determine true overlap.

## Transient resources

Transient buffers and textures exist only for the lifetime of a single graph submission. They are allocated from a shared heap, and non-overlapping lifetimes can alias onto the same memory — reducing allocation pressure for temporaries.

```rust
let mut graph = TaskGraph::new();

let tmp = graph.transient_buffer(256);

graph.node("produce", &pipeline_a)
    .bind_transient_buffer(tmp, NodeAccess::Write)
    .bind_resources_raw(&[0])
    .dispatch(1, 1, 1);

graph.node("consume", &pipeline_b)
    .bind_transient_buffer(tmp, NodeAccess::Read)
    .bind_resources_raw(&[0])
    .dispatch(1, 1, 1);

graph.dispatch(&device)?;
```

Transient textures work the same way:

```rust
let tmp_tex = graph.transient_texture(width, height, TextureFormat::Rgba8Unorm);

graph.node("render", &pipeline)
    .bind_transient_texture(tmp_tex, NodeAccess::Write)
    .bind_resources_raw(&[0])
    .dispatch(wg_x, wg_y, 1);
```

The scheduler uses wave-interval analysis to determine which transients can alias: if two transient buffers are never live in the same wave, they share the same backing memory.

By default, `Device::submit` blocks the CPU until the GPU finishes so the placement heap region is immediately reclaimable. When the graph is submitted as part of a pipelined frame via `Device::submit_pipelined`, the CPU does **not** block — the region stays in flight until the returned `TimelineValue` advances. See [Pipelined Frames](./pipelined-frames.md) for the `FrameOrchestrator` API that manages this automatically.

## Per-resource barriers on Metal

The graph emits `ResourceBarrier` commands with per-resource granularity. Each backend maps this to its native mechanism:

| Backend | Behavior |
|---|---|
| **Metal** | `memoryBarrierWithResources:count:` — precise per-resource barriers within a single compute encoder |
| **Vulkan** | Global compute pipeline barrier (per-resource `VkBufferMemoryBarrier` is a future optimization) |
| **DX12** | Global UAV barrier (per-resource `D3D12_RESOURCE_BARRIER` is a future optimization) |

On Metal — the primary beneficiary — the graph enables single-encoder submission with per-resource barriers, eliminating the per-command-buffer overhead of the one-dispatch-per-command-buffer workaround.

## Single command buffer submission

All nodes in a `TaskGraph` are submitted in a single command buffer (or compute encoder on Metal). The scheduler groups independent nodes into waves and inserts barriers only between waves that have true data dependencies. This minimizes scheduling overhead and enables the GPU to overlap independent work within a wave.

## Blocking vs non-blocking submission

**Non-blocking** — returns a `TimelineValue` for CPU-side synchronization:

```rust
let ctx = device.create_context();
let tv = graph.submit(&device)?;
// CPU work while GPU executes...
ctx.wait_until(tv)?;
```

**Blocking** — submits and waits for completion:

```rust
graph.dispatch(&device)?;
```

## Practical example: Game of Life

A ping-pong compute pattern using buffer pool views and the task graph:

```rust
let (read_view, write_view) = if use_buffer_a {
    (&view_a, &view_b)
} else {
    (&view_b, &view_a)
};

let mut graph = TaskGraph::new();
graph.node("game_of_life", &compute_pipeline)
    .bind_buffer_view(read_view, NodeAccess::Read)
    .bind_buffer_view(write_view, NodeAccess::Write)
    .bind_resources_raw(&[
        read_view.handle(ResourceAccess::Read).unwrap().index(),
        write_view.handle(ResourceAccess::Write).unwrap().index(),
    ])
    .dispatch(GRID_WIDTH.div_ceil(8), GRID_HEIGHT.div_ceil(8), 1);
graph.dispatch(&device)?;

use_buffer_a = !use_buffer_a;
```

The graph analyzes the `Read` and `Write` declarations on each view and inserts barriers only where needed. Because the two views occupy disjoint byte ranges in the same pool, the scheduler can verify they don't alias — enabling correct execution with minimal synchronization.
