# Compute Graph

The compute graph API pairs Goldy's bindless resource model with explicit dependency declarations, enabling optimal barrier insertion and dispatch parallelism across all backends. It is opt-in alongside the existing [`ComputeEncoder`](./commands.md) / [`ComputePass`](./compute.md) API.

## Why a compute graph?

Goldy's bindless model (heap-backed argument buffers, resource-slot indices) gives shaders flexible, low-overhead access to resources. But it makes the GPU's automatic dependency tracking blind — Metal cannot see through argument buffer indirection to know which resources a dispatch reads or writes.

Without the graph, each dispatch must be submitted as a separate command buffer to guarantee correct ordering. This forces total serialization with per-command-buffer overhead — worse than wgpu, which batches everything into one encoder with bind-group-based implicit hazard tracking.

The compute graph tells Goldy exactly what each dispatch reads and writes. Goldy then:

1. Builds a dependency DAG from declared resource access patterns.
2. Groups independent dispatches into **waves** that execute concurrently.
3. Inserts **per-resource barriers** only at true dependency edges (RAW, WAR, WAW).
4. Submits everything in a single command buffer/encoder.

This is strictly more powerful than implicit bind-group tracking because the caller knows the true data dependencies — the backend doesn't have to guess.

## NodeAccess: logical vs physical access

Goldy already categorizes resources by physical access pattern (`DataAccess::Scattered` for read/write storage, `Broadcast` for read-only uniform). But within a compute graph, a `Scattered` buffer might be read-only in one dispatch and read-write in another.

`NodeAccess` captures the **per-node logical access**, orthogonal to the buffer's physical type:

```rust
pub enum NodeAccess {
    Read,       // can overlap with other Reads (SWMR)
    Write,      // requires exclusive access
    ReadWrite,  // requires exclusive access
}
```

Multiple `Read` nodes on the same resource run concurrently — single-writer/multiple-reader (SWMR) parallelism.

## Tier 1: ComputeGraph — dynamic, interpreted

Build a DAG of dispatch nodes each frame. At submit time, Goldy analyzes dependencies, inserts barriers, and executes.

```rust
use goldy::{ComputeGraph, NodeAccess};

let mut graph = ComputeGraph::new();

graph.node("pathtag_reduce", &pipeline_a)
    .bind_buffer(&scene_buf, NodeAccess::Read)
    .bind_buffer(&tagmonoid_buf, NodeAccess::ReadWrite)
    .bind_resources_raw(&[scene_idx, tagmonoid_idx])
    .dispatch(64, 1, 1);

graph.node("pathtag_scan", &pipeline_b)
    .bind_buffer(&tagmonoid_buf, NodeAccess::Read)
    .bind_buffer(&reduced_buf, NodeAccess::ReadWrite)
    .bind_resources_raw(&[tagmonoid_idx, reduced_idx])
    .dispatch(32, 1, 1);

// bbox_clear is independent — can overlap with both pathtag dispatches
graph.node("bbox_clear", &pipeline_c)
    .bind_buffer(&bbox_buf, NodeAccess::Write)
    .bind_resources_raw(&[bbox_idx])
    .dispatch(16, 1, 1);

graph.dispatch(&device)?;
```

**When to use:** dynamic workloads, prototyping, small graphs, cases where topology changes per frame.

## Tier 2: ComputeProgram — compiled, specializable

Separate graph topology (static) from bindings and dimensions (dynamic). Compile once, specialize cheaply per frame. The wave grouping and barrier placement are cached at compile time.

```rust
use goldy::{ComputeProgram, NodeAccess};

// === Build phase (once, at init) ===

let mut builder = ComputeProgram::builder();

let scene     = builder.buffer_slot("scene");
let tagmonoid = builder.buffer_slot("tagmonoid");
let reduced   = builder.buffer_slot("reduced");
let wg_reduce = builder.dim_slot("wg_reduce");

builder.step("pathtag_reduce", &pipeline_a)
    .bind_buffer(scene, NodeAccess::Read)
    .bind_buffer(tagmonoid, NodeAccess::ReadWrite)
    .dispatch_slot(wg_reduce);

builder.step("pathtag_scan", &pipeline_b)
    .bind_buffer(tagmonoid, NodeAccess::Read)
    .bind_buffer(reduced, NodeAccess::ReadWrite)
    .dispatch(32, 1, 1);          // fixed dimensions are fine too

let program = builder.compile()?;

// === Execute phase (each frame) ===

let mut exec = program.specialize();
exec.bind_buffer(scene, &scene_buf);
exec.bind_buffer(tagmonoid, &tagmonoid_buf);
exec.bind_buffer(reduced, &reduced_buf);
exec.set_dim(wg_reduce, (64, 1, 1));

let tv = exec.submit(&device)?;
device.wait_until(tv)?;
```

**When to use:** fixed-topology pipelines where the sequence of shaders doesn't change frame to frame — only buffer sizes and dispatch dimensions do.

## How the tiers relate

```text
ComputeGraph::submit()               → [build graph] → [analyze] → [emit barriers] → [execute]
                                             ↑ every frame

ComputeProgram::compile()            → [build graph] → [analyze] → [cache barrier template]
ComputeProgram::specialize().submit() → [bind slots] → [replay cached template]
                                             ↑ every frame (cheap)
```

Tier 1 is conceptually "build a program, compile, specialize, and submit in one call." The graph IR is shared. Tier 2 caches the compiled form.

## Backend details

The graph emits `ComputeCommand::ResourceBarrier` with per-resource granularity:

| Backend | Behavior |
|---------|----------|
| **Metal** | `memoryBarrierWithResources:count:` — precise per-resource barriers within a single compute encoder |
| **Vulkan** | Falls back to global compute pipeline barrier (per-resource `VkBufferMemoryBarrier` is a future optimization) |
| **DX12** | Falls back to global UAV barrier (per-resource `D3D12_RESOURCE_BARRIER` is a future optimization) |

On Metal — the primary beneficiary — the graph enables single-encoder submission with per-resource barriers, eliminating the per-command-buffer overhead of the current workaround.

## Comparison with ComputeEncoder

| | `ComputeEncoder` | `ComputeGraph` / `ComputeProgram` |
|---|---|---|
| Dependency tracking | Manual (caller inserts barriers) | Automatic (from declared `NodeAccess`) |
| Barrier granularity | Global (`Barrier`) | Per-resource (`ResourceBarrier`) |
| Parallelism | None unless caller manages it | SWMR: independent nodes overlap |
| API complexity | Low — flat command list | Medium — declare access per node |
| Best for | One-off dispatches, simple workloads | Multi-dispatch pipelines with data dependencies |

The existing `ComputeEncoder` / `ComputePass` API is unchanged. The graph API is purely additive.
