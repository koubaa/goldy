# Bindless by Default

Goldy uses a **typed resource model** built on bindless descriptor heaps as substrate: there are no descriptor sets, no binding tables, and no manual layout declarations. Every GPU resource — buffers, textures, samplers — is identified at dispatch time by a small integer index packed into push constants (Vulkan/DX12) or argument buffers (Metal).

> **Terminology:** "Bindless" describes the internal descriptor-heap plumbing, not the public Rust/C API. Client code uses [`ResourceAccess`](../../resources/buffers.md), [`ResourceHandle`](../types), and `resource_index()` / `handle()` — not bindless-specific method names.

## How It Works

Traditional GPU APIs require you to declare descriptor set layouts, allocate descriptor pools, update descriptor sets, and bind them before each draw or dispatch. Goldy eliminates all of this. Instead:

1. Resources are registered in per-category descriptor heaps when created.
2. Each resource gets a `ResourceHandle` — a `(category, index)` pair — via `handle(ResourceAccess)`.
3. At dispatch time, you pass these handles as ordinary arguments. The GPU shader resolves them to live buffer/texture/sampler handles through the `goldy_exp` access functions.

```
CPU side:                         GPU side:
                                  
device.alloc_buffer_with_data(...) goldy_scattered<T>(slot)
  → ResourceHandle {                → descriptor_heap[slot]
      category: Scattered,            → RWStructuredBuffer<T>
      index: 3,
    }
```

## ResourceCategory

Goldy's descriptor heaps are organized into five pools, one per access pattern. A resource's index is only meaningful within its category:

| Category | Pool | Shader Access Function |
|----------|------|------------------------|
| `Scattered` | Storage buffers | `goldy_scattered<T>()` / `goldy_buf_ro<T>()` |
| `Broadcast` | Uniform/constant buffers | `goldy_broadcast<T>()` |
| `Texture` | Sampled textures | `goldy_interpolated<T>()` |
| `StorageImage` | Writable textures | `goldy_direct_spatial<T>()` |
| `Sampler` | Sampler states | `goldy_filter()` |

`Scattered` slot 3 and `Broadcast` slot 3 refer to different physical entries — on Metal these are `storageBuffers[3]` vs `uniformBuffers[3]`, on Vulkan they live in different descriptor array bindings.

## ResourceHandle and ResourceAccess

`ResourceAccess` (`Read`, `Write`, `ReadWrite`) selects which descriptor pool entry to use at dispatch time — SRV vs UAV, CBV vs storage, sampled vs storage image. This is distinct from [`NodeAccess`](../compute/task-graph.md) used by the task graph for scheduling hazards.

`ResourceHandle` carries both the raw index and the resource category:

```rust
use goldy::{BufferKind, ResourceAccess, ResourceHandle};

let buf = device.alloc_buffer_with_data(BufferKind::Scattered, &particles)?;
let handle: ResourceHandle = buf.handle(ResourceAccess::Write).unwrap();

assert_eq!(handle.category(), ResourceCategory::Scattered);
assert_eq!(handle.index(), 3); // assigned by the device
```

When you bind handles at dispatch time, Goldy can validate that the handle's category matches what the shader expects in that slot — a `Broadcast` handle bound to a slot the shader reads through `goldy_scattered` is caught as a type error rather than silently producing garbage.

## Typed Resource Parameters

In shader code, `goldy_exp` provides type aliases that map directly to the underlying Slang resource types. These are used as entry-point parameters in [virtual entry points](virtual-entry-points.md):

| Goldy Type | Underlying Slang Type | Usage |
|---|---|---|
| `Scattered<T>` | `RWStructuredBuffer<T>` | Read/write buffer: `data[i]`, `data[i].field = v` |
| `BufRO<T>` | `StructuredBuffer<T>` | Read-only buffer: `buf[i]` |
| `Interpolated<T>` | `Texture2D<T>` | Sampled texture: `tex.Sample(samp, uv)` |
| `DirectSpatial<T>` | `RWTexture2D<T>` | Writable texture: `img[int2(x,y)]` |
| `ByteAddress` | `RWByteAddressBuffer` | Raw byte access: `.Load()`, `.Store()`, `.Interlocked*()` |
| `Filter` | `SamplerState` | Sampler for texture filtering |

Any user-defined struct type (e.g. `MyUniforms`) declared as a parameter is automatically treated as a constant-buffer **broadcast** — no wrapper type needed.

## Dispatch-Time Type Checking

When you call `bind_resources_typed`, Goldy compares each `ResourceHandle.category` against the shader's declared parameter types (extracted via `extract_push_constant_categories`):

```rust
let uniforms = uniform_buf.handle(ResourceAccess::Read).unwrap();  // Broadcast
let data     = storage_buf.handle(ResourceAccess::Write).unwrap(); // Scattered

// Category validation happens here:
pass.bind_resources_typed(&[uniforms, data]);
pass.dispatch(workgroups, 1, 1);
```

If slot 0 expects `Broadcast` (from the shader's `MyUniforms cfg` parameter) but receives a `Scattered` handle, the dispatch fails with a clear error instead of producing undefined behavior.

## Contrast with Traditional Binding

| | Traditional (Vulkan/DX12) | Goldy Resource Model |
|---|---|---|
| **Setup** | Declare descriptor set layouts, allocate pools, create and update descriptor sets | Create resources; indices assigned automatically |
| **Binding** | Bind descriptor sets before each draw/dispatch | Pass `ResourceHandle` values as push constants |
| **Shader access** | `layout(set=0, binding=1) buffer ...` | `Scattered<T> data` as a function parameter |
| **Validation** | Runtime errors or silent corruption on mismatch | Category + stride checks at dispatch time |
| **Cross-backend** | Layout declarations differ per API | Same shader code on Vulkan, DX12, and Metal |

## Example: Compute Shader with Resource Handles

**Shader** (`particle_update.slang`):

```hlsl
import goldy_exp;

struct SimParams {
    float dt;
    uint count;
};

struct Particle {
    float2 pos;
    float2 vel;
};

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(SimParams params, Scattered<Particle> particles, ThreadId id) {
    if (id.x >= params.count) return;

    Particle p = particles[id.x];
    p.pos += p.vel * params.dt;
    particles[id.x] = p;
}
```

**Rust dispatch**:

```rust
let params_buf = device.alloc_buffer_with_data(BufferKind::Broadcast, &[sim_params])?;
let particle_buf = device.alloc_buffer_with_data(BufferKind::Scattered, &particles)?;

let shader = ShaderModule::from_slang(&device, PARTICLE_UPDATE_SOURCE)?;
let pipeline = ComputePipeline::new(&device, &shader)?;

let mut encoder = ComputeEncoder::new();
let mut pass = encoder.begin_compute_pass();
pass.set_pipeline(&pipeline);
pass.bind_resources_typed(&[
    params_buf.handle(ResourceAccess::Read).unwrap(),    // slot 0 → Broadcast → SimParams
    particle_buf.handle(ResourceAccess::Write).unwrap(), // slot 1 → Scattered → Particle
]);
pass.dispatch((particle_count + 63) / 64, 1, 1);
drop(pass);
encoder.dispatch(&device)?;
```

The shader author writes natural function parameters. The Rust side binds handles in declaration order. Goldy handles the rest — slot packing, category validation, and cross-backend descriptor plumbing.
