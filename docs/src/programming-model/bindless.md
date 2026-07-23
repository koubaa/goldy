# Bindless by Default

Goldy uses a **typed resource model** built on bindless descriptor heaps as substrate: there are no descriptor sets, no binding tables, and no manual layout declarations. Every GPU resource — buffers, textures, samplers — is registered in a per-category heap, and scheme dispatches resolve parcels to those slots internally.

> **Terminology:** "Bindless" describes the internal descriptor-heap plumbing, not the public Rust/C API. Client code binds with `Scheme::with_parcel` (or the equivalent render-pass builders). Raw heap indices are crate-private; [`ResourceHandle`](../types) is an opaque identity for equality / retention checks.

## How It Works

Traditional GPU APIs require you to declare descriptor set layouts, allocate descriptor pools, update descriptor sets, and bind them before each draw or dispatch. Goldy eliminates all of this. Instead:

1. Resources are registered in per-category descriptor heaps when created.
2. Schemes bind parcels (and samplers / present leases) by identity + access; the runtime resolves the correct descriptor slot.
3. At dispatch time, the backend packs those slots into the push-constant / frame-table ABI. Shaders resolve them through the `goldy_exp` access functions.

```
CPU side:                              GPU side:

scheme.node(...).with_parcel(&p, Write)   goldy_scattered<T>(slot)
  → internal ResourceHandle               → descriptor_heap[slot]
       (opaque identity)                    → RWStructuredBuffer<T>
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

`ResourceAccess` (`Read`, `Write`, `ReadWrite`) selects which descriptor pool entry to use at dispatch time — SRV vs UAV, CBV vs storage, sampled vs storage image. This is distinct from `NodeAccess`, which the scheme uses for scheduling hazards (`Read`, `Write`, `ReadWrite`, `Overwrite`).

`ResourceHandle` is an opaque typed identity returned by `handle(ResourceAccess)`. Callers may compare handles (for example to detect that a retained scheme must be re-recorded after a reallocation) but must not extract a raw heap index — that contract is crate-private so backends can reinterpret slots.

```rust
use goldy::{BufferKind, ResourceAccess};

let parcel = retained_pool.acquire_buffer_with_data(&particles, BufferKind::Scattered)?;
let handle = parcel.handle(ResourceAccess::Write).unwrap();
let again = parcel.handle(ResourceAccess::Write).unwrap();
assert_eq!(handle, again);
```

When you bind parcels at dispatch time, Goldy validates that the resolved handle's category matches what the shader expects in that slot — a `Broadcast` handle bound to a slot the shader reads through `goldy_scattered` is caught as a type error rather than silently producing garbage.

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

When you call `with_parcel` on a compute or render pass builder, Goldy queues each parcel as a push-constant slot. At `dispatch` / `set_pipeline` time it consults the pipeline's reflected slot kinds to pick the correct SRV vs UAV (or CBV) handle and validates categories against the shader signature. `NodeAccess` is graph semantics only (including `Overwrite` for full replace without reading prior contents); descriptor access comes from reflection.

```rust
scheme
    .node("update", &pipeline)
    .with_parcel(&params_buf, NodeAccess::Read)
    .with_parcel(&particle_buf, NodeAccess::ReadWrite)
    .dispatch((particle_count + 63) / 64, 1, 1);
```

If slot 0 expects `Broadcast` (from the shader's `MyUniforms cfg` parameter) but the parcel only exposes a scattered view, binding fails with a clear error instead of producing undefined behavior.

## Contrast with Traditional Binding

| | Traditional (Vulkan/DX12) | Goldy Resource Model |
|---|---|---|
| **Setup** | Declare descriptor set layouts, allocate pools, create and update descriptor sets | Create resources; indices assigned automatically |
| **Binding** | Bind descriptor sets before each draw/dispatch | Pass parcels via `with_parcel` on scheme nodes |
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
let params_buf = retained_pool.acquire_buffer_with_data(&[sim_params], BufferKind::Broadcast)?;
let particle_buf = retained_pool.acquire_buffer_with_data(&particles, BufferKind::Scattered)?;

let shader = ShaderModule::from_slang(&device, PARTICLE_UPDATE_SOURCE)?;
let pipeline = ComputePipeline::new(&device, &shader)?;

let mut scheme = Scheme::new(&ctx);
scheme
    .node("update", &pipeline)
    .with_parcel(&params_buf, NodeAccess::Read)
    .with_parcel(&particle_buf, NodeAccess::ReadWrite)
    .dispatch((particle_count + 63) / 64, 1, 1);
scheme.submit()?;
```

The shader author writes natural function parameters. The Rust side binds parcels in declaration order via `with_parcel`. Goldy handles the rest — slot packing, category validation, and cross-backend descriptor plumbing.
