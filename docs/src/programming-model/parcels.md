# Parcels

A **parcel** is the unit of data Goldy schemes actually operate on: a whole buffer, a range within a buffer, or a texture. Every resource you acquire from a [`RetainedPool`](../resources/retained-pool.md) or a transient allocator hands you one or more parcels, and every `with_parcel` call on a scheme node passes exactly one.

```rust
use goldy::{BufferKind, ResourceAccess};

let parcel = retained_pool.acquire_buffer_with_data(&particles, BufferKind::Scattered)?;
let handle = parcel.handle(ResourceAccess::Write).unwrap();
let again = parcel.handle(ResourceAccess::Write).unwrap();
assert_eq!(handle, again);
```

You never declare layouts, allocate descriptor pools, or manage binding slots yourself. You acquire a parcel, bind it to a node with an access mode, and Goldy figures out the rest at dispatch time.

## Categories

Every parcel belongs to one of five categories, matching the shape of access a shader can perform on it:

| Category | What it holds | Shader-side type |
|----------|------|------------------------|
| `Scattered` | Read/write structured data | `Scattered<T>` |
| `BufRO` | Read-only structured data | `BufRO<T>` |
| `Broadcast` | Small uniform data shared by every invocation | a plain struct parameter |
| `Interpolated` | Sampled texture data | `Interpolated<T>` |
| `DirectSpatial` | Read/write texture data | `DirectSpatial<T>` |
| `Filter` | Sampler state | `Filter` |

A parcel's category is fixed when it's created (`BufferKind::Scattered`, `BufferKind::Broadcast`, etc.) and determines which shader-side type it can satisfy. Categories are also independent identity spaces: a `Scattered` parcel and a `Broadcast` parcel are unrelated even if they happen to occupy "slot 3" internally — that internal indexing is not something client code ever sees or reasons about.

## ResourceHandle and ResourceAccess

`ResourceAccess` (`Read`, `Write`, `ReadWrite`) describes the kind of access a piece of shader-visible data supports — for example, whether a buffer is exposed to the shader as read-only or read/write. `parcel.handle(access)` returns a `ResourceHandle`: an opaque, comparable identity for that parcel/access pair.

`ResourceHandle` is intentionally opaque. You can compare two handles for equality (useful for deciding whether a retained scheme needs to be re-recorded after a resource was reallocated), but there's nothing else to extract from one — it's an identity, not a number you're meant to interpret.

This is distinct from `NodeAccess` (`Read`, `Write`, `ReadWrite`, `Overwrite`), which is what you actually pass to `with_parcel`. `NodeAccess` describes how a scheme *node* uses a parcel for scheduling and hazard tracking (including `Overwrite` for "I'm replacing this data wholesale, don't preserve prior contents"); `ResourceAccess` is the narrower, resolved access a shader parameter requires.

## Binding Parcels to Schemes

You bind parcels to compute or render nodes with `with_parcel`, in the same order the shader declares its resource parameters:

```rust
scheme
    .node("update", &pipeline)
    .with_parcel(&params_buf, NodeAccess::Read)
    .with_parcel(&particle_buf, NodeAccess::ReadWrite)
    .dispatch((particle_count + 63) / 64, 1, 1);
```

At dispatch time, Goldy checks each bound parcel's category against what the shader's reflected signature expects. If slot 0 expects `Broadcast` (from the shader's `SimParams params` parameter) but you bound a `Scattered` parcel there, binding fails with a clear error instead of silently producing garbage or undefined behavior.

## Typed Resource Parameters in Shaders

On the shader side, `goldy_exp` provides types that mirror the categories above and map directly to underlying Slang resource types. These appear as ordinary parameters on [virtual entry points](virtual-entry-points.md):

| Goldy Type | Underlying Slang Type | Usage |
|---|---|---|
| `Scattered<T>` | `RWStructuredBuffer<T>` | Read/write buffer: `data[i]`, `data[i].field = v` |
| `BufRO<T>` | `StructuredBuffer<T>` | Read-only buffer: `buf[i]` |
| `Interpolated<T>` | `Texture2D<T>` | Sampled texture: `tex.Sample(samp, uv)` |
| `DirectSpatial<T>` | `RWTexture2D<T>` | Writable texture: `img[int2(x,y)]` |
| `ByteAddress` | `RWByteAddressBuffer` | Raw byte access: `.Load()`, `.Store()`, `.Interlocked*()` |
| `Filter` | `SamplerState` | Sampler for texture filtering |

Any user-defined struct type (e.g. `MyUniforms`) declared as a parameter is automatically treated as `Broadcast` — no wrapper type needed.

## Contrast with Traditional Binding

| | Traditional (Vulkan/DX12) | Goldy Parcels |
|---|---|---|
| **Setup** | Declare descriptor set layouts, allocate pools, create and update descriptor sets | Acquire a parcel; category is fixed at creation |
| **Binding** | Bind descriptor sets before each draw/dispatch | Pass parcels via `with_parcel` on scheme nodes |
| **Shader access** | `layout(set=0, binding=1) buffer ...` | `Scattered<T> data` as a function parameter |
| **Validation** | Runtime errors or silent corruption on mismatch | Category checks at dispatch time |
| **Cross-backend** | Layout declarations differ per API | Same shader code on Vulkan, DX12, and Metal |

## Example: Compute Shader with Parcels

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

The shader author writes natural function parameters. The Rust side binds parcels in declaration order via `with_parcel`. Everything below that — slot packing, descriptor heaps, cross-backend plumbing — is an implementation detail you never need to think about.
