<p align="center">
  <img src="assets/goldy.png" alt="Goldy Logo" width="240">
</p>

# Goldy: GPU runtime for the Fondaco Machine

**Goldy** is a Rust GPU library that realizes the [Fondaco Machine](./fondaco/specification.md).

> **Maturity**: Goldy **0.2** is the Fondaco Machine public API. SemVer applies within 0.2.x; expect breaking changes at 0.3.

## The Fondaco model in Goldy

| Fondaco | Goldy |
|---------|-------|
| Scheme | [`Scheme`](https://docs.rs/goldy/latest/goldy/struct.Scheme.html) — retained graph, resubmitted each frame |
| Parcel | [`Parcel`](https://docs.rs/goldy/latest/goldy/struct.Parcel.html), [`Buffer`](https://docs.rs/goldy/latest/goldy/struct.Buffer.html), [`Texture`](https://docs.rs/goldy/latest/goldy/struct.Texture.html) |
| Dispatch | Compute, render, copy, and present nodes inside a scheme |
| Exchange | [`SurfaceExchange`](https://docs.rs/goldy/latest/goldy/struct.SurfaceExchange.html), [`MemoryExchange`](https://docs.rs/goldy/latest/goldy/struct.MemoryExchange.html) |
| Settlement | [`Transaction`](https://docs.rs/goldy/latest/goldy/struct.Transaction.html) → [`Claim`](https://docs.rs/goldy/latest/goldy/struct.Claim.html) |

Goldy abstracts **where bytes live** (descriptor slots, residency, relocation) but exposes **what access costs** (access patterns, resize cost, readback path). See the [design thesis](./fondaco/design-thesis.md).

## Typed bindless shaders

Shaders use Slang with `goldy_exp` virtual entry points. Resources are typed parameters — Goldy resolves bindless slots automatically:

```slang
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(MyUniforms cfg, Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + cfg.base;
}
```

| Type | Use |
|------|-----|
| `Scattered<T>` | Read/write storage |
| `BufRO<T>` | Read-only storage |
| `DirectSpatial<T>` | Read/write texture |
| `Interpolated<T>` | Sampled texture |
| `Broadcast` (struct param) | Per-dispatch constants |

## Scheme

Record once, submit every frame. Goldy inserts barriers, parallelizes independent nodes, and aliases transient resources:

```rust
let mut scheme = Scheme::new(&ctx);
scheme
    .node("simulate", &sim_pipeline)
    .with_parcel(&particles, NodeAccess::ReadWrite)
    .dispatch(group_count, 1, 1);
let submission = scheme.submit()?;
```

## Compute-to-surface

Compute shaders can write swapchain drawables directly — no graphics pipeline or raster pass:

```rust
let surface = SurfaceExchange::new(&ctx, &window, SurfaceConfig::default())?;
let mut scheme = Scheme::new(&ctx);
let (lease, present) = surface.bind_destination(&mut scheme)?;
scheme
    .node("render", &compute_pipeline)
    .with_parcel(&uniforms, NodeAccess::Read)
    .with_present(&lease)
    .dispatch(wg_x, wg_y, 1);

let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

## Backends and bindings

| Platform | Backend |
|----------|---------|
| Windows | DX12 (default), Vulkan |
| Linux | Vulkan (Wayland surfaces) |
| macOS | Metal |

CUDA and WebGPU backends are **in progress**. A Tenstorrent backend (**Torus**) is **planned**. See [Backend Architecture](./backends/overview.md).

Bindings: [Python](./bindings/python.md), [.NET](./bindings/dotnet.md), [C++](./bindings/cpp.md), [Rust FFI Client](./bindings/rust-ffi-client.md).

## Quick links

- [Installation](./tutorial/installation.md)
- [Your First Triangle](./tutorial/first-triangle.md)
- [Your First Compute Shader](./tutorial/first-compute.md)
- [Fondaco overview](./design/fondaco.md)
- [Goldy runtime mapping](./fondaco/goldy-runtime.md)
- [GitHub](https://github.com/koubaa/goldy)

## License

MIT — see [License](./license.md).
