<p align="center">
  <img src="assets/logo.jpeg" alt="Goldy Logo" width="600">
</p>

# Goldy: Modern GPU Library

**Goldy** is a Rust GPU library built around a typed bindless programming model, a dependency-driven task graph, and first-class compute support — targeting Vulkan 1.4+, DX12, and Metal Tier 2+ with native backends (no translation layers).

## Typed Bindless Programming

Shaders are written in Slang using `goldy_exp` virtual entry points (`[goldy_compute]`, `[goldy_vertex]`, `[goldy_fragment]`). Resources are declared as typed parameters — the Goldy compiler resolves bindless slots automatically:

```c
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(MyUniforms cfg, Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + cfg.base;
}
```

| Type | Maps To | Use |
|------|---------|-----|
| `Scattered<T>` | `RWStructuredBuffer<T>` | Read/write storage |
| `BufRO<T>` | `StructuredBuffer<T>` | Read-only storage |
| `DirectSpatial<T>` | `RWTexture2D<T>` | Read/write texture |
| `Interpolated<T>` | `Texture2D<T>` | Sampled texture |
| `Filter` | `SamplerState` | Texture sampler |
| `ThreadId` | `SV_DispatchThreadID` | Compute thread index |
| `VertexId` | `SV_VertexID` | Vertex index |

Struct parameters are automatically treated as broadcast (constant buffer) data.

## Task Graph

`TaskGraph` provides explicit dependency scheduling for bindless compute work. You declare what each node reads and writes; Goldy inserts optimal barriers, parallelizes independent dispatches across waves, and aliases transient resources:

```rust
let mut graph = TaskGraph::new();
graph
    .node("simulate", &sim_pipeline)
    .with_buffer(&particles, NodeAccess::ReadWrite)
    .with_resource_slots(&[particles_handle.index()])
    .dispatch(group_count, 1, 1);
```

## Compute-to-Surface

Compute shaders can write directly to swapchain textures — no graphics pipeline, no vertex buffers, no render passes. Acquire a frame, get its texture handle, dispatch, present:

```rust
let frame = surface.begin()?;
let texture = frame.texture();
// ... build TaskGraph, dispatch compute ...
frame.submit_compute(&graph)?;
frame.present()?;
```

## Multi-Backend, Single Shader Language

Goldy compiles Slang shaders to SPIR-V (Vulkan), DXIL (DX12), and Metal IR at runtime via the bundled Slang compiler. Each backend is a native implementation — Metal uses Metal idioms, not translated Vulkan.

| Platform | Backend |
|----------|---------|
| Linux | Vulkan |
| Windows | DX12 (Vulkan optional) |
| macOS | Metal |

## Quick Links

- [Installation](./tutorial/installation.md)
- [Your First Triangle](./tutorial/first-triangle.md)
- [Your First Compute Shader](./tutorial/first-compute.md)
- [GitHub Repository](https://github.com/koubaa/goldy)

## License

Goldy is licensed under the **MIT License**. See [License](./license.md) for details.
