# Goldy vs wgpu

Both Goldy and [wgpu](https://wgpu.rs/) are Rust GPU libraries with multi-backend support. They make different tradeoffs that suit different use cases.

## At a Glance

| | wgpu | Goldy |
|---|------|-------|
| **Identity** | WebGPU implementation for Rust | Modern Rust GPU library |
| **Spec governance** | W3C WebGPU specification | Independent, opinionated |
| **Browser support** | Yes (WebGPU) | No |
| **Minimum hardware** | Wide compatibility (Vulkan 1.0+) | Modern only (Vulkan 1.4+, DX12, Metal 2+) |
| **Shader language** | WGSL (primary), SPIR-V, GLSL, naga | Slang (compiles to SPIR-V, DXIL, MSL) |
| **Resource model** | Descriptor-based (bind groups) | Typed bindless |
| **Synchronization** | Manual pass ordering | Retained scheme (dependency graph) |
| **Metal support** | Via MoltenVK or wgpu-hal | Native Metal backend |
| **Compute model** | Supported but secondary | First-class (compute-to-surface) |

## Resource Binding: Descriptors vs Bindless

wgpu uses **bind groups** — the WebGPU equivalent of Vulkan descriptor sets. You declare a bind group layout, create bind groups that match it, and bind them before each draw or dispatch:

```rust
// wgpu: declare layout, create group, bind before draw
let layout = device.create_bind_group_layout(&desc);
let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    layout: &layout,
    entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
    ..
});
pass.set_bind_group(0, &group, &[]);
```

Goldy uses **bindless access**. Resources get a slot index at creation time, and shaders access them directly by index. There are no layouts, groups, or binding calls:

```rust
// Goldy: bindless parcel already has a slot; bind it in the scheme
let mut pool = RetainedPool::new(device.clone());
let parcel = pool.acquire_buffer_with_data(&data, BufferKind::Scattered)?;
pass.with_parcel(&parcel, NodeAccess::Read);
```

The bindless approach eliminates an entire layer of API surface and the pipeline layout permutations that come with it.

## Synchronization: Manual vs Dependency Graph

wgpu provides implicit synchronization within a render/compute pass but requires you to order passes correctly. Resource transitions between passes are handled by wgpu internally, following WebGPU's implicit rules.

Goldy uses a retained **scheme** — a dependency graph recorded once and submitted each frame. You declare nodes and their resource dependencies; Goldy derives barriers, layout transitions, and execution order. This gives the runtime a global view of the frame for optimal scheduling and makes synchronization bugs structurally impossible.

## Shader Language: WGSL vs Slang

wgpu's primary shader language is **WGSL**, the WebGPU Shading Language. WGSL is designed for safety and portability across web and native targets, but it lacks features like modules, generics, and automatic differentiation.

Goldy uses **Slang** exclusively. Slang compiles a single source file to SPIR-V (Vulkan), DXIL (DX12), and MSL (Metal). It provides modules with true separate compilation, generics, and HLSL-familiar syntax. The `goldy_exp` shader library builds on Slang's module system to provide shared types and utilities:

```hlsl
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<Particle> particles, ThreadId id) {
    particles[id.x].position += particles[id.x].velocity;
}
```

## Compute as First-Class Citizen

wgpu supports compute shaders, but the API is oriented around render passes. Compute-to-render workflows require manual buffer management and pass ordering.

Goldy treats compute and graphics as peers. Compute-to-surface is a built-in pattern: a compute dispatch writes to a buffer or texture, and a subsequent render pass reads from it, with the scheme handling the dependency automatically.

## Metal: Native vs MoltenVK

wgpu supports Metal through its `wgpu-hal` Metal backend or via MoltenVK (Vulkan-on-Metal translation). MoltenVK adds a translation layer that can introduce overhead and compatibility limitations.

Goldy has a **native Metal backend** that uses Metal APIs directly — Argument Buffers Tier 2 for bindless, MSL compiled from Slang, and native Metal types throughout. No translation layer sits between Goldy and the Metal driver.

## Architecture

**wgpu:**
```
Application → wgpu (WebGPU API) → wgpu-hal → Vulkan / Metal / DX12 / WebGPU
```

**Goldy:**
```
Application → Goldy (native API) → Vulkan 1.4+ / Metal 2+ / DX12 (shipped); CUDA / WebGPU (in progress); Tenstorrent (planned)
```

wgpu implements the WebGPU specification faithfully, then maps it onto each backend through an internal HAL. Goldy talks to each backend directly using native idioms.

## When to Choose Which

**Choose wgpu when:**

- You need browser deployment via WebGPU
- You need to support older GPUs or wide device compatibility
- You want the stability of a specification-driven API
- You need the wgpu ecosystem (examples, community, tooling)

**Choose Goldy when:**

- You target only modern desktop/mobile hardware (2018+)
- You want a minimal API surface with bindless as the default
- You want native Metal without a translation layer
- You want Slang's module system and shader language features
- Compute workloads are central to your application

Both libraries are valid choices — the right one depends on your hardware requirements, deployment targets, and whether you value broad compatibility or API simplicity.
