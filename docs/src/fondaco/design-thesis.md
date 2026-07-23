# Design Thesis

Why Goldy exists, and how it differs from a conventional GPU library. Machine semantics live in [Machine Specification](./specification.md); what is shipped today is in [Goldy Runtime Mapping](./goldy-runtime.md). Vocabulary: [Terminology](./terminology.md).

## Executive summary

**Goldy** is a GPU runtime for the **Fondaco Machine** — merchants are sovereign over parcels (data) and express computation as **schemes** (dispatches + ownership-derived precedences). It targets modern native APIs (Vulkan 1.4+, DX12, Metal Tier 2+) with **no translation layers**, a **single shader language** (Slang), and a **scheme-first** API that sheds descriptor sets, explicit barriers, and swapchain ceremony.

## The Fondaco model on GPU

Traditional GPU programming exposes descriptor set layouts, image layout transitions, render pass objects, pipeline layouts, and raw swapchain images with semaphores.

Fondaco instead gives the merchant:

| Concept | Role |
|---------|------|
| **Parcel** | Stable identity for data; physical medium is runtime-managed |
| **Scheme** | First-class computation graph; precedences from ownership |
| **Exchange** | Mediated foreign I/O (present, readback) via linear claims |
| **Gate** | Where the runtime may relocate, reclaim, or insert work |

Goldy's public API (`Scheme`, `Parcel`, `SurfaceExchange`, `MemoryExchange`) implements this model. Internal bindless descriptor heaps are **not** part of the merchant ABI.

## Layer A vs Layer B

From Ralph Levien's HAL post-mortem and Goldy's load-bearing invariant ([runtime mapping §12](./goldy-runtime.md#12-abstract-the-medium-expose-cost)):

> Classic HAL failed by abstracting behavior **and** cost. Modern approaches succeed by abstracting meaning and rules while exposing cost and reality.

| Layer | Abstract (Goldy) | Expose (Goldy) |
|-------|------------------|----------------|
| **A — Medium** | Parcel identity, warehouse, residency plumbing | — |
| **B — Cost** | — | Access patterns, occupancy, resize cost, readback path |

**Layer A** examples Goldy hides: descriptor slot virtualization on reslot, transient aliasing, growable buffers with stable handles.

**Layer B** examples Goldy keeps visible: `Scattered` vs `Broadcast` vs `Interpolated`, `buffer_resize_cost`, `has_zero_copy_storage_readback`, residency model per backend.

## Access patterns, not graphics categories

Goldy names resources for **what the hardware does**, not which API invented the term:

| Goldy term | Hardware behavior |
|------------|-------------------|
| **Scattered** | Any-thread read/write |
| **BufRO** | Read-only scattered (stronger cache hints) |
| **Broadcast** | Wave-broadcast constant fetch |
| **Interpolated** | Dedicated texture filtering silicon |
| **DirectSpatial** | 2D/3D indexed access, no filtering |
| **Filter** | Sampler configuration |

Shaders declare these as typed Slang parameters on `[goldy_*]` entry points. The CPU side uses matching buffer / texture kinds and scheme `NodeAccess`. See [Parcels](../programming-model/parcels.md).

## What Goldy sheds

Because Goldy requires modern baseline hardware, it drops:

| Legacy concept | Goldy approach |
|----------------|----------------|
| Render pass objects | Dynamic rendering |
| Descriptor set layouts | Bindless (backend-internal) |
| Separate transfer queues | Unified queue model |
| OpenGL fixed function | Shaders only |
| Multiple shader languages | Slang only |

Details: [What Goldy Sheds](../design/what-goldy-sheds.md) and [Goldy vs wgpu](../design/comparison.md).

## Slang and virtual entry points

Goldy uses [Slang](https://shader-slang.org/) as its sole shader language, compiled at runtime to SPIR-V / DXIL / MSL. The compiler is embedded in the crate.

```slang
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(MyUniforms cfg, Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + cfg.base;
}
```

The `virtual_main` transform generates platform entry points with bindless slot resolution — see [Virtual Entry Points](../programming-model/virtual-entry-points.md). The `goldy_exp` standard library provides access functions, math, color utilities, vertex formats, and workgroup collectives.

## Unified graphics and compute

Goldy treats graphics and compute as one scheme:

- Compute simulation → raster present in one retained scheme
- **Compute-to-surface**: compute writes swapchain drawables directly (no `RenderPipeline`) — [Compute to Surface](../compute/compute-to-surface.md)
- Cross-scheme ordering via context timeline / ledger

This matches how modern engines and CUDA-style workloads converge on the same memory-access patterns. Goldy provides the primitives; performance patterns remain the developer's responsibility.

## Backends

| Platform | Backend | Notes |
|----------|---------|-------|
| Windows | DX12 (default), Vulkan | PIX on DX12 |
| Linux | Vulkan | Wayland surfaces |
| macOS | Metal | Native, not MoltenVK |

Auto-selection with `GOLDY_BACKEND` override. Capability queries reflect backend-specific features honestly — [Backend Architecture](../backends/overview.md).

## Goldy vs wgpu

| | wgpu | Goldy |
|---|------|-------|
| Identity | WebGPU for Rust | Fondaco GPU runtime |
| Governance | W3C spec | Independent |
| Legacy floor | Vulkan 1.0+, web LCD | Vulkan 1.4+, modern only |
| Binding model | WebGPU bind groups | Typed bindless + schemes |
| Browser | Yes | No (native only) |

Use wgpu for web and maximum compatibility. Use Goldy for scheme-first Fondaco semantics and the modern feature union. Full write-up: [Goldy vs wgpu](../design/comparison.md).

## Inspirations

| Source | Contribution |
|--------|--------------|
| Sebastian Aaltonen — "No Graphics API" | Target modern hardware; drop legacy ceremony |
| Ralph Levien — piet-gpu-hal post-mortem | Abstract meaning, expose cost |
| Wayland compositor model | Complete frames, explicit sync, mediated present |
| Slang | One shader language, multi-backend |
| wgpu | Instance / device ergonomics (adapted to schemes) |
| TU Darmstadt HAL paper | Minimal necessary feature analysis |

See also [Motivation](../design/motivation.md).

## Roadmap posture

**Shipped in 0.1.x**: schemes, exchanges, compute-to-surface, growable buffers, retained replay, language bindings, Rust examples.

**Designed**: yielding scripts, scheme fusion / splitting, defragmentation, compute algorithm libraries (scan, sort, BLAS-class).

**Speculative**: WASI `goldy-host`, CUDA backend exploration, pre-2020 traditional-binding backend.

Do not treat **Designed** or **Speculative** items as shipped. Status table: [Goldy Runtime Mapping](./goldy-runtime.md).

## Further reading

1. [Machine Specification](./specification.md)
2. [Goldy Runtime Mapping](./goldy-runtime.md)
3. [Terminology](./terminology.md)
4. [Motivation](../design/motivation.md) · [What Goldy Sheds](../design/what-goldy-sheds.md) · [Target Hardware](../design/hardware.md)
