# Motivation

## The Problem with "Modern" Graphics APIs

DX12, Vulkan, and Metal are commonly called modern APIs, but they were designed over a decade ago for hardware that has since changed dramatically. Sebastian Aaltonen's **["No Graphics API"](https://www.sebastianaaltonen.com/blog/no-graphics-api)** captures the core tension:

> "DirectX 12, Vulkan, and Metal are often referred to as 'modern APIs'. These APIs are now 10 years old. They were initially designed to support GPUs that are now 13 years old, an incredibly long time in GPU history."

The GPU architectures those APIs targeted lacked coherent caches, bindless descriptors, and 64-bit pointers. The APIs compensated with layers of indirection — descriptor sets, render pass objects, explicit image layout transitions, pipeline layouts as first-class objects — that served as hints and contracts for hardware that needed them.

Modern GPUs (roughly 2018+) no longer need most of that scaffolding:

| Then (2012-era) | Now (2018+) |
|-----------------|-------------|
| Incoherent caches, manual flush | Coherent L2, automatic |
| Discrete memory, explicit copies | PCIe REBAR, unified where possible |
| 32-bit pointers, indirect | 64-bit, direct in shaders |
| CPU-bound descriptor binding | Bindless, GPU-resident |
| Render passes for tile optimization | Dynamic rendering works fine |

Yet every application using these APIs still pays the complexity cost of the old model, even when targeting only recent hardware.

## Why Bindless Matters

Traditional GPU programming organizes resources into *descriptor sets* — fixed layouts of bindings that must be declared ahead of time, allocated from pools, and swapped between draw calls. This model creates a cascade of complexity:

- **Pipeline layout explosion**: Every unique combination of descriptor set layouts produces a distinct pipeline layout, and each pipeline layout dimension multiplies the total pipeline state permutation count.
- **CPU overhead**: Updating and binding descriptor sets each frame is a significant portion of CPU-side draw call cost.
- **Shader inflexibility**: Shaders are coupled to their binding layout; changing which resources a shader accesses means changing the pipeline.

Bindless resource access replaces all of this with a single concept: resources live in GPU-visible memory, and shaders access them by index. There are no set layouts to declare, no pools to manage, no binding points to track. A shader that needs buffer #7 just reads slot 7 from a flat descriptor heap.

This isn't exotic — it's how game engines have been working internally for years. Goldy makes it the public API rather than hiding it behind compatibility abstractions.

## Why a Dependency Graph (Scheme)

Bindless access means shaders can read *any* resource at any time. The traditional model of inserting barriers at the call site ("I'm about to read this buffer, so transition it now") breaks down when the set of resources a dispatch touches isn't known until the shader runs.

Goldy uses a retained **scheme** — a dependency graph you record once and submit many times — to solve this. You declare nodes and their resource dependencies; Goldy derives the barriers, layout transitions, and execution order automatically. This is both safer (no missed barriers) and simpler (no manual synchronization) than the alternative.

The scheme also enables Goldy to batch and reorder work across the frame, which matters for compute-heavy workloads where multiple dispatches feed into each other before anything reaches the screen.

## Why Slang

The shader language landscape is fragmented. GLSL, HLSL, MSL, and WGSL each target a subset of platforms, and none is a clean superset of the others. Libraries that support multiple shading languages maintain translation layers and per-language workarounds, which is a significant source of bugs and complexity.

[Slang](https://shader-slang.org/) solves this at the source level. A single Slang source file compiles to SPIR-V (Vulkan), DXIL (DX12), and MSL (Metal). It uses HLSL-familiar syntax with additions that matter for modern GPU programming:

| Feature | Why it matters |
|---------|---------------|
| Modules and `import` | True separate compilation, no `#include` fragility |
| Generics | Type-safe reusable shader code |
| Automatic differentiation | First-class for ML and physics workloads |
| Khronos governance | Long-term stability and active development |

By committing to Slang as the sole shader language, Goldy eliminates an entire category of cross-platform bugs and keeps its codebase focused on GPU work rather than shader translation.

## Intellectual Roots

Goldy synthesizes ideas from several sources:

- **Sebastian Aaltonen, "No Graphics API"** — The primary philosophical foundation. Modern GPUs have converged enough that a dramatically simpler API is possible if you drop legacy support.
- **Ralph Levien, ["Requiem for piet-gpu-hal"](https://raphlinus.github.io/rust/gpu/2023/01/07/requiem-piet-gpu-hal.html)** — The insight that good abstractions expose *cost and reality* while abstracting *meaning and rules*. Classic HALs failed by hiding both.
- **wgpu** — Excellent API ergonomics (Instance/Device architecture, explicit command recording, pass structure). Goldy borrows patterns but is free to diverge from the WebGPU spec.
- **Wayland compositor architecture** — Frames, not commands. Explicit synchronization, not implicit state machines.
- **TU Darmstadt, ["Recursive Hardware Abstraction Layers"](https://www.kom.tu-darmstadt.de/papers/KCGS17.pdf)** — Rigorous analysis of what a minimal HAL actually needs when targeting converged modern hardware.
- **CUDA** — A composable language that exposes memory directly, with a broad library ecosystem built on that simplicity.

No single source defines Goldy. The value is in the synthesis — and the willingness to ship an opinionated library rather than wait for committee consensus.

### The Name

Goldy aspires to exist in the golden mean between wgpu's emphasis on compatibility and the vision of no-graphics-api.

## Further Reading

- [Sebastian Aaltonen: No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api)
- [Ralph Levien: Requiem for piet-gpu-hal](https://raphlinus.github.io/rust/gpu/2023/01/07/requiem-piet-gpu-hal.html)
- [TU Darmstadt: Recursive HALs](https://www.kom.tu-darmstadt.de/papers/KCGS17.pdf)
- [What Goldy Sheds](./what-goldy-sheds.md)
- [Goldy vs wgpu](./comparison.md)
