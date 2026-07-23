# Motivation

Goldy implements the **Fondaco Machine** on modern GPUs: programs describe parcels and schemes; the runtime manages the physical medium, derives precedences from ownership, and mediates present/readback through exchanges. For the normative machine spec see the [Machine Specification](../fondaco/specification.md); for what Goldy ships today see the [runtime mapping](../fondaco/goldy-runtime.md).

## The Problem with "Modern" Graphics APIs

DX12, Vulkan, and Metal are commonly called modern APIs, but they were designed over a decade ago for hardware that has since changed dramatically. The GPU architectures those APIs targeted lacked coherent caches, bindless descriptors, and 64-bit pointers. The APIs compensated with layers of indirection — descriptor sets, render pass objects, explicit image layout transitions, pipeline layouts as first-class objects — that served as hints and contracts for hardware that needed them.

Furthermore, high-performance GPU programs tend to converge on the same shape, whether they are written with PyTorch, CUDA, Metal, Vulkan, or something else. They are not best understood as a stream of API calls. They are graphs whose nodes are kernels, copies, and foreign operations, and whose edges describe data dependencies. Independent nodes may run concurrently; dependent nodes require an ordering mechanism such as a barrier, event, semaphore, or stream dependency.

PyTorch makes this especially visible. A model is a graph of tensor operations; autograd constructs another graph for the backward pass, and compilers such as TorchInductor capture, specialize, fuse, and schedule that work. Beneath it, CUDA libraries submit kernels and transfers to streams, use events to express cross-stream dependencies, reuse temporary allocations according to tensor lifetimes, and fuse adjacent operations to reduce launch and memory-traffic costs. Hand-written CUDA programs eventually acquire the same machinery: stream graphs, memory pools, dependency tracking, and explicit synchronization around shared buffers.

Graphics workloads arrive at the same structure from another direction. A frame graph records render, compute, copy, and presentation passes together with how each pass reads or writes resources. From those declarations, an engine derives execution order, barriers, transient-memory aliasing, queue placement, and opportunities for overlap. The APIs differ, but the optimization problem is the same: preserve data dependencies while minimizing synchronization, allocation, launch, and memory-traffic costs.

This convergence suggests that the graph and its resource relationships are the durable program model. Descriptor updates, barriers, command buffers, streams, and semaphores - and yes, sometimes even CPU waits - are mechanisms a runtime can derive from that model for a particular GPU.

Yet every application using graphics APIs still pays the complexity cost of the old model, and do not idiomatically map to the model of the best GPU programs.

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

By committing to Slang as the sole shader language, Goldy eliminates an entire category of cross-platform bugs and keeps its codebase focused on GPU work rather than shader translation. By embedding a verified compatible version of slang, goldy simplifies packaging.
