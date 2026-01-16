# Motivation

## The State of Graphics APIs in 2025

The complexity of graphics APIs, shader frameworks, and drivers has increased rapidly over the past decades. The pipeline state object (PSO) explosion has gotten out of hand. How did we end up with 100GB local shader pipeline caches and massive cloud servers to host them?

Sebastian Aaltonen's blog post **["No Graphics API"](https://www.sebastianaaltonen.com/blog/no-graphics-api)** articulates this problem brilliantly:

> "DirectX 12, Vulkan, and Metal are often referred to as 'modern APIs'. These APIs are now 10 years old. They were initially designed to support GPUs that are now 13 years old, an incredibly long time in GPU history."

The post demonstrates that modern GPUs have evolved significantly:

- **Complete cache hierarchies** with coherent last-level caches
- **PCIe REBAR/UMA** allowing CPUs to write directly to GPU memory
- **64-bit GPU pointers** directly supported in shaders
- **Bindless texture samplers** eliminating CPU driver descriptor binding
- **Texture descriptors** stored directly in arrays within GPU memory

If we were to design an API tailored for modern GPUs today, it wouldn't need most of the persistent "retained mode" objects that DX12, Metal 1, and Vulkan 1.0 required. The compromises made for 2013-era hardware are no longer necessary.

## The Problem with "Modern" APIs

The so-called modern APIs (DX12, Vulkan, Metal) carry significant historical baggage:

### From Vulkan
- Render pass objects (replaced by dynamic rendering in 1.4+)
- Descriptor set layouts (replaced by bindless descriptors in 1.2+)
- Complex image layout transitions
- Pipeline layouts as explicit objects
- Separate VkBuffer and VkDeviceMemory management

### From OpenGL Heritage
- Fixed-function pipeline concepts bleeding through
- Binding point abstractions (GL_TEXTURE0, etc.)
- State machine mental model

### From OpenCL
- Separate compute API from graphics
- Complex platform/device enumeration
- NDRange dispatch complexity

## Goldy's Answer

Goldy takes the position that **we can do better by doing less**. By targeting only modern hardware (Vulkan 1.4+, DX12, Metal), Goldy can:

1. **Drop compatibility complexity** - No fallback paths for 10-year-old GPUs
2. **Use modern defaults** - Bindless, dynamic rendering, unified memory
3. **Stay simple** - Fewer concepts, less API surface
4. **Move fast** - Not bound by standards committees

### The HAL Lesson

From analysis of GPU abstraction history:

> "Classic HAL failed by abstracting behavior AND cost. Modern approaches succeed by abstracting meaning and rules while exposing cost and reality."

Goldy follows this principle:

| Abstract (Goldy's job) | Expose (not Goldy's job) |
|---------------------|------------------------|
| Semantics (what operations mean) | Performance characteristics |
| Safety guarantees | Optimal usage patterns |
| Resource ownership model | Platform-specific fast paths |
| Capability queries | Cost of operations |

## Why Not Just Use wgpu?

[wgpu](https://wgpu.rs/) is excellent and serves a different purpose:

| Aspect | wgpu | Goldy |
|--------|------|-----|
| **Identity** | WebGPU implementation for Rust | Modern Rust GPU library |
| **Governance** | Bound by WebGPU spec committee | Independent, opinionated |
| **Pace** | Moves at spec speed | Moves as fast as we want |
| **Legacy** | Must support WebGPU's LCD | Assumes modern features |
| **Metal** | Via WebGPU abstraction | Native Metal idioms |
| **Vulkan** | Must work on 1.0+ | Requires 1.4+ |

wgpu is the right choice for web compatibility and broad device support. Goldy is for when you want simplicity and can require modern hardware.

## Slang: One Shader Language

Goldy uses [Slang](https://shader-slang.org/) as its sole shading language. This is a deliberate choice:

| Aspect | Benefit |
|--------|---------|
| **Portability** | Single source compiles to SPIR-V, WGSL, HLSL, MSL |
| **Familiar syntax** | HLSL-like, industry-standard |
| **Modern features** | Modules, generics, automatic differentiation |
| **Khronos governance** | Long-term stability and active development |

Rather than supporting multiple shader languages (WGSL, GLSL, HLSL) and maintaining translation layers, Goldy trusts Slang to handle cross-platform compilation. This keeps Goldy's codebase simple while providing maximum portability.

## Inspirations

Goldy synthesizes ideas from several sources, each contributing distinct principles to its design.

### Sebastian Aaltonen: "No Graphics API"

The **primary philosophical foundation** for Goldy. Aaltonen argues that modern GPUs (2018+) have evolved so far beyond what DX12/Vulkan/Metal were designed for that we could dramatically simplify if we dropped legacy support:

| Feature | Old GPUs (2012) | Modern GPUs (2018+) |
|---------|-----------------|---------------------|
| Cache model | Incoherent, manual flush | Coherent L2, automatic |
| Memory | Discrete, explicit copy | PCIe REBAR, unified where possible |
| Pointers | 32-bit, indirect | 64-bit, direct in shaders |
| Descriptors | CPU-bound binding | Bindless, GPU-resident |
| Render passes | Required for tile optimization | Dynamic rendering works fine |

Goldy applies this by requiring 2018+ hardware, using dynamic rendering, bindless descriptors, and assuming coherent caches. This isn't theoretical—it's what game engines already do internally. Goldy just makes it the public API.

### Ralph Levien: "Requiem for piet-gpu-hal"

Ralph Levien's [post-mortem](https://raphlinus.github.io/rust/gpu/2023/01/07/requiem-piet-gpu-hal.html) on building a GPU HAL provides the crucial insight:

> "Classic HAL failed by abstracting behavior AND cost. Modern approaches succeed by abstracting meaning and rules while exposing cost and reality."

Traditional HALs tried to hide everything—creating "magic black boxes" where developers couldn't understand or optimize performance. Goldy applies this by clearly separating what it abstracts (semantics, safety, ownership) from what it exposes (cost, performance characteristics, platform differences).

### Wayland Compositor Architecture

Wayland's shift from X11's distributed protocol model to local computation influences Goldy's design:

```
X11:     App → draw commands → protocol → server → GPU → display
Wayland: App → GPU renders buffer → compositor → display
```

The client renders **complete frames**. Synchronization is **explicit**, not implicit. Goldy applies this with a frame-based model, explicit synchronization, and zero-copy where possible.

### Slang: Unified Shader Language

The shader language landscape is fragmented (GLSL, HLSL, MSL, WGSL). [Slang](https://shader-slang.org/) solves this at the source level—write once, compile to any backend. Goldy accepts only Slang shaders, eliminating shader translation bugs and simplifying the codebase. Slang also provides modern features WGSL lacks: modules, generics, automatic differentiation.

### WGPU: API Patterns

[wgpu](https://wgpu.rs/) provides excellent API ergonomics that Goldy borrows: Instance/Device architecture, CommandEncoder pattern, explicit pass structure. However, wgpu must implement WebGPU exactly and support the web. Goldy is free to diverge—supporting the *union* of modern platform features rather than the lowest common denominator.

### TU Darmstadt: Recursive HAL Analysis

The paper ["Conceptual Approach Towards Recursive Hardware Abstraction Layers"](https://www.kom.tu-darmstadt.de/papers/KCGS17.pdf) by Konrad et al. provides rigorous analysis of what a minimal HAL actually needs, categorizing features as **necessary** vs **emulatable**. This validates Goldy's approach: if you target modern hardware, the abstraction almost writes itself because modern GPUs have converged on similar capabilities.

### Additional Influences

- **CUDA** - Composable language exposing memory directly with a broad library ecosystem
- **Metal's evolution** - Moving toward 64-bit pointers and simpler binding models
- **Rust's ownership model** - Explicit resource management without hidden costs

### Synthesis

These inspirations combine:

```
Sebastian Aaltonen  →  "Target modern hardware, drop legacy complexity"
Ralph Levien        →  "Abstract meaning, not cost—expose reality"
Wayland             →  "Frames not commands, explicit sync"
Slang               →  "One shader language, compiled to all backends"
wgpu                →  "Good API ergonomics, command encoder pattern"
TU Darmstadt        →  "Rigorous minimal feature analysis"
```

No single source defines Goldy. The value is in the synthesis.

### Name

Goldy aspires to exist in the golden mean between wgpu emphasis on compatibility and the vision of no-graphics-api.

## Further Reading

- [Sebastian Aaltonen: No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api) - Essential reading on modern GPU architecture
- [Ralph Levien: Requiem for piet-gpu-hal](https://raphlinus.github.io/rust/gpu/2023/01/07/requiem-piet-gpu-hal.html) - Lessons from a failed HAL
- [TU Darmstadt: Recursive HALs](https://www.kom.tu-darmstadt.de/papers/KCGS17.pdf) - Academic analysis of HAL requirements
- [What Goldy Sheds](./what-goldy-sheds.md) - Detailed breakdown of removed complexity
- [Goldy vs wgpu](./comparison.md) - When to use which

