# Motivation

## The State of Graphics APIs in 2025

The complexity of graphics APIs, shader frameworks, and drivers has increased  over the past decades. The pipeline state object (PSO) explosion has gotten out of hand. How did we end up with 100GB local shader pipeline caches and massive cloud rapidlyservers to host them?

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
- Render pass objects (replaced by dynamic rendering in 1.3+)
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

## RAG's Answer

RAG takes the position that **we can do better by doing less**. By targeting only modern hardware (Vulkan 1.4+, Metal 2+, DX12), RAG can:

1. **Drop compatibility complexity** - No fallback paths for 10-year-old GPUs
2. **Use modern defaults** - Bindless, dynamic rendering, unified memory
3. **Stay simple** - Fewer concepts, less API surface
4. **Move fast** - Not bound by standards committees

### The HAL Lesson

From analysis of GPU abstraction history:

> "Classic HAL failed by abstracting behavior AND cost. Modern approaches succeed by abstracting meaning and rules while exposing cost and reality."

RAG follows this principle:

| Abstract (RAG's job) | Expose (not RAG's job) |
|---------------------|------------------------|
| Semantics (what operations mean) | Performance characteristics |
| Safety guarantees | Optimal usage patterns |
| Resource ownership model | Platform-specific fast paths |
| Capability queries | Cost of operations |

## Why Not Just Use wgpu?

[wgpu](https://wgpu.rs/) is excellent and serves a different purpose:

| Aspect | wgpu | RAG |
|--------|------|-----|
| **Identity** | WebGPU implementation for Rust | Modern Rust GPU library |
| **Governance** | Bound by WebGPU spec committee | Independent, opinionated |
| **Pace** | Moves at spec speed | Moves as fast as we want |
| **Legacy** | Must support WebGPU's LCD | Assumes modern features |
| **Metal** | Via WebGPU abstraction | Native Metal idioms |
| **Vulkan** | Must work on 1.0+ | Requires 1.4+ |

wgpu is the right choice for web compatibility and broad device support. RAG is for when you want simplicity and can require modern hardware.

## Slang: One Shader Language

RAG uses [Slang](https://shader-slang.org/) as its sole shading language. This is a deliberate choice:

| Aspect | Benefit |
|--------|---------|
| **Portability** | Single source compiles to SPIR-V, WGSL, HLSL, MSL |
| **Familiar syntax** | HLSL-like, industry-standard |
| **Modern features** | Modules, generics, automatic differentiation |
| **Khronos governance** | Long-term stability and active development |

Rather than supporting multiple shader languages (WGSL, GLSL, HLSL) and maintaining translation layers, RAG trusts Slang to handle cross-platform compilation. This keeps RAG's codebase simple while providing maximum portability.

## Inspiration

RAG draws inspiration from:

- **Sebastian Aaltonen's "No Graphics API"** - The vision of what's possible with modern hardware
- **Slang** - A modern shader language with cross-platform compilation
- **CUDA** - A composable language exposing memory directly with a broad library ecosystem
- **Metal's evolution** - Moving toward 64-bit pointers and simpler binding models
- **Rust's ownership model** - Explicit resource management without hidden costs

## Further Reading

- [Sebastian Aaltonen: No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api) - Essential reading on modern GPU architecture
- [What RAG Sheds](./what-rag-sheds.md) - Detailed breakdown of removed complexity
- [RAG vs wgpu](./comparison.md) - When to use which

