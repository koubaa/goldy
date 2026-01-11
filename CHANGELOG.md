# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-11

Initial release of Goldy, a modern GPU library for Rust.

### Added

- **Vulkan 1.3+ backend** - Full support for modern Vulkan features including dynamic rendering
- **DX12 backend** (Windows) - DirectX 12 support for Windows platforms
- **Slang shader compilation** - Compile Slang shaders to SPIR-V at runtime
- **Shader library system** - Register reusable shader modules with `import` support
- **Built-in `goldy_exp` library** - Experimental utilities for common shader patterns
- **Surface rendering** - Window/swapchain support via `raw-window-handle`
- **Render targets** - Off-screen rendering with CPU readback
- **Compute pipelines** - GPU compute shader support
- **Bind groups** - Descriptor set abstraction for uniform buffers and textures
- **18 examples** - From basic triangle to compute particles and Game of Life

### Platforms

- Windows x86_64 (Vulkan, DX12)
- Linux x86_64 (Vulkan)
- macOS aarch64 (Vulkan via MoltenVK)

### Dependencies

- Requires Slang compiler (auto-downloaded during build, or provide via Vulkan SDK)
- Vulkan 1.3+ capable GPU (2018 or newer recommended)

