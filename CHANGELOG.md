# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-11

Initial release of Goldy, a Fondaco Machine GPU runtime for Rust.

### Added

- **Scheme-first API** — retained dependency graphs with ownership-derived ordering
- **Vulkan 1.4+ backend** — dynamic rendering, descriptor indexing (Windows, Linux)
- **DX12 backend** (Windows) — native DirectX 12
- **Metal backend** (macOS) — native Metal Tier 2+, not MoltenVK
- **Slang shader compilation** — embedded compiler; SPIR-V, DXIL, and MSL targets
- **Virtual entry points** — `[goldy_compute]`, `[goldy_vertex]`, `[goldy_fragment]` with `goldy_exp`
- **Exchanges** — `SurfaceExchange` (present), `MemoryExchange` (readback/deposit)
- **Compute-to-surface** — compute shaders write swapchain drawables directly
- **Growable buffers** — `Buffer::resize_to` with stable handles
- **Language bindings** — Python (PyPI), .NET (NuGet), C++ (FFI)
- **21 Rust examples** — triangle through multi-window and headless workflows

### Platforms

- Windows x86_64 (DX12 default, Vulkan optional)
- Linux x86_64 (Vulkan; Wayland surfaces)
- macOS aarch64 (Metal)

### Dependencies

- Slang compiler embedded at build time (override with `GOLDY_SLANG_PATH`)
- Vulkan 1.4+ capable GPU, DX12 Enhanced Barriers, or Metal Tier 2+ (2018+ recommended)
