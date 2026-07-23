<p align="center">
  <img src="assets/goldy.png" alt="Goldy Logo" width="240">
</p>

# Goldy: GPU runtime for the Fondaco Machine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Goldy** is a Rust GPU library that realizes the [Fondaco Machine](https://github.com/koba-computers/docu/blob/main/research/technical_stack/fondaco/fondaco-machine-specification.md) on modern hardware. Merchants describe **parcels** (data) and **schemes** (computation with ownership-derived ordering); Goldy manages the physical medium, schedules dispatches, and mediates all foreign interaction through **exchanges**.

> **Maturity**: Goldy **0.1.x** is pre-0.2. Breaking API changes are expected. Python bindings are alpha; Rust is the primary surface.

## Why Fondaco?

Traditional GPU APIs expose descriptors, barriers, render passes, and swapchain plumbing. The Fondaco model inverts that: the merchant is sovereign over parcels; the runtime holds them in trust, derives precedences from ownership claims, and settles foreign handoffs (present, readback) through linear exchange claims.

Goldy makes that model practical on 2020+ GPUs:

| Fondaco concept | Goldy realization |
|-----------------|-------------------|
| Scheme | [`Scheme`](https://docs.rs/goldy/latest/goldy/struct.Scheme.html) — retained dependency graph, resubmitted each frame |
| Parcel | [`Parcel`](https://docs.rs/goldy/latest/goldy/struct.Parcel.html) / [`Buffer`](https://docs.rs/goldy/latest/goldy/struct.Buffer.html) / [`Texture`](https://docs.rs/goldy/latest/goldy/struct.Texture.html) — stable handles |
| Dispatch | Compute, render, copy, and present nodes inside a scheme |
| Exchange | [`SurfaceExchange`](https://docs.rs/goldy/latest/goldy/struct.SurfaceExchange.html), [`MemoryExchange`](https://docs.rs/goldy/latest/goldy/struct.MemoryExchange.html) |
| Settlement | [`Transaction`](https://docs.rs/goldy/latest/goldy/struct.Transaction.html) → [`Claim`](https://docs.rs/goldy/latest/goldy/struct.Claim.html) → `consume()` / `discard()` |

## What Goldy ships today

- **Scheme-first API** — record once, submit every frame; barriers and transient aliasing derived automatically
- **Typed bindless shaders** — Slang with `[goldy_compute]`, `[goldy_vertex]`, `[goldy_fragment]` and `goldy_exp` access patterns (`Scattered`, `Broadcast`, `Interpolated`, …)
- **Compute-to-surface** — compute shaders write swapchain drawables directly; no raster pass required
- **Native backends** — Vulkan 1.4+ (Windows, Linux), DX12 (Windows), Metal Tier 2+ (macOS); no MoltenVK
- **Multi-language bindings** — Rust (primary), [Python](https://pypi.org/project/goldy/), [.NET](dotnet/Goldy/README.md), [C++](cpp/README.md)
- **21 Rust examples** — triangle, compute particles, Game of Life, plasma, multi-window, and more

Not yet shipped (see [runtime doc](https://github.com/koba-computers/docu/blob/main/research/technical_stack/fondaco/fondaco-machine-goldy-runtime.md) for status): yielding scripts, `$yield` petitions, scheme fusion/splitting, defragmentation, WASI host integration.

## Quick example: compute-to-surface

```rust
use goldy::{
    Buffer, BufferKind, ComputePipeline, Context, DeviceDescriptor, Instance,
    NodeAccess, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule,
    SurfaceConfig, SurfaceExchange, Transaction,
};

let instance = Instance::new()?;
let device = instance
    .request_adapter(&RequestAdapterOptions::default())?
    .request_device(&DeviceDescriptor::default())?;
let ctx = device.create_context()?;
let pool = RetainedPool::new(&device)?;
let surface = SurfaceExchange::new(&ctx, &window, SurfaceConfig::default())?;

let shader = ShaderModule::from_slang(&device, COMPUTE_SHADER)?;
let pipeline = ComputePipeline::new(&device, &shader)?;
let uniforms = pool.alloc_buffer_with_data(&device, &uniforms_data, BufferKind::Broadcast)?;

let mut scheme = Scheme::new(&ctx);
let present = surface.bind_destination(&mut scheme)?;
scheme
    .node("render", &pipeline)
    .with_parcel(&uniforms, NodeAccess::Read)
    .with_present(&present.0)
    .dispatch(wg_x, wg_y, 1);

// Each frame:
let mut submission = scheme.submit()?;
present.1.claim(&mut submission)?.consume()?;
```

Windowed examples require the `examples` feature:

```bash
cargo run --features examples --example compute_to_surface --release
cargo run --features examples --example triangle --release
```

## Architecture

```
Merchant (schemes + parcels)
        │
        ▼
   Scheme / GraphIR  ←── wave & partition analysis, retention
        │
        ▼
  RetainedPool / TransientPool / VramAllocator
        │
        ▼
  Vulkan │ DX12 │ Metal   ←── Slang → SPIR-V / DXIL / MSL
        │
        ▼
  Exchanges (present, readback)
```

Goldy abstracts **where bytes live** (Layer A: medium) but exposes **what access costs** (Layer B: coalescing, occupancy, residency). See the [design thesis](https://github.com/koba-computers/docu/blob/main/research/technical_stack/fondaco/abstract-gpu.md).

## Installation

```toml
[dependencies]
goldy = "0.1"
```

Slang is **embedded at build time** and extracted at runtime — application developers do not install Slang separately. Set `GOLDY_SLANG_PATH` only to override. Maintainer bump procedure: `slang/manifest.json` + `slang/download.sh`.

Release packaging: [PACKAGING.md](PACKAGING.md). Shader debugging: [DEBUGGING.md](DEBUGGING.md).

## Platforms

| Platform | Backend | Window surfaces |
|----------|---------|-----------------|
| Windows | DX12 (default), Vulkan | Yes |
| Linux | Vulkan | Wayland (X11 not supported) |
| macOS | Metal | Yes |

Override backend: `GOLDY_BACKEND=vulkan|dx12|metal`.

Minimum hardware: Vulkan 1.4+, DX12 with Enhanced Barriers, Metal Argument Buffers Tier 2+. See [Target Hardware](https://koubaa.github.io/goldy/design/hardware.html).

## Documentation

- 📖 **[Full mdBook](https://koubaa.github.io/goldy/)** — tutorials, programming model, backends, bindings
- 📖 **[Fondaco Machine spec](https://github.com/koba-computers/docu/blob/main/research/technical_stack/fondaco/fondaco-machine-specification.md)** — normative abstract machine
- 📖 **[Goldy runtime mapping](https://github.com/koba-computers/docu/blob/main/research/technical_stack/fondaco/fondaco-machine-goldy-runtime.md)** — what Goldy implements today vs designs in progress
- 📖 **[API reference](https://docs.rs/goldy)** — Rust docs

## Development

```bash
cargo fmt --all -- --check
cargo clippy -- -D warnings
GOLDY_VALIDATION=all cargo test
```

Run all examples: `./run_all_examples.sh` (requires `--features examples`).

## License

MIT — see [LICENSE](LICENSE).

## Author

Mohamed Koubaa
