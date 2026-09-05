# Examples Gallery

Goldy ships **23 Rust examples**, each a complete runnable program. Every example has a page
here that inlines its Rust and Slang source straight from the repository, so what you read is
always what compiles.

## Running Examples

```bash
cargo run --features examples --example <name> --release
```

All windowed examples exit on **Escape**, handle window resizes, and auto-exit after a soak
period that `GOLDY_EXAMPLE_TIMEOUT` overrides. To run every example back to back:

```bash
./run_all_examples.sh
GOLDY_BACKEND=webgpu ./run_all_examples.sh
EXAMPLE_TIMEOUT=10 ./run_all_examples.sh
```

## Backends

The examples run on every shipped backend — Vulkan 1.4+, DX12, and Metal Tier 2+ — and on the
in-progress WebGPU backend, selected with `GOLDY_BACKEND` (see
[Backend Architecture](../backends/overview.md)):

```bash
GOLDY_BACKEND=webgpu cargo run --no-default-features --features webgpu,examples --example triangle
```

Two examples probe capabilities and exit cleanly when they are missing:
[`mesh_triangle`](./mesh_triangle.md) needs mesh shaders and [`ray_query`](./ray_query.md) needs
ray query, neither of which the WebGPU backend implements. The WebGPU backend runs the examples
natively through `wgpu` — Goldy does not build for `wasm32` yet, so these pages carry source
rather than in-browser canvases.

## Bindless Basics

Fundamental Goldy patterns: vertex buffers, surfaces, uniforms, and fragment shaders.

| Example | What it demonstrates |
|---------|---------------------|
| [**`triangle`**](./triangle.md) | Minimal windowed program: retained scheme, offscreen render pass, present via `SurfaceExchange`. |
| [**`mesh_triangle`**](./mesh_triangle.md) | The `triangle` present path driven by `MeshPipeline` and `dispatch_mesh`. |
| [**`gradient`**](./gradient.md) | Animated fullscreen gradient from a time uniform, with vertex-less rendering. |
| [**`checkerboard`**](./checkerboard.md) | Procedural animated checkerboard via UV distortion in a fragment shader. |

## Compute Workflows

`ComputePipeline` and `Scheme` for GPU-side data processing, including compute-to-surface.

| Example | What it demonstrates |
|---------|---------------------|
| [**`compute_particles`**](./compute_particles.md) | Compute updates particle positions; graphics renders instanced quads. |
| [**`game_of_life`**](./game_of_life.md) | Conway's Game of Life with ping-pong sub-views in one retained mosaic parcel. |
| [**`compute_to_surface`**](./compute_to_surface.md) | Pure compute rendering — no `RenderPipeline`, writes the drawable directly. |
| [**`ray_query`**](./ray_query.md) | Triangle BLAS/TLAS with inline `RayQuery` in `[goldy_compute]`. |

## Graphics Pipelines

Classic rendering techniques: depth testing, textures, instancing, and 3D projection.

| Example | What it demonstrates |
|---------|---------------------|
| [**`solid_cube`**](./solid_cube.md) | Solid 3D cube with per-face colours and a depth buffer. |
| [**`spinning_cube`**](./spinning_cube.md) | 3D wireframe cube using line primitives. |
| [**`depth_quads`**](./depth_quads.md) | Depth buffer proves draw-order independence. |
| [**`textured_quad`**](./textured_quad.md) | Procedural texture on a quad with stage-local resources. |
| [**`instancing`**](./instancing.md) | GPU-driven instancing with compute-updated transforms. |
| [**`bouncing_lines`**](./bouncing_lines.md) | `LINE_LIST` topology with simple compute-driven physics. |
| [**`waveform`**](./waveform.md) | `LINE_STRIP` waveform visualizer. |

## Fragment Shader Effects

Screen-space effects with no geometry beyond a fullscreen triangle.

| Example | What it demonstrates |
|---------|---------------------|
| [**`plasma`**](./plasma.md) | Demoscene plasma effect. |
| [**`tunnel`**](./tunnel.md) | Flying-through-a-tunnel polar-coordinate effect. |
| [**`metaballs`**](./metaballs.md) | Metaball field rendering. |
| [**`mandelbrot`**](./mandelbrot.md) | Interactive Mandelbrot explorer. |

## Interactive and Multi-Window

Input handling, runtime state changes, and more than one surface per device.

| Example | What it demonstrates |
|---------|---------------------|
| [**`digital_clock`**](./digital_clock.md) | Seven-segment clock display with CPU-generated geometry. |
| [**`starfield`**](./starfield.md) | 3D starfield with compute-driven star recycling. |
| [**`particles`**](./particles.md) | Rain and snow particle system with a runtime mode switch. |
| [**`multi_window`**](./multi_window.md) | Three windows sharing one device. |

## Shared Code

Examples share a small amount of scaffolding — FPS reporting, run limits, hidden-window
creation, and surface-matched pipeline rebuilds. See [Shared Helpers](./shared-helpers.md).
