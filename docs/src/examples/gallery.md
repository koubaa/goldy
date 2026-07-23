# Examples Gallery

Goldy ships with **21 Rust examples** demonstrating scheme recording, compute-to-surface, graphics pipelines, and multi-window workflows. Every example uses [Slang](https://shader-slang.org/) shaders and runs on shipped backends (Vulkan 1.4+, DX12, Metal Tier 2+). CUDA and WebGPU backends are in progress; Tenstorrent is planned.

## Running Examples

```bash
cd goldy
cargo run --features examples --example <name> --release
```

All windowed examples support **Escape** to exit and automatic window-resize handling.

---

## Bindless Basics

These examples cover fundamental Goldy patterns: vertex buffers, surfaces, uniforms, and fragment shaders.

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`triangle`** | Minimal windowed program: retained scheme, offscreen render pass, present via `SurfaceExchange`. | [`triangle.rs`](https://github.com/koubaa/goldy/blob/main/examples/triangle.rs) |
| **`gradient`** | Animated full-screen gradient driven by a time uniform. Uses vertex-less rendering and optional `GOLDY_VALIDATE_LAYOUTS`. | [`gradient.rs`](https://github.com/koubaa/goldy/blob/main/examples/gradient.rs) |
| **`checkerboard`** | Procedural animated checkerboard via UV distortion in a fragment shader. | [`checkerboard.rs`](https://github.com/koubaa/goldy/blob/main/examples/checkerboard.rs) |

## Compute Workflows

Examples that use `ComputePipeline` and `Scheme` for GPU-side data processing, including compute-to-surface.

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`compute_particles`** | Compute updates particle positions; graphics renders instanced quads. Retained scheme scheduling. | [`compute_particles.rs`](https://github.com/koubaa/goldy/blob/main/examples/compute_particles.rs) |
| **`game_of_life`** | Conway's Game of Life on the GPU with ping-pong sub-views in one retained mosaic parcel. | [`game_of_life.rs`](https://github.com/koubaa/goldy/blob/main/examples/game_of_life.rs) |
| **`compute_to_surface`** | Pure compute rendering — no `RenderPipeline`. Writes swapchain via `SurfaceExchange::bind_destination`. | [`compute_to_surface.rs`](https://github.com/koubaa/goldy/blob/main/examples/compute_to_surface.rs) |

## Graphics Pipelines

Classic rendering techniques: depth testing, textures, instancing, and 3D projection.

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`solid_cube`** | Solid 3D cube with per-face colors and depth buffer. | [`solid_cube.rs`](https://github.com/koubaa/goldy/blob/main/examples/solid_cube.rs) |
| **`spinning_cube`** | 3D wireframe cube using line primitives. | [`spinning_cube.rs`](https://github.com/koubaa/goldy/blob/main/examples/spinning_cube.rs) |
| **`depth_quads`** | Depth buffer proves draw-order independence. | [`depth_quads.rs`](https://github.com/koubaa/goldy/blob/main/examples/depth_quads.rs) |
| **`textured_quad`** | Procedural checkerboard texture on a quad. | [`textured_quad.rs`](https://github.com/koubaa/goldy/blob/main/examples/textured_quad.rs) |
| **`instancing`** | GPU-driven instancing with compute-updated transforms. | [`instancing.rs`](https://github.com/koubaa/goldy/blob/main/examples/instancing.rs) |
| **`bouncing_lines`** | `LINE_LIST` topology with simple physics. | [`bouncing_lines.rs`](https://github.com/koubaa/goldy/blob/main/examples/bouncing_lines.rs) |
| **`waveform`** | `LINE_STRIP` waveform visualizer. | [`waveform.rs`](https://github.com/koubaa/goldy/blob/main/examples/waveform.rs) |

## Advanced Patterns

### Fragment Shader Effects

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`plasma`** | Demoscene plasma effect. | [`plasma.rs`](https://github.com/koubaa/goldy/blob/main/examples/plasma.rs) |
| **`tunnel`** | Flying-through-a-tunnel polar-coordinate effect. | [`tunnel.rs`](https://github.com/koubaa/goldy/blob/main/examples/tunnel.rs) |
| **`metaballs`** | Metaball field rendering. | [`metaballs.rs`](https://github.com/koubaa/goldy/blob/main/examples/metaballs.rs) |
| **`mandelbrot`** | Interactive Mandelbrot explorer. | [`mandelbrot.rs`](https://github.com/koubaa/goldy/blob/main/examples/mandelbrot.rs) |

### Interactive and Multi-Window

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`digital_clock`** | 7-segment clock display. | [`digital_clock.rs`](https://github.com/koubaa/goldy/blob/main/examples/digital_clock.rs) |
| **`starfield`** | 3D starfield with depth. | [`starfield.rs`](https://github.com/koubaa/goldy/blob/main/examples/starfield.rs) |
| **`particles`** | Rain/snow particle system. | [`particles.rs`](https://github.com/koubaa/goldy/blob/main/examples/particles.rs) |
| **`multi_window`** | Multiple windows sharing one device. | [`multi_window.rs`](https://github.com/koubaa/goldy/blob/main/examples/multi_window.rs) |

## Headless and Validation

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`headless_triangle`** | Offscreen render + CPU readback via `MemoryExchange`. | [`headless_triangle.rs`](https://github.com/koubaa/goldy/blob/main/examples/headless_triangle.rs) |
| **`scheme_screenshot`** | Scheme-based screenshot capture for tests. | [`scheme_screenshot.rs`](https://github.com/koubaa/goldy/blob/main/examples/scheme_screenshot.rs) |

Run all windowed examples interactively:

```bash
./run_all_examples.sh
```
