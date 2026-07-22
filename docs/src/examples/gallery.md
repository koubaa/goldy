# Examples Gallery

Goldy ships with 22 examples that demonstrate its core concepts. Every example uses [Slang](https://shader-slang.org/) shaders and runs on all supported backends (Vulkan 1.4+, DX12, Metal Tier 2+).

## Running Examples

```bash
cd goldy
cargo run --example <name> --release
```

All windowed examples support **Escape** to exit and automatic window-resize handling.

---

## Bindless Basics

These examples cover fundamental Goldy patterns: vertex buffers, the Surface API, uniforms, and fragment shaders.

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`triangle`** | The minimal Goldy program. Creates a vertex buffer with colored vertices, builds a render pipeline, and presents to a window via the zero-copy Surface API. | [`triangle.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/triangle.rs) |
| **`gradient`** | Animated full-screen gradient driven by a time uniform. Uses vertex-less rendering (`SV_VertexID`) and demonstrates `GOLDY_VALIDATE_LAYOUTS` for Rust ↔ Slang struct layout validation. | [`gradient.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/gradient.rs) |
| **`window`** | Triangle with continuous animation, showing the Surface API render loop and frame pacing. | [`window.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/window.rs) |
| **`checkerboard`** | Procedural animated checkerboard via UV distortion in a fragment shader. Also supports `GOLDY_VALIDATE_LAYOUTS`. | [`checkerboard.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/checkerboard.rs) |

## Compute Workflows

Examples that use `ComputePipeline` and `Scheme` for GPU-side data processing, including the compute-to-surface pattern.

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`compute_particles`** | Full compute + graphics loop. A compute shader updates 1024 particle positions and velocities each frame; a graphics shader renders them as instanced colored quads. Uses a retained scheme for dependency scheduling. | [`compute_particles.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/compute_particles.rs) |
| **`game_of_life`** | Conway's Game of Life on the GPU. A compute shader applies cellular-automaton rules on a 128×128 grid using **ping-pong sub-views** in one retained mosaic `Parcel`. A separate graphics pass renders the result. | [`game_of_life.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/game_of_life.rs) |
| **`compute_to_surface`** | Pure compute rendering — no `RenderPipeline`, no raster pass, no vertex buffers. A compute shader writes directly to a swapchain drawable via `SurfaceExchange::bind_destination` + `with_present`. Demonstrates the compute-to-surface workflow. | [`compute_to_surface.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/compute_to_surface.rs) |

## Graphics Pipelines

Classic rendering techniques: depth testing, textures, instancing, and 3D projection.

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`solid_cube`** | Solid 3D cube with per-face colors. Demonstrates 3D rendering with a depth buffer and model/view/projection matrices. | [`solid_cube.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/solid_cube.rs) |
| **`spinning_cube`** | 3D wireframe cube using line primitives. Shows 3D projection and rotation matrices without depth testing. | [`spinning_cube.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/spinning_cube.rs) |
| **`depth_quads`** | Two full-screen quads with oscillating depth values. Drawn in a fixed order, the depth buffer (`CompareFunction::Less`) ensures the nearer quad always wins — proving draw order independence. | [`depth_quads.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/depth_quads.rs) |
| **`textured_quad`** | Procedural checkerboard texture displayed on a quad. Demonstrates `Texture`, `Sampler`, cross-backend bindless resource access, and linear filtering with repeat addressing. | [`textured_quad.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/textured_quad.rs) |
| **`instancing`** | 400 rotating quads driven entirely by the GPU. A compute shader updates per-instance transforms and HSV-derived colors each frame; the graphics shader reads them from a storage buffer — no vertex buffer needed. | [`instancing.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/instancing.rs) |
| **`bouncing_lines`** | Lines bouncing off window edges. Uses the `LINE_LIST` primitive topology and simple physics. | [`bouncing_lines.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/bouncing_lines.rs) |
| **`waveform`** | Audio-style waveform visualizer using `LINE_STRIP` topology and multiple draw calls per frame. | [`waveform.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/waveform.rs) |

## Advanced Patterns

More complex examples combining multiple Goldy features or demonstrating interactive input, visual effects, and multi-window management.

### Fragment Shader Effects

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`plasma`** | Classic demoscene plasma effect using complex trigonometric math in a fragment shader with time-based animation. | [`plasma.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/plasma.rs) |
| **`tunnel`** | Flying-through-a-tunnel effect using polar coordinates and procedural checkerboard texturing in screen space. | [`tunnel.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/tunnel.rs) |
| **`metaballs`** | Organic blob simulation using distance-field evaluation and thresholding in a fragment shader. | [`metaballs.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/metaballs.rs) |
| **`starfield`** | 3D starfield fly-through simulated entirely in a fragment shader with depth-based brightness. | [`starfield.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/starfield.rs) |

### Interactive Input

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`mandelbrot`** | Real-time fractal explorer. Arrow keys pan, +/- zoom, R resets. Demonstrates interactive uniform updates driving a fragment shader. | [`mandelbrot.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/mandelbrot.rs) |
| **`particles`** | Rain and snow particle simulation. Press Space to toggle mode. Shows CPU-driven particle state with per-frame vertex buffer updates. | [`particles.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/particles.rs) |
| **`digital_clock`** | 7-segment LED display rendered from vertex data. Space pauses, click changes color. Demonstrates dynamic vertex generation for complex shapes. | [`digital_clock.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/digital_clock.rs) |

### Multi-Window

| Example | What it demonstrates | Source |
|---------|---------------------|--------|
| **`multi_window`** | Three simultaneous windows, each running an independent effect (plasma, tunnel, starfield) with its own Surface, pipeline, and input handling. Demonstrates managing multiple GPU surfaces from a single device. | [`multi_window.rs`](https://github.com/koubaa/goldy/blob/main/goldy/examples/multi_window.rs) |

---

## Common Patterns

### Present-on-Scheme Render Loop (Rust)

```rust
// Record once at init (and on resize):
let mut pass = scheme.render_pass("main", &scene_rt, TargetLoad::Clear(background_color));
pass.with_parcel(&vertices, NodeAccess::Read);
pass.set_pipeline(&pipeline);
pass.set_vertex_buffer(0, &vertices);
pass.draw(0..vertex_count, 0..1);
pass.finish();
let present = surface.bind_render_target(&mut scheme, &scene_rt)?;

// Each frame:
let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

### Compute + Graphics with Scheme

```rust
let mut scheme = Scheme::new(&ctx);
scheme
    .node("update", &compute_pipeline)
    .with_parcel(&buffer, NodeAccess::ReadWrite)
    .dispatch(workgroups, 1, 1);
scheme.submit()?;
```

### Slang Shader Template

```slang
import goldy_exp;

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv;
};

[shader("vertex")]
VertexOutput vs_main(float2 pos : POSITION, float2 uv : TEXCOORD) {
    VertexOutput output;
    output.position = float4(pos, 0.0, 1.0);
    output.uv = uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return float4(input.uv, 0.5, 1.0);
}
```
