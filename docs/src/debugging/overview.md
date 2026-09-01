# Debugging and Observability

Goldy provides validation layers, structured instrumentation, and
environment variable controls that together cover the full debugging
workflow — from catching API misuse to profiling frame timing.

## Validation

### `GOLDY_VALIDATION` Environment Variable

The primary control for runtime validation. Accepts a comma-, semicolon-,
or whitespace-separated list of categories:

| Value | Effect |
|-------|--------|
| `api` | Enable backend GPU API validation (see below) |
| `layout` | Enable Rust ↔ Slang struct layout checks and buffer stride checks |
| `host_access` | Page-protect CPU-visible GPU copies (CPU backend parcels; stray host pointers fault) |
| `fatal` | With `api`, Vulkan Khronos ERROR messages fail Goldy calls / `cargo test` (also `GOLDY_VALIDATION_FATAL=1`). Not included in `all`. |
| `all` | Enable `api`, `layout`, timeline, scheme, and `host_access` (not `fatal`) |
| `1`, `true`, `yes` | GPU API validation only (legacy shorthand; does **not** enable layout checks) |

Categories can be combined:

```bash
# API validation only
GOLDY_VALIDATION=api cargo run --example triangle

# Layout validation only
GOLDY_VALIDATION=layout cargo run --example triangle

# Both
GOLDY_VALIDATION=all cargo run --example triangle
GOLDY_VALIDATION=layout,api cargo run --example triangle
```

### API Validation

When `GOLDY_VALIDATION` includes `api` (or `1`/`true`/`yes`), Goldy
enables backend-specific validation:

| Backend | What Gets Enabled |
|---------|-------------------|
| Vulkan | `VK_LAYER_KHRONOS_validation` + `VK_EXT_debug_utils` messenger at instance creation. ERROR messages are logged; they fail Goldy calls only with `fatal` / `GOLDY_VALIDATION_FATAL`. |
| Metal | Sets `MTL_SHADER_VALIDATION=1` (if not already set) before the first device is created |
| DX12 | See [DX12 Debug Layer](#dx12-debug-layer) below |
| WebGPU | wgpu validation error scopes on shader/PSO create (always in debug builds; in release when GPU API validation is on) and on bind-group create (GPU API validation only) |

For Vulkan, validation is also enabled when `VK_INSTANCE_LAYERS` contains
`VK_LAYER_KHRONOS_validation` (the standard loader-driven workflow).

### Layout Validation

Layout validation catches mismatches between Rust struct layouts and their
Slang shader counterparts at shader compile time, and buffer element-stride
mismatches at dispatch time.

Enable via either:

```bash
GOLDY_VALIDATION=layout  cargo run
GOLDY_VALIDATE_LAYOUTS=1 cargo run   # legacy variable, equivalent
```

#### `#[derive(LayoutCheckable)]`

Annotate Rust structs that mirror Slang types to opt into automatic
validation:

```rust
#[derive(LayoutCheckable)]
#[repr(C)]
struct SceneUniforms {
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    time: f32,
}
```

The derive macro generates a `LAYOUT_CHECK` constant containing the
struct's name, total size, and per-field offsets. Pass it when creating a
shader module:

```rust
let shader = ShaderModule::from_slang_with_options(
    &device,
    source,
    &[],          // extra search paths
    &[],          // defines
    Default::default(),
    &[SceneUniforms::LAYOUT_CHECK],
)?;
```

When layout validation is enabled, Goldy compiles the Slang shader,
reflects each named struct, and compares:

- **Total struct size** — Rust `size_of` vs. Slang reflection
- **Field offsets** — each named field's byte offset

A mismatch produces an error naming the struct, the field, and the
expected vs. actual offset — immediately surfacing padding or alignment
bugs. When validation is disabled, the checks are skipped at zero cost.

#### Buffer Stride Validation

At dispatch time (when layout validation is enabled), Goldy also checks
that each bound buffer's `element_stride` matches the stride the shader
expects from Slang reflection. A mismatch produces an error like:

```
buffer element-stride mismatch in shader `my_shader`:
  slot 0: shader expects element stride 16 but buffer has 4
```

## DX12-Specific Debugging

### DX12 Debug Layer

| Variable | Values | Effect |
|----------|--------|--------|
| `GOLDY_DX12_DEBUG` | `1` | Force-enable the D3D12 debug layer (even in release builds) |
| `GOLDY_DX12_NO_DEBUG` | `1` | Disable the D3D12 debug layer (useful for parallel tests that crash the debug layer) |
| `GOLDY_DX12_GBV` | `1` | Enable GPU-Based Validation (very slow; requires the debug layer) |

GPU-Based Validation (GBV) instruments shaders on the GPU to detect issues
that the CPU-side debug layer cannot catch — such as out-of-bounds
descriptor accesses and uninitialized resource reads. Expect a significant
performance hit.

### WARP Software Rasterizer

[WARP](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/directx-warp)
is Microsoft's software implementation of D3D12. It runs on the CPU,
so it works on headless CI runners with no GPU.

```bash
GOLDY_DX12_FORCE_WARP=1 cargo nextest run
```

After the first WARP device is created, Goldy prints a confirmation:

```
[WARP] d3d10warp.dll loaded from: C:\WINDOWS\SYSTEM32\d3d10warp.dll
```

On Windows, DX12 is the default backend, so `GOLDY_DX12_FORCE_WARP=1` is
the only variable you need to run tests on a machine without a GPU.

## Structured Instrumentation

Goldy includes a structured instrumentation system built on the `tracing`
crate. It provides named observation points with hierarchical
dot-notation names and structured context data.

### Enabling Instrumentation

Instrumentation requires the `instrumentation` Cargo feature (enabled by
default). When disabled, all macros compile to no-ops at zero cost.

```bash
# Explicitly enable
cargo build --features instrumentation

# Disable (zero-cost removal)
cargo build --no-default-features --features vulkan
```

### `goldy_span!` — Timed Sections

Create a span to measure the duration of a code section:

```rust
use goldy::goldy_span;

fn compile_shader(&self) {
    let _span = goldy_span!("slang.compile", target = "metal").entered();
    // ... compilation code ...
    // Duration is recorded automatically when _span is dropped
}
```

### `goldy_event!` — Instant Markers

Emit a one-shot structured event:

```rust
use goldy::goldy_event;

goldy_event!("slang.library.load",
    path = %lib_path.display(),
    success = true
);
```

### Built-in Observation Points

Goldy instruments its own internals at these observation points:

| Category | Point Name | Emitted Data |
|----------|------------|--------------|
| **Slang** | `slang.library.load` | `path`, `success` |
| | `slang.compile.start` | `target`, `entry_points`, `bindless` |
| | `slang.compile.end` | `duration_ms`, `output_size`, `success` |
| | `slang.reflection.extract` | `parameter_blocks`, `fields` |
| **Shader** | `shader.module.create` | `backend`, `shader_type` |
| | `shader.pipeline.create` | `pipeline_type`, `bind_groups` |
| **Resource** | `resource.buffer.create` | `size`, `usage` |
| | `resource.texture.create` | `dimensions`, `format` |
| | `resource.bind_group.create` | `bindings_count` |
| **Render** | `render.frame.start` | `frame_id` |
| | `render.compute.dispatch` | `workgroups`, `pipeline` |
| | `render.draw` | `vertices`, `instances` |
| | `render.frame.end` | `frame_id`, `duration_ms` |

### JSON Logging

Install a JSON file logger to capture all instrumentation output as
structured JSON:

```rust
use goldy::instrumentation::install_json_logger;

install_json_logger("/tmp/goldy-debug.json")?;

// All subsequent goldy_span!/goldy_event! calls are written to the file
```

### Filtering with `RUST_LOG`

Use the standard `RUST_LOG` environment variable to control verbosity.
All Goldy instrumentation uses the `goldy` target:

```bash
RUST_LOG=goldy=debug cargo run --example triangle
RUST_LOG=goldy::render=trace cargo run --example triangle
```

## Environment Variables Summary

| Variable | Values | Effect |
|----------|--------|--------|
| `GOLDY_BACKEND` | `vulkan`/`vk`, `dx12`/`d3d12`/`directx`, `metal`/`mtl` | Override backend selection |
| `GOLDY_VALIDATION` | `api`, `layout`, `host_access`, `fatal`, `all`, `1`/`true`/`yes` | Enable validation categories (`fatal` is opt-in; not part of `all`) |
| `GOLDY_VALIDATION_FATAL` | `1`, `true`, `yes` | Fail on Vulkan Khronos ERROR messages |
| `GOLDY_VALIDATE_LAYOUTS` | `1`, `true`, `yes` | Enable layout validation (legacy; prefer `GOLDY_VALIDATION=layout`) |
| `GOLDY_DX12_FORCE_WARP` | `1` | Use WARP software rasterizer |
| `GOLDY_DX12_DEBUG` | `1` | Force-enable D3D12 debug layer in release |
| `GOLDY_DX12_NO_DEBUG` | `1` | Disable D3D12 debug layer |
| `GOLDY_DX12_GBV` | `1` | Enable GPU-Based Validation |
| `GOLDY_SHADER_TIMING` | `1` | Print Slang/PSO startup timings to stderr |
| `GOLDY_CPU_SHADERS` | `1` | Documented gate for debug CPU host-callable kernels (`goldy::cpu_shaders`) |
| `RUST_LOG` | e.g. `goldy=debug` | Filter instrumentation output |

## Common Debugging Patterns

### Catch API misuse early

```bash
GOLDY_VALIDATION=api cargo run --example my_app
```

Turn on API validation during development to catch invalid GPU API calls.
On Vulkan this enables the Khronos validation layer; on Metal it enables
shader validation.

### Diagnose struct layout bugs

```bash
GOLDY_VALIDATION=layout cargo test
```

If a `LayoutCheckable` struct diverges from its Slang counterpart (due to
padding, alignment, or a field being added on only one side), the error
message names the exact struct and field.

### Catch stray CPU pointers into GPU copies

```bash
GOLDY_VALIDATION=host_access cargo test --test cpu_backend
GOLDY_VALIDATION=all cargo run
```

When `host_access` is on, the CPU backend allocates each parcel in its own
page-aligned mapping (plus a guard page) and leaves it inaccessible except
during upload, kernel dispatch, and withdraw. A leftover host pointer then
faults instead of silently reading GPU-owned bytes. This is a debug allocator:
slower, not complete (native device-local VRAM is not mapped), and meant to
grow to staging buffers on other backends.

### Headless CI on Windows

```bash
GOLDY_DX12_FORCE_WARP=1 cargo nextest run
```

WARP gives you a fully functional D3D12 device on machines with no GPU.
Combine with `GOLDY_VALIDATION=api` for maximum coverage.

### Profile frame timing

```rust
use goldy::instrumentation::install_json_logger;

install_json_logger("/tmp/goldy-profile.json")?;

// Run your application, then inspect the JSON output for
// render.frame.start / render.frame.end durations
```

### Deep DX12 debugging

```bash
GOLDY_DX12_DEBUG=1 GOLDY_DX12_GBV=1 cargo run --example my_app
```

GPU-Based Validation catches GPU-side issues the CPU debug layer cannot
see, at a significant performance cost. Use it when you suspect
descriptor or resource access bugs.
