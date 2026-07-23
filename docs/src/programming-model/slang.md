# Slang in One Source

Goldy uses [Slang](https://shader-slang.org/) as its single shader language across all backends. You write one `.slang` file and Goldy compiles it to the native format for whichever GPU API is in use — no manual HLSL/GLSL/MSL translation, no per-backend shader files.

## Compilation Targets

| Backend | Target Format | API Requirement | Status |
|---------|---------------|-----------------|--------|
| Vulkan | SPIR-V | Vulkan 1.4+ | Shipped |
| DirectX 12 | DXIL | Windows 10+ | Shipped |
| Metal | Metal IR | Metal Tier 2+ (Argument Buffers) | Shipped |
| CUDA | PTX | NVIDIA CUDA | In progress |
| WebGPU | WGSL | WebGPU (via wgpu) | In progress |

Slang compiles through its native `slang.dll` / `libslang.dylib` — the same compiler used by NVIDIA, Khronos, and major game engines. Goldy links it directly; there is no intermediate translation step.

## Why Slang

- **One source**: Vertex, fragment, and compute shaders all live in a single `.slang` file. No preprocessor gymnastics to target different backends.
- **HLSL-compatible syntax**: If you know HLSL, you already know Slang. Standard types (`float4`, `uint3`, `Texture2D`), standard intrinsics (`mul`, `lerp`, `smoothstep`), standard semantics (`SV_Position`, `SV_Target`).
- **Modern language features**: Modules (`import`), generics, interfaces, operator overloading, and automatic differentiation — features that HLSL and GLSL lack.
- **Khronos governance**: Long-term stability under open-source stewardship.

## Cross-Backend Matrix Layout Consistency

Slang normalizes matrix layout across all backends. HLSL defaults to column-major storage, GLSL to column-major, and Metal to column-major — but the conventions for how `mul(matrix, vector)` is interpreted differ. Slang's compilation ensures that a `float4x4` in your shader has identical memory layout and multiplication semantics whether it compiles to SPIR-V, DXIL, or Metal IR.

This means your Rust-side `#[repr(C)]` matrix types can use the same byte layout regardless of which backend the application runs on.

## Shader Module Creation

### Basic Compilation

`ShaderModule::from_slang()` compiles a Slang source string into GPU bytecode:

```rust
let shader = ShaderModule::from_slang(&device, r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(64, 1, 1)]
    void cs_main(Scattered<float> data, ThreadId id) {
        data[id.x] = data[id.x] * 2.0;
    }
"#)?;
```

The `goldy_exp` library is pre-registered on every device — `import goldy_exp` works without any setup.

### Additional Search Paths

`ShaderModule::from_slang_with_paths()` adds filesystem directories to the Slang module search path:

```rust
let shader = ShaderModule::from_slang_with_paths(
    &device,
    source,
    &["my_project/shaders"],
)?;
```

### Preprocessor Defines

`ShaderModule::from_slang_with_paths_and_defines()` passes preprocessor defines for shader variants:

```rust
let shader = ShaderModule::from_slang_with_paths_and_defines(
    &device,
    source,
    &[],
    &[("msaa", "1"), ("SAMPLE_COUNT", "4")],
)?;
```

### Full Options

`ShaderModule::from_slang_with_options()` provides complete control — search paths, defines, optimization level, and layout validation checks:

```rust
let shader = ShaderModule::from_slang_with_options(
    &device,
    source,
    &["shaders/"],
    &[("DEBUG", "1")],
    OptimizationLevel::Default,
    &[TimeUniforms::LAYOUT_CHECK],
)?;
```

## Built-in Shader Modules

Goldy ships a few complete shaders as Rust string constants in `goldy::shader::builtins`:

| Constant | Description |
|----------|-------------|
| `VERTEX_COLOR_2D` | 2D vertex+fragment shader with per-vertex color |
| `SOLID_COLOR` | Solid color fragment shader with a uniform |

These are self-contained (no `import` needed) and useful for bootstrapping:

```rust
use goldy::shader::builtins;

let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
```

## Shader Libraries

Shader libraries are reusable Slang modules registered with a `Device`. Once registered, any shader compiled on that device can `import` the library.

### The Built-in `goldy_exp` Library

Every device comes with `goldy_exp` pre-registered. It provides:

- Resource type aliases (`Scattered<T>`, `BufRO<T>`, `Interpolated<T>`, etc.)
- System-value wrappers (`ThreadId`, `VertexId`, `InstanceId`, etc.)
- Vertex formats (`FullscreenVarying`, `ColoredVarying`, etc.)
- Math utilities (`hash()`, `center_uv()`, `smootherstep()`, etc.)
- Color utilities (`rainbow()`, `palette()`, `hsv_to_rgb()`, etc.)
- Procedural geometry (`quad_position()`, `billboard_position()`, etc.)

### Registering Custom Libraries

```rust
use goldy::ShaderLibrary;

device.register_library(ShaderLibrary::from_source("myutils", r#"
    module myutils;
    public float3 my_effect(float t) { return float3(t, t * 0.5, 1.0 - t); }
"#))?;
```

Now any shader can `import myutils`:

```hlsl
import myutils;

[goldy_fragment]
float4 fs_main(FullscreenVarying input) : SV_Target {
    return float4(my_effect(input.uv.x), 1.0);
}
```

### Multi-Module Libraries

For larger libraries with internal sub-modules:

```rust
let lib = ShaderLibrary::from_embedded("effects", &[
    ("effects", r#"
        module effects;
        __include "effects/blur";
        __include "effects/bloom";
    "#),
    ("effects/blur", r#"
        implementing effects;
        public float4 gaussian_blur(Texture2D<float4> tex, SamplerState s, float2 uv) { ... }
    "#),
    ("effects/bloom", r#"
        implementing effects;
        public float4 bloom(Texture2D<float4> tex, SamplerState s, float2 uv, float threshold) { ... }
    "#),
]);

device.register_library(lib)?;
```

### Loading from the Filesystem

```rust
let lib = ShaderLibrary::from_directory("effects", Path::new("shaders/effects/"))?;
device.register_library(lib)?;
```

### Library Management

```rust
device.has_library("goldy_exp");       // true — always registered
device.list_libraries();               // ["goldy_exp", "myutils", ...]
device.unregister_library("myutils");  // remove a custom library
```

## Layout Validation

When Rust structs are passed to shaders as uniform data (e.g. via `Broadcast`), the memory layout must match exactly. Goldy can validate this at compile time using Slang reflection.

### Setup

1. Derive `LayoutCheckable` on your Rust struct:

```rust
#[derive(LayoutCheckable)]
#[repr(C)]
struct TimeUniforms {
    time: f32,
    delta_time: f32,
    frame: u32,
    _pad: u32,
}
```

2. Pass the layout check to shader compilation:

```rust
let shader = ShaderModule::from_slang_with_options(
    &device,
    source,
    &[],
    &[],
    OptimizationLevel::Default,
    &[TimeUniforms::LAYOUT_CHECK],
)?;
```

3. Enable validation via environment variable:

```bash
GOLDY_VALIDATE_LAYOUTS=1 cargo run
# or
GOLDY_VALIDATION=layout cargo run
# or enable everything:
GOLDY_VALIDATION=all cargo run
```

### What Gets Validated

- **Field offsets**: Each field's byte offset in the Rust struct is compared against the Slang reflection data.
- **Struct size**: Total size must match.
- **Buffer element stride**: At dispatch time, the buffer's recorded element stride is checked against what the shader expects.

Validation is zero-cost when disabled — the checks are skipped entirely, not compiled out. The environment variable is read at runtime so it can be toggled without recompiling.

### GOLDY_VALIDATION

The `GOLDY_VALIDATION` environment variable controls multiple validation categories:

| Value | Layout Checks | GPU API Validation |
|-------|:---:|:---:|
| `layout` | Yes | No |
| `api` | No | Yes |
| `layout,api` | Yes | Yes |
| `all` | Yes | Yes |
| `1` / `true` / `yes` | No | Yes |

`GOLDY_VALIDATE_LAYOUTS=1` is a standalone toggle that enables layout checks regardless of `GOLDY_VALIDATION`.
