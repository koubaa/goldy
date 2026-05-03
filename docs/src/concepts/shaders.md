# Shaders

Goldy uses [Slang](https://shader-slang.org/) as its sole shading language. Slang is compiled to:

- **SPIR-V** for Vulkan
- **DXIL/HLSL** for DirectX 12
- **MSL** for Metal

## Why Slang?

Slang offers:

1. **Portability**: Single shader source for all backends
2. **Familiar Syntax**: HLSL-like, industry-standard
3. **Modern Features**: Modules, generics, automatic differentiation
4. **Khronos Governance**: Long-term stability

## Two Ways to Work with Shaders

Goldy provides two complementary approaches to shaders:

### 1. Built-in Shaders

Complete, self-contained shaders embedded in the Rust library. No imports, no file system access needed. Perfect for simple use cases.

```rust
use goldy::shader::builtins;

// Ready-to-use 2D colored vertex shader
let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
```

Available built-ins:
- `VERTEX_COLOR_2D` - 2D vertices with per-vertex color
- `SOLID_COLOR` - Solid color with uniform

### 2. Shader Libraries

Reusable Slang modules that your shaders can import. Libraries are registered with a `Device` and automatically available to all shaders compiled for that device.

```rust
// Shaders can import the built-in goldy_exp library
let shader = ShaderModule::from_slang(&device, r#"
    import goldy_exp;

    [shader("vertex")]
    FullscreenVarying vs_main(FullscreenVertex input) {
        return vs_fullscreen(input);
    }

    [shader("fragment")]
    float4 fs_main(FullscreenVarying input) : SV_Target {
        return float4(rainbow(input.uv.x), 1.0);
    }
"#)?;
```

## Rust vs Slang struct layout (optional)

If your Rust `#[repr(C)]` types must match Slang `struct` layouts (uniforms, structured buffers), you can validate them on the **same** compile that produces GPU bytecode—no extra Slang invocation.

1. Name the Rust struct like the Slang type (reflection uses `FindTypeByName`).
2. Add **`#[derive(LayoutCheckable)]`** (from the `goldy` crate).
3. Pass **`&[YourType::LAYOUT_CHECK]`** to **`ShaderModule::from_slang_with_options`** as the last argument.

Validation runs only when **`GOLDY_VALIDATE_LAYOUTS`** is set to `1`, `true`, or `yes`; otherwise the checks are skipped.

The **`gradient`** and **`checkerboard`** examples use this pattern with `TimeUniforms`. For tables of other environment variables, logging, and shader dumps, see **[DEBUGGING.md](https://github.com/koubaa/goldy/blob/main/DEBUGGING.md)** in the repository.

## The `goldy_exp` Library (Experimental)

Every `Device` comes with the `goldy_exp` shader library pre-registered.

> ⚠️ **Experimental**: This library's API is unstable and may change significantly
> as we learn what abstractions work best for shader development.

### Bindless Resource Access — Virtual Entry Points

The primary way to write bindless shaders with `goldy_exp` is the **virtual main** system.
Tag your entry point with `[goldy_compute]`, `[goldy_vertex]`, or `[goldy_fragment]` and
declare resources as typed parameters. Goldy generates the GPU-level `[shader(...)]`
entry point — with `uniform uint` push constants and `SV_*` system-value semantics — automatically.

```hlsl
import goldy_exp;

struct Particle { float2 pos; float2 vel; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(SimParams params, Scattered<Particle> particles, ThreadId id) {
    Particle particle = particles[id.x];
    particle.pos += particle.vel * params.dt;
    particles[id.x] = particle;
}
```

**Resource types** (each resolved from a single `uint` slot push constant by the codegen):

| Type | Is / Backing resource | Access |
|------|-----------------------|--------|
| `Scattered<T>` | = `StorageBuffer<T>` (read/write) | `buf[i]`, `buf[i].field = v` |
| `BufRO<T>` | = `ReadOnlyBuffer<T>` (read-only) | `buf[i]` |
| `Interpolated<T>` | = `Texture2D<T>` | `tex.Sample(samp, uv)`, `tex.Load(loc)` |
| `DirectSpatial<T>` | = `RWTexture2D<T>` | `img[int2(x,y)]`, `img.GetDimensions(w,h)` |
| `ByteAddress` | = `ByteAddressView` (`RWByteAddressBuffer`) | `.Load/.Store/.Interlocked*` |
| `Filter` | = `SamplerState` | `tex.Sample(samp, uv)` |
| `MyUniforms` (any struct) | constant buffer broadcast | `cfg.field` directly |

**System-value types** (codegen maps to `SV_*` semantics automatically):

| Type | Maps to | Fields |
|------|---------|--------|
| `ThreadId` | `SV_DispatchThreadID` | `.x .y .z` |
| `GroupThreadId` | `SV_GroupThreadID` | `.x .y .z` |
| `GroupId` | `SV_GroupID` | `.x .y .z` |
| `VertexId` | `SV_VertexID` | `.value` |
| `InstanceId` | `SV_InstanceID` | `.value` |
| `IsFrontFace` | `SV_IsFrontFace` | `.value` |

Plain scalar types (`uint`, `float`, `int`, etc.) become `uniform` push constants too:

```hlsl
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, uint offset, ThreadId id) {
    data[id.x + offset] *= 2;
}
```

It provides:

### Math Utilities (`goldy_exp/math`)

```hlsl
// Constants
static const float PI = 3.14159265359;
static const float TAU = 6.28318530718;

// Hash functions
float hash(float p);
float hash2(float2 p);

// UV manipulation
float2 center_uv(float2 uv);    // Remap [0,1] to [-0.5, 0.5]
float2 scale_uv(float2 uv, float scale);

// Coordinate transforms
float2 to_polar(float2 cartesian);

// Interpolation
float smootherstep(float edge0, float edge1, float x);
```

### Color Utilities (`goldy_exp/color`)

```hlsl
// Palettes
float3 rainbow(float t);        // Smooth rainbow gradient
float3 heat(float t);           // Black → Red → Yellow → White
float3 palette(float t, float3 freq, float3 phase);

// Color space
float3 hsv_to_rgb(float3 hsv);
float luminance(float3 rgb);
float3 gamma_correct(float3 linear_rgb);
```

### Vertex Formats (`goldy_exp/vertex`)

```hlsl
// Fullscreen quad rendering
struct FullscreenVertex {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct FullscreenVarying {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

FullscreenVarying vs_fullscreen(FullscreenVertex input);

// Colored 2D vertices
struct ColoredVertex {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct ColoredVarying {
    float4 position : SV_Position;
    float4 color : COLOR;
};

ColoredVarying vs_colored(ColoredVertex input);
float4 fs_colored(ColoredVarying input);

// Fullscreen with time uniform
struct FullscreenTimeVertex {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
    float time : TEXCOORD1;
};

FullscreenTimeVarying vs_fullscreen_time(FullscreenTimeVertex input);
```

## Custom Libraries

You can create and register your own shader libraries:

```rust
use goldy::ShaderLibrary;

// Create a library from source
let effects = ShaderLibrary::from_source("effects", r#"
    module effects;
    
    public float3 glow(float intensity) {
        return float3(intensity, intensity * 0.8, intensity * 0.3);
    }
    
    public float3 neon(float t, float3 base_color) {
        float pulse = sin(t * 3.14159) * 0.5 + 0.5;
        return base_color * (1.0 + pulse * 2.0);
    }
"#);

// Register with device
device.register_library(effects)?;

// Now your shaders can import it
let shader = ShaderModule::from_slang(&device, r#"
    import goldy_exp;
    import effects;

    [shader("fragment")]
    float4 fs_main(FullscreenVarying input) : SV_Target {
        float t = input.uv.x;
        return float4(neon(t, rainbow(t)), 1.0);
    }
"#)?;
```

### Multi-Module Libraries

For larger libraries, use `from_embedded` with multiple modules:

```rust
let mylib = ShaderLibrary::from_embedded("mylib", &[
    ("mylib", r#"
        module mylib;
        __include "mylib/utils";
        __include "mylib/effects";
    "#),
    ("mylib/utils", r#"
        implementing mylib;
        public float remap(float v, float lo, float hi) {
            return lo + v * (hi - lo);
        }
    "#),
    ("mylib/effects", r#"
        implementing mylib;
        public float3 vignette(float2 uv, float strength) {
            float d = distance(uv, float2(0.5, 0.5));
            return float3(1.0 - d * strength);
        }
    "#),
]);

device.register_library(mylib)?;
```

### Loading from Filesystem

For development, load libraries from disk:

```rust
use std::path::Path;

let lib = ShaderLibrary::from_directory("mylib", Path::new("shaders/mylib"))?;
device.register_library(lib)?;
```

Expected directory structure:
```
shaders/mylib/
├── mylib.slang        # Primary module: module mylib;
└── mylib/
    ├── utils.slang    # implementing mylib;
    └── effects.slang  # implementing mylib;
```

## Creating Custom Shaders

### Basic Shader Structure

```rust
const MY_SHADER: &str = r#"
struct VertexOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vs_main(float2 pos : POSITION) {
    VertexOutput output;
    output.position = float4(pos, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fs_main() : SV_Target {
    return float4(1.0, 0.0, 0.0, 1.0);  // Red
}
"#;

let shader = ShaderModule::from_slang(&device, MY_SHADER)?;
```

### With Goldy Library

```rust
const EFFECT_SHADER: &str = r#"
import goldy_exp;

[[vk::binding(0, 0)]]
cbuffer Uniforms { float time; };

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = center_uv(input.uv);
    float d = length(uv);
    float t = d + time * 0.5;
    return float4(rainbow(t), 1.0);
}
"#;

let shader = ShaderModule::from_slang(&device, EFFECT_SHADER)?;
```

## Slang Basics

### Types

```hlsl
// Scalars
float a = 1.0;
int b = -5;
uint c = 10;
bool d = true;

// Vectors
float2 v2 = float2(1.0, 2.0);
float3 v3 = float3(1.0, 2.0, 3.0);
float4 v4 = float4(1.0, 2.0, 3.0, 4.0);

// Matrices
float4x4 m = float4x4(...);
```

### Shader Entry Points

Entry points are marked with `[shader("type")]`:

```hlsl
[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    // Vertex processing
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    // Fragment/pixel processing
}

[shader("compute")]
void cs_main() {
    // Compute processing
}
```

### Semantics

Slang uses HLSL-style semantics:

```hlsl
struct VertexInput {
    float2 position : POSITION;      // Vertex attribute 0
    float4 color : COLOR;             // Vertex attribute 1
    uint vertexId : SV_VertexID;      // Built-in vertex index
};

struct VertexOutput {
    float4 position : SV_Position;    // Clip-space position
    float4 color : COLOR;             // Interpolated to fragment
};

// Fragment output
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;               // Output to render target 0
}
```

### Uniform Buffers

```hlsl
[[vk::binding(0, 0)]]
cbuffer Uniforms {
    float4x4 modelViewProj;
    float time;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = mul(modelViewProj, float4(input.position, 1.0));
    return output;
}
```

### Textures and Samplers

```hlsl
[[vk::binding(0, 0)]]
Texture2D myTexture;

[[vk::binding(1, 0)]]
SamplerState mySampler;

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return myTexture.Sample(mySampler, input.uv);
}
```

## Common Patterns

### Fullscreen Effect

```hlsl
import goldy_exp;

[[vk::binding(0, 0)]]
cbuffer Uniforms { float time; };

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = input.uv;
    // Apply your effect using uv (0-1) and time
    return float4(rainbow(uv.x + time), 1.0);
}
```

### Animated Color

```hlsl
import goldy_exp;

cbuffer TimeData { float time; };

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float r = sin(time * 2.0) * 0.5 + 0.5;
    float g = cos(time * 3.0) * 0.5 + 0.5;
    return float4(r, g, 0.5, 1.0);
}
```

## Library Management API

```rust
// Check if a library is registered
if device.has_library("goldy_exp") {
    println!("goldy_exp library available");
}

// List all registered libraries
for name in device.list_libraries() {
    println!("  - {}", name);
}

// Unregister a library (not recommended for goldy_exp)
device.unregister_library("mylib");
```

## Resources

- [Slang Documentation](https://shader-slang.com/slang/user-guide/)
- [Slang GitHub](https://github.com/shader-slang/slang)
- [HLSL Reference](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-reference)
