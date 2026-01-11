# Goldy Shader Library

Slang shader sources shared between native (Vulkan) and web (WebGPU) platforms.

## Goldy Module (Experimental)

> ⚠️ **EXPERIMENTAL**: This library's API is unstable and may change significantly
> as we learn what abstractions work best for shader development.

The `goldy_exp` module provides shared utilities that shaders can import:

```slang
import goldy_exp;

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = center_uv(input.uv);
    return float4(rainbow(uv.x), 1.0);
}
```

### Module Contents

| File | Contents |
|------|----------|
| `goldy_exp.slang` | Primary module entry point |
| `goldy_exp/math.slang` | Math utilities: `PI`, `TAU`, `hash()`, `hash2()`, `center_uv()`, `scale_uv()`, `to_polar()`, `smootherstep()` |
| `goldy_exp/color.slang` | Color utilities: `rainbow()`, `palette()`, `heat()`, `hsv_to_rgb()`, `luminance()`, `gamma_correct()` |
| `goldy_exp/vertex.slang` | Vertex formats and shaders (see below) |

### Vertex Formats

**Fullscreen Quad** (position + UV):
```slang
FullscreenVertex   // Input: float2 position, float2 uv
FullscreenVarying  // Output: float4 position, float2 uv
vs_fullscreen()    // Standard vertex shader
```

**Colored Vertices** (position + color):
```slang
ColoredVertex      // Input: float2 position, float4 color
ColoredVarying     // Output: float4 position, float4 color
vs_colored()       // Standard vertex shader
fs_colored()       // Pass-through fragment shader
```

**Fullscreen with Time** (position + UV + time):
```slang
FullscreenTimeVertex   // Input: float2 position, float2 uv, float time
FullscreenTimeVarying  // Output: float4 position, float2 uv, float time
vs_fullscreen_time()   // Standard vertex shader
```

## Usage

**Native (Rust):**

The `goldy_exp` library is automatically registered when you create a `Device`:

```rust
use goldy::{Instance, DeviceType, ShaderModule, shaders};

let instance = Instance::new()?;
let device = instance.create_device(DeviceType::DiscreteGpu)?;

// The goldy_exp library is pre-registered - just use import goldy_exp;
let shader = ShaderModule::from_slang(&device, shaders::PLASMA)?;
```

**Custom Libraries:**

```rust
use goldy::ShaderLibrary;

// Register your own library
device.register_library(ShaderLibrary::from_source("myutils", r#"
    module myutils;
    public float3 effect() { return float3(1, 0, 0); }
"#))?;

// Now your shaders can use: import myutils;
```

**Web (JavaScript + slang-wasm):**
```javascript
const slangSource = SHADERS['plasma'];
const wgsl = slangCompiler.compileToWgsl(slangSource);
const module = device.createShaderModule({ code: wgsl });
```

## Shader Files

| File | Description | Uses Module |
|------|-------------|-------------|
| `plasma.slang` | Classic demoscene plasma | ✓ `import goldy_exp` |
| `mandelbrot.slang` | Fractal explorer with zoom | ✓ `import goldy_exp` |
| `vertex_color_2d.slang` | Basic 2D position + color | — |
| `digital_clock.slang` | 7-segment display shader | — |
| `triangle.slang` | Procedural triangle from vertex ID | — |
| `gradient.slang` | Animated color gradient | — |
| `tunnel.slang` | Demoscene tunnel effect | — |
| `checkerboard.slang` | Animated checker pattern | — |
| `metaballs.slang` | Blending distance fields | — |
| `starfield.slang` | 3D starfield flying forward | — |
| `particles.slang` | Particle system rendering | — |
| `game_of_life.slang` | Conway's Game of Life compute | — |

## Compilation Targets

All shaders compile to:
- **SPIR-V** (via native slang.dll) → Vulkan
- **WGSL** (via slang-wasm) → WebGPU
- **HLSL** (future) → DirectX 12
- **MSL** (future) → Metal

## Module System

Goldy uses Slang's module system for code sharing:

```
shaders/
├── goldy_exp.slang           # module goldy_exp;
├── goldy_exp/
│   ├── math.slang            # implementing goldy_exp;
│   ├── color.slang           # implementing goldy_exp;
│   └── vertex.slang          # implementing goldy_exp;
├── plasma.slang              # import goldy_exp;
└── mandelbrot.slang          # import goldy_exp;
```

### How Library Registration Works

1. When a `Device` is created, the `goldy_exp` library is automatically registered
2. The library source files are written to a temp directory
3. The Slang compiler uses this directory to resolve `import` statements
4. When the `Device` is dropped, the temp files are cleaned up

### Creating a Shader That Uses the Library

```rust
// The goldy_exp library is pre-registered - just import it in your shader
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

### Library Management API

```rust
// Check what libraries are available
for name in device.list_libraries() {
    println!("Library: {}", name);
}

// Query specific library
assert!(device.has_library("goldy_exp"));

// Register your own
device.register_library(ShaderLibrary::from_source("mylib", "..."))?;

// Unregister (not recommended for goldy_exp)
device.unregister_library("mylib");
```
