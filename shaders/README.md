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
| `tunnel.slang` | Demoscene tunnel effect | ✓ `import goldy_exp` |
| `checkerboard.slang` | Animated checker pattern | — |
| `metaballs.slang` | Blending distance fields | — |
| `starfield.slang` | 3D starfield flying forward | — |
| `particles.slang` | Particle system rendering | — |
| `game_of_life.slang` | Conway's Game of Life compute | — |

## Compilation Targets

All shaders compile via native slang.dll:
- **SPIR-V** → Vulkan
- **DXIL** → DX12
- **MSL** → Metal

## Preprocessor Defines

When compiling shaders, Goldy passes backend-specific preprocessor defines:

| Define | When Set | Use For |
|--------|----------|---------|
| `__METAL__` | Targeting Metal | Metal-specific code (ParameterBlock for argument buffers) |
| `__SPIRV__` | Targeting Vulkan | Vulkan-specific code (push constants, descriptor arrays) |
| `__DX12__` | Targeting DX12 | DX12-specific code (root constants, ResourceDescriptorHeap) |

### Cross-Platform Shader Pattern

```slang
#if defined(__METAL__)
    // Metal: use ParameterBlock for argument buffers
    struct MyResources {
        ConstantBuffer<MyUniforms> uniforms;
    };
    ParameterBlock<MyResources> gResources;
    #define TIME gResources.uniforms.time
#elif defined(__SPIRV__)
    // Vulkan: use push constants + descriptor arrays
    [[vk::binding(1, 0)]] ConstantBuffer<MyUniforms> g_UniformBuffers[];
    #define TIME g_UniformBuffers[getBufferIndex(0)].time
#elif defined(__DX12__)
    // DX12: use root constants + ResourceDescriptorHeap
    cbuffer BufferIndices : register(b0) { uint uniformsIndex; };
    #define TIME (*DescriptorHandle<ConstantBuffer<MyUniforms>>(uint2(uniformsIndex, 0))).time
#endif
```

### Tips

1. **Preprocessor defines don't propagate to imported modules** — this is a deliberate design decision in Slang, not a limitation. The module system (`import`, `__include`) isolates preprocessor state to enable true separate compilation and clean module boundaries. Think of `import foo;` as closer to `using namespace foo;` in C++ rather than `#include`.

   **However, session-level macros DO propagate!** Macros passed via `-D` flag or the API's `PreprocessorMacroDesc` are visible to ALL modules, including imports. This is why Goldy's backend defines (`__METAL__`, `__SPIRV__`, `__DX12__`) work — they're passed at the session level.

   | Macro Source | Visible in Imports? |
   |--------------|---------------------|
   | `#define` in source | ❌ No — isolated to that file |
   | `-D` flag / session macros | ✅ Yes — visible everywhere |
   | Traditional `#include` | ✅ Yes — behaves like C/C++ |

   **Solutions for module-level configuration:**
   - Use session-level `-D` flags for feature toggles (Goldy does this automatically)
   - Use functions instead of macros (functions export, macros don't)
   - Don't use guards in modules that are only imported when needed

2. **Push constants require specific syntax** for Vulkan SPIR-V:
   ```slang
   // WRONG - cbuffer doesn't generate push constants
   [[vk::push_constant]]
   cbuffer MyData { ... };
   
   // CORRECT - struct + ConstantBuffer pattern
   struct MyDataBlock { ... };
   [[vk::push_constant]] ConstantBuffer<MyDataBlock> myData;
   ```

3. **Macros don't export from modules**. Use functions instead:
   ```slang
   // In module - this WON'T be visible to importers:
   #define GET_INDEX(slot) indices[slot]
   
   // Use a function instead - this WILL be visible:
   public uint getIndex(uint slot) { return indices[slot]; }
   ```

4. **Metal uses `ParameterBlock`, not push constants**. Unlike Vulkan/DX12 which use push/root constants to pass buffer indices, Metal uses `ParameterBlock` which Slang compiles to argument buffers:
   ```slang
   // Metal pattern
   struct MyResources {
       ConstantBuffer<MyUniforms> uniforms;
       Texture2D myTexture;
   };
   ParameterBlock<MyResources> gResources;
   
   // Access via: gResources.uniforms.time, gResources.myTexture
   ```
   
   The Goldy backend writes GPU addresses directly to the argument buffer and binds it at the slot specified by Slang reflection. When using `set_push_constants()` in Rust with a Metal ParameterBlock shader, the backend automatically handles this translation.

## Module System

Goldy uses Slang's module system for code sharing. Unlike `#include`, Slang modules provide true separate compilation — modules can be precompiled to `.slang-module` files and imported code behaves consistently regardless of where it's imported from.

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
