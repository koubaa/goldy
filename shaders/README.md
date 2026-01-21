# Goldy Shader Library

Slang shader sources shared between native (Vulkan) and web (WebGPU) platforms.

## Goldy Idioms

Goldy encourages specific patterns that leverage modern GPU capabilities:

### 1. Vertex-less Fullscreen Rendering

For fullscreen effects (plasma, mandelbrot, etc.), use `vs_fullscreen_triangle()` instead of vertex buffers:

```slang
import goldy_exp;

[shader("vertex")]
FullscreenVarying vs_main(uint vertex_id : SV_VertexID) {
    return vs_fullscreen_triangle(vertex_id);  // No vertex buffer needed!
}
```

```rust
// Rust side - no vertex buffer
pass.draw_fullscreen();  // Draws 3 vertices using SV_VertexID
```

This is more efficient than a fullscreen quad (3 verts vs 6) and eliminates buffer creation overhead.

### 2. Compute + Graphics Buffer Sharing

For particle systems and instancing, let compute shaders update data that graphics shaders read:

```slang
// Compute shader updates instances
[shader("compute")]
void cs_main(uint3 id : SV_DispatchThreadID) {
    Instance inst = INSTANCES[id.x];
    inst.rotation += delta_time;
    INSTANCES[id.x] = inst;
}

// Graphics shader reads instances, generates geometry
[shader("vertex")]
VSOutput vs_main(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID) {
    Instance inst = INSTANCES[instance_id];
    float2 pos = quad_position_rotated(vertex_id, inst.position, inst.size, inst.rotation);
    // ...
}
```

```rust
// Rust side
compute_pass.dispatch(workgroups, 1, 1);
render_pass.draw_quads(num_instances);  // 6 vertices per quad
```

### 3. Everything is a Buffer

Goldy's bindless architecture treats all GPU data as indexed buffers:

- **Uniforms**: `ConstantBuffer<T>` arrays with push constants for indices
- **Instance data**: `StructuredBuffer<T>` indexed by `SV_InstanceID`
- **Geometry**: Generated from `SV_VertexID` + `quad_position()` helpers

This minimizes CPU-GPU synchronization and enables GPU-driven rendering.

## Goldy Module (Experimental)

> ⚠️ **EXPERIMENTAL**: This library's API is unstable and may change significantly
> as we learn what abstractions work best for shader development.

The `goldy_exp` module provides shared utilities that shaders can import:

```slang
import goldy_exp;

[shader("vertex")]
FullscreenVarying vs_main(uint vertex_id : SV_VertexID) {
    return vs_fullscreen_triangle(vertex_id);
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
| `goldy_exp/types.slang` | Common data types: `Particle2D`, `Particle3D`, `FrameUniforms`, `Transform2D`, `Instance2D` |
| `goldy_exp/primitives.slang` | Procedural geometry: `quad_local()`, `quad_position()`, `quad_position_rotated()`, `billboard_position()`, `fullscreen_position()`, `fullscreen_uv()` |
| `goldy_exp/descriptor_handle.slang` | Cross-platform `DescriptorHandle<T>` support (routes by access pattern) |
| `goldy_exp/access.slang` | Unified access functions: `goldy_broadcast<T>()`, `goldy_scattered<T>()`, etc. |
| `goldy_exp/bindless_resources.slang` | Metal Tier 2 ParameterBlock for bindless resources |

### Vertex Formats

**Fullscreen Triangle** (vertex-less, recommended):
```slang
FullscreenVarying          // Output: float4 position, float2 uv
vs_fullscreen_triangle()   // Generates fullscreen triangle from SV_VertexID (no buffer needed!)
```

**Fullscreen Quad** (position + UV, legacy):
```slang
FullscreenVertex   // Input: float2 position, float2 uv
FullscreenVarying  // Output: float4 position, float2 uv
vs_fullscreen()    // Standard vertex shader (requires vertex buffer)
```

**Colored Vertices** (position + color):
```slang
ColoredVertex      // Input: float2 position, float4 color
ColoredVarying     // Output: float4 position, float4 color
vs_colored()       // Standard vertex shader
fs_colored()       // Pass-through fragment shader
```

**Fullscreen with Time** (position + UV + time, legacy):
```slang
FullscreenTimeVertex   // Input: float2 position, float2 uv, float time
FullscreenTimeVarying  // Output: float4 position, float2 uv, float time
vs_fullscreen_time()   // Standard vertex shader
```

### Procedural Geometry (from goldy_exp.primitives)

For instanced rendering, use these helpers instead of vertex buffers:

```slang
quad_local(vertex_id)                                  // Get quad vertex [-1,1]
quad_uv(vertex_id)                                     // Get quad UV [0,1]
quad_position(vertex_id, center, size)                 // Quad at position
quad_position_rotated(vertex_id, center, size, rot)    // Rotated quad
billboard_position(vertex_id, center, size, cam_right, cam_up)  // 3D billboard
fullscreen_position(vertex_id)                         // Fullscreen clip-space
fullscreen_uv(vertex_id)                               // Fullscreen UV
```

### Common Types (from goldy_exp.types)

These types have matching Rust structs in `goldy::common_types`:

```slang
Particle2D     // float2 position, float2 velocity
Particle3D     // float3 position, float3 velocity, float age, float lifetime
FrameUniforms  // float time, float delta_time, uint frame, uint _pad
Transform2D    // float2 position, float rotation, float2 scale, float _pad
Instance2D     // float2 position, float rotation, float scale, float4 color
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
| `plasma.slang` | Classic demoscene plasma (uses `goldy_broadcast<T>()`) | ✓ `import goldy_exp` |
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
| `__METAL_BINDLESS__` | Metal + Tier 2 argument buffers | Enables `goldy_broadcast<T>()` on Metal (see below) |
| `__SPIRV__` | Targeting Vulkan | Vulkan-specific code (push constants, descriptor arrays) |
| `__DX12__` | Targeting DX12 | DX12-specific code (root constants, ResourceDescriptorHeap) |

### Metal Tier 1 vs Tier 2

Metal has two argument buffer tiers with different capabilities:

| Tier | Hardware | Capabilities | Define |
|------|----------|--------------|--------|
| **Tier 1** | Older Intel (GitHub runners) | Traditional bindings only | `__METAL__` only |
| **Tier 2** | Apple Silicon, Intel 2017+, AMD 2015+ | Full argument buffers, bindless | `__METAL__` + `__METAL_BINDLESS__` |

Goldy automatically detects the tier at runtime and passes the appropriate defines.
This allows CI tests to run on GitHub runners (Tier 1) while production builds
get full bindless support on Tier 2 hardware.

### Cross-Platform Resource Binding

Goldy provides **unified access functions** that work across all platforms:

```slang
import goldy_exp;

struct TimeUniforms { float time; };

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    // Unified access - works on SPIRV, DX12, and Metal Tier 2!
    float t = goldy_broadcast<TimeUniforms>(0).time;
    return float4(rainbow(t), 1.0);
}
```

#### Access Pattern Functions

| Function | Access Pattern | Use For |
|----------|----------------|---------|
| `goldy_broadcast<T>(slot)` | All threads read same address | Uniforms, material params |
| `goldy_scattered<T>(slot)` | Any thread, any address | Particle buffers, compute storage |
| `goldy_interpolated<T>(slot)` | Hardware-filtered texture reads | Material textures |
| `goldy_direct_spatial<T>(slot)` | Unfiltered read/write texture | Compute output, framebuffer effects |
| `goldy_filter(slot)` | Sampler state for filtering | Texture sampling config |

**Platform support:**
- **SPIRV**: Routes to Goldy's custom binding layout (bindings 0-4)
- **DX12**: Uses `DescriptorHandle<T>` → `ResourceDescriptorHeap`
- **Metal Tier 2**: Uses `ParameterBlock` with typed resource arrays
- **Metal Tier 1**: Not supported (use fallback pattern below)

#### Metal Tier 1 Fallback

For GitHub CI compatibility (Metal Tier 1), use the fallback pattern:

```slang
import goldy_exp;

struct TimeUniforms { float time; };

// Metal Tier 1 fallback for CI - traditional binding
#if defined(__METAL__) && !defined(__METAL_BINDLESS__)
ConstantBuffer<TimeUniforms> uniforms : register(b0);
#define TIME uniforms.time
#else
// SPIRV, DX12, Metal Tier 2 - unified access
#define TIME goldy_broadcast<TimeUniforms>(0).time
#endif

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float t = TIME;  // Works everywhere!
    // ...
}
```

This 6-line pattern replaces the old 30+ lines of platform-specific code while
maintaining compatibility with CI runners.

**See `plasma.slang` and `test_access_functions.slang` for complete examples.**

#### Alternative: Direct `DescriptorHandle<T>`

For more control, you can also use Slang's `DescriptorHandle<T>` directly:

```slang
// Works on SPIRV and DX12 (not Metal Tier 1)
uniform ConstantBuffer<TimeUniforms>.Handle uniforms;
float t = (*uniforms).time;  // Dereference to access
```

The `goldy_exp` module overrides `getDescriptorFromHandle` for SPIRV to route
to Goldy's binding layout. DX12 uses Slang's default `ResourceDescriptorHeap`.

<details>
<summary>Technical Details: Vulkan Descriptor Override</summary>

The `goldy_exp` module includes `goldy_exp/descriptor_handle.slang` which provides a custom
`getDescriptorFromHandle<T>` override for SPIRV. Slang's default bindings don't match Goldy's
Vulkan descriptor layout.

**Goldy's bindings are organized by ACCESS PATTERN:**

| Binding | Access Pattern | What Hardware Does | Slang Types |
|---------|----------------|-------------------|-------------|
| 0 | **Scattered** | Any thread, any address, read/write | `StructuredBuffer<T>`, `RWStructuredBuffer<T>` |
| 1 | **Broadcast** | All threads same address (cache optimized) | `ConstantBuffer<T>` |
| 2 | **Interpolated** | Hardware filtering between neighbors | `Texture2D<T>` with sampler |
| 3 | **Direct Spatial** | 2D/3D indexing, no filtering, read/write | `RWTexture2D<T>` |
| 4 | **Filter Config** | Settings for interpolated access (not data) | `SamplerState` |

The custom `getDescriptorFromHandle<T>` uses Slang's `__DynamicResource<T>` type to declare
arrays at Goldy's specific bindings, then routes based on `T.kind` (Sampler, Texture,
ConstantBuffer, StorageBuffer, etc.) and `T.descriptorAccess` (Read vs ReadWrite).

This is transparent to shader code — just `import goldy_exp` and use `DescriptorHandle<T>`.

</details>

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

3. **DX12: Use named indices, not arrays** in cbuffers for buffer indices:
   ```slang
   // WORKS - direct named indices
   cbuffer BufferIndices : register(b0, space0) {
       uint instanceBufferIndex;
       uint paramsBufferIndex;
   };
   #define DATA (*DescriptorHandle<StructuredBuffer<T>>(uint2(instanceBufferIndex, 0)))
   
   // MAY NOT WORK - array access has issues on some DX12 configurations
   cbuffer BufferIndices : register(b0, space0) {
       uint indices[16];
   };
   #define DATA (*DescriptorHandle<StructuredBuffer<T>>(uint2(indices[0], 0)))
   ```

4. **Macros don't export from modules**. Use functions instead:
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
