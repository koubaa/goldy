# Slang Quick Reference

Goldy uses [Slang](https://shader-slang.org/) as its sole shading language. This page covers what you need to write Goldy shaders — not a full Slang language reference.

## Basics

Slang uses HLSL-style syntax. If you've written HLSL or GLSL, most of it will look familiar.

### Scalar Types

```hlsl
float f = 1.0;
int   i = -5;
uint  u = 10;
bool  b = true;
```

### Vector and Matrix Types

```hlsl
float2 v2 = float2(1.0, 2.0);
float3 v3 = float3(1.0, 2.0, 3.0);
float4 v4 = float4(1.0, 2.0, 3.0, 4.0);

// Swizzling
float2 xy  = v4.xy;
float3 rgb = v4.rgb;

// Matrices
float4x4 mvp;
float4 transformed = mul(mvp, float4(pos, 1.0));
```

### Structs

```hlsl
struct Particle {
    float2 position;
    float2 velocity;
    float  age;
};
```

### Functions

```hlsl
float square(float x) { return x * x; }

// Public functions are exported from modules
public float3 my_effect(float2 uv) { return float3(uv, 0.5); }
```

### Modules

Slang has a real module system (not `#include`). Modules are separate compilation units:

```hlsl
// In mylib.slang
module mylib;
public float3 effect(float2 uv) { return float3(uv, 1.0); }

// In shader.slang
import mylib;
float3 c = effect(uv);
```

## `goldy_exp` Resource Types

The `goldy_exp` module defines type aliases that map to native Slang buffer and texture types. When used as parameters in `[goldy_*]` entry points, the Goldy compiler automatically resolves slot indices to live resource handles.

### Buffer Types

| Type alias | Underlying type | Access pattern | Usage |
|------------|----------------|----------------|-------|
| `Scattered<T>` | `StorageBuffer<T>` (`RWStructuredBuffer<T>`) | Read/write, any thread, any address | `data[i]`, `data[i].field = v` |
| `BufRO<T>` | `ReadOnlyBuffer<T>` (`StructuredBuffer<T>`) | Read-only, hardware read-cache hint | `data[i]` |
| `ByteAddress` | `ByteAddressView` (`RWByteAddressBuffer`) | Raw byte-level access | `.Load(addr)`, `.Store(addr, v)`, `.InterlockedMin(...)` |

### Texture Types

| Type alias | Underlying type | Access pattern | Usage |
|------------|----------------|----------------|-------|
| `Interpolated<T>` | `Texture2D<T>` | Hardware-filtered sampling | `tex.Sample(samp, uv)`, `tex.Load(loc)` |
| `DirectSpatial<T>` | `RWTexture2D<T>` | Direct 2D read/write, no filtering | `img[int2(x,y)]`, `img.GetDimensions(w,h)` |

### Sampler Type

| Type alias | Underlying type | Usage |
|------------|----------------|-------|
| `Filter` | `SamplerState` | Pass to `tex.Sample(filter, uv)` |

### Broadcast (Constant Buffer)

To pass uniform data (same value for all threads), declare a struct type directly as a parameter — no wrapper needed. The codegen recognizes any non-resource, non-system-value struct as a constant-buffer broadcast:

```hlsl
struct TimeUniforms { float time; float delta_time; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(TimeUniforms cfg, Scattered<Particle> particles, ThreadId id) {
    particles[id.x].position += particles[id.x].velocity * cfg.delta_time;
}
```

## System-Value Types

Declare these as parameters in `[goldy_*]` entry points to receive GPU-provided values. The codegen maps each type to its `SV_*` semantic automatically.

### Compute

| Type | Maps to | Components |
|------|---------|------------|
| `ThreadId` | `SV_DispatchThreadID` | `.x`, `.y`, `.z`, `.xy`, `.xyz` |
| `GroupThreadId` | `SV_GroupThreadID` | `.x`, `.y`, `.z`, `.xy`, `.xyz` |
| `GroupId` | `SV_GroupID` | `.x`, `.y`, `.z`, `.xy`, `.xyz` |

### Graphics

| Type | Maps to | Components |
|------|---------|------------|
| `VertexId` | `SV_VertexID` | `.value` |
| `InstanceId` | `SV_InstanceID` | `.value` |
| `IsFrontFace` | `SV_IsFrontFace` | `.value` |

## Entry Point Attributes

### `[goldy_compute]`

Marks a compute shader entry point. The Goldy compiler generates the real `[shader("compute")]` wrapper that resolves resource slots and system values.

```hlsl
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, uint offset, ThreadId id) {
    data[id.x + offset] += 1;
}
```

### `[goldy_vertex]`

Marks a vertex shader entry point.

```hlsl
import goldy_exp;

struct VSOutput {
    float4 position : SV_Position;
    float4 color    : COLOR;
};

[goldy_vertex]
VSOutput vs_main(BufRO<Vertex> verts, VertexId vid) {
    Vertex v = verts[vid.value];
    VSOutput o;
    o.position = float4(v.pos, 0.0, 1.0);
    o.color    = v.color;
    return o;
}
```

### `[goldy_fragment]`

Marks a fragment shader entry point.

```hlsl
import goldy_exp;

[goldy_fragment]
float4 fs_main(Interpolated<float4> tex, Filter samp, float2 uv : TEXCOORD0) : SV_Target {
    return tex.Sample(samp, uv);
}
```

## Common Patterns

### Accessing Buffers by Index

All `Scattered<T>` and `BufRO<T>` parameters support standard array indexing. Field-level writes work directly on `Scattered<T>`:

```hlsl
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<Particle> particles, ThreadId id) {
    Particle p = particles[id.x];
    p.position += p.velocity;
    particles[id.x] = p;

    // Or field-level write:
    particles[id.x].age += 1.0;
}
```

### Sampling Textures

```hlsl
[goldy_fragment]
float4 fs_main(Interpolated<float4> albedo, Filter samp, float2 uv : TEXCOORD0) : SV_Target {
    return albedo.Sample(samp, uv);
}
```

### Writing to Storage Images

```hlsl
[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(DirectSpatial<float4> output, ThreadId id) {
    output[int2(id.x, id.y)] = float4(float(id.x) / 512.0, float(id.y) / 512.0, 0.5, 1.0);
}
```

### Fullscreen Triangle (Vertex-less)

Use `vs_fullscreen_triangle()` from `goldy_exp` to render fullscreen effects without a vertex buffer:

```hlsl
import goldy_exp;

[shader("vertex")]
FullscreenVarying vs_main(uint vertex_id : SV_VertexID) {
    return vs_fullscreen_triangle(vertex_id);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    return float4(input.uv, 0.5, 1.0);
}
```

### Compute + Render Buffer Sharing

Compute shaders and graphics shaders share the same bindless buffers. The task graph handles the dependency:

```hlsl
// Compute: update particles
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_update(TimeUniforms cfg, Scattered<Particle> particles, ThreadId id) {
    particles[id.x].position += particles[id.x].velocity * cfg.delta_time;
}

// Vertex: read particles for rendering
[goldy_vertex]
VSOutput vs_draw(BufRO<Particle> particles, InstanceId iid, VertexId vid) {
    Particle p = particles[iid.value];
    // Generate quad geometry from particle position...
}
```

## Rust-Side Resource Binding

Resources are bound in declaration order (left to right in the shader signature):

```rust
pass.with_resource_slots(&[
    cfg_buf.resource_index(ResourceAccess::Read).unwrap(),
    particle_buf.resource_index(ResourceAccess::Write).unwrap(),
]);
```

Plain scalar parameters (`uint offset`) are also push-constant bindings — no wrapper struct needed.

## `goldy_exp` Utility Modules

| Module | Contents |
|--------|----------|
| `goldy_exp/math.slang` | `PI`, `TAU`, `hash()`, `hash2()`, `center_uv()`, `scale_uv()`, `to_polar()`, `smootherstep()` |
| `goldy_exp/color.slang` | `rainbow()`, `palette()`, `heat()`, `hsv_to_rgb()`, `luminance()`, `gamma_correct()` |
| `goldy_exp/primitives.slang` | `quad_position()`, `quad_position_rotated()`, `billboard_position()`, `fullscreen_position()`, `fullscreen_uv()` |
| `goldy_exp/types.slang` | `Particle2D`, `Particle3D`, `FrameUniforms`, `Transform2D`, `Instance2D` |
| `goldy_exp/vertex.slang` | `FullscreenVarying`, `ColoredVertex`, `ColoredVarying`, `vs_fullscreen_triangle()` |
| `goldy_exp/access.slang` | Resource type aliases and system-value types (documented above) |

## Further Reading

- [Slang User Guide](https://shader-slang.com/slang/user-guide/)
- [Slang GitHub](https://github.com/shader-slang/slang)
- [HLSL Reference](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-reference)
