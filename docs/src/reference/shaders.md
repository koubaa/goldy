# WGSL Shader Reference

RAG uses WGSL (WebGPU Shading Language) for shaders.

## Shader Structure

```wgsl
// Structs
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// Vertex shader
@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
```

## Attributes

### Built-in Inputs

| Attribute | Type | Description |
|-----------|------|-------------|
| `@builtin(vertex_index)` | `u32` | Vertex index (0, 1, 2, ...) |
| `@builtin(instance_index)` | `u32` | Instance index |
| `@builtin(position)` | `vec4<f32>` | Fragment position (in fragment shader) |
| `@builtin(front_facing)` | `bool` | Is front face |

### Built-in Outputs

| Attribute | Type | Description |
|-----------|------|-------------|
| `@builtin(position)` | `vec4<f32>` | Clip-space position (vertex shader) |
| `@builtin(frag_depth)` | `f32` | Fragment depth |

### User Locations

```wgsl
// Vertex inputs (match VertexAttribute.location)
@location(0) position: vec2<f32>
@location(1) color: vec4<f32>

// Vertex outputs / Fragment inputs
@location(0) uv: vec2<f32>
@location(1) normal: vec3<f32>

// Fragment outputs
@location(0) color: vec4<f32>
```

## Types

### Scalars

```wgsl
let f: f32 = 1.0;
let i: i32 = -5;
let u: u32 = 10u;
let b: bool = true;
```

### Vectors

```wgsl
let v2: vec2<f32> = vec2<f32>(1.0, 2.0);
let v3: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
let v4: vec4<f32> = vec4<f32>(1.0, 2.0, 3.0, 4.0);

// Swizzling
let xy = v4.xy;     // vec2
let rgb = v4.rgb;   // vec3
let rr = v4.rr;     // vec2
```

### Matrices

```wgsl
let m2: mat2x2<f32>;
let m3: mat3x3<f32>;
let m4: mat4x4<f32>;

// Matrix * vector
let transformed = m4 * vec4<f32>(pos, 1.0);
```

### Arrays

```wgsl
let arr: array<f32, 4> = array<f32, 4>(1.0, 2.0, 3.0, 4.0);
let elem = arr[0];
```

## Control Flow

```wgsl
// If
if condition {
    // ...
} else if other {
    // ...
} else {
    // ...
}

// Loop
var i = 0;
loop {
    if i >= 10 { break; }
    // ...
    i = i + 1;
}

// For
for (var i = 0; i < 10; i = i + 1) {
    // ...
}

// While (via loop)
loop {
    if !condition { break; }
    // ...
}
```

## Math Functions

### Trigonometry

```wgsl
sin(x)   cos(x)   tan(x)
asin(x)  acos(x)  atan(x)
atan2(y, x)
sinh(x)  cosh(x)  tanh(x)
```

### Exponential

```wgsl
pow(x, y)  // x^y
exp(x)     // e^x
exp2(x)    // 2^x
log(x)     // ln(x)
log2(x)    // log2(x)
sqrt(x)
inverseSqrt(x)
```

### Common

```wgsl
abs(x)
sign(x)      // -1, 0, or 1
floor(x)
ceil(x)
round(x)
fract(x)     // x - floor(x)
trunc(x)
modf(x)      // returns (fract, whole)
```

### Clamping

```wgsl
min(a, b)
max(a, b)
clamp(x, low, high)
saturate(x)  // clamp(x, 0.0, 1.0)
```

### Interpolation

```wgsl
mix(a, b, t)       // Linear interpolation: a*(1-t) + b*t
step(edge, x)      // 0 if x < edge, 1 otherwise
smoothstep(e0, e1, x)  // Smooth Hermite interpolation
```

### Vector Operations

```wgsl
length(v)
distance(a, b)
dot(a, b)
cross(a, b)        // vec3 only
normalize(v)
reflect(v, n)
refract(v, n, eta)
faceForward(n, i, nref)
```

### Component-wise

```wgsl
// These work on vectors component-wise
abs(v)
sign(v)
floor(v)
// etc.
```

## Examples

### Solid Color

```wgsl
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);  // Red
}
```

### UV Gradient

```wgsl
@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(uv, 0.5, 1.0);
}
```

### Time Animation

```wgsl
@fragment
fn fs_main(@location(0) uv: vec2<f32>, @location(1) time: f32) -> @location(0) vec4<f32> {
    let r = sin(time) * 0.5 + 0.5;
    let g = cos(time) * 0.5 + 0.5;
    return vec4<f32>(r, g, uv.x, 1.0);
}
```

### Distance Field Circle

```wgsl
@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let center = vec2<f32>(0.5, 0.5);
    let dist = distance(uv, center);
    let circle = 1.0 - smoothstep(0.2, 0.21, dist);
    return vec4<f32>(circle, circle, circle, 1.0);
}
```

### Checkerboard

```wgsl
@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let scale = 8.0;
    let checker = floor(uv.x * scale) + floor(uv.y * scale);
    let color = (checker % 2.0);
    return vec4<f32>(color, color, color, 1.0);
}
```

## Resources

- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [Tour of WGSL](https://google.github.io/tour-of-wgsl/)
- [naga (WGSL compiler)](https://github.com/gfx-rs/wgpu/tree/trunk/naga)

