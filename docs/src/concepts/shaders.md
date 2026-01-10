# Shaders

RAG uses WGSL (WebGPU Shading Language) for shaders, compiled to SPIR-V for Vulkan.

## Creating Shaders

```rust
use rag::ShaderModule;

const MY_SHADER: &str = r#"
@vertex
fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);  // Red
}
"#;

let shader = ShaderModule::from_wgsl(&device, MY_SHADER)?;
```

## Built-in Shaders

RAG includes common shaders:

```rust
use rag::shader::builtins;

// 2D colored vertices
let shader = ShaderModule::from_wgsl(&device, builtins::VERTEX_COLOR_2D)?;
```

### VERTEX_COLOR_2D

```wgsl
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>
) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
```

## WGSL Basics

### Types

```wgsl
// Scalars
let a: f32 = 1.0;
let b: i32 = -5;
let c: u32 = 10u;
let d: bool = true;

// Vectors
let v2: vec2<f32> = vec2<f32>(1.0, 2.0);
let v3: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
let v4: vec4<f32> = vec4<f32>(1.0, 2.0, 3.0, 4.0);

// Matrices
let m: mat4x4<f32> = mat4x4<f32>(...);
```

### Vertex Inputs

```wgsl
@vertex
fn vs_main(
    @location(0) position: vec2<f32>,  // First attribute
    @location(1) color: vec4<f32>,      // Second attribute
    @builtin(vertex_index) idx: u32,    // Built-in vertex index
) -> VertexOutput {
    // ...
}
```

### Fragment Outputs

```wgsl
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);  // RGBA output
}
```

### Structs

```wgsl
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}
```

## Common Patterns

### Pass-through Vertex Shader

```wgsl
@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos, 0.0, 1.0);
}
```

### Full-screen Quad

```wgsl
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    // Generate full-screen triangle
    let x = f32(i32(idx) - 1);
    let y = f32(i32(idx & 1u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}
```

### Time Animation

Pass time through vertex attributes:

```wgsl
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) time: f32,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) time: f32) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.time = time;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = in.time;
    return vec4<f32>(sin(t), cos(t), 0.5, 1.0);
}
```

## Math Functions

WGSL includes standard math:

```wgsl
sin(x), cos(x), tan(x)
asin(x), acos(x), atan(x), atan2(y, x)
pow(x, y), exp(x), log(x), sqrt(x)
abs(x), sign(x), floor(x), ceil(x), fract(x)
min(a, b), max(a, b), clamp(x, low, high)
mix(a, b, t)  // Linear interpolation
length(v), distance(a, b), dot(a, b), cross(a, b)
normalize(v), reflect(v, n)
```

## Shader Compilation

RAG uses [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) to compile WGSL to SPIR-V:

```
WGSL source → naga → SPIR-V → Vulkan
```

Compilation happens at `ShaderModule::from_wgsl()`. Errors are returned if the shader is invalid:

```rust
let result = ShaderModule::from_wgsl(&device, bad_shader);
match result {
    Ok(shader) => { /* use shader */ }
    Err(e) => eprintln!("Shader error: {}", e),
}
```

## Shader Error Messages

naga provides helpful error messages:

```
error: unknown function 'sine'
  ┌─ wgsl:10:13
  │
10 │     let x = sine(t);
  │             ^^^^ unknown function
  │
  = note: did you mean 'sin'?
```

## Further Reading

- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [naga Documentation](https://docs.rs/naga)

