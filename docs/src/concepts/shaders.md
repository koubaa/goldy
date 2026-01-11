# Shaders

RAG uses [Slang](https://shader-slang.org/) as its sole shading language. Slang is compiled to:

- **SPIR-V** for Vulkan
- **DXIL/HLSL** for DirectX 12
- **MSL** for Metal (future)
- **WGSL** for documentation demos (via slang-wasm)

## Why Slang?

Slang offers:

1. **Portability**: Single shader source for all backends
2. **Familiar Syntax**: HLSL-like, industry-standard
3. **Modern Features**: Modules, generics, automatic differentiation
4. **Khronos Governance**: Long-term stability

## Creating Shaders

```rust
use rag::ShaderModule;

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

## Built-in Shaders

RAG includes common shaders:

```rust
use rag::shader::builtins;

// 2D colored vertices
let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
```

### VERTEX_COLOR_2D

```hlsl
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
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
Texture2D myTexture;
SamplerState mySampler;

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return myTexture.Sample(mySampler, input.uv);
}
```

## Common Patterns

### Fullscreen Quad

```hlsl
struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vs_main(uint vertexId : SV_VertexID) {
    VertexOutput output;
    // Generate fullscreen triangle
    output.uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.position = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    // Post-processing effect using input.uv
    return float4(input.uv, 0.0, 1.0);
}
```

### Animated Effects

```hlsl
cbuffer TimeData {
    float time;
};

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float r = sin(time * 2.0) * 0.5 + 0.5;
    float g = cos(time * 3.0) * 0.5 + 0.5;
    return float4(r, g, 0.5, 1.0);
}
```

## Resources

- [Slang Documentation](https://shader-slang.com/slang/user-guide/)
- [Slang GitHub](https://github.com/shader-slang/slang)
- [HLSL Reference](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-reference)
