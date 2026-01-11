# Slang Shader Reference

Goldy uses [Slang](https://shader-slang.org/) as its sole shading language.

## Shader Structure

```hlsl
// Structs
struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

// Vertex shader
[shader("vertex")]
VertexOutput vs_main(float2 pos : POSITION) {
    VertexOutput output;
    output.position = float4(pos, 0.0, 1.0);
    return output;
}

// Fragment shader
[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
```

## Semantics

### System Value Semantics (Input)

| Semantic | Type | Description |
|----------|------|-------------|
| `SV_VertexID` | `uint` | Vertex index (0, 1, 2, ...) |
| `SV_InstanceID` | `uint` | Instance index |
| `SV_Position` | `float4` | Fragment position (in fragment shader) |
| `SV_IsFrontFace` | `bool` | Is front face |

### System Value Semantics (Output)

| Semantic | Type | Description |
|----------|------|-------------|
| `SV_Position` | `float4` | Clip-space position (vertex shader) |
| `SV_Target` | `float4` | Render target output (fragment shader) |
| `SV_Depth` | `float` | Fragment depth |

### User Semantics

```hlsl
// Vertex inputs (match VertexAttribute.location)
float2 position : POSITION;     // Location 0
float4 color : COLOR;           // Location 1

// Interpolated values
float2 uv : TEXCOORD0;
float3 normal : NORMAL;
```

## Types

### Scalars

```hlsl
float f = 1.0;
int i = -5;
uint u = 10;
bool b = true;
```

### Vectors

```hlsl
float2 v2 = float2(1.0, 2.0);
float3 v3 = float3(1.0, 2.0, 3.0);
float4 v4 = float4(1.0, 2.0, 3.0, 4.0);

// Swizzling
float2 xy = v4.xy;
float3 rgb = v4.rgb;
float2 rr = v4.rr;
```

### Matrices

```hlsl
float2x2 m2;
float3x3 m3;
float4x4 m4;

// Matrix * vector
float4 transformed = mul(m4, float4(pos, 1.0));
```

### Arrays

```hlsl
float arr[4] = { 1.0, 2.0, 3.0, 4.0 };
float elem = arr[0];
```

## Control Flow

```hlsl
// If
if (condition) {
    // ...
} else if (other) {
    // ...
} else {
    // ...
}

// For
for (int i = 0; i < 10; i++) {
    // ...
}

// While
while (condition) {
    // ...
}

// Do-while
do {
    // ...
} while (condition);
```

## Math Functions

### Trigonometry

```hlsl
sin(x)   cos(x)   tan(x)
asin(x)  acos(x)  atan(x)
atan2(y, x)
sinh(x)  cosh(x)  tanh(x)
```

### Exponential

```hlsl
pow(x, y)   // x^y
exp(x)      // e^x
exp2(x)     // 2^x
log(x)      // ln(x)
log2(x)     // log2(x)
sqrt(x)
rsqrt(x)    // 1/sqrt(x)
```

### Common

```hlsl
abs(x)
sign(x)      // -1, 0, or 1
floor(x)
ceil(x)
round(x)
frac(x)      // x - floor(x)
trunc(x)
```

### Clamping

```hlsl
min(a, b)
max(a, b)
clamp(x, low, high)
saturate(x)  // clamp(x, 0.0, 1.0)
```

### Interpolation

```hlsl
lerp(a, b, t)      // Linear interpolation: a*(1-t) + b*t
step(edge, x)      // 0 if x < edge, 1 otherwise
smoothstep(e0, e1, x)  // Smooth Hermite interpolation
```

### Vector Operations

```hlsl
length(v)
distance(a, b)
dot(a, b)
cross(a, b)        // float3 only
normalize(v)
reflect(v, n)
refract(v, n, eta)
```

### Component-wise

```hlsl
// These work on vectors component-wise
abs(v)
sign(v)
floor(v)
// etc.
```

## Constant Buffers

```hlsl
cbuffer Uniforms {
    float4x4 modelViewProj;
    float time;
};

[shader("vertex")]
VertexOutput vs_main(float3 pos : POSITION) {
    VertexOutput output;
    output.position = mul(modelViewProj, float4(pos, 1.0));
    return output;
}
```

## Textures and Samplers

```hlsl
Texture2D myTexture;
SamplerState mySampler;

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return myTexture.Sample(mySampler, input.uv);
}
```

## Examples

### Solid Color

```hlsl
[shader("fragment")]
float4 fs_main() : SV_Target {
    return float4(1.0, 0.0, 0.0, 1.0);  // Red
}
```

### UV Gradient

```hlsl
[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target {
    return float4(uv, 0.5, 1.0);
}
```

### Time Animation

```hlsl
cbuffer TimeData {
    float time;
};

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target {
    float r = sin(time) * 0.5 + 0.5;
    float g = cos(time) * 0.5 + 0.5;
    return float4(r, g, uv.x, 1.0);
}
```

### Distance Field Circle

```hlsl
[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target {
    float2 center = float2(0.5, 0.5);
    float dist = distance(uv, center);
    float circle = 1.0 - smoothstep(0.2, 0.21, dist);
    return float4(circle, circle, circle, 1.0);
}
```

### Checkerboard

```hlsl
[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target {
    float scale = 8.0;
    float checker = floor(uv.x * scale) + floor(uv.y * scale);
    float color = fmod(checker, 2.0);
    return float4(color, color, color, 1.0);
}
```

## Resources

- [Slang User Guide](https://shader-slang.com/slang/user-guide/)
- [Slang GitHub](https://github.com/shader-slang/slang)
- [HLSL Reference](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-reference)
