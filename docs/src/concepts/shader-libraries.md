# Shader Libraries

`ShaderLibrary` packages reusable Slang modules that can be `import`-ed by other shaders. Goldy ships with a built-in `goldy` library that is automatically registered on every device.

## Built-in `goldy` Library

The `goldy` Slang library provides bindless resource arrays, common vertex types, and utility functions:

```slang
import goldy;

// Available after import:
// RWByteAddressBuffer g_StorageBuffers[]  — bindless storage buffers
// ConstantBuffer<...> g_UniformBuffers[]  — bindless uniform buffers
// Texture2D<float4>   g_Textures[]        — bindless textures
// SamplerState        g_Samplers[]        — bindless samplers
//
// FullscreenVertex / FullscreenVarying    — helpers for fullscreen passes
// vs_fullscreen(input)                    — pre-built fullscreen vertex shader
```

A minimal fullscreen fragment effect using the built-in library:

```slang
import goldy;

struct PushConstants { float time; };
[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[shader("fragment")]
float4 fs_main(FullscreenVarying input) : SV_Target {
    float2 uv = input.uv;
    return float4(uv, sin(pc.time) * 0.5 + 0.5, 1.0);
}
```

## Custom Libraries

Register your own reusable Slang modules:

```rust
use goldy::ShaderLibrary;

let my_lib = ShaderLibrary::from_source("myutils", r#"
    module myutils;

    public float3 hsv_to_rgb(float3 c) {
        float4 K = float4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
        float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }
"#);

device.register_library(my_lib)?;
```

Now any shader can `import myutils`:

```slang
import myutils;

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD) : SV_Target {
    float3 color = hsv_to_rgb(float3(uv.x, 1.0, 1.0));
    return float4(color, 1.0);
}
```

## Libraries from Files

For larger projects, load a library from a directory of `.slang` files:

```rust
use goldy::ShaderLibrary;
use std::path::Path;

let lib = ShaderLibrary::from_directory("effects", Path::new("shaders/effects/"))?;
device.register_library(lib)?;
```

All `.slang` files in the directory become available under the library name.

## Checking Registration

```rust
if device.has_library("myutils") {
    println!("myutils library is registered");
}
```
