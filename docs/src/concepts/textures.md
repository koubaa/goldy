# Textures and Samplers

Textures hold image data on the GPU. Samplers control how that data is read in shaders (filtering, addressing).

## Creating a Texture

```rust
use goldy::{Texture, SpatialAccess, TextureFormat, TextureFlags};

let texture = Texture::new(
    &device,
    512,                        // width
    512,                        // height
    TextureFormat::Rgba8Unorm,
    SpatialAccess::Interpolated, // hardware bilinear filtering
    TextureFlags::empty(),
)?;
```

## Spatial Access Patterns

| Access | Maps to | Use when |
|--------|---------|----------|
| `Interpolated` | `Texture2D` + sampler | Image data filtered between texels |
| `Direct` | `RWTexture2D` | Storage images, compute output, exact pixel reads |

## Uploading Image Data

```rust
// Data must be RGBA8 bytes (width * height * 4 bytes)
let pixels: Vec<u8> = load_image_rgba("my_texture.png");
texture.write(&pixels)?;
```

## Creating a Sampler

`Sampler` objects describe how texture coordinates outside `[0, 1]` are handled and how texels are filtered.

```rust
use goldy::{Sampler, SamplerDesc, FilterMode, AddressMode};

let sampler = Sampler::new(&device, &SamplerDesc {
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    address_mode_u: AddressMode::Repeat,
    address_mode_v: AddressMode::Repeat,
    ..Default::default()
})?;
```

### Filter Modes

| Mode | Effect |
|------|--------|
| `Nearest` | Pixelated (no interpolation) |
| `Linear` | Smooth bilinear interpolation |

### Address Modes

| Mode | Effect when UV is outside [0, 1] |
|------|-----------------------------------|
| `Repeat` | Tiles the texture |
| `MirrorRepeat` | Tiles with alternating flips |
| `ClampToEdge` | Stretches the border pixel |
| `ClampToBorder` | Fills with a border color |

## Using Textures in Shaders

Textures and samplers are bound via **bindless** descriptors. Each texture has a `bindless_index()` that you pass to the shader through push constants:

```rust
let tex_idx = texture.bindless_index().unwrap();
let smp_idx = sampler.bindless_index().unwrap();

// In a render pass:
pass.set_push_constants_raw(&[tex_idx, smp_idx]);
```

In the Slang shader:

```slang
import goldy;

struct PushConstants { uint tex_idx; uint smp_idx; };
[[vk::push_constant]] PushConstants pc;

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD) : SV_Target {
    Texture2D<float4> tex = g_Textures[pc.tex_idx];
    SamplerState smp = g_Samplers[pc.smp_idx];
    return tex.Sample(smp, uv);
}
```

## TextureFlags

Use `TextureFlags` for textures that participate in copy operations or are used as render targets:

```rust
use goldy::TextureFlags;

// A texture that can be a copy destination (e.g., updated each frame)
let tex = Texture::new(
    &device, width, height, format,
    SpatialAccess::Interpolated,
    TextureFlags::COPY_DST,
)?;
```

## Vertex2DUv

For 2D rendering with texture coordinates, use the built-in `Vertex2DUv` type:

```rust
use goldy::Vertex2DUv;

let verts = vec![
    Vertex2DUv::new(-0.5, -0.5, 0.0, 1.0),
    Vertex2DUv::new( 0.5, -0.5, 1.0, 1.0),
    Vertex2DUv::new( 0.0,  0.5, 0.5, 0.0),
];
```

See the [`textured_quad`](../examples/triangle.md) example for a complete usage.
