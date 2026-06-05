# Textures and Samplers

`Texture` holds image data on the GPU. `Sampler` controls how that data is filtered and addressed when read in shaders. Together, they provide the standard texture sampling pipeline.

## Creating a Texture

```rust
use goldy::{Texture, TextureKind, TextureFormat, TextureFlags};

let texture = Texture::new(
    &device,
    512, 512,
    TextureFormat::Rgba8Unorm,
    TextureKind::Interpolated,
    TextureFlags::COPY_DST,
)?;
```

### With Initial Data

Data must be raw bytes matching `width × height × bytes_per_pixel`:

```rust
let pixels: Vec<u8> = load_image_rgba("sprite.png");
let texture = Texture::with_data(
    &device,
    &pixels,
    256, 256,
    TextureFormat::Rgba8Unorm,
    TextureKind::Interpolated,
    TextureFlags::COPY_DST,
)?;
```

## Spatial Access Patterns

The access pattern determines how the texture is bound and accessed in shaders:

| Access | Shader Mapping | Use When |
|--------|---------------|----------|
| `Interpolated` | `Texture2D` with sampler | Image data filtered between texels — sprites, materials, UI |
| `Direct` | `RWTexture2D` | Storage images, compute output, exact pixel reads/writes |

## Texture Formats

| Format | BPP | Notes |
|--------|-----|-------|
| `R8Unorm` | 1 | Single-channel (masks, SDFs) |
| `Rg8Unorm` | 2 | Two-channel (normal maps, motion vectors) |
| `Rgba8Unorm` | 4 | Standard 8-bit RGBA |
| `Rgba8UnormSrgb` | 4 | sRGB color space |
| `Bgra8UnormSrgb` | 4 | sRGB, swapped channels (common swapchain format) |
| `Bgra8Unorm` | 4 | Linear, swapped channels |
| `Rgba16Float` | 8 | HDR |
| `Rgba32Float` | 16 | Full precision |

## TextureFlags

```rust
bitflags! {
    pub struct TextureFlags: u32 {
        const COPY_SRC       = 1 << 0;
        const COPY_DST       = 1 << 1;
        const RENDER_TARGET  = 1 << 2;
    }
}
```

| Flag | Purpose |
|------|---------|
| `COPY_SRC` | Texture can be a copy source (needed for `read_to_cpu`) |
| `COPY_DST` | Texture can be a copy destination (needed for `write` / `write_region`) |
| `RENDER_TARGET` | Texture can be used as a color attachment |

## Writing Data

Prefer `TaskGraph::write_texture()` for batched, non-blocking uploads. The synchronous methods below stall the GPU:

```rust
#[allow(deprecated)]
texture.write(&pixels)?;

#[allow(deprecated)]
texture.write_region(x, y, width, height, &region_pixels)?;
```

## Reading Data

Read texture contents back to CPU memory. The texture must have been created with `TextureFlags::COPY_SRC`:

```rust
let mut output = vec![0u8; texture.byte_size()];
texture.read_to_cpu(&mut output)?;
```

## Texture Queries

```rust
texture.width();
texture.height();
texture.format();
texture.byte_size();  // width * height * bytes_per_pixel
texture.access();     // TextureKind
texture.flags();      // TextureFlags
texture.is_owned();   // true if dropping destroys the GPU resource
```

## Bindless Descriptors

Textures are registered in the global bindless descriptor set. The category depends on the access pattern: `Interpolated` maps to `ResourceCategory::Texture`, `Direct` maps to `ResourceCategory::StorageImage`.

```rust
// Typed handle (preferred)
let handle = texture.handle(ResourceAccess::Read).unwrap();

// Raw index
let index = texture.resource_index(ResourceAccess::Read).unwrap();
```

## Texture Borrowing

`Texture::borrow()` creates a non-owning view that shares the GPU resource. Dropping a borrowed texture does not destroy the underlying resource. Use this when handing a texture reference into a system that may drop it before the owner is done.

```rust
let borrowed = texture.borrow();
assert!(!borrowed.is_owned());
// dropping `borrowed` does not free GPU memory
```

## Depth Textures

Depth textures are created through `SurfaceConfig` or `RenderTarget`, not directly via `Texture::new`. Available depth formats:

| Format | Bits | Stencil |
|--------|------|---------|
| `Depth16Unorm` | 16 | No |
| `Depth24Plus` | 24 | No |
| `Depth24PlusStencil8` | 24 + 8 | Yes |
| `Depth32Float` | 32 | No |
| `Depth32FloatStencil8` | 32 + 8 | Yes |

```rust
let surface = Surface::new_with_config(&device, &window, SurfaceConfig {
    depth_format: Some(DepthFormat::Depth32Float),
    ..Default::default()
})?;
```

## Texture as Render Target

A texture created with `TextureFlags::RENDER_TARGET` can be used as a color attachment for offscreen rendering.

```rust
let offscreen = Texture::new(
    &device,
    1920, 1080,
    TextureFormat::Rgba16Float,
    TextureKind::Interpolated,
    TextureFlags::RENDER_TARGET | TextureFlags::COPY_SRC,
)?;
```

---

## Samplers

A `Sampler` defines how texture coordinates are interpreted — filtering between texels and handling coordinates outside `[0, 1]`.

### Creating a Sampler

```rust
use goldy::{Sampler, SamplerDesc, FilterMode, AddressMode};

let sampler = Sampler::new(&device, &SamplerDesc {
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    mipmap_filter: FilterMode::Linear,
    address_mode_u: AddressMode::Repeat,
    address_mode_v: AddressMode::Repeat,
    ..Default::default()
})?;
```

### Convenience Constructors

```rust
let nearest   = Sampler::nearest(&device)?;        // nearest filter, clamp to edge
let linear    = Sampler::linear(&device)?;          // linear filter, clamp to edge
let tiling    = Sampler::linear_repeat(&device)?;   // linear filter, repeat addressing
let default   = Sampler::default_sampler(&device)?; // nearest filter, clamp to edge
```

### SamplerDesc

```rust
pub struct SamplerDesc {
    pub address_mode_u: AddressMode,    // default: ClampToEdge
    pub address_mode_v: AddressMode,    // default: ClampToEdge
    pub address_mode_w: AddressMode,    // default: ClampToEdge
    pub mag_filter: FilterMode,         // default: Nearest
    pub min_filter: FilterMode,         // default: Nearest
    pub mipmap_filter: FilterMode,      // default: Nearest
    pub max_anisotropy: f32,            // default: 1.0 (disabled)
    pub compare: Option<CompareFunction>, // default: None
    pub lod_min_clamp: f32,             // default: 0.0
    pub lod_max_clamp: f32,             // default: 32.0
}
```

### Filter Modes

| Mode | Effect |
|------|--------|
| `Nearest` | Pixelated — nearest texel, no interpolation |
| `Linear` | Smooth — bilinear interpolation between neighbors |

### Address Modes

| Mode | Effect for UVs outside [0, 1] |
|------|-------------------------------|
| `ClampToEdge` | Stretches the border texel |
| `Repeat` | Tiles the texture |
| `MirrorRepeat` | Tiles with alternating mirror flips |

### Depth Comparison Samplers

For shadow mapping and depth-based effects, set the `compare` field:

```rust
let shadow_sampler = Sampler::new(&device, &SamplerDesc {
    compare: Some(CompareFunction::LessEqual),
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    ..Default::default()
})?;
```

### Bindless Descriptors

Samplers are registered under `ResourceCategory::Sampler`:

```rust
let handle = sampler.handle(ResourceAccess::Read).unwrap();
let index = sampler.resource_index(ResourceAccess::Read).unwrap();
```

## Binding Textures and Samplers in Shaders

Pass texture and sampler indices together through resource bindings:

```rust
let tex = texture.handle(ResourceAccess::Read).unwrap();
let samp = sampler.handle(ResourceAccess::Read).unwrap();
pass.bind_resources_typed(&[tex, samp]);
```

In Slang:

```slang
import goldy_exp;

[goldy_fragment]
float4 fs_main(Interpolated<float4> tex, Filter smp, float2 uv : TEXCOORD) {
    return tex.Sample(smp, uv);
}
```
