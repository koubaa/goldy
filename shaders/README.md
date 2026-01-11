# Goldy Shader Library

Slang shader sources shared between native (Vulkan) and web (WebGPU) platforms.

## Usage

**Native (Rust):**
```rust
use goldy::ShaderModule;

let shader = ShaderModule::from_slang(&device, include_str!("../../shaders/plasma.slang"))?;
```

**Web (JavaScript + slang-wasm):**
```javascript
const slangSource = SHADERS['plasma'];
const wgsl = slangCompiler.compileToWgsl(slangSource);
const module = device.createShaderModule({ code: wgsl });
```

## Shader Files

| File | Description | Demo |
|------|-------------|------|
| `vertex_color_2d.slang` | Basic 2D position + color | particles, starfield, bouncing_lines, etc. |
| `digital_clock.slang` | 7-segment display shader | digital_clock |
| `triangle.slang` | Procedural triangle from vertex ID | triangle |
| `plasma.slang` | Classic demoscene plasma | plasma |
| `mandelbrot.slang` | Fractal explorer with zoom | mandelbrot |
| `gradient.slang` | Animated color gradient | gradient |
| `tunnel.slang` | Demoscene tunnel effect | tunnel |
| `checkerboard.slang` | Animated checker pattern | checkerboard |
| `metaballs.slang` | Blending distance fields | metaballs |

## Compilation Targets

All shaders compile to:
- **SPIR-V** (via native slang.dll) → Vulkan
- **WGSL** (via slang-wasm) → WebGPU
- **HLSL** (future) → DirectX 12
- **MSL** (future) → Metal

