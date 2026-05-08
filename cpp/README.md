# Goldy C++ Package

C and C++ bindings for the Goldy GPU library.

## Features

- **goldy.h** - Auto-generated C API header
- **goldy.hpp** - Modern C++ RAII wrapper (C++20)
- **goldy_ffi.dll/so/dylib** - Native library built from Rust

## Quick Start

```cpp
#include <goldy.hpp>

int main() {
    // Create instance and device
    goldy::Instance instance;
    goldy::Device device = instance.create_device(GoldyDeviceType::DiscreteGpu);
    
    // Create render target
    goldy::RenderTarget target(device, 800, 600);
    
    // Compile shader and create pipeline
    goldy::ShaderModule shader(device, R"(
        [shader("vertex")]
        float4 vs_main(float2 pos : POSITION) : SV_Position {
            return float4(pos, 0.0, 1.0);
        }
        
        [shader("fragment")]
        float4 fs_main() : SV_Target {
            return float4(1.0, 0.0, 0.0, 1.0);
        }
    )");
    
    GoldyRenderPipelineDesc desc{};
    desc.target_format = GoldyTextureFormat::Rgba8Unorm;
    goldy::RenderPipeline pipeline(device, shader, shader, desc);
    
    // Render
    goldy::CommandEncoder encoder;
    encoder.clear(goldy::Color::cornflower_blue());
    encoder.set_pipeline(pipeline);
    encoder.draw(3);
    target.render(std::move(encoder));
    
    // Read back pixels
    auto pixels = target.read_to_cpu();
}
```

## Installation

### vcpkg

```bash
# Add to your vcpkg.json
{
    "dependencies": ["goldy"]
}

# Or install directly
vcpkg install goldy
```

### Conan

```bash
# Add to your conanfile.txt
[requires]
goldy/0.1.0

# Or conanfile.py
def requirements(self):
    self.requires("goldy/0.1.0")
```

### Manual Build

```bash
# Requires Rust toolchain: https://rustup.rs

# Build the native library
cd /path/to/goldy
cargo build --package goldy-ffi --release

# Configure with CMake
cd cpp
cmake -B build -DGOLDY_BUILD_FROM_SOURCE=ON
cmake --build build
```

## Requirements

- **C++20** compiler (MSVC 2019+, GCC 10+, Clang 12+)
- **Rust** toolchain (for building from source)
- **Vulkan SDK** with Slang compiler (slang.dll required at runtime)

## Platform Support

| Platform | Status |
|----------|--------|
| Windows x64 | ✅ Supported |
| Linux x64 | ✅ Supported |
| macOS x64 | ✅ Supported |
| macOS ARM64 | ✅ Supported |

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `goldy::Instance` | Entry point, creates devices |
| `goldy::Device` | GPU device handle |
| `goldy::Buffer` | GPU buffer (vertex, uniform, etc.) |
| `goldy::ShaderModule` | Compiled Slang shader |
| `goldy::RenderPipeline` | Graphics pipeline |
| `goldy::RenderTarget` | Offscreen render target |
| `goldy::CommandEncoder` | Records render commands |
| `goldy::ComputePipeline` | Compute shader pipeline |
| `goldy::ComputeEncoder` | Records compute commands |
| `goldy::Texture` | GPU texture |
| `goldy::Sampler` | Texture sampler |

### Error Handling

All operations that can fail throw `goldy::Exception`:

```cpp
try {
    goldy::Instance instance;
    // ...
} catch (const goldy::Exception& e) {
    std::cerr << "Goldy error: " << e.what() << "\n";
}
```

### Raw C API

For C or when you need low-level control, use the C API directly:

```c
#include <goldy.h>

GoldyInstance* instance = goldy_instance_create();
if (!instance) {
    const char* error = goldy_get_last_error();
    // handle error
}

GoldyDevice* device = goldy_instance_create_device(instance, DiscreteGpu);
// ...

goldy_device_destroy(device);
goldy_instance_destroy(instance);
```

## License

LGPL-2.1-or-later. A commercial license is also available; contact [koubaa on github](permament email tbd) for terms.

