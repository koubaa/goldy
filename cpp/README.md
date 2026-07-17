# Goldy C++ Package

C and C++ bindings for the Goldy GPU library.

## Features

- **goldy.h** - Auto-generated C API header
- **goldy.hpp** - Modern C++ RAII wrapper (C++20)
- **goldy_ffi.dll/so/dylib** - Native library built from Rust

## Quick Start

Headless offscreen render (all platforms with a GPU):

```cpp
#include <goldy.hpp>

#include <cstdint>
#include <iostream>

struct Vertex {
    float position[2];
    float color[4];
};

int main() {
    try {
        goldy::Instance instance;
        goldy::Device device = instance.request_adapter().request_device();
        goldy::Context ctx(device);

        const Vertex vertices[] = {
            {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f, 1.0f}},
            {{-0.5f, 0.5f}, {0.0f, 1.0f, 0.0f, 1.0f}},
            {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f, 1.0f}},
        };

        goldy::RetainedPool pool(device);
        goldy::Buffer vertex_buffer = pool.acquire_buffer_with_data(
            std::span<const Vertex>(vertices),
            goldy::BufferKind::Scattered);

        goldy::ShaderModule shader(device, goldy::ShaderModule::builtin_vertex_color_2d());

        GoldyVertexAttribute attributes[] = {
            {0, GOLDY_VERTEX_FORMAT_FLOAT32X2, 0},
            {1, GOLDY_VERTEX_FORMAT_FLOAT32X4, static_cast<uint32_t>(sizeof(float) * 2)},
        };

        GoldyRenderPipelineDesc desc{};
        desc.vertex_attributes = attributes;
        desc.vertex_attribute_count = static_cast<uint32_t>(std::size(attributes));
        desc.vertex_stride = sizeof(Vertex);
        desc.topology = GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        desc.target_format = GOLDY_TEXTURE_FORMAT_RGBA8_UNORM;

        goldy::RenderPipeline pipeline(device, shader, shader, desc);

        GoldyTextureFlags readback_flags{};
        readback_flags._0 = goldy::TextureFlags::CopySrc | goldy::TextureFlags::CopyDst;
        goldy::Texture readback = pool.acquire_texture(
            800, 600, GOLDY_TEXTURE_FORMAT_RGBA8_UNORM,
            GOLDY_TEXTURE_KIND_DIRECT, readback_flags);

        goldy::Scheme scheme(ctx);
        goldy::SchemeRenderTargetLease rt = scheme.lease_render_target(
            800, 600, GOLDY_TEXTURE_FORMAT_RGBA8_UNORM, nullptr);
        {
            auto pass = scheme.render_pass("triangle", rt);
            pass.with_field(vertex_buffer, 0, goldy::NodeAccess::Read)
                .clear(goldy::Color::cornflower_blue())
                .set_pipeline(pipeline)
                .set_vertex_buffer(0, vertex_buffer)
                .draw(0, 3);
        }
        scheme.copy_to_texture(rt, readback);
        goldy::ReadGrant grant = scheme.grant_read_texture(readback);
        goldy::SchemeSubmission submission = scheme.submit();
        auto pixels = grant.consume(submission);
        std::cout << "Rendered " << pixels.size() << " bytes\n";
        return 0;
    } catch (const goldy::Exception& e) {
        std::cerr << "Goldy error: " << e.what() << '\n';
        return 1;
    }
}
```

Windowed rendering: see `examples/triangle.cpp` (Win32 / macOS). Check
`goldy::Surface::is_supported()` before using `Surface`.

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
cmake --build build --target triangle
```

On Windows, if MSVC cannot find `stdarg.h`, either:

- Re-configure from any shell (CMake auto-detects MSVC/SDK paths for Ninja), or
- Use **x64 Native Tools Command Prompt for VS 2022**, or
- Run `cpp/build.bat` which calls `vcvars64.bat` first.

## Requirements

- **C++20** compiler (MSVC 2019+, GCC 10+, Clang 12+)
- **Rust** toolchain (for building from source)
- **Vulkan SDK** with Slang compiler (slang.dll required at runtime)

## Platform Support

| Platform | Headless Scheme | Windowed Surface |
|----------|-------------------|------------------|
| Windows x64 | ✅ | ✅ |
| Linux x64 | ✅ | ❌ (use headless or custom window + C API) |
| macOS x64 / ARM64 | ✅ | ✅ |

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `goldy::Instance` | Entry point, creates devices |
| `goldy::Device` | GPU device handle |
| `goldy::RetainedPool` | Deed-governed pool for retained GPU parcels |
| `goldy::Parcel` | Retained buffer or texture parcel |
| `goldy::RecordBuilder` | Build partitioned buffer records (multiple field parcels) |
| `goldy::ShaderModule` | Compiled Slang shader |
| `goldy::RenderPipeline` | Graphics pipeline |
| `goldy::RenderTarget` | Offscreen render target (readback) |
| `goldy::Scheme` | Retained dependency graph (render passes, compute, present) |
| `goldy::Surface` | Window swapchain (Win32 / macOS only) |
| `goldy::ComputePipeline` | Compute shader pipeline |
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

GoldyAdapterInfo info = {};
goldy_instance_get_adapter(instance, 0, &info);
GoldyDevice* device = goldy_instance_create_device_for_adapter(instance, info.id);
// ...

goldy_device_destroy(device);
goldy_instance_destroy(instance);
```

## License

MIT License. See the [goldy repository](https://github.com/koubaa/goldy/blob/main/LICENSE) for the full text.
