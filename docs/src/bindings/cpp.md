# C++ Bindings

Goldy provides C and C++ bindings over the native `goldy-ffi` library. The C++ layer (`goldy.hpp`) wraps the auto-generated C API (`goldy.h`) with RAII types and exceptions.

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
# conanfile.txt
[requires]
goldy/0.2.0
```

### Building from Source

```bash
# Build the native library (requires Rust: https://rustup.rs)
cargo build --package goldy-ffi --release

# Configure and build examples
cd cpp
cmake -B build -DGOLDY_BUILD_FROM_SOURCE=ON
cmake --build build --target triangle_headless
```

On Windows, if MSVC cannot find standard headers, use **x64 Native Tools Command Prompt for VS 2022** or run `cpp/build.bat`, which sets up the MSVC environment before invoking CMake.

### Requirements

- **C++20** compiler (MSVC 2019+, GCC 10+, Clang 12+)
- **Rust** toolchain (for building `goldy-ffi` from source)
- A GPU with Vulkan 1.4+, DX12, or Metal Tier 2+ support (CUDA and WebGPU backends are in progress; Tenstorrent is planned)
- Slang is **embedded** in the Goldy build — no separate SDK install for normal use

### Native Library Deployment

The `goldy_ffi` shared library and Slang runtime DLLs must be on the loader path at runtime. CMake post-build steps copy them next to example binaries when building from source. For your own applications, ship `goldy_ffi.dll` / `libgoldy_ffi.so` / `libgoldy_ffi.dylib` alongside your executable, with the Slang shared libraries in the same directory.

## Quick Start

### Headless Rendering

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
            auto pass = scheme.render_pass("triangle", rt, goldy::TargetLoad::clear(goldy::Color::cornflower_blue()));
            pass.with_field(vertex_buffer, 0, goldy::NodeAccess::Read)
                .set_pipeline(pipeline)
                .set_vertex_buffer(0, vertex_buffer)
                .draw(0, 3);
        }
        scheme.copy_to_texture(rt, readback);
        goldy::MemoryExchange memory(ctx);
        goldy::WithdrawTransaction withdraw = memory.bind_withdraw_texture(scheme, readback);
        goldy::SchemeSubmission submission = scheme.submit();
        goldy::WithdrawBytes bytes = withdraw.claim(submission).consume();
        std::cout << "Rendered " << bytes.size() << " bytes\n";
        return 0;
    } catch (const goldy::Exception& e) {
        std::cerr << "Goldy error: " << e.what() << '\n';
        return 1;
    }
}
```

See `cpp/examples/triangle_headless.cpp` for the full example.

### Windowed Rendering

Use `goldy::SurfaceExchange` for swapchain presentation. See `cpp/examples/triangle.cpp` (Win32 / macOS).

### Shaders (Slang)

Goldy uses [Slang](https://shader-slang.org/) as its shader language across all backends:

```cpp
const char* source = R"(
import goldy_exp;

[goldy_vertex]
float4 vs_main(Vertex2D v) : SV_Position {
    return float4(v.position, 0.0, 1.0);
}

[goldy_fragment]
float4 fs_main(Vertex2D v) : SV_Target {
    return float4(v.color);
}
)";

goldy::ShaderModule shader(device, source);
```

## Resource Management

All C++ wrapper types use RAII — destructors release GPU handles automatically. Operations that can fail throw `goldy::Exception`:

```cpp
try {
    goldy::Instance instance;
    // ...
} catch (const goldy::Exception& e) {
    std::cerr << "Goldy error: " << e.what() << "\n";
}
```

## Key Differences from Rust

| Aspect | Rust | C++ |
|--------|------|-----|
| Instance creation | `Instance::new()?` | `goldy::Instance instance` |
| Error handling | `Result<T, GoldyError>` | `goldy::Exception` |
| Device lifetime | `Arc<Device>` | RAII destructor |
| Retained buffer | `pool.acquire_buffer_with_data(&data, access)` | `pool.acquire_buffer_with_data(span, access)` |
| Render pass | `scheme.render_pass(...)` | `scheme.render_pass(...)` (RAII scope) |
| Readback | `claim.consume(&submission)` | `withdraw.claim(submission).consume()` |

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `goldy::Instance` | Entry point, adapter enumeration |
| `goldy::Device` / `goldy::Context` | GPU device and execution context |
| `goldy::RetainedPool` | Retained buffer/texture acquisition |
| `goldy::RecordBuilder` | Partitioned buffer records (ping-pong fields) |
| `goldy::Scheme` | Retained dependency graph |
| `goldy::MemoryExchange` | CPU↔GPU withdraw/deposit |
| `goldy::SurfaceExchange` | Window swapchain (Win32 / macOS / Wayland) |
| `goldy::ShaderModule` | Compiled Slang shader |
| `goldy::RenderPipeline` / `goldy::ComputePipeline` | Graphics/compute pipelines |
| `goldy::Sampler` | Texture sampler |

### Scheme

```cpp
goldy::Scheme scheme(ctx);
goldy::SchemeRenderTargetLease rt = scheme.lease_render_target(w, h, format, nullptr);

{
    auto pass = scheme.render_pass("main", rt, goldy::TargetLoad::clear(color));
    pass.with_field(buf, 0, goldy::NodeAccess::Read)
        .set_pipeline(pipeline)
        .set_vertex_buffer(0, buf)
        .draw(0, 3);
}

auto node = scheme.compute_node("update", compute_pipeline);
node.with_field(buf, 0, goldy::NodeAccess::ReadWrite)
    .dispatch(wg_x, wg_y, 1);

goldy::SchemeSubmission submission = scheme.submit();
```

### MemoryExchange / SurfaceExchange

```cpp
goldy::MemoryExchange memory(ctx);
goldy::WithdrawTransaction withdraw = memory.bind_withdraw_texture(scheme, texture);
goldy::SchemeSubmission submission = scheme.submit();
goldy::WithdrawBytes pixels = withdraw.claim(submission).consume();

goldy::SurfaceExchange surface(ctx, window_handle, width, height);
auto present = surface.bind_render_target(scheme, rt);
goldy::SchemeSubmission submission = scheme.submit();
present.claim(submission).consume();
```

### Raw C API

For C code or when you need low-level control, use `goldy.h` directly. Failed calls return null or error codes; call `goldy_get_last_error()` for details:

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

## Platform Support

| Platform | Headless Scheme | Windowed Surface |
|----------|-----------------|------------------|
| Windows x64 | Yes | Yes |
| Linux x64 | Yes | Yes (Wayland; X11 not supported) |
| macOS x64 / ARM64 | Yes | Yes |

## Backend Selection

Goldy auto-selects the best backend per platform. Override with `GOLDY_BACKEND` (set before creating an `Instance`):

```bash
GOLDY_BACKEND=vulkan ./my_app
```

When building `goldy-ffi` for a specific platform, pass backend features through:

```bash
cargo build -p goldy-ffi --no-default-features --features vulkan
```

## Examples

| Example | Description |
|---------|-------------|
| `cpp/examples/triangle_headless.cpp` | Offscreen triangle + readback |
| `cpp/examples/triangle.cpp` | Windowed triangle (GLFW) |
| `cpp/examples/compute_simple.cpp` | Compute dispatch |
| `cpp/examples/game_of_life.cpp` | Hybrid compute + render |
