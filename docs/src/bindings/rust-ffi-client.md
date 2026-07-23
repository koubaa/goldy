# Rust FFI Client

`goldy-ffi-client` is a Rust crate that loads the `goldy-ffi` native library at runtime and exposes the same RAII API as the core `goldy` crate. Instead of statically linking the `goldy` library, it calls the stable C ABI through `libloading` (`LoadLibrary` on Windows, `dlopen` on Unix).

This is the same native boundary used by the [C++](./cpp.md) and [.NET](./dotnet.md) bindings. Python is different — it links the core `goldy` crate directly via PyO3.

## When to Use

| Use case | Crate |
|----------|-------|
| Normal Rust applications | `goldy` (static link, published on crates.io) |
| FFI integration tests | `goldy-ffi-client` |
| Validating the C ABI from Rust | `goldy-ffi-client` |
| Swapping the native library without recompiling the client | `goldy-ffi-client` |

The ffi-client API mirrors the core Rust crate: `Instance`, `Scheme`, `RetainedPool`, `MemoryExchange`, `SurfaceExchange`, and the rest of the Fondaco programming model are available with the same names and patterns.

## Installation

`goldy-ffi-client` is a workspace crate — it is not published to crates.io. Add it as a path dependency:

```toml
[dependencies]
goldy-ffi-client = { path = "../ffi-client" }
```

Build the native library first:

```bash
cargo build -p goldy-ffi
```

Then build or run ffi-client examples:

```bash
cd ffi-client
cargo run --example triangle_headless
```

### Requirements

- Rust 2021 edition
- A built `goldy_ffi` shared library (`goldy_ffi.dll` / `libgoldy_ffi.so` / `libgoldy_ffi.dylib`)
- A GPU with Vulkan 1.4+, DX12, or Metal Tier 2+ support (CUDA and WebGPU backends are in progress; Tenstorrent is planned)

## Library Discovery

At runtime, ffi-client searches for the native library in this order:

1. `GOLDY_FFI_PATH` — full path to the `goldy_ffi` dylib
2. `GOLDY_FFI_LIB_DIR` — compile-time directory from the `goldy-ffi` build
3. The directory containing the running executable

On Windows, ffi-client also calls `SetDllDirectoryW` so Slang DLLs next to `goldy_ffi.dll` are found.

```bash
# Point at a specific build of the native library
GOLDY_FFI_PATH=/path/to/libgoldy_ffi.so cargo run --example triangle_headless
```

## Quick Start

### Headless Rendering

```rust
use goldy_ffi_client::{
    shader::builtins, BufferKind, Color, Context, DeviceDescriptor, Instance, NodeAccess,
    RenderPipeline, RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme,
    ShaderModule, TargetLoad, TextureFlags, TextureFormat, TextureKind, Vertex2D,
};

fn main() -> goldy_ffi_client::Result<()> {
    let instance = Instance::new()?;
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())?
        .request_device(&DeviceDescriptor::default())?;
    let ctx = Context::new(&device)?;

    let vertices = [
        Vertex2D { position: [0.0, -0.5], color: [1.0, 0.0, 0.0, 1.0] },
        Vertex2D { position: [-0.5, 0.5], color: [0.0, 1.0, 0.0, 1.0] },
        Vertex2D { position: [0.5, 0.5], color: [0.0, 0.0, 1.0, 1.0] },
    ];
    let mut pool = RetainedPool::new(&device)?;
    let vertex_buffer = pool.acquire_buffer_with_data(&vertices, BufferKind::Scattered)?;

    let readback = pool.acquire_texture(
        64, 64, TextureFormat::Rgba8Unorm, TextureKind::Direct,
        TextureFlags::COPY_SRC.union(TextureFlags::COPY_DST), None,
    )?;

    let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
    let pipeline = RenderPipeline::new(
        &device, &shader, &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )?;

    let mut scheme = Scheme::new(&ctx)?;
    let rt = scheme.lease_render_target(64, 64, TextureFormat::Rgba8Unorm, None)?;
    {
        let mut pass = scheme.render_pass("triangle", &rt, TargetLoad::Clear(Color::BLACK));
        pass.with_buffer(&vertex_buffer, NodeAccess::Read);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.draw(0..3, 0..1);
        pass.finish_recorded();
    }
    scheme.copy_to_texture(&rt, &readback)?;

    let memory = goldy_ffi_client::MemoryExchange::new(&ctx)?;
    let withdraw = memory.bind_withdraw_texture(&mut scheme, &readback)?;
    let mut submission = scheme.submit()?;
    let pixels = withdraw.claim(&mut submission)?.consume()?;

    println!("Rendered {} bytes", pixels.len());
    Ok(())
}
```

See `ffi-client/examples/triangle_headless.rs` for the full example.

### Windowed Rendering

See `ffi-client/examples/triangle.rs` and `ffi-client/examples/game_of_life.rs` (winit).

### Compute

```rust
use goldy_ffi_client::{ComputePipeline, Context, Instance, MemoryExchange, NodeAccess, Scheme, ShaderModule};

let mut scheme = Scheme::new(&ctx)?;
let mut node = scheme.compute_node("double", &pipeline);
node.with_buffer(&buf, NodeAccess::ReadWrite);
node.dispatch(1, 1, 1);

let memory = MemoryExchange::new(&ctx)?;
let withdraw = memory.bind_withdraw(&mut scheme, &buf.field(0)?)?;
let mut submission = scheme.submit()?;
let bytes = withdraw.claim(&mut submission)?.consume()?;
```

See `ffi-client/examples/compute_simple.rs`.

## Resource Management

All ffi-client types use RAII via `Drop`. Errors are returned as `goldy_ffi_client::Result<T>` with `GoldyError` — the same pattern as the core `goldy` crate.

## Key Differences from Core `goldy`

| Aspect | `goldy` | `goldy-ffi-client` |
|--------|---------|-------------------|
| Linking | Static (compiled into your binary) | Dynamic (`libloading` at runtime) |
| Distribution | crates.io | Workspace path dependency |
| API surface | Reference implementation | Mirrors core API over C ABI |
| Native library | Embedded in your binary | Separate `goldy_ffi` dylib required |
| Crate name | `goldy` | `goldy_ffi_client` |

Functionally, application code looks nearly identical. The main difference is build and deployment: ffi-client binaries need the `goldy_ffi` shared library (and Slang DLLs) available at runtime.

## Backend Selection

`GOLDY_BACKEND` works the same as with the core crate — set it before creating an `Instance`:

```bash
GOLDY_BACKEND=vulkan cargo run --example triangle_headless
```

When building `goldy-ffi`, pass backend features through:

```bash
cargo build -p goldy-ffi --no-default-features --features vulkan
```

## Examples

| Example | Description |
|---------|-------------|
| `ffi-client/examples/triangle_headless.rs` | Offscreen triangle + readback |
| `ffi-client/examples/triangle.rs` | Windowed triangle (winit) |
| `ffi-client/examples/compute_simple.rs` | Compute dispatch |
| `ffi-client/examples/game_of_life.rs` | Hybrid compute + render (windowed) |
| `ffi-client/examples/game_of_life_headless.rs` | Game of Life readback |
