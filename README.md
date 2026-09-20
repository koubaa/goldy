<p align="center">
  <img src="assets/goldy.png" alt="Goldy Logo" width="240">
</p>

# Goldy: Rust GPU runtime

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Goldy is a cross-platform, opinionated GPGPU and Graphics library for modern hardware written entirely in Rust with bindings in Python, C++, and dotnet and backends for Vulkan, Metal, D3D12, wgpu, and CUDA. Shaders can be authored in Slang (full support) or Rust (partial support).

Goldy realizes an abstract machine (https://koubaa.github.io/goldy/fondaco/specification.html) that can be said to be both data-oriented and functional.

The machine model admits _schemes_, or graphs of computations over data and does not provide any synchronization to user-space. Instead, synchronization is derived inside the runtime by data access patterns and submission order, using a ledger. The runtime modifies shaders to pass data it uses to its entrypoint, and introspects the IR of shaders as part of its execution. Operations on data not owned by the abstract machine happen at the edge of the runtime in _exchanges_.

Shader entry points are modified so that shaders see data they operate on according to the ownership model of the invocation.

GPU memory is completely virtualized, and the runtime may relocate objects between _dispatches_ or shader invocations, as long as it does not change the behavior of the program.

## Quick example: compute-to-surface

```rust
#[goldy::gpu]
struct Uniforms {
    width: u32,
    height: u32,
    time: f32,
}

#[goldy::compute(workgroup_size = [8, 8, 1])]
fn plasma(uniforms: &[Uniforms], output: gpu::DirectSpatial<gpu::Float4>) {
    let tid = gpu::global_id();
    let u: Uniforms = uniforms[0];
    if tid.x >= u.width || tid.y >= u.height {
        return;
    }

    let uv = gpu::float2(tid.x as f32 / u.width as f32, tid.y as f32 / u.height as f32);
    let mut p = uv * 2.0 - 1.0;
    p.x *= u.width as f32 / u.height as f32;

    let mut v = 0.0;
    v += gpu::sin(p.x * 6.0 + u.time);
    v += gpu::sin(p.y * 6.0 + u.time * 1.3);
    v += gpu::sin((p.x + p.y) * 4.0 + u.time * 0.7);
    v += gpu::sin(gpu::length(p) * 8.0 - u.time * 2.0);
    v *= 0.25;

    let col = gpu::float3(
        0.5 + 0.5 * gpu::sin(v * 3.14159 + 0.0),
        0.5 + 0.5 * gpu::sin(v * 3.14159 + 2.094),
        0.5 + 0.5 * gpu::sin(v * 3.14159 + 4.188),
    );
    output[tid.xy] = gpu::float4(col.x, col.y, col.z, 1.0);
}

let instance = Instance::new()?;
let runtime = instance
    .request_adapter(&RequestAdapterOptions::default())?
    .request_runtime(&RuntimeDescriptor::default())?;
let ctx = runtime.create_context()?;

let uniforms = runtime.acquire_buffer_with_data(&uniforms_data, BufferKind::Scattered)?;

let surface = SurfaceExchange::new(&ctx, &window, SurfaceConfig::default())?;

// Compile the Rust kernel to [goldy_compute] Slang (or hit the shader cache).
let kernel = plasma::Kernel::prepare(&runtime)?;


let mut scheme = Scheme::new(&ctx);
// Lease is the drawable the kernel writes; transaction is how the frame is presented.
let (lease, present) = surface.bind_destination(&mut scheme)?;
kernel
    .record(&mut scheme, "render", &uniforms, &lease)
    .over_2d(width, height); // workgroups: ceil(width/8) × ceil(height/8) × 1

// Each frame: submit the recorded graph, then present the claimed drawable.
let mut submission = scheme.submit()?;
present.claim(&mut submission)?.consume()?;
```

## Installation

```toml
[dependencies]
goldy = "0.2"
```

Slang is **embedded at build time** and extracted at runtime — application developers need not install Slang separately. Set `GOLDY_SLANG_PATH` only to override.

Release packaging and shader debugging notes live in the [GitHub repo](https://github.com/koubaa/goldy).

## Platforms

| Platform | Backend | Window surfaces |
|----------|---------|-----------------|
| Windows | DX12 (default), Vulkan | Yes |
| Linux | Vulkan | Wayland (X11 not supported) |
| macOS | Metal | Yes |

Override backend: `GOLDY_BACKEND=vulkan|dx12|metal`.

Minimum hardware: Vulkan 1.4+, DX12 with Enhanced Barriers, Metal Argument Buffers Tier 2+. See [Target Hardware](https://koubaa.github.io/goldy/design/hardware.html).

## Documentation

- 📖 **[Documentation](https://koubaa.github.io/goldy/)** — tutorials, programming model, backends, bindings
- 📖 **[Fondaco Machine spec](https://koubaa.github.io/goldy/fondaco/specification.html)** — normative abstract machine
- 📖 **[Goldy runtime mapping](https://koubaa.github.io/goldy/fondaco/goldy-runtime.html)** — what Goldy implements today vs designs in progress
- 📖 **[API reference](https://docs.rs/goldy)** — Rust docs

## License

MIT — see [LICENSE](LICENSE).

## Author

Mohamed Koubaa
