# mesh_triangle

The same present path as [`triangle`](./triangle.md), but the geometry is produced by a
`[goldy_mesh]` entry point driven by `dispatch_mesh` instead of a vertex buffer.
`MeshOutput` and `FsIn` deliberately use different struct names so the example also shows
Goldy linking stages by semantic (`SV_Position`, `COLOR`) rather than by type identity.

*No recording: the capture runs on the WebGPU backend, which has no mesh shaders, so the example skips.*

```bash
cargo run --features examples --example mesh_triangle
```

## What it demonstrates

- `MeshPipeline` and `dispatch_mesh`
- Automatic payload linking between mesh and fragment stages
- Capability probing — the example exits 0 when `DeviceCapabilities::mesh_shaders` is false

## Notes

Mesh shaders are not implemented on the WebGPU backend, and adapters without mesh-shader
support skip the example rather than failing.

## Source

`examples/mesh_triangle.rs`:

```rust,noplayground
{{#include ../../../examples/mesh_triangle.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

The Slang source is inline in the example above.
