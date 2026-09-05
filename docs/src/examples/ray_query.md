# ray_query

Builds a BLAS for one triangle plus a TLAS, then traces primary rays with inline `RayQuery`
from a `[goldy_compute]` entry point and writes hits straight into the swapchain. No ray
tracing pipeline or shader binding table is involved.

```bash
cargo run --features examples --example ray_query
```

## What it demonstrates

- Acceleration structure build (BLAS and TLAS) inside a scheme
- Inline ray query from a compute entry point
- Compute-to-surface output

## Notes

The example exits 0 when `DeviceCapabilities::ray_query` is false, and on the WebGPU backend,
where Slang's WGSL target has no `TraceRayInline`.

## Source

`examples/ray_query.rs`:

```rust,noplayground
{{#include ../../../examples/ray_query.rs}}
```

The example pulls in `examples/common.rs` — see [Shared Helpers](./shared-helpers.md).

The Slang source is inline in the example above.
