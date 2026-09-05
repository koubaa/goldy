# Transient Allocation

Rendering pipelines allocate many short-lived GPU buffers and textures each
frame — scratch storage, per-pass intermediates, filter pyramids. After
submission those resources are dead until the GPU finishes, at which point the
memory can be recycled. Clients must not poll timeline clocks to decide when
reuse is safe.

## Client door: `TransientPool`

Goldy exposes one transient door per [`Context`](../compute/settlement.md):

| Acquire | Return |
|---------|--------|
| `Context::acquire_transient_buffer` | `Context::return_transient_buffer` |
| `Context::acquire_transient_texture` | `Context::return_transient_texture` |

Scheme leases (`Scheme::lease_buffer` / `lease_texture`) realize through the same
pool. Relinquished retained parcels enter via `StampedParcel` / `ready_after`;
the pool reissues only after every stamped epoch has retired.

```rust
let scratch = ctx.acquire_transient_buffer(
    size,
    BufferKind::Scattered,
    BufferFlags::GPU_ONLY,
    Some(stride),
)?;
// ... bind, submit ...
ctx.return_transient_buffer(scratch);
```

See also [Pooling and Sub-Allocation](./pooling.md).

## What was removed

The former public `TransientAllocator` strategies (`BumpReset`, `Heap`) and the
internal scattered bump arena (`BufferPool`) were deleted: they had no in-tree
consumers once in-tree callers moved to `RetainedPool` / `TransientPool`.
Whole-object epoch-gated recycle bins are the supported transient path; any
future suballocation belongs behind that door, not as a parallel public API.
