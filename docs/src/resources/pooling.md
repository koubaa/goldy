# Pooling and Sub-Allocation

GPU resource allocation is expensive. Creating many small buffers or textures
each frame produces allocation overhead, descriptor churn, and VRAM
fragmentation. Goldy routes client allocation through two doors:

| Door | Permanence | Acquire |
|------|------------|---------|
| [`RetainedPool`](./retained-pool.md) | Cross-submission identity (deeds) | `acquire_texture` / `acquire_buffer` / `acquire_record` |
| Context transient pool | One-submission tenancy (leases) | `Context::acquire_transient_texture` / `acquire_transient_buffer`, or `Scheme::lease_texture` / `lease_buffer` |

The runtime owns reclaim: retained release transfers into the transient pool
with a `ready_after` stamp; transient bins reissue only after GPU retirement.

Partitioned retained buffers (one backing, many bindable fields) use internal
scattered suballocation — see `RetainedPool::acquire_record`. Do not construct
bump arenas or whole-object texture free-lists in client code.
