# CPU Dispatches

A **CPU dispatch** is a scheme node whose body runs on the host instead of the GPU. It is
the "virtual main" idea from [Virtual Entry Points](./virtual-entry-points.md) applied to a
plain Rust function: the parameter list *is* the entry point. Every buffer parcel you bind
arrives as a whole `&[T]` or `&mut [T]` slice, followed by any scalar parameters.

```rust
use goldy::{NodeAccess, Scheme};

scheme
    .cpu_node("integrate")
    .with_parcel(&velocities, NodeAccess::Read)
    .with_parcel(&positions, NodeAccess::ReadWrite)
    .with_param(dt.to_bits())
    .dispatch(|vel: &[f32], pos: &mut [f32], dt: f32| {
        for (p, v) in pos.iter_mut().zip(vel) {
            *p += v * dt;
        }
    })?;
```

CPU dispatches exist so a host program with a Fondaco shape — a set of functions over
parcels with declared access — can move into a scheme one node at a time. Each node can
later be rewritten as a wave-based compute dispatch without touching its neighbours, because
the scheme sees the same parcels and the same access declarations either way.

## The virtual main

Any `Fn + Send + Sync + 'static` with up to sixteen [`CpuArg`](https://docs.rs/goldy/latest/goldy/cpu_dispatch/trait.CpuArg.html) parameters is a valid main:

| Parameter type | Bound by | Notes |
|---|---|---|
| `&[T]` where `T: bytemuck::Pod` | `with_parcel(.., NodeAccess::Read)` | whole parcel, `byte_size / size_of::<T>()` elements |
| `&mut [T]` where `T: bytemuck::Pod` | `with_parcel(.., Write / ReadWrite / Overwrite)` | same |
| `u32`, `i32`, `f32`, `bool` | `with_param(u32)` | wire word; `f32` via `to_bits()` |

Slice parameters come first, in `with_parcel` order, then scalars in `with_param` order.
`dispatch` validates the function against the bindings at record time and fails without
recording anything when the arity, mutability, or element size does not match.

There is no thread id and no workgroup. The function runs once per submission and sees the
complete parcel. It must not hold mutable state between submissions (it is `Fn`, not
`FnMut`); everything it needs comes through its parameters.

## Access and staging

Host visibility is a property of the **node**, not of the parcels. Bound parcels keep their
device-resident allocation; the runtime stages them around the host call:

| `NodeAccess` | Before the call | Slice contents | After the call |
|---|---|---|---|
| `Read` | device → host copy | current parcel bytes | nothing |
| `Write`, `ReadWrite` | device → host copy | current parcel bytes | host → device copy |
| `Overwrite` | nothing | zeroed | host → device copy |

Use `Overwrite` when the function produces every element; use `Write` when it touches only
some of them and the rest must keep their previous values.

Because the staging is a fence wait, a CPU dispatch is a full pipeline drain: every GPU node
it depends on has finished before it runs, and every GPU node that depends on it starts
only after its upload copy. A scheme with a CPU dispatch in the middle costs at least two
extra GPU submits and one host wait per submission. This is the intended price of the
migration path, not a steady-state design; on unified-memory backends a later pass may skip
the copies without changing the node's contract.

## What stays the same

- **Ordering.** CPU dispatches take part in the same conflict analysis as GPU nodes. Two
  CPU dispatches on disjoint parcels are independent (they still run serially on the host);
  a CPU dispatch that reads a parcel written by a compute node runs after it.
- **Cross-scheme sync.** Bound parcels are stamped like any other binding, so other schemes
  and contexts see the host's writes through the normal ledger.
- **Retention.** A clean scheme with CPU dispatches still resubmits without re-recording;
  the GPU partitions around the host node are retained as usual. The host partition itself
  is never retained.
- **Leases.** `with_lease` binds a scheme-held buffer lease the same way `with_parcel`
  binds a retained parcel.

Textures are not supported as CPU dispatch parameters in 0.2.x.
