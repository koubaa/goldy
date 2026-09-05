# Yielding Scripts

A **yielding script** is a compute shader whose lanes may *suspend* — hand a request to
the host (or to another dispatch), let the runtime service it, and *resume* later with
the answer. In Fondaco terms the lane **petitions** the runtime at a **yield point**; the
runtime **resolves** the petition and re-enters the script in a **continuation**.

Nothing about a GPU wave can actually pause, so the runtime implements this the only way
a GPU allows: the suspend point is a **dispatch boundary**. A lane that yields appends a
small record (its petition payload and whatever state it wants back) to a mailbox and
returns. After the dispatch retires, the runtime services the mailbox and launches the
continuation over the recorded lanes. You write the two halves as ordinary Slang
functions; the lowering, the mailboxes, the read-backs, and the resume dispatches are
Goldy's job.

```slang
import goldy_exp;

[goldy_petition(Result = BufRO<uint>)]
struct Fetch { uint key; };

struct St { uint lane; uint acc; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, uint scale, ThreadId tid) {
    uint v = data[tid.x];
    if (v % 2u == 1u) {
        $yield(cs_resume, Fetch { v }, St { tid.x, v * scale });
        return;
    }
    data[tid.x] = v * 2u;
}

[goldy_resume]
[numthreads(32, 1, 1)]
void cs_resume(Scattered<uint> data, Resolved<uint> r, St s, ThreadId tid) {
    data[s.lane] = r.is_null() ? 0xFFFFFFFFu : r[0] + s.acc;
}
```

```rust
use goldy::{NodeAccess, Petition, Promised, Scheme, YieldPoint};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Fetch { key: u32 }

impl Petition for Fetch {
    const SLANG_NAME: &'static str = "Fetch";
    type Result = u32;
}

let node = scheme
    .node("fetch", &pipeline)
    .with_parcel(&data, NodeAccess::ReadWrite)
    .with_param(3)
    .yield_point(
        "cs_resume",
        YieldPoint::cpu(1024, 4096, |p: &Fetch, promised: Promised<'_, u32>| {
            match table.get(&p.key) {
                Some(value) => promised.fulfil(&[*value]),
                None => promised.reject(),
            }
        }),
    )
    .dispatch(16, 1, 1);
scheme.submit()?;
let stats = scheme.yield_stats(node).unwrap();
```

The scheme sees one node. Inside it the runtime runs as many rounds as the script needs.

`dispatch(16, 1, 1)` is the prologue launch: 16 groups × `[numthreads(64,1,1)]` = 1024 lanes, which matches the mailbox **capacity**. They are chosen independently — capacity is “how many lanes may be suspended at once”, not a dispatch size — and `Backpressure::Stall` chunks a wider prologue so the live population never exceeds capacity. `arena_len` (4096) is a third knob: how many result elements all fulfilments of one round may occupy, not a lane count.

## The Slang side

Three constructs. `import goldy_exp` pulls in `Resolved<T>` and the petition helpers
when Goldy sets `GOLDY_YIELD` — automatically, for yielding scripts and for GPU
handlers that call `goldy_resolve` / `goldy_reject`:

| Construct | Meaning |
|---|---|
| `[goldy_petition(Result = BufRO<E>)] struct P { .. };` | `P` is a petition payload; a handler answers it with zero or more `E` elements. |
| `$yield(continuation, payload, state);` | Suspend this lane. `payload` is a `P`, `state` is any struct the continuation wants back. Follow it with `return;`. |
| `[goldy_resume] void c(program params.., Resolved<E> r, S s [, ThreadId tid])` | A continuation. Runs once per resumed lane. |

`Resolved<E>` is a read-only window into the yield point's result arena: `r.is_null()` is
true when the handler rejected the petition, `r.len()` is the element count, and `r[i]`
reads an element. Indexing a null view is undefined.

The `[goldy_compute]` entry is the **prologue**. Continuations take a *subset* of its
program parameters, matched **by name and type** — that is how the runtime knows which of
your bindings to hand to the resume dispatch. `payload` and `state` may be written as
`P { a, b }`, `{ a, b }`, or any expression of the right type. A continuation may `$yield`
again, to itself or to another continuation; the petition type of a continuation is
inferred from the payloads yielded to it, or spelled out as `[goldy_resume(P)]`.

Continuations declare their own `[numthreads(N, 1, 1)]` (default 64). Only `ThreadId` is
available in a continuation and it counts resumed records, not the original lanes; carry
anything you need from the prologue in the state struct.

### Restrictions in v0

- Petition and state structs hold only `uint` / `int` / `float`, fixed arrays of those,
  and nested structs of the same shape. This is what lets the host size mailboxes without
  a reflection round-trip; the Rust `Petition` type is then a `#[repr(C)]` mirror.
- Program parameters are buffers (`Scattered<T>`, `BufRO<T>`, broadcast structs) and
  scalars. No textures or samplers yet, and no `#if` inside a parameter list.
- One `[goldy_compute]` entry per source, and `$yield` must appear directly in the
  prologue or a continuation body, not in a helper.
- Indirect dispatch (`dispatch_shape_parcel`) is not supported for yielding nodes.
- `goldy_buf_len` is not portable: on Metal, WebGPU, and CUDA it currently returns
  `0xFFFFFFFF`. Pass an explicit `uint count` (or a constant) for bounds checks and
  table wraps; do not use `key % goldy_buf_len(table)` in a handler.

## The host side

`ComputePipeline::new` on a yielding script compiles the prologue *and* one entry point per
continuation; `ComputePipeline::is_yielding()` reports it. Recording differs from a plain
dispatch in one call: every continuation needs a `yield_point(name, YieldPoint)` before
`dispatch`. Missing, duplicate, or mistyped yield points are reported on `submit` as
`GoldyError::Validation`.

A [`YieldPoint`](https://docs.rs/goldy/latest/goldy/petition/struct.YieldPoint.html) has
three parts:

- **capacity** — the mailbox size: how many lanes may be suspended at this continuation
  at once.
- **arena_len** — how many `E` elements all fulfilments of one round may use together.
- **a handler** — `YieldPoint::cpu(..)` runs a Rust closure once per petition on the
  submitting thread; `YieldPoint::node(.., &pipeline)` runs a compute dispatch instead.

### CPU handlers

```rust
YieldPoint::cpu(capacity, arena_len, |p: &P, promised: Promised<'_, E>| { .. })
```

The closure receives each payload as `&P` and a `Promised<E>`. `fulfil(&[E])` copies the
elements into the arena and the continuation sees them through `Resolved<E>`; `reject()`
(or dropping the `Promised`) resumes the lane with a null view. A fulfilment that does not
fit the arena is treated as a rejection and counted in `YieldStats::arena_overflow`.

The runtime checks `P::SLANG_NAME` against the continuation's petition and
`size_of::<P>()` against the Slang struct at record time.

### GPU handlers

```rust
YieldPoint::node(capacity, arena_len, &handler_pipeline).with_parcel(&table, NodeAccess::Read)
```

The handler shader is an ordinary `[goldy_compute]` entry with the signature

```slang
void cs_main(BufRO<P> petitions, Scattered<Resolution> resolutions, Scattered<E> arena,
             /* parcels from with_parcel, in order */ uint count, ThreadId tid)
```

dispatched over `count` lanes along x. It must write one resolution per petition with
`goldy_resolve(resolutions, i, offset, len)` or `goldy_reject(resolutions, i)`, placing
results in `arena` itself. Nothing in the round touches the host.

### Backpressure

What happens when more lanes yield than `capacity`:

| Policy | Behaviour |
|---|---|
| `Backpressure::Stall` (default) | Never lose a lane. The prologue is launched in chunks of at most `capacity` lanes, each chunk drained before the next starts. Since every lane yields at most once per body, the live population never exceeds the smallest `Stall` capacity. Needs `capacity >= numthreads.x`, and — when more than one chunk is needed — a `ThreadId` parameter on the prologue, no `GroupId` / `GroupThreadId`, and a one-dimensional dispatch. |
| `Backpressure::Drop` | Launch once; lanes that find the mailbox full write nothing and their continuation never runs. Counted in `YieldStats::dropped`. |

### Statistics

`Scheme::yield_stats(node)` returns the counters of the last submission: chunks, rounds,
petitions serviced, lanes resumed, rejections, drops, and arena overflows. They are what
the tests assert on and a cheap way to see whether a capacity is sized right.

## Execution model and cost

The yielding node is recorded as a host-driven node with the user's bindings for graph
ordering. On each submission the driver:

1. clears the yield counters and launches the prologue (or one chunk of it), writing
   mailbox set A;
2. reads back the counters and the payloads of CPU-handled mailboxes;
3. services every pending continuation — CPU handlers deposit their resolution table and
   arena bytes, GPU handlers are recorded as a dispatch;
4. launches each pending continuation over its records, reading set A and writing any
   re-yields into set B;
5. repeats from 2 with the sets swapped until no lane is suspended.

Each round is one sub-scheme submission on the same context, with a fence wait in the
middle. Like a [CPU dispatch](./cpu-dispatch.md), a yielding node is therefore a full
pipeline drain, and the round count — not the lane count — is what costs. Scripts that
yield once are cheap; a traversal that yields per step pays one round per step. That is
the honest shape of the feature: it makes host round-trips *expressible* inside a scheme,
with the same parcels and access declarations as every other node, so they can later be
replaced by a device-side handler (`YieldPoint::node`) without touching the script.

## Where it fits

Yielding scripts are the Fondaco *petition* mechanism made concrete on today's GPUs. The
mailbox is the runtime's, the arena is the runtime's, and the only imperative host code is
the handler body — the scheme stays declarative and the parcels stay in trust. See
[Design Thesis](../fondaco/design-thesis.md) for the model this implements and
[CPU Dispatches](./cpu-dispatch.md) for the simpler host-visible node it builds on.
