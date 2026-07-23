# Settlement

Goldy makes GPU completion observable as **settlement** of concrete objects — submissions,
parcels, and exchange claims — not as raw timeline numbers.

Internal clearing still uses a monotonic device clock. That clock is crate-private.
Clients wait for *work* or *resources* to settle.

## Submission settlement

Every successful [`Scheme::submit`](https://docs.rs/goldy/latest/goldy/struct.Scheme.html)
returns a [`Submission`](https://docs.rs/goldy/latest/goldy/struct.Submission.html):

```rust
let submission = scheme.submit()?;

if !submission.is_settled() {
    submission.wait_until_settled()?;
}
```

Bounded wait:

```rust
let done = submission.wait_until_settled_timeout(1000)?; // milliseconds
if !done {
    // GPU has not finished yet
}
```

The submission owns the context it was submitted on; callers do not pass a `Context` to wait.

## Parcel and resource settlement

Before reusing or dropping a resource that may still be referenced by in-flight GPU work:

```rust
if !parcel.is_settled() {
    parcel.wait_until_settled()?;
}
```

The same methods exist on [`Buffer`](https://docs.rs/goldy/latest/goldy/struct.Buffer.html)
and [`Texture`](https://docs.rs/goldy/latest/goldy/struct.Texture.html).

Direct host writes on CPU-writable buffers require the buffer to be settled (or never
GPU-referenced). Prefer [`MemoryExchange`](https://docs.rs/goldy/latest/goldy/struct.MemoryExchange.html) deposits for uploads.

## Exchange claims (unchanged)

Surface and memory exchanges still settle occurrences via consume/discard:

```rust
let mut submission = scheme.submit()?;

// Present
transaction.claim(&mut submission)?.consume()?;

// Readback — consume waits for the submission internally
let bytes = withdraw.claim(&mut submission)?.consume()?;
```

A live linear claim is unsettled until `consume` or `discard`. Dropping an unsettled claim
discards it.

## Multi-frame pipelining

For production renderers, use [`FrameOrchestrator`](./pipelined-frames.md). It bounds
CPU/GPU depth using submissions — not raw epochs:

```rust
let mut orch = FrameOrchestrator::new(&ctx, 3);

loop {
    let handle = orch.begin_frame()?;
    let submission = scheme.submit()?;
    orch.end_frame_standalone(handle, &submission)?;
}

orch.drain_all()?;
```

## How this differs from fence-based APIs

Traditional GPU APIs expose fence objects or timeline counters to the application.
Goldy keeps those as runtime clearing instruments (finance analogy: sequence numbers in
a clearinghouse). Application code holds **receipts** (`Submission`) and **property**
(`Parcel`) and asks when those are settled.

| | Fence / timeline counter | Settlement |
|---|---|---|
| Query | Poll a fence or compare `u64` | `obj.is_settled()` |
| Wait | Wait on fence / `wait_until(tv)` | `obj.wait_until_settled()` |
| Identity | Opaque fence or epoch number | Concrete submission or parcel |
| Portability | Tied to native timeline primitives | Backend may use fences, events, or `onSubmittedWorkDone` |

## Resource lifetime

Dropping a `Buffer` or `Texture` may be deferred internally until GPU work that referenced
it has retired. Prefer settling before dropping when you need deterministic reclaim timing
(for example Metal heap-sensitive resize paths).
