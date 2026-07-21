# Pipelined Frames

Goldy's `FrameOrchestrator` manages CPU/GPU frame pacing: an in-flight timeline
ring, a depth cap, and present-path timeline patching. It does **not** own GPU
bytes or run cleanup callbacks — recycle lives in [`TransientPool`](../resources/transient-allocation.md)
and [`RetainedPool`](../resources/pooling.md).

## The problem it solves

Every pipelined renderer needs the same bookkeeping:

1. A ring of in-flight frame timelines.
2. A pipeline-depth cap — block the CPU when the ring is full to prevent unbounded memory growth.
3. Deferred retirement of slots when `gpu_progress() >= epoch`.
4. Present-path timeline patching — the epoch is only known *after* [`Claim::consume`](https://docs.rs/goldy/latest/goldy/struct.Claim.html), so the most recent slot must be stamped retroactively.

Without shared infrastructure, every consumer reimplements this independently. `FrameOrchestrator` centralizes the pacing half of it.

## Core API

```rust
use goldy::{FrameOrchestrator, FrameHandle};

// max_depth: how many frames may be in-flight before begin_frame blocks
let mut orch = FrameOrchestrator::new(&ctx, 3);
```

### Standalone (headless / render-to-texture) path

```rust
loop {
    // 1. Open a new frame slot; drains completed older slots.
    //    Blocks if max_depth frames are already in flight.
    let handle = orch.begin_frame()?;

    // 2. Submit retained scheme work (recorded earlier or this frame).
    let submission = scheme.submit()?;

    // 3. Register the slot with the submission timeline.
    orch.end_frame_standalone(handle, submission.timeline_value())?;
}
```

### Present-on-scheme (swapchain) path

```rust
loop {
    let handle = orch.begin_frame()?;

    let mut submission = scheme.submit()?;
    present.claim(&mut submission)?.consume()?;

    // Register slot; timeline from present may arrive via note_presented.
    orch.end_frame_for_present(handle, submission.timeline_value())?;
}
```

`note_presented` fills in the `None` timeline on the most recent slot when the present epoch differs from the submit epoch. If it is never called (e.g. the window closes before present), `drain_all` falls back to the internal high-water timeline as a safe fence.

### Externally ordered path

When scheme submit sidecars / present easement already enforce cross-frame
ordering, close with `end_frame_externally_ordered` so no ring slot is created
and the next `begin_frame` does not wait on a coarse frame timeline.

## Mid-frame submit boundaries

Split a frame into multiple scheme submissions so the GPU can begin earlier phases while the CPU records later ones:

```rust
let handle = orch.begin_frame()?;

// Coarse phase
let coarse = coarse_scheme.submit()?;

// Fine phase — GPU executes coarse while CPU records/submits this
let fine = fine_scheme.submit()?;

orch.end_frame_standalone(handle, fine.timeline_value())?;
```

Each `Scheme::submit` creates a real command-buffer boundary on all backends. Because Metal (and Vulkan/DX12) execute command buffers on the same queue in submission order, the fine submission automatically waits for the coarse one — no explicit fence is required.

## CPU/GPU overlap

`FrameOrchestrator` enables two distinct layers of CPU/GPU overlap:

**Frame-level** — `begin_frame` drains completed slots without blocking when under the depth cap, so the CPU can start recording frame N+1 while the GPU executes frame N. The depth cap (`max_depth`) prevents the CPU from running too far ahead.

**Intra-frame** — multiple `scheme.submit()` calls in one frame split the command stream into multiple GPU submissions. The GPU starts executing the first submission before the CPU finishes the last one.

## Inspecting orchestrator state

```rust
orch.pending_frames();   // slots currently in the ring
orch.max_depth();        // cap configured at construction
orch.has_open_frame();   // true between begin_frame and end_frame_*
```

## Design notes

### Present path timeline is always deferred

On the swapchain path the final scanout timeline may arrive only after `Claim::consume`. The orchestrator holds the slot in a `timeline: None` state until `note_presented` arrives.

### Relationship to resource recycling

`FrameOrchestrator` owns the frame-slot ring only. Transient buffer/texture recycling lives in the per-context [`TransientPool`](../resources/transient-allocation.md) (leases via `acquire_transient_*` / `return_transient_*`). They are independent: the orchestrator does not call into the pool, and clients must not hang byte reclaim on orchestrator callbacks (there are none).
