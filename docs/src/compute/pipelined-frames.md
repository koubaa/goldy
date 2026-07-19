# Pipelined Frames

Goldy's `FrameOrchestrator<T>` manages the lifecycle of multiple in-flight GPU frames so your CPU can record frame N+1 while the GPU executes frame N, without any manually written cleanup rings or deferred-timeline-patching code.

## The problem it solves

Every pipelined renderer needs the same bookkeeping:

1. A ring of in-flight frame slots, each holding per-frame GPU resources.
2. A pipeline-depth cap — block the CPU when the ring is full to prevent unbounded memory growth.
3. Deferred retirement — pop completed slots from the front when `gpu_progress() >= epoch`.
4. Present-path timeline patching — the epoch is only known *after* [`Claim::consume`](https://docs.rs/goldy/latest/goldy/struct.Claim.html), so the most recent slot must be stamped retroactively.

Without shared infrastructure, every consumer reimplements this independently. `FrameOrchestrator` centralizes all of it.

## Core API

```rust
use goldy::{FrameOrchestrator, FrameHandle};

// max_depth: how many frames may be in-flight before begin_frame blocks
let mut orch: FrameOrchestrator<MyCleanup> = FrameOrchestrator::new(&ctx, 3);
```

`T` is your per-frame payload type — whatever data you need to clean up when a slot retires (buffer views, textures, readback buffers, etc.).

### Standalone (headless / render-to-texture) path

```rust
loop {
    // 1. Open a new frame slot; retires completed older slots via your closure.
    //    Blocks if max_depth frames are already in flight.
    let handle = orch.begin_frame(|dev, retired| {
        my_cleanup(dev, retired.timeline, retired.data)
    })?;

    // 2. Submit retained scheme work (recorded earlier or this frame).
    let submission = scheme.submit()?;

    // 3. Collect per-frame resources that should live until the GPU is done.
    let cleanup = MyCleanup { /* views, buffers, etc. */ };

    // 4. Register the slot with the submission timeline.
    orch.end_frame_standalone(handle, submission.timeline_value(), cleanup)?;
}
```

### Present-on-scheme (swapchain) path

```rust
loop {
    let handle = orch.begin_frame(|dev, retired| {
        my_cleanup(dev, retired.timeline, retired.data)
    })?;

    let mut submission = scheme.submit()?;
    present.claim(&mut submission)?.consume()?;

    let cleanup = MyCleanup { /* ... */ };

    // Register slot; timeline from present may arrive via note_presented.
    orch.end_frame_for_present(handle, submission.timeline_value(), cleanup)?;
}
```

`note_presented` fills in the `None` timeline on the most recent slot when the present epoch differs from the submit epoch. If it is never called (e.g. the window closes before present), `drain_all` falls back to the internal high-water timeline as a safe fence.

## Mid-frame submit boundaries

Split a frame into multiple scheme submissions so the GPU can begin earlier phases while the CPU records later ones:

```rust
let handle = orch.begin_frame(|dev, retired| { /* ... */ })?;

// Coarse phase
let coarse = coarse_scheme.submit()?;

// Fine phase — GPU executes coarse while CPU records/submits this
let fine = fine_scheme.submit()?;

let cleanup = MyCleanup { /* ... */ };
orch.end_frame_standalone(handle, fine.timeline_value(), cleanup)?;
```

Each `Scheme::submit` creates a real command-buffer boundary on all backends. Because Metal (and Vulkan/DX12) execute command buffers on the same queue in submission order, the fine submission automatically waits for the coarse one — no explicit fence is required.

### Transient resources across submit boundaries

Scheme submits are non-blocking by default. Transient buffers and textures allocated within a scheme stay in flight until `gpu_progress()` advances past the returned timeline value. When splitting work across multiple submits in one frame, each boundary acquires its own placement heap region so coarse-phase and fine-phase transients coexist without aliasing.

## CPU/GPU overlap

`FrameOrchestrator` enables two distinct layers of CPU/GPU overlap:

**Frame-level** — `begin_frame` retires completed slots without blocking, so the CPU immediately starts recording frame N+1 while the GPU executes frame N. The depth cap (`max_depth`) prevents the CPU from running too far ahead.

**Intra-frame** — multiple `scheme.submit()` calls in one frame split the command stream into multiple GPU submissions. The GPU starts executing the first submission before the CPU finishes the last one.

## Inspecting orchestrator state

```rust
orch.pending_frames();   // slots currently in the ring
orch.max_depth();        // cap configured at construction
orch.has_open_frame();   // true between begin_frame and end_frame_*
```

## Design notes

### Retirement callback

`begin_frame`, `reclaim`, and `drain_all` all accept a fallible closure `FnMut(&Device, RetiredFrame<T>) -> Result<(), E>`. The orchestrator converts errors into `GoldyError`. This keeps the orchestrator generic over your cleanup payload while allowing cleanup itself to fail.

`RetiredFrame<T>` carries:

```rust
pub struct RetiredFrame<T> {
    pub timeline: TimelineValue,  // epoch at which this frame's GPU work completed
    pub data: T,                  // your per-frame payload
}
```

### Present path timeline is always deferred

On the swapchain path the final scanout timeline may arrive only after `Claim::consume`. The orchestrator holds the slot in a `timeline: None` state until `note_presented` arrives. The `Heap` transient allocator documents the same invariant — `end_frame` may legally arrive after the next `begin_frame` (mid-frame frees are stamped in `end_frame`).

### Relationship to `TransientAllocator`

`FrameOrchestrator` owns the frame-slot ring and retirement callbacks. `TransientAllocator` owns the per-frame bump region and advances its epoch via `begin_frame` / `end_frame`. They are independent: the orchestrator does not call into the allocator. Call `allocator.begin_frame()` before recording and `allocator.end_frame(tv)` in your retirement closure — or immediately after the standalone submit where `tv` is known synchronously.
