# Pipelined Frames

Goldy's `FrameOrchestrator<T>` manages the lifecycle of multiple in-flight GPU frames so your CPU can record frame N+1 while the GPU executes frame N, without any manually written cleanup rings or deferred-timeline-patching code.

## The problem it solves

Every pipelined renderer needs the same bookkeeping:

1. A ring of in-flight frame slots, each holding per-frame GPU resources.
2. A pipeline-depth cap — block the CPU when the ring is full to prevent unbounded memory growth.
3. Deferred retirement — pop completed slots from the front when `gpu_progress() >= epoch`.
4. Surface-path timeline patching — the epoch is only known *after* `Frame::present()`, so the most recent slot must be stamped retroactively.

Without shared infrastructure, every consumer reimplements this independently. `FrameOrchestrator` centralizes all of it.

## Core API

```rust
use goldy::{FrameOrchestrator, FrameHandle};

// max_depth: how many frames may be in-flight before begin_frame blocks
let mut orch: FrameOrchestrator<MyCleanup> = FrameOrchestrator::new(&device, 3);
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

    // 2. Record compute work.
    let mut graph = TaskGraph::new();
    // ... add dispatches ...

    // 3. Collect per-frame resources that should live until the GPU is done.
    let cleanup = MyCleanup { /* views, buffers, etc. */ };

    // 4. Submit & register the slot.  Returns the timeline value.
    let tv = orch.end_frame_standalone(handle, graph, None, cleanup)?;
}

// Shutdown: drain all remaining slots.
orch.drain_all(|dev, retired| my_cleanup(dev, retired.timeline, retired.data))?;
```

### Surface (swapchain) path

```rust
loop {
    let frame = surface.begin()?;

    let handle = orch.begin_frame(|dev, retired| {
        my_cleanup(dev, retired.timeline, retired.data)
    })?;

    let mut graph = TaskGraph::new();
    // ... write into frame.texture() ...

    let cleanup = MyCleanup { /* ... */ };

    // Submit compute into the frame bracket; timeline is unknown until present.
    orch.end_frame_for_surface(handle, &graph, &frame, cleanup)?;

    // Present; stamp the pending slot with the returned epoch.
    let tv = frame.present()?;
    orch.note_presented(tv);
}

orch.drain_all(|dev, retired| my_cleanup(dev, retired.timeline, retired.data))?;
```

`note_presented` fills in the `None` timeline on the most recent slot pushed by `end_frame_for_surface`. If it is never called (e.g. the window closes before present), `drain_all` falls back to the internal high-water timeline as a safe fence.

## Mid-frame flush

`flush()` submits the current graph and starts a fresh one, letting the GPU begin earlier phases while the CPU records later ones:

```rust
let handle = orch.begin_frame(|dev, retired| { /* ... */ })?;

let mut graph = TaskGraph::new();
let mut last_tv = None;

// Coarse phase
record_coarse(&mut graph);
orch.flush(handle, &mut graph, None, &mut last_tv)?;

// Fine phase — GPU executes coarse while CPU records this
record_fine(&mut graph);

let cleanup = MyCleanup { /* ... */ };
let tv = orch.end_frame_standalone(handle, graph, last_tv, cleanup)?;
```

For surface frames pass `Some(&frame)` instead of `None`:

```rust
orch.flush(handle, &mut graph, Some(&frame), &mut last_tv)?;
```

### Transient resources across flush boundaries

Each `flush()` call uses `Device::submit_pipelined` rather than `Device::submit`. The difference matters for graphs that contain transient buffers or textures:

| Method | Transient heap behaviour |
|---|---|
| `Device::submit` | Blocks the CPU until the GPU finishes so the placement heap region is immediately reclaimable |
| `Device::submit_pipelined` | Returns immediately; the placement heap region stays in flight and is reclaimed once `gpu_progress()` advances past the returned timeline |

This means transient-buffer graphs can now be flushed mid-frame. Each flush boundary acquires its own placement heap region, so coarse-phase transients and fine-phase transients coexist without aliasing.

## `Device::submit_pipelined`

`submit_pipelined` is also available as a standalone method when you want non-blocking transient submit without the full orchestrator:

```rust
// Blocks until transients are reclaimable:
let tv = device.submit(&graph)?;

// Does not block — region stays in flight until tv retires:
let tv = device.submit_pipelined(&graph)?;
```

Only use `submit_pipelined` when you have a mechanism (such as `FrameOrchestrator`, or your own tracking ring) to ensure you don't reuse the placement heap region before the GPU is done with it.

## CPU/GPU overlap

`FrameOrchestrator` enables two distinct layers of CPU/GPU overlap:

**Frame-level** — `begin_frame` retires completed slots without blocking, so the CPU immediately starts recording frame N+1 while the GPU executes frame N. The depth cap (`max_depth`) prevents the CPU from running too far ahead.

**Intra-frame** — `flush()` splits a single frame's command stream into multiple submissions. The GPU starts executing the first submission before the CPU finishes recording the last one.

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

### Surface path timeline is always deferred

On the swapchain path the timeline value comes from `Frame::present`, which fires *after* `end_frame_for_surface`. The orchestrator holds the slot in a `timeline: None` state until `note_presented` arrives. The `EpochRegions` transient allocator documents the same invariant — `end_frame` may legally arrive after the next `begin_frame`.

### Relationship to `TransientAllocator`

`FrameOrchestrator` owns the frame-slot ring and retirement callbacks. `TransientAllocator` owns the per-frame bump region and advances its epoch via `begin_frame` / `end_frame`. They are independent: the orchestrator does not call into the allocator. Call `allocator.begin_frame()` before recording and `allocator.end_frame(tv)` in your retirement closure — or immediately after the standalone submit where `tv` is known synchronously.
