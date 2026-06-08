# Device Timeline

Goldy tracks GPU completion with a monotonic **timeline counter** — a `u64` value (`TimelineValue`) that increments with each submission. This replaces fence-per-submission models with a single, always-increasing counter on the device.

Timeline read/wait APIs live on [`Context`](https://docs.rs/goldy/latest/goldy/struct.Context.html), created via `device.create_context()`.

## TimelineValue

Every non-blocking submission returns a `TimelineValue`:

```rust
let ctx = device.create_context();
let tv: TimelineValue = graph.submit(&device)?;
```

This value represents a point on the device's timeline. When the GPU finishes executing that submission, the timeline advances past `tv`.

`TaskGraph::submit` returns a timeline value. Surface presentation via `Frame::present` also returns one.

## Querying GPU progress

`ctx.gpu_progress()` returns the latest completed timeline value without blocking:

```rust
let current = ctx.gpu_progress();
if current >= tv {
    // submission has finished — safe to read back results
}
```

This is a lightweight query (single atomic read on most backends) suitable for polling in a loop or checking once per frame.

## Waiting for completion

`ctx.wait_until(value)` blocks the current thread until the GPU timeline reaches at least `value`:

```rust
let tv = graph.submit(&device)?;

// CPU work while GPU executes...
prepare_next_frame();

// Block until this specific submission completes
ctx.wait_until(tv)?;
```

For bounded waits, use `wait_until_timeout`:

```rust
let completed = ctx.wait_until_timeout(tv, 1000)?; // 1 second timeout
if !completed {
    // GPU hasn't finished yet — handle timeout
}
```

## Blocking dispatch

For simple cases where you don't need CPU/GPU overlap, `dispatch` combines submit + wait:

```rust
graph.dispatch(&device)?; // submits and blocks until complete
```

`dispatch` waits internally via the device's context; you do not need a separate `Context` for that path.

## How this differs from fence-based synchronization

Traditional GPU APIs use one fence object per submission. You create a fence, attach it to a submit call, then query or wait on that specific fence. Managing multiple in-flight submissions means tracking multiple fence objects.

Goldy's timeline is a single monotonic counter shared across all submissions on a device:

| | Fence-based | Timeline-based |
|---|---|---|
| Tracking | One fence per submission | One counter for the device |
| Query | Poll each fence individually | `gpu_progress() >= value` |
| Wait | Wait on a specific fence | `wait_until(value)` |
| Ordering | Fences are independent | Values are monotonically ordered |
| Multi-frame | Track N fence objects | Compare N `u64` values |

Because timeline values are ordered, you can reason about completion transitively: if `gpu_progress() >= tv_b` and `tv_b > tv_a`, then `tv_a` has also completed.

## Practical use cases

### CPU readback after compute

```rust
let ctx = device.create_context();
let tv = graph.submit(&device)?;
ctx.wait_until(tv)?;

let result: Vec<f32> = buffer.read_data(0)?;
```

### Multi-frame pipelining

For production renderers, use [`FrameOrchestrator`](./pipelined-frames.md). It manages the in-flight slot ring, depth cap, retirement callbacks, and surface-path timeline patching with no boilerplate:

```rust
let mut orch: FrameOrchestrator<MyCleanup> = FrameOrchestrator::new(&device, 3);

loop {
    let handle = orch.begin_frame(|dev, retired| my_cleanup(dev, retired))?;
    // ... record and submit ...
    orch.end_frame_standalone(handle, &mut graph, None, cleanup)?;
}

orch.drain_all(|dev, retired| my_cleanup(dev, retired))?;
```

When you only need a one-off overlap without full frame management, the raw `TimelineValue` pattern works:

```rust
let ctx = device.create_context();
let mut pending: Option<TimelineValue> = None;

loop {
    // Wait for the previous frame to finish before reusing its resources
    if let Some(tv) = pending {
        ctx.wait_until(tv)?;
    }

    // Prepare frame N+1 on the CPU
    update_uniforms(&uniform_buffer)?;

    // Submit frame N+1 — GPU starts working, CPU continues
    let tv = graph.submit(&device)?;
    pending = Some(tv);

    // CPU work for the next iteration...
}
```

### Polling without blocking

Check completion in a non-blocking render loop:

```rust
let ctx = device.create_context();
let tv = graph.submit(&device)?;

loop {
    if ctx.gpu_progress() >= tv {
        break; // done
    }
    // do other work, yield, etc.
}
```

### Resource lifetime

Dropping a `Buffer` or `Texture` may be deferred internally: the GPU memory stays alive until all submissions that reference it have completed. Submit (or present a frame) before dropping resources that must outlive those commands.
