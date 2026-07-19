# Device Timeline

Goldy tracks GPU completion with a monotonic **timeline counter** — a `u64` value (`TimelineValue`) that increments with each submission. This replaces fence-per-submission models with a single, always-increasing counter on the device.

Timeline read/wait APIs live on [`Context`](https://docs.rs/goldy/latest/goldy/struct.Context.html), created via `device.create_context()`.

## TimelineValue

Every scheme submission returns a `TimelineValue` via [`Submission::timeline_value`](https://docs.rs/goldy/latest/goldy/struct.Submission.html):

```rust
let ctx = device.create_context();
let mut scheme = Scheme::new(&ctx);
// ... record nodes ...
let submission = scheme.submit()?;
let tv: TimelineValue = submission.timeline_value();
```

This value represents a point on the device's timeline. When the GPU finishes executing that submission, the timeline advances past `tv`.

Present-on-scheme paths also stamp a timeline when [`Claim::consume`](https://docs.rs/goldy/latest/goldy/struct.Claim.html) completes scanout.

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
let submission = scheme.submit()?;

// CPU work while GPU executes...
prepare_next_frame();

// Block until this specific submission completes
ctx.wait_until(submission.timeline_value())?;
```

Or call `submission.wait(&ctx)?` directly.

For bounded waits, use `wait_until_timeout`:

```rust
let completed = ctx.wait_until_timeout(tv, 1000)?; // 1 second timeout
if !completed {
    // GPU hasn't finished yet — handle timeout
}
```

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
let grant = scheme.grant_read(&buffer);
let submission = scheme.submit()?;
submission.wait(&ctx)?;

let result = grant.consume(&submission)?;
```

### Multi-frame pipelining

For production renderers, use [`FrameOrchestrator`](./pipelined-frames.md). It manages the in-flight slot ring, depth cap, retirement callbacks, and present-path timeline patching with no boilerplate:

```rust
let mut orch: FrameOrchestrator<MyCleanup> = FrameOrchestrator::new(&ctx, 3);

loop {
    let handle = orch.begin_frame(|dev, retired| my_cleanup(dev, retired))?;
    let submission = scheme.submit()?;
    orch.end_frame_standalone(handle, submission.timeline_value(), cleanup)?;
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
    let submission = scheme.submit()?;
    pending = Some(submission.timeline_value());

    // CPU work for the next iteration...
}
```

### Polling without blocking

Check completion in a non-blocking render loop:

```rust
let ctx = device.create_context();
let submission = scheme.submit()?;
let tv = submission.timeline_value();

loop {
    if ctx.gpu_progress() >= tv {
        break; // done
    }
    // do other work, yield, etc.
}
```

### Resource lifetime

Dropping a `Buffer` or `Texture` may be deferred internally: the GPU memory stays alive until all submissions that reference it have completed. Submit (or consume a present grant) before dropping resources that must outlive those commands.
