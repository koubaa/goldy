# GpuFuture

`GpuFuture` represents pending GPU compute work submitted with [`ComputeEncoder::submit`](./compute.md). It lets you overlap CPU and GPU work without blocking the CPU unnecessarily.

## Creating a GpuFuture

`GpuFuture` is returned by `ComputeEncoder::submit`:

```rust
use goldy::ComputeEncoder;

let mut encoder = ComputeEncoder::new();
{
    let mut pass = encoder.begin_compute_pass();
    pass.set_pipeline(&pipeline);
    pass.set_push_constants(&[&buffer]);
    pass.dispatch(groups, 1, 1);
}

let future = encoder.submit(&device)?;
```

## Polling (Non-Blocking)

```rust
// Check if GPU is done without blocking
if future.is_complete() {
    println!("GPU finished early!");
}
```

## Blocking Wait

```rust
// Block the CPU until the GPU finishes
future.wait()?;

// Now safe to read buffer results
let results = buffer.read_to_cpu()?;
```

## Wait with Timeout

`wait_timeout` is useful for detecting hung shaders or TDR (Timeout Detection & Recovery) scenarios:

```rust
match future.wait_timeout(5000)? { // 5 second timeout
    true  => println!("GPU completed"),
    false => eprintln!("Timeout! Shader may be hung"),
}
```

Returns:
- `Ok(true)` — GPU finished before the timeout
- `Ok(false)` — Timeout elapsed, GPU still running
- `Err(_)` — Device lost

## CPU–GPU Overlap Pattern

The canonical pattern for pipelining CPU preparation with GPU compute:

```rust
// Submit frame N compute work
let future = encoder.submit(&device)?;

// Do CPU work for frame N+1 while GPU runs frame N
let next_data = prepare_next_frame();
let next_buffer = Buffer::with_data(&device, &next_data, DataAccess::Scattered)?;

// Now wait for frame N to complete before reading results
future.wait()?;

let results = buffer.read_to_cpu()?;
```

## Comparison: `dispatch` vs `submit`

| Method | Blocking? | Returns |
|--------|-----------|---------|
| `encoder.dispatch(&device)` | Yes — waits for GPU | `Result<()>` |
| `encoder.submit(&device)` | No — fire and forget | `Result<GpuFuture>` |

Use `dispatch` when simplicity matters; use `submit` when you need to overlap work.
