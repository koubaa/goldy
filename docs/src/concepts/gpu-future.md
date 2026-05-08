# Device timeline (`TimelineValue`)

Non-blocking GPU compute uses an explicit **timeline counter** (`TimelineValue`, a `u64`) instead of a heap handle.

## Submitting work

[`TaskGraph::submit`](./compute-graph.md) and [`ComputeEncoder::submit`](./compute.md) return `Result<TimelineValue>`:

```rust,ignore
let tv = graph.submit(&device)?;
device.wait_until(tv)?;
```

[`Device::gpu_progress`](../../src/device.rs) returns the latest completed value; when `gpu_progress() >= tv`, the submission has finished.

## Compared to blocking `dispatch`

| API | Blocks CPU? | Completion |
|-----|-------------|------------|
| `device.dispatch(&graph)` | Yes | N/A |
| `graph.submit(&device)` | No | `TimelineValue` + `wait_until` |
| `encoder.submit(&device)` | No | `TimelineValue` + `wait_until` |

Swapchain frames return a timeline value from [`Frame::present`](../surface.rs); use it the same way for pipelined windowed rendering.
