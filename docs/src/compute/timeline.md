# Device Timeline (internal)

> **Prefer [Settlement](./settlement.md).** Goldy's public completion API is
> `is_settled` / `wait_until_settled` on submissions and parcels. Raw timeline
> counters are crate-private clearing instruments.

Internally, Goldy tracks GPU completion with a monotonic `u64` counter per context.
That clock backs settlement, deferred destruction, and exchange claim readiness.
It is not part of the public Rust, Python, FFI, or .NET surfaces.

See [Settlement](./settlement.md) for application patterns and
[Pipelined Frames](./pipelined-frames.md) for CPU/GPU depth pacing.
