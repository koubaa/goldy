# VRAM Allocator

All GPU buffer and texture allocations route through the device's internal
allocator (pools call `Device::alloc_*`). Clients obtain bytes via
[`RetainedPool`](./retained.md) / [`TransientPool`](./transient-allocation.md);
the allocator itself is not a public customization point.

## Allocation Policy (Tracking and Budget)

Install a [`BudgetPolicy`](../../src/allocation_policy.rs) to track live GPU
bytes and optionally enforce a cap:

```rust
use goldy::BudgetPolicy;
use std::sync::Arc;

let policy = Arc::new(BudgetPolicy::with_budget(512 * 1024 * 1024)); // 512 MiB
device.ensure_allocation_policy(policy)?;

println!("GPU memory in use: {} bytes", device.tracked_vram_bytes());
```

Use `BudgetPolicy::new()` when you only need telemetry without a hard budget.

## Relationship to pools

- **`RetainedPool` / `TransientPool`** — recycling policy (when deeds and leases
  may be reissued after GPU retirement).
- **Device allocator + `BudgetPolicy`** — provenance and optional byte budget.

Both pools allocate through the device, so an installed `BudgetPolicy` covers
retained and transient parcels automatically.
