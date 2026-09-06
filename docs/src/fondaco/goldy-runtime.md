# Goldy Runtime Mapping

**Status**: Implementation note for Goldy 0.2.x. Claims use the labels in [Terminology](./terminology.md).

How Goldy realizes the Fondaco machine from [Machine Specification](./specification.md). Goldy is *a* runtime, not the machine. Where this chapter disagrees with the spec, the spec governs.

Hardware terms (GPU, Vulkan, Metal, DX12, shader, fence) appear because Goldy must speak them. They have no normative meaning in the Fondaco machine.

For *why* Goldy looks this way, see [Design Thesis](./design-thesis.md). For day-to-day usage, start at the [Introduction](../introduction.md).

## 1. Goldy realizes a Fondaco machine

**Shipped.** Goldy is a Rust library that admits dispatches, honors scheme partial orders, maintains parcel identity across physical activity, and acts at gates as the machine requires.

It targets 2020-era heterogeneous compute: a host processor plus one or more GPUs via Vulkan 1.4+, DX12, or native Metal (macOS).

The spec's runtime is a single agent. Goldy splits it into:

- **Host** (Rust): parcel identity, schemes, ledger analysis, gates, exchanges
- **Device** (GPU queue): executes admitted dispatches

That split is a substrate artifact, not a machine requirement.

## 2. Status overview

| Machine concept | Goldy realization | Status |
|-----------------|-------------------|--------|
| Scheme | `Scheme` + internal `GraphIR` | **Shipped** |
| Dispatch | Compute / render / copy / clear / present nodes | **Shipped** |
| CPU dispatch | `Scheme::cpu_node` — serial host function over staged parcels | **Shipped** (correctness first; staging always copies) |
| Script | Slang via `[goldy_*]` virtual entry points | **Shipped** |
| Parcel | `Parcel`, `Buffer`, `Texture` (stable handles) | **Shipped** |
| Ownership / claims | `NodeAccess` → derived precedences | **Shipped** |
| Ledger | Cross-submission sync (`ParcelStamp`, timeline) | **Shipped** (internal) |
| Gate | Submission gate, `Context::boundary_crossed` | **Shipped** |
| Exchange | `SurfaceExchange`, `MemoryExchange`, `PixelExchange` | **Shipped** |
| Exchange claim | `Claim`, `WithdrawClaim`, `PixelClaim` → `consume` / `discard` | **Shipped** |
| Warehouse / budget | `BudgetPolicy`, `VramAllocator` | **Shipped** (partial) |
| Growable buffers | `Buffer::resize_to`, stable handles | **Shipped** |
| Retained resubmit | Clean schemes replay with zero re-record | **Shipped** |
| Compute-to-surface | `SurfaceExchange::bind_destination` | **Shipped** |
| Pipelined frames | `FrameOrchestrator`, surface depth | **Shipped** |
| Yielding scripts / `$yield` | Slang intrinsic + petition servicing | **Designed** |
| Scheme fusion (mega-kernel) | Merge adjacent dispatches | **Designed** |
| Scheme splitting (wavefront) | Split at yield points | **Designed** |
| Defragmentation | `VramAllocator::defragment` | **Designed** |
| Memory-pressure events | `MemoryPressureEvent` | **Designed** |
| Promise / continuation API | Indirect continuation dispatch | **Designed** |
| WASI host (`goldy-host`) | GPU to WASM guests | **Speculative** |
| Pre-2020 bindless-free backend | Traditional binding backend | **Speculative** |

## 3. Dispatches

**Shipped.** A Goldy dispatch is a compute or graphics submission to the accelerator.

- **Script**: Slang compiled through `virtual_main` to SPIR-V, DXIL, or Metal IR. Goldy fixes Slang; the machine does not require it. See [Virtual Entry Points](../programming-model/virtual-entry-points.md).
- **Execution**: Workgroup grid (threadgroups on Metal). `dispatch_indirect` where the backend allows.
- **Claims**: `NodeAccess` on scheme nodes — read, write, read-write — mapped to public / private / private-inaugural ownership.

Non-computing dispatches also **Shipped**: buffer copy, buffer write, texture upload, buffer clear, present / copy-to-swapchain.

**CPU dispatches** (**Shipped**, 0.2.x): `Scheme::cpu_node` admits a serial host function whose parameter list is the virtual main (`&[T]` / `&mut [T]` per bound parcel, then scalars). The machine does not distinguish where a dispatch executes; Goldy realizes host execution by staging bound parcels through readback/upload copies around a fence wait, so the node is a full drain of the device pipeline. See [CPU Dispatches](../programming-model/cpu-dispatch.md).

Goldy does not preserve shader invocation identity across dispatch gates; logical threads must persist state in parcels.

## 4. Parcels and bindless internals

**Shipped.** A Goldy parcel is a **stable handle**. Programs never author raw `(category, index)` bindless slots.

Bindless descriptor indexing is **backend-internal**:

- **Rust**: Public types are `Parcel`, `Buffer`, `Texture`, `Scheme`, exchanges. Bindless resolution happens at scheme record / submit.
- **Slang**: Typed parameters (`Scattered<T>`, `BufRO<T>`, …). `virtual_main` generates slot packing.

Program-visible are access-pattern **categories** (Layer B): `Scattered`, `BufRO`, `Broadcast`, `Interpolated`, `DirectSpatial`, `Filter`. See [Parcels](../programming-model/parcels.md) and [Design Thesis](./design-thesis.md).

### Parcel identity and reslot

**Shipped.** Identity is the handle, not the descriptor slot. Handles stay stable across physical growth (`Buffer::resize_to`), transient aliasing within an epoch, and backend pool rotation.

When backing changes (**reslot**), Goldy:

1. Keeps the handle unchanged
2. Gives the new allocation a **new** descriptor slot; old slots remain valid until in-flight work retires
3. Invalidates retained command buffers that embedded stale slots

This follows DX12 / Vulkan / Metal descriptor versioning. See [Buffers](../resources/buffers.md) and [VRAM Allocator](../resources/vram-allocator.md).

## 5. Warehouse and memory

**Shipped (partial).** Physical medium is managed by `VramAllocator`, `RetainedPool`, and `TransientPool`. See [RetainedPool and Parcel](../resources/retained-pool.md) and [Transient Allocation](../resources/transient-allocation.md).

Goldy distinguishes three quantities:

| Quantity | Owner | Meaning |
|----------|-------|---------|
| **Logical warehouse** | Program + runtime | Sum of parcel extents (Fondaco warehouse) |
| **Committed** | Runtime | Bytes handed out (commit charge) |
| **Resident** | OS | Bytes in the fast tier now |

Budget enforcement keys on **committed**. **Resident** enters reactively via OS memory-pressure signals.

**Shipped** residency models per backend: `ManagedAllocation` (discrete Vulkan / DX12), `PageOnFault` (Apple Metal), plus capability queries for resize cost (`Constant`, `PageBind`, `Copy`).

**Designed**: defragmentation, proactive memory-pressure petition delivery at gates.

## 6. Exchanges

**Shipped.** Primary exchange: **surface presentation**.

```rust
let transaction = surface_exchange.bind_render_target(&mut scheme, &scene_rt)?;
let mut submission = scheme.submit()?;
let claim = transaction.claim(&mut submission)?;
claim.consume()?; // present
```

- Binding does not acquire a drawable; acquire runs at submit when the partition needs it
- `Claim::consume` is terminal
- The program never passes raw GPU addresses to the compositor

**Shipped** CPU readback: `MemoryExchange` with `WithdrawTransaction` / `WithdrawClaim`. See [Settlement](../compute/settlement.md) and [Compute to Surface](../compute/compute-to-surface.md).

**Shipped** pixel blit: `PixelExchange` withdraws a buffer pixmap and
`PixelClaim::consume` copies it into a `PixelSink` that is not a Goldy device
(`HostPixelSink`, or `foreign::vulkan` offscreen). See [Pixel Exchange](../surfaces/pixel-exchange.md).
The CPU backend stays compute-only; graphics is a foreign singleton behind the
exchange verb.

**Designed**: video-encoder exchange (foreign read continues after enqueue);
windowed WSI on the foreign Vulkan/DX12/Metal singleton.

## 7. Schemes and GraphIR

**Shipped.** Public type: `Scheme`. Internally Goldy holds **GraphIR** — nodes, ownership-derived edges, wave / partition analysis, retention fingerprints.

On `Scheme::submit`:

- Dependency analysis inserts barriers
- Transient regions are colored for aliasing
- Partitions may acquire exchange backing
- Retained command buffers replay when bindings are unchanged

**Designed** scheme transformations (spec §8): fusion (mega-kernel), splitting at yield points (wavefront), dead-dispatch elision beyond basic analysis.

Goldy **refuses ill-formed schemes** (conflicting unordered private claims) rather than producing unspecified results — a deliberate narrowing of spec latitude.

## 8. Gates and ordering

**Shipped.** A gate is the interval between `Scheme::submit` calls and retirement via `Context::boundary_crossed(T)`.

At a gate Goldy may reclaim deferred allocations, flush VRAM deferred rings, and service timeline signals.

**Shipped** cross-submission ordering: the runtime enforces ledger precedences across schemes on the same `Context`, using GPU barriers or host waits as needed. Clients must not assume which lever is used.

Pipeline depth (in-flight submissions) is **client pacing** — surface depth, `FrameOrchestrator`, when to `consume` claims. See [Pipelined Frames](../compute/pipelined-frames.md).

## 9. Scripts: non-yielding today

**Shipped.** Public shaders today are non-yielding: `[goldy_compute]`, `[goldy_vertex]`, `[goldy_fragment]`.

**Designed** yielding scripts:

- `$yield` intrinsic in the virtual-entry-point transform
- Script-state preservation (register spin-wait, workgroup-local, or parcel-backed)
- Yield-point petitions (limited runtime power, not full gate powers)

## 10. Calling conventions

**Shipped:**

- **Slang** as sole script language — [Slang in One Source](../programming-model/slang.md)
- **Virtual entry points** — typed parameters; `virtual_main` generates platform wrappers
- **Push-constant layout** — bindless indices + scalars prepended per dispatch (backend-internal)
- **Access categories** — validated at scheme record time

**Designed**: `$yield` petition descriptors, promise / continuation bindless category, paged-parcel fault servicing.

Portable programs depend on typed access categories and scheme structure, not on bindless heap layout.

## 11. Where Goldy constrains the spec

Deliberate restrictions for modern desktop / laptop workloads:

- Refuses ill-formed schemes
- Fixed Slang scripts
- Closed typed-access category set
- Workgroup-grid execution model
- Single accelerator queue per device (heterogeneous multi-queue: **Designed**)
- 2020+ hardware floor (Vulkan 1.4+, DX12 Enhanced Barriers, Metal Tier 2+)

For older hardware or maximum portability, use **wgpu**. See [Goldy vs wgpu](../design/comparison.md) and [Target Hardware](../design/hardware.md).

## 12. Abstract the medium, expose cost

**Normative for Goldy design.**

- **Layer A (medium)**: VRAM, residency, relocation, descriptor slots — abstracted; runtime-owned
- **Layer B (cost)**: Registers, occupancy, coalescing, access patterns, first-touch latency — exposed and queryable

Goldy must not present Layer A operations as uniform-cost or hide them entirely. Access-pattern types (`Scattered` vs `Broadcast` vs `Interpolated`) exist because hardware treats them differently.

Capability queries report backend, residency model, resize cost, zero-copy readback, and optional features honestly.

## Appendix: Fondaco ↔ Goldy

| Fondaco term | Goldy / GPU analogue |
|--------------|----------------------|
| Scheme | `Scheme`, `GraphIR` |
| Dispatch | Kernel launch, draw / dispatch command |
| Script | Slang shader |
| Parcel | `Buffer` / `Texture` handle |
| Merchant | Program |
| Exchange | `SurfaceExchange`, `MemoryExchange`, `PixelExchange` |
| Claim (exchange) | `Claim`, `WithdrawClaim` |
| Gate | Fence epoch, `boundary_crossed` |
| Warehouse | `BudgetPolicy`, `VramAllocator` |
| Ledger | Cross-submit sync analysis |

Analogues are not equivalences. A scheme is the program's computation, not merely a scheduling artifact. An exchange preserves program sovereignty over parcels; a raw swapchain handle does not.

Full vocabulary: [Terminology](./terminology.md).
