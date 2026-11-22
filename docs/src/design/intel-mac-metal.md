# Intel and Discrete Macs on Metal

**Status**: Proposal (**Designed**, not shipped). Tracks [#330](https://github.com/koubaa/goldy/issues/330); reviews [#333](https://github.com/koubaa/goldy/pull/333). The full per-site inventory is in [Metal on Intel and AMD Macs: Audit](./metal-discrete-audit.md).

Goldy's Metal backend aborts on every Intel Mac at device creation:

```text
-[MTLHeapDescriptorInternal validateWithDevice:]: failed assertion `Heap Descriptor Validation
Shared storage mode disallowed'
```

This chapter explains why, describes the placement contract the other backends already follow, and proposes mapping Metal onto that contract instead of adding a Metal-only storage plan. The proposal is staged so the abort can be fixed first without locking in the wrong default.

## 1. What Metal allows

| Fact | Source |
|------|--------|
| Shared `MTLHeap`s exist only on Apple silicon. Every other Mac can create **Private** heaps only. | [`MTLHeapDescriptor.storageMode`](https://developer.apple.com/documentation/metal/mtlheapdescriptor/storagemode) |
| `hasUnifiedMemory` is "typically an integrated GPU". Intel Iris / UHD report `true`. | [`hasUnifiedMemory`](https://developer.apple.com/documentation/metal/mtldevice/hasunifiedmemory); device logs in [llama.cpp#9873](https://github.com/ggml-org/llama.cpp/issues/9873), [llama.cpp#11463](https://github.com/ggml-org/llama.cpp/issues/11463) |
| Using `hasUnifiedMemory` as a proxy for "Apple silicon" aborts on Intel iGPUs. | [gogpu/wgpu#271](https://github.com/gogpu/wgpu/issues/271) (texture storage-mode abort on Iris Plus 655) |
| Shared *buffers* are legal on every Mac. On a discrete GPU they live in system memory and the GPU reads them over PCIe. Private resources live in VRAM. | [Choosing a resource storage mode for Intel and AMD GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-intel-and-amd-gpus) |
| Managed resources are a CPU/GPU pair that needs `didModifyRange:` / `synchronizeResource:`. Heaps never accept Managed. | Same Apple guide |

Two device facts matter, and they are independent:

| Device | Apple GPU family | `hasUnifiedMemory` | Shared heaps | Shared textures |
|--------|------------------|--------------------|--------------|-----------------|
| Apple silicon (M1+) | yes | true | yes | yes |
| Intel iGPU (Iris, UHD) | no | **true** | **no** | no |
| AMD discrete / Intel Mac default high-performance GPU | no | false | no | no |
| GitHub `macos-15-intel` runner (Apple Paravirtual device) | no | false | no heaps at all, no Tier 2 | n/a |

The legality of Shared heaps follows the **Apple GPU family**, not `hasUnifiedMemory`. Whether host-visible memory costs nothing for the GPU follows `hasUnifiedMemory`.

Goldy defaults to `PowerPreference::HighPerformance` ([`src/runtime.rs`](../../src/runtime.rs)), so a dual-GPU MacBook Pro picks the AMD GPU. A single-GPU Intel MacBook (Air, 13" Pro), `PowerPreference::LowPower`, and tests that call `preferring(DeviceType::IntegratedGpu)` get the Intel iGPU.

## 2. The placement contract Goldy already has

Goldy has no explicit placement enum. The contract lives in `BufferKind` + `BufferFlags` intent ([`src/types.rs`](../../src/types.rs)) and in how each backend resolves it at allocation:

| Intent | Vulkan | DX12 | CUDA | Metal today |
|--------|--------|------|------|-------------|
| `Scattered`, no CPU flag (parcel storage; tensors, Ekrano, most compute) | `DEVICE_LOCAL` | `DEFAULT` | device memory | **Shared heap** (CPU-mapped) |
| `CPU_READABLE` (host-claim hint) | `HOST_VISIBLE \| HOST_COHERENT`, mapped | `DEFAULT` + mapped `READBACK` twin | staged | Shared heap |
| `CPU_WRITABLE` (deposit / staging) | `HOST_VISIBLE \| HOST_COHERENT`, mapped | `DEFAULT` + mapped `UPLOAD` twin | device + pinned host staging | Shared heap |
| `Broadcast` (uniform) | `HOST_VISIBLE \| HOST_COHERENT` | `UPLOAD` | device | Shared heap |
| Readback staging (`alloc_readback_buffer`) | host-visible, coherent | `READBACK` | cacheable pinned host | standalone Shared |
| Textures | `DEVICE_LOCAL`, staged upload | `DEFAULT`, staged upload | — | **Shared** texture heap, staged upload |
| `GPU_ONLY` | ignored | ignored | ignored | Private, outside the heap |

Three backend-neutral mechanisms sit on top of that and are already placement-agnostic:

- **Deposits** (`MemoryExchange::bind_deposit`) always write into a pooled `CPU_WRITABLE` staging backing and copy into the destination parcel inside graph execution ([`src/deposit_pool.rs`](../../src/deposit_pool.rs), [`src/exchange.rs`](../../src/exchange.rs)).
- **Host claims** (`(&mut submission >> &parcel).take()`) map the parcel when `host_read_mapping` returns a pointer, use a READBACK twin when `host_read_twin_mapping` does, and otherwise copy through the context `HostReadStagingPool` ([`src/host_claim.rs`](../../src/host_claim.rs) `realize_buffer`).
- **Texture uploads** always go staging buffer → blit on every backend, Metal included ([`src/backend/metal/texture.rs`](../../src/backend/metal/texture.rs) `write`).

Every backend requires coherent host-visible memory. Vulkan only requests `HOST_VISIBLE | HOST_COHERENT` and never flushes or invalidates; [Target Hardware](./hardware.md) lists coherent caches as part of Goldy's floor.

So the de facto rule is: **unflagged parcels are device-local; CPU access goes through exchange-owned staging or an opt-in host-coherent hint.** Metal breaks the rule in one direction: it treats every unflagged parcel as CPU-mapped Shared memory. That is the right choice on Apple silicon, where Shared memory is the GPU's memory, and the wrong one anywhere else. Vulkan, DX12, and CUDA break it in the other direction: they never use UMA, ReBAR, `D3D12_HEAP_TYPE_GPU_UPLOAD`, or CUDA integrated memory even when it is available.

## 3. Where Metal hard-codes Apple silicon

| Site | Assumption |
|------|-----------|
| `create_heaps`, `HeapAllocator::create_heap`, `TextureHeapAllocator::create_heap` ([`device.rs`](../../src/backend/metal/device.rs), [`types.rs`](../../src/backend/metal/types.rs)) | Shared heaps (the #330 abort) |
| `mtl_resource_options` ([`buffer.rs`](../../src/backend/metal/buffer.rs)) | Every buffer that is not `GPU_ONLY` is Shared |
| `texture::create` | Shared textures |
| `write`, `clear`, `resize`, `hint_unused_above` (`buffer.rs`), `host_read_mapping` ([`mod.rs`](../../src/backend/metal/mod.rs)), scheme `WriteBuffer` fast path ([`compute.rs`](../../src/backend/metal/compute.rs)) | "Not `GPU_ONLY`" means `contents()` is valid. The flag stands in for the storage the buffer actually got. |
| `adapter_capabilities` (`device.rs`) | `buffer_decommit_supported: true` (`madvise` on `contents()`), `buffer_page_size: 16 KiB` (x86_64 macOS pages are 4 KiB), `buffer_resize_cost: Constant`, for every device |
| `enumerate` (`device.rs`) | `DeviceType` from `isLowPower`: Apple silicon reports `DiscreteGpu` |
| [Goldy Runtime Mapping](../fondaco/goldy-runtime.md) §5 | Claims shipped residency models `ManagedAllocation` / `PageOnFault`; neither exists in code |

`BufferFlags::GPU_ONLY` exists only because Metal's default is Shared. No in-tree caller sets it outside `heap_tests.rs`, and Ekrano never sets it.

## 4. Assessment of #333

[#333](https://github.com/koubaa/goldy/pull/333) adds a pure `MetalStoragePlan` keyed on `hasUnifiedMemory`: Shared heaps on unified devices; otherwise Private heaps, standalone Shared buffers for every non-`GPU_ONLY` buffer, and Private textures. It is **partially aligned**:

- **Aligned.** Private heaps and Private textures on non-Apple GPUs, with uploads already done by blit. Standalone Shared buffers for exchange-owned staging (deposit pool, staging belt, readback). Device capabilities resolved once in a function that can be tested off-device.
- **Wrong predicate (blocker).** `hasUnifiedMemory` is true on Intel iGPUs, so single-GPU Intel MacBooks still get Shared heaps and Shared textures and still abort. The unit test `mac2_without_unified_memory_does_not_allow_shared_heaps` asserts Shared heaps for `(unified=true, apple=false)`, which is that exact device.
- **Wrong default for parcels.** Unflagged parcels stay host-visible and become standalone `newBuffer` allocations. On AMD the GPU then reads every parcel (Ekrano's scene, tile, and PTCL buffers; every tensor) over PCIe. The Private heap serves only `GPU_ONLY`, which nothing sets, so the heap allocator, its overflow and drain logic, and the `Oversubscribed` signal are effectively idle on these devices.
- **Speculative coherency code.** `coherency.rs` adds Managed `didModifyRange:` / blocking `synchronizeResource:` hooks for buffers that are never created Managed. That conflicts with the coherent-memory floor and adds a storage-mode query to every CPU write.
- **Capabilities unchanged.** Decommit, page size, and resize cost stay Apple silicon values on Intel.
- **No evidence from CI.** The `macos-15-intel` runner reports an Apple Paravirtual device with no Tier 2 and no heaps of any storage mode. The job skips every Goldy GPU test and runs only the five pure plan tests.

## 5. Proposal

### 5.1 Two device facts, not one bit

The Metal device records both facts when it is created:

```rust
struct MetalMemoryFacts {
    /// `supportsFamily(MTLGPUFamilyApple1)`. Gates Shared heaps and Shared textures.
    apple_gpu: bool,
    /// `hasUnifiedMemory`. Host-visible memory is the GPU's memory; zero-copy is free.
    unified_memory: bool,
}
```

### 5.2 A backend-neutral placement vocabulary

A crate-private module (for example `src/backend/placement.rs`) defines one resolution table that every backend maps onto native memory:

```rust
pub(crate) enum MemoryModel { Unified, Discrete }

pub(crate) enum Placement {
    /// GPU-optimal, not CPU-mapped. Vulkan DEVICE_LOCAL, DX12 DEFAULT,
    /// Metal Private (heap), CUDA device memory.
    DeviceLocal,
    /// CPU-mapped and coherent. Vulkan HOST_VISIBLE|HOST_COHERENT,
    /// DX12 UPLOAD/READBACK, Metal Shared, CUDA pinned host.
    HostCoherent,
    /// Both at once: Apple silicon Shared, Vulkan DEVICE_LOCAL|HOST_VISIBLE
    /// (UMA / ReBAR), DX12 GPU_UPLOAD / cache-coherent UMA.
    Unified,
}

pub(crate) fn resolve(kind: BufferKind, flags: BufferFlags, model: MemoryModel) -> Placement;
```

| Intent | `Discrete` | `Unified` |
|--------|------------|-----------|
| `Scattered`, no CPU flag | `DeviceLocal` | `Unified` |
| `CPU_READABLE` | `HostCoherent` | `Unified` |
| `CPU_WRITABLE`, readback staging, staging belt | `HostCoherent` | `Unified` |
| `Broadcast` | `HostCoherent` | `Unified` |
| `GPU_ONLY` | `DeviceLocal` | `DeviceLocal` |
| Textures | `DeviceLocal` | `DeviceLocal` |

The rows match what Vulkan and DX12 do today, so step one changes nothing for them. Backends that never report `Unified` keep their current behavior. Textures are `DeviceLocal` everywhere because every backend uploads them by blit, and Apple GPUs compress Private textures losslessly.

### 5.3 Metal mapping

| Placement | Apple silicon (`apple_gpu`) | Intel iGPU, AMD |
|-----------|-----------------------------|-----------------|
| `Unified` | Shared, from the Shared buffer heap | not produced: an Intel iGPU reports `Discrete` to the resolver because it cannot place Shared resources in heaps |
| `DeviceLocal` buffer | Private, standalone (today's `GPU_ONLY` path) | Private, **from the Private buffer heap** |
| `HostCoherent` buffer | Shared heap | standalone Shared `newBuffer` |
| Texture | Private texture heap | Private texture heap |

Memory model for Metal: `Unified` if and only if `apple_gpu && unified_memory`. An Intel iGPU has unified memory physically, but Metal's heap rules make it behave like a discrete device for placement, and its Shared buffers outside heaps are cheap anyway.

Standalone Shared buffers on Intel / AMD are acceptable because almost all of them are **exchange media**, not parcels. Deposit backings, the staging belt, and readback staging never enter the parcel ledger, and Goldy already pools them above the backend (`DepositExchangePool`, `HostReadStagingPool`, `StagingBelt`). Parcels stay in the heap allocator.

Each `BufferState` records its resolved placement and an optional host pointer at allocation, as Vulkan's `host_mapped: Option<usize>` already does. Every "not `GPU_ONLY`, so `contents()` is valid" check reads that field instead. This is also a correctness requirement: `resize` writes through `contents()` for any non-`GPU_ONLY` buffer, which would be a null-pointer write once unflagged parcels are Private.

### 5.4 Capabilities (Layer B)

The Fondaco mapping requires Goldy to report costs honestly ([Goldy Runtime Mapping §12](../fondaco/goldy-runtime.md#12-abstract-the-medium-expose-cost)). Additive changes, allowed within 0.3.x:

- `RuntimeCapabilities::unified_memory: bool` (or `memory_model`): whether unflagged parcels are host-coherent, so a host claim on them is a wait plus a pointer instead of a staged copy.
- Metal `buffer_decommit_supported` and `buffer_page_size` come from the resolved placement and `sysconf(_SC_PAGESIZE)`, not constants.
- `has_zero_copy_storage_readback` keeps its documented meaning (`CPU_READABLE` buffers) and stays `true` on all Macs.
- Either implement the residency models §5 of the runtime mapping claims, or correct that paragraph.
- Metal `DeviceType` comes from `hasUnifiedMemory` / `isRemovable`, not `isLowPower`.

### 5.5 Coherency

Goldy stays coherent-only. Metal Shared buffers are coherent at command-buffer boundaries, which is what the ledger already enforces: host writes land before commit, and host reads happen after a completion wait. Managed storage is not introduced, and `debug_assert!`s reject it so a future change cannot add it silently. If Managed is ever useful (for example a VRAM-resident argument buffer on AMD), it should be its own `Placement` with explicit flush semantics, matching a future non-coherent Vulkan path.

## 6. Fondaco implications

- **Physical medium is runtime-private (spec §5).** Placement is Layer A. Programs keep expressing intent (`CPU_READABLE` as a hint, deposits for uploads, host claims for reads) and never choose a storage mode. `GPU_ONLY` is a Metal leak into the public API; it becomes redundant and should be deprecated.
- **Exchanges own their media.** The split in §5.3 between parcel media (heap, `DeviceLocal`) and exchange media (pooled, `HostCoherent`) is the spec's split between parcels and foreign handoffs. Staging never enters the ledger, so it never needs to live in the parcel heap.
- **Host claims stay valid; only their cost changes.** On Intel and AMD, a host claim on an unflagged parcel becomes a staged copy, as on Vulkan and DX12. Parcel states are identical, which is the only thing conformance (spec §10) requires.
- **Gates may relocate medium (spec §6).** A later step can move a parcel that the program host-claims every submission from `DeviceLocal` to `HostCoherent` at a gate, using the existing reslot machinery. That makes `CPU_READABLE` something the runtime can infer rather than a flag programs must set. **Designed**, not required for #330.
- **Warehouse.** A discrete device has two pools (VRAM and host-visible system memory). `BudgetPolicy` should eventually account committed bytes per placement, and Metal's default budget should come from `recommendedMaxWorkingSetSize`.

## 7. Ekrano implications

[Ekrano](https://github.com/koubaa/ekrano) is a well-behaved consumer. Every buffer it acquires uses `BufferFlags::empty()`, uploads go through `MemoryExchange` deposits, and reads are host claims (bump feedback in robust mode, headless output). It never sets `GPU_ONLY` or `CPU_*` and never checks `has_zero_copy_storage_readback`.

- Under this proposal, Ekrano's parcels become Private heap allocations on Intel / AMD with no Ekrano change. Under #333 they become standalone Shared buffers read over PCIe on AMD.
- Robust-mode bump reads and `render_to_buffer` become staged copies on those devices. They are correct and cost one blit per read.
- Ekrano branches on `BackendType::Metal` in three places: `metal_fused_upload` (duplicates `fuse_upload_with_compute_partitions`), `direct_present` (compute writes the drawable), and `metal_heap_sensitive` (overflow texture heaps pinned across resize). Each should become a capability query so Intel Macs and future backends get the right behavior. That is follow-up work in Ekrano.

## 8. Staged plan

Each stage can land on its own.

### Stage 0: stop the abort (revise #333)

- Gate Shared heaps and Shared textures on `supportsFamily(Apple1)`, not `hasUnifiedMemory`. Log both facts.
- Keep Private heaps, Private textures, and standalone Shared for host-visible buffers on non-Apple GPUs (#333's structure).
- Drop the Managed hooks in `coherency.rs`; assert that no buffer is Managed.
- Fix the plan tests so `(unified=true, apple=false)` (Intel iGPU) gets Private heaps.
- Keep the `macos-15-intel` probe only as an informational step, and do not read a green job as coverage. Ask the reporter of #330 to run `cargo test --lib test_metal_device_creation` and one example on their hardware.

### Stage 1: device-local parcels on non-Apple GPUs

- Record resolved placement and host pointer per `BufferState`. Replace every `GPU_ONLY` proxy check (`write`, `clear`, `resize`, `hint_unused_above`, `host_read_mapping`, scheme `WriteBuffer` fast path).
- Unflagged `Scattered` parcels on non-Apple GPUs: Private, from the buffer heap. Imperative writes and scheme `WriteBuffer` use the existing staging belt + blit path; host claims fall back to the staged path through the existing trait methods.
- Move textures to Private on every Metal device, then compare screenshots and benchmarks on Apple silicon.
- Make decommit, page size, and `DeviceType` per device.
- **Test on existing CI.** Private heaps and standalone Shared buffers are legal on Apple silicon too. A crate-private override (for example `GOLDY_METAL_PLACEMENT=discrete`) runs the whole Metal suite on the `macos-15` arm64 runner with the Intel / AMD placement, giving real coverage without Intel hardware.

### Stage 2: shared vocabulary

- Add `backend/placement.rs` (`MemoryModel`, `Placement`, `resolve`) with one table test. Metal, Vulkan, DX12, and CUDA call `resolve` and map the result. No behavior change for Vulkan, DX12, or CUDA.
- Add `RuntimeCapabilities::unified_memory`. Correct the residency claim in the runtime mapping. Deprecate `GPU_ONLY` in CHANGELOG (still accepted; same meaning as `DeviceLocal`).

### Stage 3: use UMA elsewhere, and Fondaco follow-ups

- Vulkan: report `Unified` and prefer `DEVICE_LOCAL | HOST_VISIBLE` on UMA / ReBAR. DX12: `GPU_UPLOAD` heaps and cache-coherent UMA. CUDA: integrated devices (Jetson).
- `BudgetPolicy` per placement; Metal budget from `recommendedMaxWorkingSetSize`.
- Gate-time placement migration for host-claimed parcels (**Designed**).
- Ekrano: replace `BackendType::Metal` checks with capabilities.

## 9. Open questions

- Is a Private texture heap measurably better on Apple silicon (lossless compression), or should Apple silicon keep Shared textures to avoid a perf regression in Stage 1?
- Should Intel iGPUs (unified memory, no Shared heaps) keep unflagged parcels as standalone Shared buffers for zero-copy host claims, at the cost of bypassing the heap allocator? This proposal says no, for one allocator path; benchmarks on real hardware should decide.
- The argument buffer, frame table, and dispatch-timing buffers are standalone Shared. On AMD, shaders read the argument buffer over PCIe on every bindless access. Measure before considering a Managed or Private-with-blit argument buffer.
- Which real hardware will validate Stage 0 and Stage 1: the #330 reporter, a maintainer's Intel Mac, or a self-hosted runner?
