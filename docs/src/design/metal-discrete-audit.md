# Metal on Intel and AMD Macs: Audit

**Status**: Audit (no code changes). Tracks [#330](https://github.com/koubaa/goldy/issues/330). Companion to [Intel and Discrete Macs on Metal](./intel-mac-metal.md) and to [#333](https://github.com/koubaa/goldy/pull/333).

This chapter lists every place the Metal backend, and the shared code around it, assumes an Apple-family GPU with unified memory. For each one it gives the location, the assumption, what happens on an Intel iGPU and on an AMD discrete GPU, the fix (with the Vulkan or DX12 code that already does the discrete-GPU version), a severity, and whether #333 fixes it.

Line numbers are for `main` at `19a383f3`. "#333" means the PR head `c9444503` (*gate Metal Shared heaps on Apple-family GPUs*).

## Verdict

The backend cannot start on any Intel or AMD Mac today. Both heaps and every texture are created with Shared storage, which Metal rejects on any GPU outside the Apple family. #333 fixes that abort correctly: it uses `supportsFamily(Apple1)` as the predicate, Private heaps and Private textures elsewhere, and standalone Shared buffers for anything the CPU may map.

After #333 the backend would probably run, but it would not yet behave like a discrete-GPU backend:

- Unflagged parcels live in system memory, so on AMD every shader access crosses PCIe.
- Capabilities still report Apple silicon values.
- Render passes never declare textures resident.
- Three imperative CPU paths write through `contents()` without a null check. They crash the moment unflagged parcels become Private, so they block the device-local step.

The hardware floor is also stated wrongly. Argument Buffers Tier 2 is not the limiting requirement, because every Mac-family GPU has it. The real floor is Metal 3 plus macOS 13.

46 findings: **13 crash**, **5 wrong results**, **11 perf**, **17 cosmetic**. #333 fixes 5 of the crashes. Three more are latent: they appear only once Stage 1 makes unflagged parcels Private.

Recommendation: **support both** Intel iGPUs and AMD dGPUs through one non-Apple-family placement path. Gate support on Metal 3, macOS 13, Tier 2, and read-write texture Tier 2, and label it best-effort. Do not build an Intel-specific unified-memory path. Intel macOS ends with macOS 26 Tahoe.

## Device classes

| Class | Examples (on macOS 13+) | Apple family | `hasUnifiedMemory` | Metal 3 | Shared heaps / textures |
|-------|-------------------------|--------------|--------------------|---------|-------------------------|
| Apple silicon | M1–M4 | yes | true | yes | yes |
| Intel iGPU | UHD 630 (2018–2020 Mac mini, MacBook Pro 15"/16", iMac), Iris Plus G7 (2020 MacBook Pro 13" four-port / MacBook Air) | no | **true** | yes | **no** |
| AMD dGPU, Metal 3 | Vega (Vega 20, Vega 56/64, Vega II), Radeon Pro 5300M–5600M, 5300–5700XT, W5500X–W6900X | no | false | yes | no |
| AMD dGPU, pre-Metal 3 | Polaris: Radeon Pro 4xx/5xx, 555X/560X, 580X | no | false | **no** | no |
| GitHub `macos-15` (arm64) | Apple Paravirtual device | yes | true | yes (no mesh) | yes |
| GitHub `macos-15-intel` | Apple Paravirtual device | no | false | — | no heaps at all, no Tier 2 (#333 probe) |

Metal 3 membership comes from Apple's Metal 3 device list: M1+, AMD Vega, AMD 5000/6000, Intel UHD 630, Intel Iris Plus. Apple's macOS support notes exclude Intel Macs whose only discrete GPU is a Radeon Pro 500 (Polaris).

## Severity

| Severity | Meaning |
|----------|---------|
| crash | Objective-C assertion or abort, Rust panic, or a write through a null pointer |
| wrong | Wrong output, GPU fault, a Metal validation error, or refusing to run on a supported GPU |
| perf | Correct but slow, or wasting memory |
| cosmetic | Capability misreported, a misleading comment or doc, a log, or a test or CI gap |

## Inventory

### A. Device creation, floor, adapter info

| ID | Where | Assumption, and what breaks | Fix (analogue) | Sev | Status |
|----|-------|-----------------------------|----------------|-----|--------|
| A1 | `src/backend/metal/device.rs:106-113` | The only device gate is Argument Buffers Tier 2. The message says "Intel 2017+, AMD 2015+". There is no Metal 3, macOS 13, or read-write-texture check, although `mod.rs:3` says the backend targets Metal 3 on macOS 13+. **Intel:** UHD 630 and Iris Plus pass, which is correct. **AMD:** Polaris passes too, then fails later (A3). | Gate with `ensure!` on `supportsFamily(Metal3)`, macOS ≥ 13, Tier 2, and `readWriteTextureSupport == Tier2` (Vulkan API-version gate `vulkan/device.rs:275-280`; DX12 Enhanced Barriers gate `dx12/device.rs:24`). | wrong | open |
| A2 | `device.rs:290-300` | `assert_eq!(sampler_stride, 8)` panics instead of returning an error. Only the sampler encoder is checked. The buffer, texture, storage-image, and acceleration-structure encoders (`device.rs:227-287`) all write at `index × 8`, and their `encoded_length` is logged but never checked. 8 bytes per slot holds for macOS 13+ Tier 2. On older systems the layout is implementation-defined. **Intel and AMD on macOS 13+:** expected to be 8, not verified on hardware. | Check all five encoders and return `Err`, so adapter fallback still works (DX12 asks the device for descriptor strides through `GetDescriptorHandleIncrementSize` rather than assuming them). | crash | open |
| A3 | `device.rs:79-86` | Mesh support is `!paravirtual && (Apple7 \|\| Mac2)`. Mac2 includes Polaris, which is not Metal 3, so Goldy advertises mesh shaders and the first mesh pipeline fails validation or aborts. **Intel:** UHD 630 and Iris Plus are Metal 3, so this is correct but untested. The Paravirtual check matches on the device name string. | AND with `supportsFamily(Metal3)` and keep the Paravirtual guard (Vulkan checks the `VK_EXT_mesh_shader` feature bit; DX12 checks `MeshShaderTier`). A1 makes this moot. | crash | open |
| A4 | `device.rs:35-39` | `DeviceType` comes from `isLowPower`. Apple silicon reports `DiscreteGpu`, Intel reports `IntegratedGpu`, and AMD reports `DiscreteGpu`. Apple silicon is the misreported one here. | Apple1 → `IntegratedGpu`. Leave Intel and AMD as they are (Vulkan reads `properties.device_type`). | cosmetic | open |
| A5 | `device.rs:44` | The vendor is hard-coded to `"Apple"`. Intel and AMD adapters report "Apple", and matmul tuning (J1) is keyed on the adapter. | Take the vendor from the device name or registry ID (Vulkan maps `vendor_id`, `vulkan/device.rs:28`). | cosmetic | open |
| A6 | `device.rs:57-72` | The capability block is the same for every Metal device: `has_zero_copy_storage_readback: true`, `buffer_resize_cost: Constant` (oversized buffers relying on demand paging), `buffer_page_size: 16 KiB`, `buffer_decommit_supported: true`. On Intel and AMD, once parcels are Private: zero-copy readback is false, Private VRAM is committed rather than demand-paged, and `madvise` does nothing (C4). x86_64 pages are 4 KiB, so 16 KiB is still a valid multiple. The doc comment at `src/types.rs:256-266` describes `Constant` in terms of demand paging. | Derive the capabilities from the placement plan. A discrete GPU should report what DX12 does (`dx12/device.rs:149`, `has_zero_copy_storage_readback: false`) and what CUDA does (`cuda/mod.rs:3880-3881`, `Copy`, no decommit). | cosmetic | open (#333 review) |
| A7 | `metal/mod.rs:199-206`, `device.rs:93-101`, `src/runtime.rs:166-179` | Adapters come from `MTLDevice::all()` with no filtering by floor. `HighPerformance` picks the first `DiscreteGpu`. Example: a 2018/2019 MacBook Pro 15" on Ventura with UHD 630 and a Radeon Pro 560X (Polaris). Goldy picks the 560X and, with A1 in place, fails, although the UHD 630 is supported. | Filter enumeration by the floor, so selection falls back to the supported GPU (Vulkan enumeration skips devices below API 1.4). | wrong | open |
| A8 | `metal/mod.rs:199` | No device-removal observer. An eGPU unplug, or a GPU switch on a Mac Pro, leaves `device_lost` unset, and the next submission faults or hangs. Applies to AMD only. | `MTLCopyAllDevicesWithObserver` sets `device_lost` (DX12 handles `DXGI_ERROR_DEVICE_REMOVED`). Needs hardware to verify. | crash | open |

### B. Heaps and storage modes

| ID | Where | Assumption, and what breaks | Fix (analogue) | Sev | Status |
|----|-------|-----------------------------|----------------|-----|--------|
| B1 | `device.rs:171-176` | The buffer heap is `MTLStorageMode::Shared`. **Intel and AMD:** `newHeap` asserts `Shared storage mode disallowed` (#330). | Private heap unless Apple1 (Vulkan uses `DEVICE_LOCAL` memory blocks). | crash | fixed in #333 |
| B2 | `device.rs:199-201` | The texture heap is Shared. Same abort. | Private texture heap. | crash | fixed in #333 |
| B3 | `metal/types.rs:341-349` | Overflow buffer heaps are hard-coded Shared. The abort comes at the first overflow, once B1 is fixed. | Use the plan's heap mode. | crash | fixed in #333 |
| B4 | `types.rs:469-477` | Overflow texture heaps are hard-coded Shared. | Use the plan's heap mode. | crash | fixed in #333 |
| B5 | `texture.rs:144-151`, fallback at `texture.rs:97` | Texture descriptors are Shared, both for heap textures and for the standalone pressure-valve fallback. Shared textures are illegal on non-Apple macOS GPUs (gogpu/wgpu#271: Iris Plus 655). Uploads already go through a staging blit (`texture.rs:250-351`), so Private textures need no new upload path. | Private textures (Vulkan `vulkan/texture.rs:95`, `DEVICE_LOCAL` plus staging). | crash | fixed in #333 |
| B6 | `buffer.rs:11-17`, `buffer.rs:36` | Every buffer without `GPU_ONLY` is Shared. **Main:** the Shared heap aborts (B1). **With #333:** each one becomes a standalone Shared buffer. **Intel:** acceptable, since Shared really is the same DRAM. **AMD:** every unflagged storage parcel sits in system memory, so every shader read and write crosses PCIe, roughly 16–32 GB/s instead of hundreds of GB/s. This is the owner's main complaint. | Stage 1 placement: unflagged storage buffers go to the Private heap. `CPU_READABLE`, `CPU_WRITABLE`, and Broadcast stay Shared (Vulkan `vulkan/buffer.rs:151-162`; DX12 `dx12/buffer.rs:38-42`, DEFAULT vs UPLOAD). | perf | partial (#333 avoids the abort only) |
| B7 | `buffer.rs:36`; #333 `allocate_mtl_storage_buffer` | `GPU_ONLY` is the only route to Private storage, and it stands in for "not mapped". On Apple silicon `GPU_ONLY` skips the heap. With #333 on discrete GPUs, `GPU_ONLY` goes into the Private heap and the meaning of `is_device_allocated` flips. One flag drives two opposite conventions, and the tests pin the Apple one (L3). | Use a placement enum (`DeviceLocal` / `HostCoherent` / `Unified`, [intel-mac-metal.md §5.2](./intel-mac-metal.md)) and stop using flags as a proxy. | cosmetic | open (#333 review) |
| B8 | #333 `HostVisibleBufferPlacement::SharedStandalone` | On discrete GPUs every host-visible allocation is its own `newBuffer`, with no suballocation. That means a kernel allocation and page wiring per parcel, and the residency list grows with each one (F2). | Suballocate Shared buffers from a pool (Vulkan suballocates `HOST_VISIBLE` blocks; DX12 places resources in an UPLOAD heap). Bindless slots would then need to carry offsets. | perf | open |
| B9 | `buffer.rs:15`, `staging.rs:50,149`, `frame_table.rs:85`, `device.rs:116` | Buffers the CPU only writes (staging belt, texture staging, frame table, argument buffer) use `DefaultCache`. Apple's guidance for discrete GPUs is write-combined memory for CPU-write, GPU-read data. **Intel:** minor. **AMD:** slower uploads. | `CPUCacheModeWriteCombined` for upload-only buffers, never for readback or `CPU_READABLE` (DX12's UPLOAD heap is write-combined). | perf | open |
| B10 | `types.rs:240-275`, `types.rs:307-331` | Heap growth is capped at `recommendedMaxWorkingSetSize / 2`. **Intel:** the recommended size is the VRAM carve-out macOS reports, about 1.5 GB (not verified per model), so the cap is about 768 MB. **AMD:** the recommended size is about VRAM, but standalone Shared buffers use system RAM and are not counted. There is no `VideoMemoryInfo` for Metal (`src/runtime.rs:62-76`). | Track a budget per pool (device-local and host-visible) and report `VideoMemoryInfo` (DX12 `QueryVideoMemoryInfo`, `dx12/mod.rs:717-731`; Metal has `currentAllocatedSize` and `recommendedMaxWorkingSetSize`). | perf | open |
| B11 | `types.rs:64-74`, `buffer.rs:36-37`, `buffer.rs:90` | Allocations above `MAX_HEAP_SIZE` (1 GB) and the pressure-valve fallback call `newBuffer` without checking `maxBufferLength`. The comment cites Apple silicon limits. Mac2 limits differ (not verified). If `newBuffer` returns nil, the nil `MTLBuffer` gets used. | Check `device.max_buffer_length()` and return an error (DX12's committed-resource failure path). | crash | open |

### C. CPU access through `contents()`

| ID | Where | Assumption, and what breaks | Fix (analogue) | Sev | Status |
|----|-------|-----------------------------|----------------|-----|--------|
| C1 | `buffer.rs:468-475` (`resize`) | `write_bytes` into `new_buffer.contents()` whenever the old buffer is not `GPU_ONLY`, with no null check. Safe while unflagged buffers are Shared. Once Stage 1 makes them Private, this writes through a null pointer on Intel and AMD. | Skip the memset when `contents()` is null. The `fill_buffer` blit at line 477 already zeroes the tail (Vulkan resize uses `vkCmdFillBuffer` only). | crash | open, latent until Stage 1 |
| C2 | `buffer.rs:669-675` (`write`) | `copy_nonoverlapping` into `contents()` whenever the buffer is not `GPU_ONLY`, with no null check. The staging blit and wait that follow at 691-704 are the part that works on Private memory. | Gate on the placement. The staged path is already correct (Vulkan lazy staging, `vulkan/buffer.rs:1424`). This imperative API is due for removal anyway. | crash | open, latent until Stage 1 |
| C3 | `buffer.rs:772-777` (`clear`) | `write_bytes` into `contents()` whenever the buffer is not `GPU_ONLY`, with no null check. The doc comment at 720-725 says "Metal buffers on Apple Silicon use StorageModeShared". | Same as C2. The `fill_buffer` blit is enough on Private memory. | crash | open, latent until Stage 1 |
| C4 | `buffer.rs:319-348` (`hint_unused_above`) | `madvise(MADV_FREE)` on `contents()`. Does nothing for Private buffers, while A6 advertises decommit support. | Report `buffer_decommit_supported` per placement (Vulkan and DX12 decommit by unbinding sparse pages). | cosmetic | open |

### D. Textures and formats

| ID | Where | Assumption, and what breaks | Fix (analogue) | Sev | Status |
|----|-------|-----------------------------|----------------|-----|--------|
| D1 | `shaders/goldy_exp/bindless_resources.slang:43`, `device.rs:250-262` | Storage images are `RWTexture2D<float4>`, meaning MSL `access::read_write`, over RGBA8 and RGBA16F, including the drawable. Read-write texture Tier 1 only allows R32 formats. Nothing checks `readWriteTextureSupport`. Apple silicon is Tier 2. Metal 3 Intel and AMD GPUs are expected to be Tier 2, but this is not verified. | Add `readWriteTextureSupport == Tier2` to the floor (A1). Vulkan and DX12 have no matching check today, so this is a Metal-only gate. | wrong | open |

### E. Argument buffers and shaders

| ID | Where | Assumption, and what breaks | Fix (analogue) | Sev | Status |
|----|-------|-----------------------------|----------------|-----|--------|
| E1 | `bindless_resources.slang:30-51`, `types.rs:31-35`, `types.rs:776-783` | One 192 KiB `ParameterBlock` with 4096-entry arrays, writable textures included, laid out at 8 bytes per entry. This needs Tier 2 for the counts and writable textures, plus the C-struct layout Apple guarantees only on macOS 13+. **Intel and AMD on macOS 13+ Tier 2:** within limits. On macOS 12, the layout is implementation-defined and bindless lookups read the wrong slots. | Put macOS 13 in the floor (A1) and check strides (A2). Longer term, write `gpuAddress` and `gpuResourceID` directly on macOS 13+ instead of encoding with a per-slot `MTLArgumentEncoder` (Vulkan descriptor sizes come from device properties). | wrong | open |
| E2 | `device.rs:114-116` | The argument buffer is a standalone Shared buffer. **AMD:** descriptor fetches go to system memory, cached after first touch. **Intel:** fine. | Measure first. Tier 2 allows a Private argument buffer with blit updates (DX12's shader-visible descriptor heap is in device memory). | perf | open |
| E3 | `types.rs:776-783`, `bindless_resources.slang:46` | 4096 sampler slots. Apple limits unique argument-buffer samplers per app (`maxArgumentBufferSamplerCount`, "at least 1024"). The comment compares against the Tier 2 limit of 500K, which is the buffer and texture limit. Not specific to discrete GPUs; latent on every Mac. | Clamp sampler slots to the device limit and fail cleanly beyond it. | cosmetic | open |

### F. Residency and hazards

| ID | Where | Assumption, and what breaks | Fix (analogue) | Sev | Status |
|----|-------|-----------------------------|----------------|-----|--------|
| F1 | `render_commands.rs:82-143` | Render encoders call `useHeap` and `useResource` for buffers and the frame table, but never `useResource` for textures. That leaves the drawable, standalone fallback textures (`texture.rs:97`), and fragment writes to heap textures undeclared, and Apple requires `useResource` for writable resources even inside heaps. Apple silicon tolerates this in practice. **Intel and AMD:** GPU page faults or stale reads are likely. There is also a debug `eprintln!` in the mesh branch at `:95`. | Mirror the compute path, `compute.rs:543-591` (read-write and read-only texture tiers). Applies to Apple silicon too. | wrong | open |
| F2 | `compute.rs:521-595` | Every compute encoder calls `useResources` over every live buffer and texture. With #333 on discrete GPUs, unflagged buffers are standalone (B8), so heaps cover none of them and the list grows with the parcel count. | After Stage 1, Private-heap parcels are covered by `useHeap`. Pool the Shared buffers, and use `MTLResidencySet` where available. | perf | partial (#333 makes it larger) |

### G. Frame table, staging, readback

| ID | Where | Assumption, and what breaks | Fix (analogue) | Sev | Status |
|----|-------|-----------------------------|----------------|-----|--------|
| G1 | `frame_table.rs:1`, `frame_table.rs:85-86`, `frame_table.rs:131`, `frame_table.rs:269`; `shaders/goldy_exp/access.slang:249-253` | The frame table and selector are standalone Shared buffers written with a CPU memcpy at each submission. The module doc says "UMA CPU write, no GPU copy". **AMD:** every dispatch reads its slot indices over PCIe. **Intel:** fine. | Vulkan `vulkan/frame_table.rs` stages the upload and copies it to device-local memory in the prologue. | perf | open |
| G2 | `compute.rs:1289-1290` | Twin readback (`CopyToCpuReadableTwin`) exists only on DX12, because Metal `CPU_READABLE` buffers are Shared. **AMD:** GPU writes to a `CPU_READABLE` storage buffer go to system memory on every kernel. | Optionally give `CPU_READABLE` storage that the GPU writes heavily a device-local buffer plus a readback twin (DX12 `dx12/compute.rs:1822`). | perf | open |

### H. Present, drawable, `CAMetalLayer`

| ID | Where | Assumption, and what breaks | Fix (analogue) | Sev | Status |
|----|-------|-----------------------------|----------------|-----|--------|
| H1 | `surface.rs:96-101` | `CAMetalLayer.pixelFormat = RGBA8Unorm`, chosen because BGRA8 cannot be a storage target on Apple silicon. RGBA8Unorm is not in Apple's documented list of layer formats (BGRA8Unorm, BGRA8Unorm_sRGB, RGBA16Float, RGB10A2Unorm, BGR10A2Unorm, BGRA10_XR, BGR10_XR, and their sRGB variants). It works on Apple silicon. **Intel and AMD:** unknown; the layer may raise an exception or composite wrongly. | Try RGBA8Unorm; otherwise use BGRA8Unorm with shader write if the family allows it; otherwise compute into a Private RGBA8 texture and blit to the drawable (Vulkan picks from the formats the surface reports, `vulkan/surface.rs:166-178`, and keeps a scratch storage image, `:1380-1396`). Needs hardware. | crash (unverified) | open |
| H2 | `surface.rs:94-95` | The layer's device is the rendering device, with no handling for the case where a different GPU drives the display (Mac Pro, eGPU, or a dual-GPU MacBook with an external display). WindowServer copies the frame across GPUs every frame. | When the app asks for presentation, prefer the adapter that drives the window's display. Needs hardware. | perf | open |

### I. Synchronisation and timing

| ID | Where | Assumption, and what breaks | Fix (analogue) | Sev | Status |
|----|-------|-----------------------------|----------------|-----|--------|
| I1 | `metal/mod.rs:1177-1186` | The blocking wait uses `wfe` on aarch64 and a bare `spin_loop` everywhere else. On x86_64 that busy-spins a core for the whole GPU wait. On Intel, the iGPU shares package power with the CPU, so the spinning core takes turbo headroom from the GPU. The comment cites M1 timings. | Spin briefly, then park on the completion handler's condvar or `MTLSharedEvent` notify (Vulkan `vkWaitSemaphores`; DX12 `SetEventOnCompletion`). | perf | open |
| I2 | `dispatch_timing.rs:59` | Per-dispatch timing requires `AtStageBoundary` counter sampling, which only Apple GPUs offer. Intel and AMD sample at dispatch and blit boundaries, so timing is silently disabled there. | Use `sampleCountersInBuffer` at dispatch boundaries on Mac2 (Vulkan `vkCmdWriteTimestamp`; DX12 timestamp queries). | cosmetic | open |

### J. Ops and MPS

| ID | Where | Assumption, and what breaks | Fix (analogue) | Sev | Status |
|----|-------|-----------------------------|----------------|-----|--------|
| J1 | `src/ops/matmul.rs:237-242`, `src/ops/matmul.rs:281-283` | `MatMulHardware::detect` maps every Metal device to `AppleSilicon`, which gets `APPLE_M1` tuning (stdlib GEMV, `library_min_n: 4`). The tuning is unmeasured on Intel and AMD, where MPS and stdlib cost differently. | Map devices outside Apple1 to `Other` until `tools/gemv_bench` has measured them (CUDA tuning is measured). | perf | open |

### K. Shared code, comments, docs

| ID | Where | Assumption | Fix | Sev | Status |
|----|-------|------------|-----|-----|--------|
| K1 | `src/types.rs:439-470` | The public `GPU_ONLY` docs say "Metal: MTLStorageMode::Private", and the `CPU_WRITABLE` docs say "On Vulkan/Metal persistently mapped". Once unflagged parcels are device-local on discrete Metal, `GPU_ONLY` is redundant there, and the docs expose Metal storage modes in the public API. | Describe intent rather than storage modes. Note in the CHANGELOG if behaviour changes. | cosmetic | open |
| K2 | `src/buffer.rs:116`, `src/buffer.rs:129`, `src/buffer.rs:454` | "oversize virtual reservations (e.g. Metal)", "Use GPU_ONLY for device-local frame scratch pools on Metal", "On Metal (shared memory), may return physical pages". | Qualify as Apple silicon, or describe in terms of capabilities. | cosmetic | open |
| K3 | `src/runtime.rs:455-468` | The `has_zero_copy_storage_readback` docs say "Metal (Shared storage)". | Make it per device. | cosmetic | open |
| K4 | `docs/src/design/hardware.md:50-64`, `shaders/README.md:229-236`, `device.rs:107,110`, `types.rs:688` | "Intel Macs 2017+, AMD 2015+, pre-2017 Intel lacks Tier 2." This is wrong in both directions; see [Hardware floor](#hardware-floor). | Restate the floor as Metal 3 plus macOS 13. | cosmetic | partial (#333 edits hardware.md) |
| K5 | `hardware.md:70-77` | The floor table lists "Unified or REBAR memory". AMD Macs have neither: macOS gives apps no resizable BAR, and Shared means system memory. Vulkan already handles discrete GPUs without ReBAR by staging. | Change the floor row to "unified, ReBAR, or staged discrete", or state that AMD Macs are excluded. | cosmetic | open |
| K6 | `metal/mod.rs:3`, `types.rs:1-11`, `device.rs:120`, `buffer.rs:720-725`, `frame_table.rs:1`, `texture.rs:144-149`, `surface.rs:96-100` | Comments state Apple-silicon behaviour as Metal behaviour. | Rewrite them when the code changes. | cosmetic | partial (#333 rewrites some) |

### L. Tests and CI

| ID | Where | Gap | Fix | Sev | Status |
|----|-------|-----|-----|-----|--------|
| L1 | `.github/workflows/ci.yml:169-248` | `test-metal` runs only on `macos-15`, an arm64 Apple Paravirtual GPU in the Apple family. Private heaps, Private textures, standalone Shared buffers, and the staged paths are never exercised. | Add a job with forced discrete placement (see [Test strategy](#test-strategy)). | cosmetic | open (#333 adds an Intel probe only) |
| L2 | `ci.yml:213-214` | `MTL_DEBUG_LAYER` and `MTL_SHADER_VALIDATION` are set only on the mesh step. The main `cargo test` runs without them and without `GOLDY_VALIDATION` (the Vulkan job sets it at `:104`). | Turn validation on for the whole Metal job. | cosmetic | open |
| L3 | `src/heap_tests.rs:72-131` | `buffer_allocation_increments_count`, `gpu_only_buffer_does_not_use_heap` (`:88`), and `buffer_drop_frees_heap_space_after_flush` encode Apple placement. With #333 on a discrete GPU, unflagged buffers are standalone (no count increase, no overflow) and `GPU_ONLY` goes into the heap, so all three fail. | Assert against the device's plan. | cosmetic | open |
| L4 | `src/buffer_alloc_tests.rs:143-152` | Asserts `Constant` and decommit for every Metal device. | Assert against the plan (A6). | cosmetic | open |

### Release exposure (not counted)

`.github/workflows/publish.yml:129-155` builds and publishes **macOS x86_64 Python wheels** on `macos-15-intel`. Intel Mac users can therefore install today a Goldy build that aborts at device creation (B1). This is scope rather than a defect: it is why Stage 0 matters even if Intel is best-effort.

### Checked and portable

These were checked and need no discrete-specific change:

- `staging.rs` (belt and texture staging, Shared with a cached `contents()`): the `HOST_VISIBLE` staging equivalent. B9 covers the write-combined option.
- `alloc_readback_buffer`, `buffer.rs:793-825`, and texture readback at `buffer.rs:827-897`: Shared readback, the DX12 READBACK equivalent.
- `host_read_mapping` (`metal/mod.rs:563-573`): returns `None` for a null `contents()`, so a Private buffer falls back to the staged copy correctly.
- The scheme `WriteBuffer` fast path (`compute.rs:799-850`, mirrored in the pre-pass at `:1495-1506`): already null-checked, and falls back to the staging belt.
- `CPU_WRITABLE` paths (`cpu_writable_flat_slice`, `buffer.rs:623-639`; deferred host writes, `pending_submit.rs:68`; `CopyBufferToTexture`, `compute.rs:1541-1575`): `CPU_WRITABLE` is always Shared.
- Uniform (Broadcast) buffers are Shared, the same as Vulkan's `HOST_VISIBLE | HOST_COHERENT` uniforms (`vulkan/buffer.rs:153-161`).
- Depth targets and render targets are Private (`surface.rs:155-163`, `surface.rs:664-670`, `render_target.rs:26-38`). The `Depth24Plus` → `Depth32Float` mapping is valid on every Mac.
- Acceleration structures: Private scratch (`accel.rs:81`), Shared instance upload (`accel.rs:245`).
- Tracked heaps (`device.rs:178-193`): hazard tracking is per heap on every GPU, so the false serialization is the same everywhere.
- Serial encoders with no `memoryBarrier` calls; `fuse_upload_with_compute_partitions` and `compute_partition_split: None`. None of these depend on unified memory.
- MPS (`metal/matmul.rs`): runs on Mac2 GPUs. Adding an `MPSSupportsMTLDevice` guard would be cheap.
- Subgroup code (`shaders/goldy_exp/collectives.slang:100-140`) reads `WaveGetLaneCount` and handles 8 to 64 lanes, so Intel SIMD8/16/32 and AMD wave32/64 both work. The stdlib GEMV (`src/ops/matmul_kernel.rs:65-112`) uses workgroup memory, not lane intrinsics. `subgroup_width` is `None` on Metal, which is safe.
- Examples and `tests/` contain no Apple-silicon assumptions. The Metal skips in `tests/` concern resubmit retention and ray-tracing pipelines, not memory.

## Counts

| Severity | Count | IDs |
|----------|-------|-----|
| crash | 13 | A2, A3, A8, B1, B2, B3, B4, B5, B11, C1, C2, C3, H1 |
| wrong | 5 | A1, A7, D1, E1, F1 |
| perf | 11 | B6, B8, B9, B10, E2, F2, G1, G2, H2, I1, J1 |
| cosmetic | 17 | A4, A5, A6, B7, C4, E3, I2, K1–K6, L1–L4 |
| **total** | **46** | |

Of the 13 crashes, #333 fixes 5 (B1–B5). Three (C1–C3) appear only after Stage 1. A3 affects only Polaris, A2 only systems before macOS 13, A8 only eGPUs and Mac Pros, B11 only allocations above `maxBufferLength`. H1 is unverified.

## Top 10

1. **Shared heaps and textures abort** (B1–B5). Fixed by #333. Land it.
2. **Render encoders never declare textures resident** (F1). This is a latent fault on every GPU and the likeliest source of wrong pixels on Intel and AMD. A small fix: mirror the compute path.
3. **Unchecked `contents()` writes in `resize`, `write`, and `clear`** (C1–C3). They block Stage 1: a null-pointer write the moment unflagged parcels become Private.
4. **Unflagged parcels in system memory on AMD** (B6, with B7 and B8). This is the owner's "discrete like Vulkan and DX12" requirement. Unflagged storage belongs in a Private heap, and `GPU_ONLY` should stop deciding placement.
5. **The floor gate does not match what the encoding needs** (A1, A2, A3, D1, E1). Gate on Metal 3, macOS 13, Tier 2, and read-write texture Tier 2. Return an error, not a panic, for every encoder stride.
6. **`CAMetalLayer` uses RGBA8Unorm, which Apple does not document for layers** (H1). Unverified on Intel and AMD. Needs a fallback.
7. **Capabilities hard-code Apple silicon values** (A6). Zero-copy readback, resize cost, and decommit are reported wrongly on discrete GPUs.
8. **Adapter selection on dual-GPU MacBooks** (A7). `HighPerformance` picks a Polaris dGPU that the floor rejects, instead of the supported UHD 630.
9. **Frame table and argument buffer in system memory on AMD** (G1, E2). Every dispatch reads indices over PCIe. Vulkan already stages the upload and copies it to device-local memory.
10. **CI never runs discrete placement or Metal validation** (L1, L2). Nothing catches a regression in any of the above.

Next in line: per-encoder residency growth (F2), the x86 spin wait (I1), M1 matmul tuning on every Metal device (J1), and heap budget accounting (B10).

## Hardware floor

**Tier 2 is not the binding constraint.** Apple's feature set tables put Tier 2 on Apple6+ and on every Mac2 GPU. Tier 1 was the pre-A13 iOS GPUs. Public `metal2caps`-style reports show Skylake HD 530 and GCN AMD parts at Tier 2. The premise that many Intel iGPUs are Tier 1 does not hold for any Mac that can run macOS 13.

What Goldy actually needs:

| Requirement | Why | Source in Goldy |
|-------------|-----|-----------------|
| Argument Buffers Tier 2 | 4096-entry arrays and writable textures in one argument buffer | E1, `device.rs:106` |
| macOS 13+ | C-struct argument buffer layout. Goldy writes 8-byte slots and Slang emits a C struct. | E1, A2, `mod.rs:3` |
| Metal 3 family | Stated target (`mod.rs:3`); mesh shaders; excludes Polaris | A1, A3 |
| `readWriteTextureSupport` Tier 2 | `RWTexture2D<float4>` storage images and drawable writes | D1 |
| Private heaps | All non-Apple placement (#333 probe) | B1–B4 |

Against that floor:

- **Intel UHD 630 and Iris Plus (Ice Lake) are in.** Both are Metal 3 and run macOS 13+.
- **AMD Vega, 5000-series, and 6000-series are in.**
- **Polaris is out** (Radeon Pro 4xx/5xx, 555X/560X, 580X): Tier 2 but not Metal 3.
- **Older Intel GPUs on macOS 12 or earlier are out** because of the OS requirement, not Tier 2.

`hardware.md` should be corrected (K4, K5). Its "Intel 2017+, AMD 2015+" admits Polaris and older OSes, and its "pre-2017 lacks Tier 2" gives the wrong reason for excluding them.

**OS lifetime.** macOS 26 Tahoe is the last release for Intel Macs, and macOS 27 runs only on Apple silicon. Apple has said Intel Macs get about three years of security updates. Tahoe runs on four Intel models: MacBook Pro 16" 2019 (UHD 630 plus 5300M/5500M/5600M), MacBook Pro 13" 2020 four-port (Iris Plus G7), iMac 2020 (5300/5500XT/5700/5700XT), and Mac Pro 2019 (580X Polaris, which is out, or W5500X–W6900X/Vega II). Ventura through Sequoia cover more 2017–2019 machines.

## Recommendation

Support **both** Intel iGPUs and AMD dGPUs through **one** placement path for GPUs outside the Apple family:

- **AMD dGPU** is the case the owner describes, and it gains the most from device-local parcels.
- **Intel iGPU** gets the same placement. It cannot use Shared heaps or textures. Shared buffers are genuinely zero-copy there, but Private heaps cost little, and one path is testable through forced placement. A dedicated `Unified` placement for Intel (zero-copy readback, Shared parcels) is not worth building for a platform with a fixed end of life.
- **Polaris and anything below the floor** get a clean error at adapter enumeration, and selection falls back to a supported GPU.
- Label Intel and AMD support **best-effort / community-tested** until someone runs Stage 3.

## Staged plan

The stages follow [intel-mac-metal.md §8](./intel-mac-metal.md).

**Stage 0: stop the abort and fail cleanly.** Land #333 (B1–B5), then make these small follow-ups:

- Gate on Metal 3, macOS 13, and read-write texture Tier 2 (A1, A3, D1).
- Return an error rather than panic for all five encoder strides (A2).
- Filter adapters by the floor (A7).
- Declare textures resident in render encoders (F1). This also fixes Apple silicon.
- Turn validation on for the whole Metal CI job (L2).
- Correct `hardware.md` and `shaders/README.md` (K4).

Stage 0 needs no hardware.

**Stage 1: device-local parcels.**

- Add a placement enum ([intel-mac-metal.md §5.2](./intel-mac-metal.md)) resolved from `MetalMemoryFeatures`, with a `GOLDY_METAL_PLACEMENT=apple|discrete` override.
- Unflagged storage buffers go to the Private heap. `CPU_READABLE`, `CPU_WRITABLE`, and Broadcast stay Shared (B6, B7).
- Make C1–C3 null-safe.
- Derive capabilities from the plan (A6, C4).
- Stage the frame table (G1).
- Rewrite L3 and L4 to assert against the plan.
- Add the forced-discrete CI job (L1).
- Note the `GPU_ONLY` semantics change in the CHANGELOG (K1).

**Stage 2: performance and polish.**

- Shared-buffer suballocator (B8) and residency growth (F2).
- Write-combined upload buffers (B9).
- Budgets and `VideoMemoryInfo` (B10).
- `maxBufferLength` check (B11).
- x86 wait (I1).
- Matmul tuning (J1).
- Counter sampling (I2).
- Argument buffer placement (E2) and the `CPU_READABLE` twin (G2), both measured first.
- Vendor and device type (A4, A5), samplers (E3), and docs (K2, K3, K5, K6).

**Stage 3: hardware validation.** See [What still needs hardware](#what-still-needs-hardware).

## Test strategy

These tests need no Intel or AMD hardware:

1. **Unit tests of the selection logic.** Make floor checks, placement, and capabilities pure functions of a `MetalDeviceFacts` struct (Apple1, Mac2, Metal3, unified memory, argument-buffer tier, read-write texture tier, OS version, encoder strides). Test them against fixed profiles: M1, UHD 630, Iris Plus G7, 5500M, W6800X, 580X, Paravirtual arm64, and Paravirtual x86. #333's `metal_storage.rs` tests are the start.
2. **Forced discrete placement on the arm64 runner.** `GOLDY_METAL_PLACEMENT=discrete` gives Private heaps, Private textures, Private unflagged parcels, and standalone Shared host-visible buffers. All of these are legal on Apple silicon, so the full `cargo test` and FFI suites exercise the discrete code paths: C1–C3, staged readback, frame-table staging, and `host_read_mapping` returning `None`.
3. **Storage-mode tripwire.** Route every `newHeap`, `newTexture`, and `newBuffer` call through helpers that debug-assert the mode against the plan. Then a forced-discrete run fails on any Shared heap or texture, even though Apple silicon would accept it. Back this with a grep lint that rejects `MTLStorageMode::Shared` and `StorageModeShared` outside the helper module.
4. **Metal API and shader validation** (`MTL_DEBUG_LAYER=1`, `MTL_SHADER_VALIDATION=1`) on the forced-discrete job. Shader validation reports residency violations (F1) and out-of-bounds access.
5. **Clean failure on `macos-15-intel`.** That runner's GPU has no Tier 2 and no heaps. Assert that creating a runtime returns `Err` and does not abort. Also run the x86_64 unit tests there.
6. **x86_64 under Rosetta on the arm64 runner**, if the image has Rosetta. This exercises x86-only code (I1) against a real Metal device.
7. **Capability tests** (L4) that compare reported capabilities with the plan.

### What still needs hardware

- That Intel and AMD drivers accept Private heaps and textures as Goldy creates them, with the right alignment and sizing.
- Argument-buffer `encoded_length` for every encoder on real Mac2 GPUs (A2).
- Whether `CAMetalLayer` accepts RGBA8Unorm and allows drawable writes (H1).
- Real values of `readWriteTextureSupport` and `maxBufferLength` (D1, B11).
- Mesh shaders on AMD and Intel (A3).
- PCIe cost of the remaining Shared data (E2, G1, G2). Matmul tuning (J1).
- Dual-GPU switching, eGPU removal, cross-GPU present (A7, A8, H2).
- Numerical differences under the default `fastMath` (`shader.rs:446-447`) on Intel and AMD compilers against existing tolerances.

The cheapest coverage is a single **MacBook Pro 16" 2019** (UHD 630 plus Radeon Pro 5300M/5500M/5600M), which covers both GPU classes and runs Tahoe. AWS EC2 `mac1.metal` instances are Intel Mac minis with a UHD 630, so they give Intel iGPU coverage without owning hardware. They are not GitHub-hosted and have a 24-hour minimum allocation. No hosted option for AMD is known.
