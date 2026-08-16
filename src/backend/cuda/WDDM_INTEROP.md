# CUDA + DX12 presentation on WDDM (internal)

**Status:** state of the art for Goldy CUDA on Windows consumer GPUs as of August 2026.
This is the supported public interop path, not a temporary workaround we expect to beat
with queue reordering alone.

## Pipeline (head-chasing tail)

To get more throughput, some GPU programs try to run multiple submissions in parallel. For presentation purposes, it's often better to do **one compute frame in flight**: stage X of frame N+1 starts immediately after stage X of frame N retires, not alongside it.

Per frame, on CUDA+DX12:

```text
HtoA scene upload
CUDA graph islands
stream fine cs_main
CUDA AtoA export → imported RGBA8 scratch (depth-3 ring slot)
DX12 DIRECT CopyResource(scratch → DXGI backbuffer) + Present
```

Scratch is a **depth-3 imported staging ring** independent of the DXGI image index.
Two shareable fences (`dx12_companion.rs`):

| Fence | Producer | Consumer | Purpose |
|-------|----------|----------|---------|
| **ready** | CUDA | DX12 | Scratch ready for present copy |
| **recycle** | DX12 | CUDA | Scratch slot safe to reuse (ring wrap only) |

CUDA→DX12 ready waits run on a **COPY hop queue**; the DXGI **DIRECT** queue waits a native
hop fence then `CopyResource` + `Present`. Flip-model backbuffers are DWM-shared; this
driver stack returns `DXGI_ERROR_ACCESS_DENIED` if `CopyResource` is attempted from a COPY
queue — the backbuffer copy must stay on DIRECT.

Schemes write a CUDA-owned `out_image` then `CopyTexture` into scratch (same local-then-copy
pattern as native DX12). Direct launches onto imported scratch remain supported but costlier.

**CUDA is ~2× slower than native DX12** despite the measured compute being very similar (CUDA should theoretically be faster but slang's CUDA backend may not be fully optimized yet. This is one direction to advance the state of the art but it is not currently the bottleneck for CUDA+present workloads)

### Experiments that did not move FPS materially

| Change | Effect |
|--------|--------|
| Reorder acquire (DXGI waitable before slot fence) | `fence_wait` instant; `dxgi_wait` still dominates CPU pacing |
| Remove `present_stream` CUDA-event bridge | 1.9% median interval |
| COPY hop for CUDA fence wait (not copy) | No meaningful FPS change; `CopyResource` still on DIRECT |
| Depth-3 scratch ring + ready/recycle fences | zero overlap of DX12 copy with CUDA N+1 |

The ring removes scratch **reuse** hazards but does not hide WDDM context transitions on the
critical path because the dependency remains serial:

```text
CUDA writes scratch → DX12 copies scratch → CUDA continues
```

WDDM schedules CUDA and DX12 in separate contexts; the gaps bracketing the DX12 copy are consistent with context transitions plus fence/queue scheduling.

## Can CUDA blit to the DXGI surface directly?

**No** on the public API stack.

- CUDA cannot create or own an `IDXGISwapChain`.
- Flip-model backbuffers are DXGI/DWM-owned; they are not normal exportable D3D12 resources.
- CUDA imports only explicitly shareable D3D12 textures via `cuImportExternalMemory`.
- There is no public CUDA API to import an arbitrary WDDM allocation or DXGI backbuffer.
- WDDM is a scheduler, not a blit/present API.

Mandatory boundary today:

```text
CUDA-writable shared D3D12 UAV (scratch)
    → cuImportExternalMemory / surface writes
D3D12 DIRECT CopyResource
    → DXGI flip-model backbuffer
Present
```

DX12 is not the only graphics API that could own the swapchain (D3D11, Vulkan+Win32, etc.),
but **some** graphics API must own presentation; CUDA cannot replace that role on WDDM
without a private driver interface.

TCC mode avoids WDDM scheduling but is generally unavailable for consumer display GPUs and
incompatible with presenting from the same GPU.

## What graphics APIs are missing

To remove or amortize the per-frame CUDA↔graphics context switch we need at least one of:

1. **Unified submission queue** — submit CUDA compute and graphics/present commands into one
   WDDM context/queue without an API boundary (no public CUDA↔D3D12 equivalent of a single
   Vulkan queue or Metal command buffer mixing compute and blit).

2. **Exportable flip-model backbuffers** — DXGI swapchain images importable into CUDA (or
   any compute runtime) as writable surfaces, with defined synchronization against `Present`.

3. **Bidirectional fence/semaphore without mixed-producer hazards** — today's D3D12 shared
   fences + CUDA external semaphores work but require separate ready/recycle fences when
   both sides signal; a single timeline with clean cross-API ordering would simplify staging.

4. **COPY-queue or compute-queue present path** — ability to write flip-model backbuffers
   without a DIRECT-queue context transition, or a documented fast blit from a shareable
   intermediate that does not ping-pong contexts.

5. **Optional: CUDA graph nodes in graphics command lists** — or D3D12 Work Graphs / equivalent
   that retain partitioned dispatch model without small launches and updater
   kernels per frame on the CUDA side.

Until then, Goldy documents the depth-3 scratch ring and dual fences as the interop tradeoff
for CUDA+DX12 presentation.
