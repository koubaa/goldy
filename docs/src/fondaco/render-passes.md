# Render Passes and Schemes

**Status**: Research note. Complements [Machine Specification](./specification.md) and [Goldy Runtime Mapping](./goldy-runtime.md). Day-to-day recording: [Render Pass Nodes](../graphics/commands.md).

**Claim.** In the Fondaco machine, a draw can be a dispatch. Goldy records many draws as **one** scheme node because Vulkan / DX12 / Metal in 2026 only expose a gate at the **framebuffer epoch** (the render pass), not at each draw. That clustering is a runtime transformation (spec §8 fusion), not a statement that draws are invisible to the machine.

## Two grains

A **dispatch** is the finest unit the ledger can wait on: claims, gates, parcel retirement. A merchant may name a kernel launch, a copy, or a single raster draw as a dispatch.

A **framebuffer epoch** (native “render pass”, Goldy `render_pass` node) is the interval during which one or more attachments are **checked out** of the warehouse medium into on-chip tile / ROP storage. Load ops happen at checkout; store (and MSAA resolve, if any) happen at check-in. Other dispatches may not observe the attachment’s medium until check-in.

These are different objects:

| | Dispatch | Framebuffer epoch |
|---|---|---|
| Machine | Atomic work; gate on either side | Not a machine primitive |
| 2026 APIs | Kernel, copy, *or a whole pass* | `vkCmdBeginRendering` … `EndRendering`, Metal render encoder, DX12 `BeginRenderPass` |
| Waitable from outside | Yes, if the runtime admits it | Yes — this is what the APIs actually fence |
| Private storage | Script registers, groupshared | Live color/depth tile, bound pipeline / VB / IB |

Goldy’s `finish()` is not a GPU opcode. It **commits a fused dispatch**: the epoch plus the draw list recorded inside it. Compute already terminates a node with `dispatch(x, y, z)` because that node *is* one launch. A raster node has no single “the” draw, so the terminator is explicit (Rust) or RAII (FFI / Python).

A pass has **no native handle**. Identity, if any, is the retained scheme node (or the command list it was recorded into). Vulkan 1.0 `VkRenderPass` objects had identity; Goldy sheds them ([What Goldy Sheds](../design/what-goldy-sheds.md)). Dynamic rendering is two marks in a command stream.

## Machine view: draws are dispatches

Spec §3: a dispatch runs to completion and is not reordered internally. Spec §8: the runtime may **fuse** adjacent dispatches if parcel states are preserved, and merchants should express natural grain.

Nothing in that forbids:

1. Draw A (reads `warm`, writes attachment `rt`)
2. Draw B (reads `cool`, writes `rt`)
3. Compute C (overwrites `warm`)

with precedences `A → B` (both private on `rt`) and `A → C` (C private on `warm`). After A’s vertex/mesh stage retires, `warm` could be recycled while B still shades — **if** a gate existed after A that did not check `rt` back in.

The command processor (CP / firmware) already has that state machine: pipeline and VB binds are register writes; timestamp / event packets mark stage retirement per draw; TBDR hardware splits a pass into a binning job (vertex fetch for *all* draws) then per-tile fragment. Metal’s `updateFence(afterStages: .vertex)` is the one portable-ish leak of that vertex-job boundary. The machine is allowed to model it. The 2026 **portable ABI** is not required to.

Scripts remaining opaque (spec §4) still holds for each **draw’s shader**. It does not imply the *sequence of draws* is one script. Treating the sequence as opaque is Goldy’s lowering, not Fondaco ontology.

A kernel’s registers are private by physics. Pass-internal sticky state (current pipeline, current VB) is private **by API contract**. Drivers and firmware fiddle with it constantly; applications may not place a fence, event, or compute dispatch between two draws inside a Vulkan render pass instance. DX12 without `BeginRenderPass` is looser (interleaved compute + barrier) and thereby often flushes the tile. Variation across APIs is the proof that the intra-pass wall is policy, not the Fondaco machine.

## What is physics

**Tile residency of the attachment** is the honest constraint. During the epoch, `rt` is not a stable VRAM image: on tilers it lives in GMEM; on immediate-mode GPUs it is spread across ROP caches and compression metadata. A compute dispatch that samples or overwrites `rt` mid-epoch forces a store/load — the same cost as splitting the pass.

So:

- Fusing draws that **share an attachment epoch** is often required for cost, even on a runtime that could name each draw.
- Fusing **buffer** claims (vertex buffers, bindless reads) with that epoch is **not** required by physics. It is required by Vulkan/Metal encoder rules: you cannot barrier `warm` and dispatch compute without ending the encoder, which checks `rt` in.

Recycling `warm` the moment draw A’s vertex stage is done, while draw B continues, is a real hardware opportunity. Portable 2026 APIs do not give Goldy a gate that releases `warm` without ending the epoch on `rt`. Practical lifetime for vertex memory is therefore **frame-in-flight rings**, not per-draw retirement.

## Goldy view (shipped)

Goldy admits **one** render-pass node per epoch:

- **Claims** are declared on the node (`with_parcel`, `with_buffer_dependency`, `TargetLoad` → public / private / private-inaugural on the target).
- **Body** is a list of render commands: set pipeline, set VB/IB, draw, mesh dispatch, clear depth. Sequential sticky state. Not graph edges.
- **Gates** land at node boundaries: after the native end-rendering, then copies, compute, present.

That is spec §8 fusion applied at record time because the substrate cannot address a finer gate. The scheme cannot insert a node between two draws, cannot fence one draw, and cannot start compute on `warm` until the whole pass’s graphics in the barrier’s source stages have retired — typically the entire pass, because the barrier is recorded after end-rendering.

`with_parcel` on the pass is the seam: the fused body still names parcels (`set_vertex_buffer(warm)`), so the program must echo those names as node claims or the ledger is a lie.

**Designed** (not shipped): per-draw dispatches plus a fusion hint “keep this attachment tile-resident.” A firmware-level or fondaco-native backend could split buffer retirement from attachment check-in. Until then, Goldy will not pretend a `draw()` is a waitable epoch.

## Consequences for scheme authors

- One-shot `set_pipeline` + `draw(0..3)` + `finish()` is a fused dispatch that happens to contain one statement. That is fine.
- Two draws that must depth-test against each other **must** share an epoch (or the second pass `Load`s depth and pays a round trip).
- Compute that consumes the color target is a **later** scheme node, after the pass — never inside the builder.
- Overwriting a vertex parcel used by draw 1 while draw 2 of the same pass still runs is not expressible in Goldy 0.2. Split the pass (`Load` the target) if you need that gate; expect a store/load.
