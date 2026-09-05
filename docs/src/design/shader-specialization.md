# Shader Specialization Prediction

Programs know things about a frame that a shader cannot see: whether any image in the
scene carries a tint, which antialiasing path the surface needs, whether a filter is
active, what the output format is. Each of those facts could pick a smaller, faster GPU
program — but only if something decides, before a retained scheme is recorded, which
program the next hundred frames will want.

This note describes how Goldy makes that decision generically, without growing a named
feature flag per specialization and without caching the full combinatorial set of
variants.

Specialization is an implementation detail of the runtime, not an API. Every program that
submits a retained scheme is a potential beneficiary without changing a line, and the
only knob is an environment variable that turns the whole thing off.

## Why not a flag per feature

The obvious approach is to let the caller name the fact — a `has_tint` boolean on a worker
description, a second retained scheme for the tinted case — and it does not scale. Tint is
one of many statically determinable specializations. Antialiasing mode, filters,
non-default blend, and output format are others, and a higher-level compiler will emit
facts Goldy has never heard of. Each named flag doubles the number of retained schemes a
client has to hold, and each retained scheme carries its own command lists, topology
edges, and partition bookkeeping.

Goldy already resists shader permutations rather than industrializing them (see
[What Goldy Sheds](./what-goldy-sheds.md)). The mechanism here follows that stance: the
runtime holds **at most one specialized program per dispatch site**, plus a universal
program that is always correct.

A CPU branch predictor is the precedent. It does not expose a named flag for each `if` in
user code, and it does not ask the program what it expects; it indexes opaque history by
instruction identity and outcome.

| CPU branch predictor | Goldy |
|---|---|
| Branch PC | Stable dispatch-site identity (`NodeId` on a scheme) |
| Branch outcome | Scalar param wire words the site last dispatched with |
| Hidden history table | Small per-site predictor state |
| Generic code path | Universal shader (reads its params at runtime) |
| Optimized target | Specialized PSO plus the re-recorded partition that binds it |

## Where the facts come from

Goldy does not need to be told what to specialize on, because it already holds the facts:
the scalar params of every dispatch site.

`Scheme::with_param` takes a `u32` wire word. Those words are the program's own encoding of
whatever it decided — a tint factor, a mode enum, a filter toggle, a count — and Goldy
stores them on the dispatch node, hashes them into the emission fingerprint, and writes
them into push constants on every record. A value that has been the same for the last ten
frames is a fact about the scene, and the runtime can see that without being told.

So the specialization key for a site is derived, not supplied: it is the tuple of scalar
wire words the site dispatches with. Goldy never interprets a word. It only compares words
for equality across frames, which is exactly what makes the mechanism generic — a new
specialization axis is a new `with_param` in the caller, not a new field in a Goldy struct.

### Baking a param

The virtual-main transform reads every scalar param through a preprocessor macro that
defaults to the push-constant word:

```hlsl
#ifndef _GOLDY_SPEC_CS_MAIN_UW0
#define _GOLDY_SPEC_CS_MAIN_UW0 _uw0
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uniform uint _bw0, /* ... */ uniform uint _uw0, /* ... */) {
    uint factor = _GOLDY_SPEC_CS_MAIN_UW0;   // was: _uw0
    // ...
}
```

Defining that macro to a wire-word literal bakes the value. `ShaderModule::variant` already
merges defines onto a retained module, so a specialized program is
`variant(&[("_GOLDY_SPEC_CS_MAIN_UW0", "10u")])` and nothing more. The macro is named after
the entry point so that sources with several entries specialize independently, and the
define always carries the raw `u32` wire word, so the runtime needs no knowledge of the
param's Slang type — the existing `asfloat` / `asint` / `!= 0u` decode applies to the
literal exactly as it applied to the push-constant word.

Two properties make this safe to swap under an already-recorded dispatch:

- **The push-constant ABI does not move.** The wrapper's signature always declares all
  eight `_uw` words regardless of use, so baking one leaves the others at their original
  offsets. A site with a baked `factor` and a runtime `bias` still reads `bias` correctly.
- **The recorded payload does not change.** Baking rewrites the program, not the command
  list. The same wire words stay baked into the partition; the specialized program simply
  stops reading them.

What the driver receives is a statically decidable branch. Slang does not run its
dead-code pass at the default optimization level, so the unreachable block survives into
the SPIR-V, but the push-constant load is gone and the condition is `OpConstantFalse` —
folding that is the first thing any driver backend does.

### Per-slot, not per-tuple

Stability is tracked per param slot, not over the whole tuple. A site that passes a stable
mode flag and a frame counter can still bake the flag. Keying on the whole tuple would
give up on any site with a single volatile param, which in practice is most of them.

### Why `with_param` and not a bound buffer

A fact in a buffer is invisible: Goldy sees a parcel read, not a value, and it would have
to read back GPU memory to learn anything. A fact in a `with_param` is already on the CPU,
already hashed, and already known to be stable or not.

The apparent objection is that scalar params are part of the partition retention
fingerprint, so flipping one re-records. That is not a cost of this mechanism, it is the
signal that drives it: a param that changes every frame re-records anyway and will never
earn promotion, and a param that is stable is both free to retain and profitable to bake.

## Two caches, deliberately separate

| Cache | Keyed by | Holds | Evicting it costs |
|---|---|---|---|
| Variant PSO cache | `(shader identity, baked slots and values)` | Compiled pipeline | A recompile |
| Per-site prediction | Dispatch-site identity | Which slots are baked | A re-record |

Keeping them separate means a site can be demoted (or its scheme dropped) without throwing
away the compiled pipeline, so re-promotion later is nearly free. The PSO cache is
device-scoped and bounded; the prediction lives with the scheme node.

The Slang bytecode disk cache already sits underneath both, keyed on post-transform source
plus defines, so a cold process still avoids full recompiles.

## The predictor

The hard part is not compiling two pipelines. It is first-frame uncertainty: when a scheme
is recorded, nothing knows whether the next hundred frames will use the same params. So the
predictor never speculates about the *current* frame — the params for the current frame are
already known exactly, and history only decides whether to prepare a specialization for
*future* frames.

### States

Compilation is expensive and uncancellable once started, so promotion is staged. Compiling
is separated from swapping, with different thresholds:

| State | Entered when | Behaviour |
|---|---|---|
| `Observing` | Site recorded, or after a miss | Universal runs; count consecutive frames with unchanged params |
| `Warming` | 2 consecutive clean submits with unchanged params | Universal still runs; variant compiles on a worker thread |
| `Promoted` | 10 consecutive matches **and** the compile succeeded | Site's pipeline swapped to the variant |
| `Pinned` | Repeated failed promotions | Universal only; no further compiles until reset |

The two-hit warm threshold exists to skip the common ping-pong case, where a value
alternates every frame and no specialization would ever pay off. The ten-hit promotion
threshold buys confidence that the streak is a scene property rather than a coincidence,
and it usually gives the compile enough time to finish before the swap is wanted.

Both thresholds are counted only on submits where the scheme was otherwise clean. A frame
that re-records for unrelated reasons tells us nothing about param stability.

### Demotion is mandatory, not optional

A promoted site is only correct while its baked words match what the program is
dispatching. `Scheme::set_node_param` is the single place a param can change, so it is also
the place that must un-promote: if the new word differs from the baked one, the site's
pipeline reverts to universal in the same call that marks the scheme params-dirty. The
partition then re-records with the universal program bound, and no submit ever runs a
program whose baked value disagrees with the frame.

This is the one invariant the implementation cannot get wrong, and it is why the swap lives
inside the runtime rather than in a caller's hands.

### Cancellation is best-effort

A param change while `Warming` sends a cancel signal. The signal is checked at three
points: before the Slang compile starts, between Slang and pipeline creation, and before
the swap. It cannot interrupt work already in flight — Slang compilation runs behind a
process-global lock and the driver's pipeline creation is not interruptible — so a
cancelled compile may still run to completion.

When it does, the resulting pipeline is still inserted into the variant PSO cache. It is
not swapped in, but it is not wasted either: if those words come back and earn promotion
later, there is nothing left to compile.

### Cost-aware promotion

A streak alone is not a reason to specialize. Recording a partition costs CPU time and, on
backends that require it, a wait for the previous retained command storage to retire.
Promotion should additionally require that the expected saving — a function of dispatch
size, since a tiny dispatch cannot repay anything — exceeds the record cost. This is what
keeps the predictor from recording extra command lists for small or short-lived surfaces.

### First frame, oscillation, reset

- **First frames** always run universal. There is no speculation before any history exists.
- **A miss** demotes immediately, as above.
- **Oscillation** is absorbed by the two-hit warm gate, and repeated failed promotions move
  the site to `Pinned` so a pathological caller cannot make Goldy compile forever.
- **Reset** happens on a topology epoch (a foreign scheme changing shared-parcel
  interaction topology, which already forces a re-record) or on an explicit purge. Both
  clear prediction state; neither needs to clear the PSO cache.

## Reflection compatibility

Baking can make a bound resource unreachable, and a shader compiler is entitled to drop an
unused binding from reflection. A dispatch node captures `slot_access` from its pipeline
when the node is built, and swapping a pipeline does not refresh it, so a variant whose
reflection disagrees with the universal program would leave the node choosing descriptors
by a stale table.

Promotion therefore compares the variant's `slot_access` against the universal program's
and refuses to promote when they differ. A site that cannot be specialized without moving
its bindings stays universal; that is a missed optimization, not a bug.

## Off switch

`GOLDY_SPECIALIZATION=0` disables prediction entirely: no history, no warm compiles, no
swaps, and every site stays on the program it was recorded with. The gate follows the
convention of [`GOLDY_DISABLE_CB_REUSE`](../appendix/environment-variables.md) — read once
through `validation_env`, with a thread-local override for tests.

## What this rests on

The mechanism depends on runtime properties that are already in place:

- **Params-only dirtiness.** Swapping a node's pipeline marks a scheme params-dirty rather
  than structurally dirty, so only partitions whose baked payload changed re-record. The
  schedule cache keys on bindings, which a pipeline swap does not touch.
- **`Scheme::set_node_pipeline`.** The swap itself, addressed by a stable node identity.
- **`ShaderModule::variant`.** A new module with merged defines, reusing retained source,
  search paths, optimization level, and layout checks.
- **Overridable scalar reads.** The virtual-main macro described above, which is what makes
  a param bakeable without the shader author participating.
- **Compile off the device lock.** `ComputePipeline::new` runs Slang outside the backend
  mutex on Vulkan and DX12, which is what makes an off-thread warm compile possible
  without stalling submits. Pipeline creation itself still holds the lock, and on Vulkan
  that is the expensive half (see
  [goldy#175](https://github.com/koubaa/goldy/issues/175)), so a warm compile can still
  contend with unrelated submits until pipeline creation moves out too.

### Shader provenance

One prerequisite is missing. A dispatch node holds a `ComputePipelineHandle`, a
`ComputePipeline` holds a backend handle, and neither can be traced back to the
`ShaderModule` that produced it — so the runtime cannot recompile the program a site is
running. Backend `ShaderState` still holds the source, but only while the caller's
`ShaderModule` is alive, and nothing links a PSO to it.

Specialization needs `ComputePipeline` to retain its compile inputs (an `Arc` of source,
search paths, defines, optimization level, and layout checks are all already `Arc`-shaped
on `ShaderModule`), and it needs the variant cache to own the pipelines it creates. This is
part of the pipeline-creation work tracked in
[goldy#175](https://github.com/koubaa/goldy/issues/175).

Ownership matters more than it looks: on Vulkan, `destroy_compute_pipeline` waits for
device idle, so evicting a variant is not a background operation. The cache has to be
bounded and evict rarely, and it must outlive any retained command list that still binds
the handle.

### Backend differences

Metal and WebGPU do not retain command lists, so partition-level resubmit counters are not
a usable predictor signal there. Scheme cleanliness is: whether a submit found the scheme
`Clean` is tracked on every backend, independent of retention, and that is what the streak
counts. The specialization mechanism therefore behaves the same everywhere; only the size
of the saving differs.

The overridable macro is emitted by the native transform. CUDA and WebGPU lower scalars
through their own paths (`_goldy_cuda_user_N`, a uniform buffer field), so prediction is a
no-op there until those paths grow the same indirection.

## Non-goals

- **Per-image or per-command program switching inside a mixed command-stream walk.** A
  single dispatch over a heterogeneous command buffer cannot pick a different program per
  element; that stays a branch inside the shader.
- **Speculating incorrectly.** A miss always runs the universal program. There is no
  "probably fine" path.
- **Named feature flags as the long-term API.** `has_tint` and its successors do not belong
  in worker topology.
- **A public specialization API.** Callers do not name keys, supply hints, or opt sites in.
  The only control is the off switch.
- **Caching every combination.** The runtime holds one promoted variant per site and a
  bounded PSO cache, not the cross product of every axis.
