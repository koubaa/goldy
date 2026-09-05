# Shader Specialization Prediction

Programs know things about a frame that a shader cannot see: whether any image in the
scene carries a tint, which antialiasing path the surface needs, whether a filter is
active, what the output format is. Each of those facts could pick a smaller, faster GPU
program — but only if something decides, before a retained scheme is recorded, which
program the next hundred frames will want.

This note describes how Goldy makes that decision generically, without growing a named
feature flag per specialization and without caching the full combinatorial set of
variants.

Nothing in this note is implemented yet. It is the contract that the implementation is
expected to satisfy, and the rationale for the parts that look surprising.

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
user code; it indexes opaque history by instruction identity and outcome.

| CPU branch predictor | Goldy |
|---|---|
| Branch PC | Stable dispatch-site identity (`NodeId` on a scheme) |
| Branch outcome | Opaque specialization key |
| Hidden history table | Small per-site predictor state |
| Generic code path | Universal shader (runtime values, full opcode set) |
| Optimized target | Specialized PSO plus the re-recorded partition that binds it |

## The mechanism

### Opaque keys

A specialization key is `(shader identity, hash(specialization bytes))`. Goldy does not
interpret the bytes. The caller produces them from whatever facts it has — preprocessor
define values today, Slang `[SpecializationConstant]` values once those are wired — and
Goldy only ever compares keys for equality.

This is what keeps the mechanism generic. Adding a new specialization axis is a change in
the caller's key derivation, not a new field in a Goldy struct.

### The universal fallback always exists

Every dispatch site is recorded against a universal program that is correct for any key.
Reading a specialization fact at runtime (from a bound buffer, not from a baked parameter)
is slower than branching at compile time, but it is never wrong. A misprediction costs
performance, never correctness, and the runtime must always be able to fall back to it.

One caveat matters for callers: the universal program must take its specialization facts
from a **bound parcel**, not from a `with_param` scalar. Scalar params are baked into the
emitted command list and are part of the partition retention fingerprint, so flipping one
re-records the partition. A fact that lives in a buffer keeps the universal path free
across frames, which is exactly what the first frames of any site depend on.

### One promoted variant per site

Promotion *replaces* the previous specialization for that site rather than accumulating
variants. A site oscillating between two keys does not end up holding two retained
recordings; it ends up pinned to universal (see below).

### Two caches, deliberately separate

| Cache | Keyed by | Holds | Evicting it costs |
|---|---|---|---|
| Variant PSO cache | `(shader identity, specialization key)` | Compiled pipeline | A recompile |
| Per-site prediction | Dispatch-site identity | Which key is promoted | A re-record |

Keeping them separate means a site can be demoted (or its scheme dropped) without throwing
away the compiled pipeline, so re-promotion later is nearly free. The PSO cache is
device-scoped and bounded; the prediction lives with the scheme node.

The Slang bytecode disk cache already sits underneath both, keyed on post-transform source
plus defines, so a cold process still avoids full recompiles.

## The predictor

The hard part is not compiling two pipelines. It is first-frame uncertainty: when a scheme
is recorded, nothing knows whether the next hundred frames will use the same key. So the
predictor never speculates about the *current* frame — the caller supplies the current
frame's exact key before submit, and history only decides whether to prepare a
specialization for *future* frames.

### States

Compilation is expensive and uncancellable once started, so promotion is staged. Compiling
is separated from swapping, with different thresholds:

| State | Entered when | Behaviour |
|---|---|---|
| `Observing` | Site recorded, or after a miss | Universal runs; count consecutive matching keys |
| `Warming` | 2 consecutive clean submits with the same key | Universal still runs; variant compiles on a worker thread |
| `Promoted` | 10 consecutive matches **and** the compile succeeded | Site's pipeline swapped to the variant |
| `Pinned` | Repeated failed promotions | Universal only; no further compiles until reset |

The two-hit warm threshold exists to skip the common ping-pong case, where a value
alternates every frame and no specialization would ever pay off. The ten-hit promotion
threshold buys confidence that the streak is a scene property rather than a coincidence,
and it usually gives the compile enough time to finish before the swap is wanted.

Both thresholds are counted only on submits where the scheme was otherwise clean. A frame
that re-records for unrelated reasons tells us nothing about key stability.

### Cancellation is best-effort

A key mismatch while `Warming` sends a cancel signal. The signal is checked at three
points: before the Slang compile starts, between Slang and pipeline creation, and before
the swap. It cannot interrupt work already in flight — Slang compilation runs behind a
process-global lock and the driver's pipeline creation is not interruptible — so a
cancelled compile may still run to completion.

When it does, the resulting pipeline is still inserted into the variant PSO cache. It is
not swapped in, but it is not wasted either: if that key comes back and earns promotion
later, there is nothing left to compile.

### Cost-aware promotion

A streak alone is not a reason to specialize. Recording a partition costs CPU time and, on
backends that require it, a wait for the previous retained command storage to retire.
Promotion should additionally require that the expected saving — a function of dispatch
size, since a tiny dispatch cannot repay anything — exceeds the record cost. This is what
keeps the predictor from recording extra command lists for small or short-lived surfaces.

### First frame, oscillation, reset

- **First frames** always run universal. There is no speculation before any history exists.
- **A miss** demotes immediately: the site returns to universal and the streak resets. The
  runtime never runs a program that does not match the current key.
- **Oscillation** is absorbed by the two-hit warm gate, and repeated failed promotions move
  the site to `Pinned` so a pathological caller cannot make Goldy compile forever.
- **Reset** happens on a topology epoch (a foreign scheme changing shared-parcel
  interaction topology, which already forces a re-record) or on an explicit purge. Both
  clear prediction state; neither needs to clear the PSO cache.

### Caller hints

A caller may constrain the predictor per site with `Disabled` / `Enabled` / `Auto`.
`Disabled` pins universal, `Enabled` skips the streak thresholds for a fact the caller
knows is scene-static, `Auto` is the default. Hints constrain policy only — they are not a
place to add Goldy feature booleans, and they never change what a key means.

## What this rests on

The mechanism depends on runtime properties that are already in place:

- **Params-only dirtiness.** Swapping a node's pipeline marks a scheme params-dirty rather
  than structurally dirty, so only partitions whose baked payload changed re-record. The
  schedule cache keys on bindings, which a pipeline swap does not touch.
- **`Scheme::set_node_pipeline`.** The swap itself, addressed by a stable node identity.
- **`ShaderModule::variant`.** A new module with merged defines, reusing retained source,
  search paths, optimization level, and layout checks.
- **Compile off the device lock.** `ComputePipeline::new` runs Slang outside the backend
  mutex on Vulkan and DX12, which is what makes an off-thread warm compile possible
  without stalling submits. Pipeline creation itself still holds the lock, and on Vulkan
  that is the expensive half (see
  [goldy#175](https://github.com/koubaa/goldy/issues/175)), so a warm compile can still
  contend with unrelated submits until pipeline creation moves out too.

### Backend differences

Metal and WebGPU do not retain command lists, so partition-level resubmit counters are not
a usable predictor signal there. Scheme cleanliness is: whether a submit found the scheme
`Clean` is tracked on every backend, independent of retention, and that is what the streak
counts. The specialization mechanism therefore behaves the same everywhere; only the size
of the saving differs.

## Non-goals

- **Per-image or per-command program switching inside a mixed command-stream walk.** A
  single dispatch over a heterogeneous command buffer cannot pick a different program per
  element; that stays a branch inside the shader.
- **Speculating incorrectly.** A miss always runs the universal program. There is no
  "probably fine" path.
- **Named feature flags as the long-term API.** `has_tint` and its successors do not belong
  in worker topology.
- **Caching every combination.** The runtime holds one promoted variant per site and a
  bounded PSO cache, not the cross product of every axis.
