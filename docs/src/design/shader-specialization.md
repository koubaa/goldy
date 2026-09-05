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
only knob is an environment variable that turns the whole thing off. Baking is a full
recompile: the specialized program can elide whole code paths, not merely load a
constant. How optimistic to be depends on whether the baked word *gates* work; see
[What baking actually compiles](#what-baking-actually-compiles).

**Status:** implemented in `src/specialization.rs`, driven from `Scheme::submit` and
`Scheme::set_node_param`. Sections below describe the shipped behaviour; the closing
[Follow-ups](#follow-ups) list what was deliberately left out.

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
the author's `[goldy_compute]` function (`cs_main` above; `_GOLDY_SPEC_TINT_UW0` for
`void tint(...)`) so that sources with several entries specialize independently, and the
define always carries the raw `u32` wire word, so the runtime needs no knowledge of the
param's Slang type — the existing `asfloat` / `asint` / `!= 0u` decode applies to the
literal exactly as it applied to the push-constant word. The runtime recovers the function
name from the retained source (`single_compute_entry_name`); a source with zero or several
compute entries has no unambiguous macro to define and is left alone.

Two properties make this safe to swap under an already-recorded dispatch:

- **The parameter ABI does not move.** The wrapper's signature always declares all eight
  `_uw` words regardless of use, so baking one leaves the others at their original offsets.
  A site with a baked `factor` and a runtime `bias` still reads `bias` correctly. The
  WebGPU and CUDA lowerings keep their uniform field and kernel argument for the same
  reason.
- **The recorded payload does not change.** Baking rewrites the program, not the command
  list. The same wire words stay baked into the partition; the specialized program simply
  stops reading them.

What baking must not do is change the pipeline's *binding* layout, which is a real
constraint on some backends and is covered below.

### What baking actually compiles

Baking is a full recompile, not a constant patch. The specialized program is a new
`ShaderModule::variant` with the `_GOLDY_SPEC_*` macros defined to wire-word literals,
so Slang, the backend IR (SPIR-V / DXIL / MSL / PTX), and the driver's ISA compiler all
see a compile-time constant. That constant can participate in folding anywhere it flows:
branch conditions, loop trip counts, array indices, `asfloat(...)` arithmetic, resource
selection. Nothing is rewritten at bind time, and the universal program is left alone.

Elision is split across two compilers, and that split is easy to misread.

Goldy's default `OptimizationLevel::Default` does not run Slang's dead-code pass, so a
branch whose condition has become `false` still appears in the dumped SPIR-V — as
`OpConstantFalse` plus the unreachable block. The push-constant load is already gone at
that point. Folding a constant condition, deleting the dead block, and reallocating
registers is the first thing any production driver compiler does (NVIDIA, AMD's LLVM,
Apple's, DXC, Mesa lavapipe). Dumping SPIR-V instruction counts therefore understates
the specialized program: the variant that looks almost the same as the universal one at
the SPIR-V level can be a fraction of the size after the driver.

Raising Slang to `OptimizationLevel::Maximal` does the DCE in SPIR-V itself. That moves
compile time; it does not improve the ISA the GPU runs. Variants inherit the module's
optimization level and there is no reason to raise it for baking.

Measured on lavapipe through Goldy's real Vulkan compile path, a compute shader whose
`has_tint` scalar gated a tinting loop:

| Program | SPIR-V instructions | SPIR-V still has the branch+loop | Driver NIR instructions | `if` / `fmul` |
|---|---|---|---|---|
| Universal | 212 | yes | 201 | 1 / 55 |
| Baked `has_tint = 0` | 210 | yes, on `OpConstantFalse` | 36 | 0 / 0 |
| Baked `has_tint = 1` | 207 | yes | 194 | 0 / 55 |
| Baked `has_tint = 0`, Slang `Maximal` | 91 | no | 36 | 0 / 0 |

The specialized-off program dropped the extra buffer loads, the extra push-constant
loads, the unrolled loop, and the registers they held. The specialized-on program
dropped only the `if` and the two unused push-constant loads, which is the expected
"only what the constant makes dead goes away" behaviour. Numbers will differ on a
hardware compiler; the split between Slang and the driver will not.

### When this is worth expecting

A uniform branch on a push constant is already non-divergent and cheap. Removing the
`if` itself is not the win. The win is whatever sits behind it.

| Kind of `with_param` | What baking does | How optimistic to be |
|---|---|---|
| Feature / mode flag that gates whole code paths (`has_tint`, AA mode, filter on) | The driver deletes the unused path, its memory traffic, and the registers it held — occupancy can rise | High, if the gated work is real |
| Loop bound or "which of N resources" index | Trip count or index becomes a literal; unrolling and addressing follow | High when the bound is small and the body is heavy |
| Arithmetic coefficient (`x * factor + bias`) | One less push-constant load; maybe a strength reduction | Modest |
| Value that changes every few frames | Predictor raises the bake threshold and leaves the slot dynamic | None — it will not stay promoted |
| Value that lives in a bound buffer | Invisible to the predictor | None — put scene facts in `with_param` if they should bake |

The cost is a full Slang compile plus PSO creation on a worker thread, which is why
warm waits for two clean hits and promotion waits for ten. A scheme whose many nodes
all stabilize at once will spend real CPU in the background for a moment; the 16-entry
per-scheme cache bounds the steady state.

Shader authors do not opt sites in, but they do choose where facts live. A mode flag
in a `Scattered` buffer cannot bake. A mode flag passed with `with_param`, with the
expensive work behind `if (has_tint != 0u)`, is exactly the shape the predictor was
built for. That is also why Goldy still [resists permutation systems](./what-goldy-sheds.md):
the runtime holds at most one specialized program per dispatch site, not the cross
product of every flag.

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

Keeping them separate means a site can be demoted without throwing away the compiled
pipeline, so re-promotion later is nearly free: a demoted site whose words come back is
promoted straight from the cache once its streak recovers, with no compile.

The PSO cache is **scheme-scoped** and bounded (16 variants, LRU). A device-scoped cache
was the first design and does not work as stated: a `ComputePipeline` holds a strong
`Device`, so a cache inside the device would form a reference cycle and the device would
never be dropped. Scoping the cache to the scheme also settles ownership — the scheme holds
an `Arc` to the variant it has promoted, and the cache holds another, so eviction from the
cache never destroys a pipeline a node still binds. The price is that two schemes running
the same shader with the same stable words each compile their own variant; the Slang disk
cache and the driver's PSO cache make the second compile cheap, and device-level sharing is
listed as a follow-up.

Compile workers write into the cache directly, so a variant that finishes after its site
lost interest still lands there rather than being dropped.

The Slang bytecode disk cache already sits underneath both, keyed on post-transform source
plus defines, so a cold process still avoids full recompiles.

## The predictor

The hard part is not compiling two pipelines. It is first-frame uncertainty: when a scheme
is recorded, nothing knows whether the next hundred frames will use the same params. So the
predictor never speculates about the *current* frame — the params for the current frame are
already known exactly, and history only decides whether to prepare a specialization for
*future* frames.

### Per-slot streaks

Every dispatch node with at least one `with_param` gets a site record when it is declared
(`Scheme::node`, `ComputeNodeRecord::commit_dispatch_scheme`, or a caller-side
`set_node_pipeline`, which starts the site over against the new pipeline). The record holds
the caller's pipeline (the *universal*), the shader's provenance, the words seen at the
last submit, and per slot:

- a **streak** — consecutive clean submits during which the word held its value. A changed
  word resets its slot to zero on any submit; an unchanged word advances only when the
  scheme was otherwise clean, so a scheme that re-records every frame for unrelated
  reasons keeps its history but does not earn promotions from it. Topology dirtiness
  (a foreign scheme changing shared-parcel interaction) resets every slot.
- a **bake threshold** — the streak the slot needs before it is baked. It starts at the
  warm threshold and grows every time the slot invalidates a compile or a promotion (to
  the promote threshold, then doubling), so a fact that flips every few frames is baked
  once, disproved once, and thereafter left dynamic while its neighbours specialize.

The set of slots at or past their threshold is the site's **bake target**. It is per-slot:
a site with a stable mode flag and a moving counter bakes the flag.

### Stages

Compilation is expensive and uncancellable once started, so promotion is staged. Compiling
is separated from swapping, with different thresholds:

| Stage | Entered when | Behaviour |
|---|---|---|
| Observing | Site declared, or after a demotion | Universal runs; streaks accumulate |
| Warming | Bake target non-empty and not what is already promoted | Universal still runs; a variant baking the target compiles on a worker thread (or is taken from the cache) |
| Ready | The compile landed | Universal still runs until every baked slot's streak reaches the promote threshold |
| Promoted | Every baked slot at or past the promote threshold | Node rebound to the variant as a params-only re-record; the site keeps observing the slots it did not bake and may warm a *wider* variant |
| Pinned | Three failed compiles | Universal only; the site is never consulted again |

Defaults are warm at 2, promote at 10 (`SpecializationPolicy`). The two-hit warm threshold
skips the common ping-pong case, where a value alternates every frame and no specialization
would ever pay off. The ten-hit promotion threshold buys confidence that the streak is a
scene property rather than a coincidence, and it usually gives the compile enough time to
finish before the swap is wanted.

The predictor runs at the top of `Scheme::submit`, before dirtiness is read for recording,
so a promotion is recorded by the very submit that decided it and the scheme is clean again
afterwards. It never touches a node whose current words differ from the variant's baked
words — a swap is only ever to a program that agrees with the frame.

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

A baked word changing while a compile is in flight — in `set_node_param`, or observed at
the next submit — drops the job and raises its cancel flag. The flag is honoured before the
compile starts; it cannot interrupt work already running, because Slang compilation runs
behind a process-global lock and the driver's pipeline creation is not interruptible, so a
cancelled compile may still run to completion.

When it does, the resulting pipeline is still inserted into the variant PSO cache. It is
not swapped in, but it is not wasted either: if those words come back and earn promotion
later, there is nothing left to compile.

### Cost model

A streak alone is not the whole story. Recording a partition costs CPU time and, on
backends that require it, a wait for the previous retained command storage to retire; a
tiny dispatch cannot repay that. The shipped predictor has no dispatch-size term — the
thresholds are the only cost model — so a small, long-lived, stable dispatch pays one
re-record it may never recoup. That is bounded (one record per promotion, promotions are
rare by construction) and is listed as a follow-up rather than guessed at without profiles.

### First frame, oscillation, reset

- **First frames** always run universal. There is no speculation before any history exists.
- **A miss** demotes immediately, as above.
- **Oscillation** is absorbed by the two-hit warm gate and the growing per-slot bake
  threshold, and three failed compiles move the site to Pinned so a pathological shader
  cannot make Goldy compile forever.
- **Reset** happens on topology dirtiness (a foreign scheme changing shared-parcel
  interaction topology, which already forces a re-record) and on a caller-side
  `set_node_pipeline`, which replaces the universal. Both clear streaks; neither clears the
  PSO cache.
- **Turning the feature off** at runtime (the environment variable is read every submit)
  demotes every promoted site on the next submit and drops in-flight compiles.

## Baking must not change the binding layout

A specialized variant is swapped under bindings that were already resolved when the node
was recorded, so it has to agree with the universal program about what those bindings are.
Baking can break that agreement, because making a value constant can make a *resource*
unreachable, and a compiler is entitled to stop expecting a binding nothing reads.

This is not hypothetical, and it is not uniform across backends. Vulkan, DX12, and Metal
bind through a bindless heap, so a pipeline's expectations follow the shader signature and
survive baking. WebGPU does not: wgpu derives the bind group layout from the compiled WGSL,
so it reflects *usage*. Two consequences, both observed:

- Baking a value that was the only reason a bound buffer was read drops that buffer from
  the layout, and the recorded dispatch then supplies a binding the pipeline no longer
  declares.
- Baking **every** scalar of an entry point leaves the generated user-params uniform buffer
  unreferenced, which drops it the same way — so on those backends at least one scalar slot
  has to stay dynamic.

Neither constraint is enforced by the predictor, because the predictor does not run where
they apply (next section). On the backends where it does run, the layout follows the
signature and a variant binds exactly like its universal, so the site's bake target can be
every stable slot.

A caller-side `set_node_pipeline` on WebGPU is a different matter: `slot_access` cannot
check it — that table is derived from the shader signature, so it agrees across a variant
that a usage-derived layout would reject — and wgpu exposes no entry count for a
`BindGroupLayout`, so an incompatible hand swap surfaces as a bind group mismatch inside
the backend rather than as a `set_node_pipeline` error.

A site that cannot be specialized without moving its bindings must stay universal. That is
a missed optimization, not a bug — but it has to be detected rather than assumed.

### Gate the backend, do not tier the feature

The awkward part is that the specializations most worth having are the ones that drop a
binding. `if (has_tint) { read the tint buffer; blend }` is the motivating case, and baking
`has_tint` to zero is exactly what makes that buffer unreachable. On a backend with
usage-derived layouts, the predictor would decline promotion precisely where it would have
paid off most.

That is a reason to gate WebGPU, not to make specialization a property of "better"
backends. Usage-derived layouts are not a WebGPU limitation: the backend passes
`layout: None` and takes wgpu's auto layout. An explicit pipeline layout built from the
signature-derived `WgpuComputeLayout` the backend already computes would be stable under
baking, because a layout may declare bindings the shader does not reference. Until that
lands, the honest position is that Goldy's WebGPU backend cannot support specialization,
not that WebGPU cannot.

So promotion is conditional on a backend capability —
`GpuBackend::compute_pipeline_layout_follows_signature` — rather than on a backend
allow-list. Vulkan, DX12, Metal, CUDA, and the mock backend report yes; WebGPU reports no
while it uses auto layouts, and flips on by itself once it does not; the CPU backend
reports no because its host-callable lowering has no bake macros, so a variant would be the
same kernel. The predictor stays backend-agnostic either way, and the capability is
internal: specialization is an implementation detail, so it does not belong in
`DeviceCapabilities` where a program would branch on it. A scheme queries it once, on its
first submit.

## Off switch

Specialization is on by default. `GOLDY_SPECIALIZATION=0` (or `false` / `no` / `off`)
disables prediction entirely: no history, no warm compiles, no swaps, and every site stays
on the program it was recorded with. The gate follows the convention of
[`GOLDY_DISABLE_CB_REUSE`](../appendix/environment-variables.md) — read through
`validation_env`, with a thread-local override (`test_support::SpecializationOverride`) so
tests that count records across many frames can pin it.

This is separate from the backend capability above. The environment variable is a global
kill switch for when prediction is suspected of causing a problem; the capability decides
where prediction can be correct at all.

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

A dispatch node holds a `ComputePipelineHandle` and a `ComputePipeline` holds a backend
handle; neither, on its own, says which program it is. So every `ShaderModule` keeps its
compile inputs — source, search paths, defines, optimization level, layout checks — in a
shared `ShaderProvenance` with a process-unique id, and every `ComputePipeline` built from
the module carries an `Arc` to it. The runtime can therefore compile a variant of the
program a site is running after the caller has dropped the module
(`ShaderModule::from_provenance`), and variants are keyed by provenance id plus baked
words.

Ownership matters more than it looks: on Vulkan, `destroy_compute_pipeline` waits for
device idle, so dropping a variant is not a background operation. The scheme holds the
variant it has promoted; when a variant is unbound (demotion, a wider promotion, a caller
swap) it is parked for two further submits before its `Arc` is released, by which point no
retained command list that bound it is still the one being submitted. The bounded cache
usually still holds it after that.

### Telemetry

`ReplayStats` gains `specialization_warms`, `specialization_promotions`, and
`specialization_demotions`; `Scheme::node_is_specialized(NodeId)` answers for one site.
Demotions are visible in the stats immediately after the `set_node_param` that caused
them. Each transition also emits a `tracing` event under the `goldy` target (`debug` for
warm / promote / demote, `warn` for a failed compile or a pinned site).

### Backend differences

Metal and WebGPU do not retain command lists, so partition-level resubmit counters are not
a usable predictor signal there. Scheme cleanliness is: whether a submit found the scheme
`Clean` is tracked on every backend, independent of retention, and that is what the streak
counts. The specialization mechanism therefore behaves the same everywhere; only the size
of the saving differs.

The overridable macro is emitted by every compute lowering — the native push-constant path,
the WebGPU uniform-buffer path, and the CUDA kernel-argument path — each defaulting the
macro to its own read expression, so baking works the same way everywhere and the launch
layout is unchanged in all three. Graphics stages lower scalars without the indirection;
specialization is defined for dispatch sites, so that is deliberate.

## Follow-ups

- **Cost model.** A dispatch-size (or measured-duration) term so tiny dispatches are never
  promoted. Needs profiles from a real consumer before the threshold is more than a guess.
- **Device-level variant sharing.** Many schemes running one shader with the same stable
  words compile it once each today. Sharing needs an owner that does not cycle with
  `Device`, e.g. variants that hold a weak device reference and a cache that drains on
  device drop.
- **WebGPU explicit layouts.** Building the pipeline layout from the signature-derived
  `WgpuComputeLayout` would let the backend report `compute_pipeline_layout_follows_signature`
  and enable prediction there with no predictor changes.
- **Graphics stages.** Draw sites with scalar params lower without the macro indirection;
  specialization is defined for compute dispatch sites only.
- **Pipeline creation off the lock.** Warm compiles still take the backend mutex for PSO
  creation ([goldy#175](https://github.com/koubaa/goldy/issues/175)).

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
