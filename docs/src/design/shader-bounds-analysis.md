# Static Shader Bounds Analysis

> Status: prototype, opt-in (`GOLDY_VALIDATION=bounds`). Warnings only; a
> finding never fails a compile. Tracking issue:
> [#290](https://github.com/koubaa/goldy/issues/290).

Slang rejects *constant* out-of-bounds array indices at compile time, but a
dynamic index such as `links[link]` is only "checked" at runtime — and on most
GPUs that means an undefined read, a hang, or a device loss that is hard to
attribute back to a shader line. Goldy can do better for a well-defined class
of bugs: prove, conservatively, that every dynamic index into a statically
sized array stays inside `0 <= index < length`, and report everything it
cannot prove with the Slang source location.

## The motivating bug

```slang
groupshared int links[256];

int link = searchPredecessor(gtid.x);            // may be -1 / -2
int parent = select(link >= 0, links[link], link - 1);
```

`select` is an eager intrinsic: both arms are evaluated, so `links[link]` is
read even when `link < 0`. The explicitly guarded form is safe and must not be
reported:

```slang
int parent = link - 1;
if (link >= 0) {
    parent = links[link];
}
```

With Slang 2026.13 the scalar ternary `cond ? a : b` is lowered to control
flow (a branch plus a phi / memory merge) at every optimization level, so it
behaves like the guarded form; only `select(...)` produces an `OpSelect`. The
analysis operates on what the compiler emitted, not on the surface syntax, so
it follows whatever Slang decides.

## Integration point: SPIR-V, in Goldy

The issue asked where in Slang's lowering pipeline such an analysis belongs.
The options considered:

| Option | Verdict |
|--------|---------|
| Slang IR pass inside a Slang fork | Slang's IR has SSA, dominators and loop analysis, but it is not a public API — Goldy consumes `libslang.so` through the C API only. A fork would have to be rebuilt and re-validated on every Slang bump. Rejected for now. |
| Upstream Slang contribution | Most attractive long-term home for the *generic* parts (interval propagation on Slang IR would cover every target). The prototype here is the evidence for that conversation; see [Upstream vs Goldy-owned](#upstream-vs-goldy-owned). |
| **SPIR-V output, analyzed by Goldy** | SPIR-V is already SSA (`OpPhi`, structured control flow, `OpAccessChain` with explicit indices), Slang embeds `NonSemantic.Shader.DebugInfo.100` line info that maps back to Slang source, and the analysis sees the IR *after* all of Slang's lowering choices (select vs. branch, inlining, promotion). No compiler fork; a pure Rust module with no new dependencies. **Chosen.** |

The analysis compiles the shader a second time as a separate Slang request with
`DebugInfoLevel::Standard` (set as a session compiler option; the request-level
`spSetDebugInfoLevel` is ignored by the COM session) and analyzes that blob. The
production bytecode handed to the driver is untouched. The analysis compile runs
at the shader's optimization level, but never below `Default`: at `None` Slang
keeps aggregates in memory and inlines nothing, and the analysis only
reconstructs SSA for scalar integer locals, so nearly every index would come
through untracked memory.

Non-SPIR-V targets (DXIL, MSL, WGSL, PTX) are not analyzed. Since Goldy shaders
are single-source, a SPIR-V analysis run on a machine with the Vulkan feature
still validates the shader that will run on the other backends.

## Analysis model

Implemented in `src/slang/bounds_analysis.rs` (`analyze_spirv_bytes` /
`SlangCompiler::analyze_spirv_bounds`).

1. **SSA reconstruction.** Slang emits locals as `OpVariable` + `OpLoad` /
   `OpStore` rather than `OpPhi` at the default level. Non-escaping scalar
   integer locals are promoted to SSA (phis at the iterated dominance frontier)
   so a guard on one load refines every other load of the same variable in the
   guarded region. Local names are recovered from `OpName`, `DebugDeclare`, or
   the `DebugValue` of the value stored into the variable.
2. **Interval propagation.** Every integer SSA value gets a flow-insensitive
   interval from an ascending fixpoint with widening followed by a narrowing
   pass. Phi operands are evaluated under the facts that hold on their incoming
   edge, so the back edge of `for (i = 0; i < 8; i++)` contributes `[1, 8]`
   instead of a wrapped increment, and `verts[i]` is proven. Built-ins are
   seeded from the module: `SV_GroupThreadID` / `SV_GroupIndex` from
   `LocalSize`, `WaveGetLaneCount()` from Vulkan's `[1, 128]`, everything else
   from the type range. Arithmetic that may wrap collapses to the type range
   (sound, not precise).
3. **Path-sensitive refinement.** For each dynamic index the analysis collects
   the conditions on `OpBranchConditional` edges that dominate the access
   (`index >= 0`, `index < N`, `a >= b`, Slang's bool-phi lowering of `&&` /
   `||`, ...) and re-evaluates the index expression under them. This includes
   the relational rule `a >= b ⇒ a - b ∈ [0, hi(a) - lo(b)]` that workgroup
   scans rely on (`scratch[i - stride]` under `if (i >= stride)`). `OpSelect`
   and phi operands are evaluated under their own edge conditions.
4. **Check.** Every `OpAccessChain` index into a fixed-length array, vector or
   matrix column must satisfy `0 <= index <= length - 1`. Constant indices are
   skipped (Slang already validates them). Runtime arrays (buffers) are not
   checked: their length is a runtime property of the binding.

Every finding carries the array name (or `<unnamed float2 array>` for
`static const` tables Slang folds into anonymous temporaries), the array
length, the proven index range when it is narrower than the type range, the
Slang source location, and a *provenance note* naming what the index depends
on that the analysis cannot bound (`SV_VertexID`, `WaveGetLaneCount()`, a
buffer load, a float-to-int conversion, an un-inlined call, a widened
loop-carried value).

## What the prototype proves and reports

Covered by `src/slang/bounds_analysis/tests.rs` (every case compiles real
Slang):

| Pattern | Result |
|---------|--------|
| `select(link >= 0, links[link], link - 1)` | **reported**, range `[-2, 253] (may be negative)` |
| `if (link >= 0) parent = links[link];` | proven |
| `cond ? links[link] : x` (scalar ternary) | proven — lowered to control flow |
| `links[max(link, 0)]`, `links[i & 255]`, `links[min(i, 255)]`, `links[i % 256]` | proven |
| `if (i < 256)` alone on a signed index from a buffer | **reported** (lower bound missing) |
| `if (i >= 0 && i < 64)` on a signed index from a buffer | proven (bool-phi conjunction) |
| Workgroup inclusive scan `if (lid >= s) x += scratch[lid - s]`, reduce, XOR butterfly | proven |
| Same scan with the `lid >= s` guard removed | **reported** |
| Padded dispatch: `if (gid >= n) return;` then `scratch[gtid.x]` | proven |
| Padded dispatch: sentinel `uint idx = gid < n ? gid : 0xffffffff; scratch[idx]` | **reported**, `depends on SV_DispatchThreadID` |
| `for (int i = 0; i < 8; i++) verts[i]` (function-local array) at `None` / `Default` / `High` | proven |
| `static const float2 quad[6]; quad[vertex_id]` | **reported**, `depends on SV_VertexID`; `quad[vertex_id % 6]` proven |
| `palette[int(hue * 6.0)]` | **reported**, `depends on a float-to-int conversion` |
| `totals[gtid.x / WaveGetLaneCount()]` with `totals[32]`, 256 threads | **reported**, `[0, 255]` — safe only if the subgroup has ≥ 8 lanes |
| Call result at `OptimizationLevel::None` (not inlined) | **reported** as unknown; documented intraprocedural limit |

## Corpus evaluation

All 33 shaders under `shaders/` (54 entry points) were analyzed at `Default`
and `High`; the two levels agree. Of 44 dynamic indices into fixed-length
aggregates, 31 are proven and 13 are reported:

| Shader | Finding | Assessment |
|--------|---------|------------|
| `game_of_life_render`, `particle_render`, `rain_snow_render`, `starfield_render` (6 sites) | `static const` vertex table indexed by `SV_VertexID` | Unprovable from the shader: correctness depends on the draw's vertex count. The idiomatic `quad[vertex_id % 6]` is proven; documented as the recommended form. |
| `starfield_render` (2 sites) | `galaxyColors[int(hue)]`, `galaxyColors[int(hue) + 1]` | Safe if `hash1 < 1.0`; the analysis has no float intervals. Real dependency on a float invariant the code does not state. |
| `goldy_exp/collectives.slang` `workgroup_inclusive_scan_wave_uint_sum` (3 sites) | `scratch_wave_totals[32]` indexed by `local_ix / lanes` and `i < 256 / lanes` | True precondition: the function's own comment says sizes `<= 32` cover 256 threads only for subgroups of 8+ lanes. Vulkan only guarantees `[1, 128]`. Encoding the assumption (`lanes = max(WaveGetLaneCount(), 8)`) would make it provable. |
| `goldy_exp/collectives.slang` `workgroup_upper_bound` (1 site) | `prefix_sums[probe - 1]` in a branchless binary search | **False positive.** `probe - 1 ∈ [0, N - 2]` holds by a summation invariant over the loop (`ix` accumulates halving strides) that interval analysis with widening cannot express. Reported with `depends on a loop-carried value the analysis could not bound`. |
| `test_goldy_exp_interlocked.slang` (1 site) | `groupshared sh[64]` indexed by `SV_DispatchThreadID` | Latent bug in a compile-only test: correct only when exactly one workgroup is dispatched. |

Barrier-heavy algorithms (`test_collectives`, `test_monoids`, `test_algebra`:
scans, reduces, butterflies, prefix sums — 30 dynamic indices) are otherwise
clean. Before the corpus run, every counted loop over a local array
(`spinning_cube.slang`) was a false positive; that drove the edge-refined phi
evaluation and the narrowing phase.

The full run log is attached to the pull request that introduced this document.

## Known limitations

- **Intraprocedural.** Un-inlined call results are unknown. Slang inlines
  nearly everything at `Default`+, so this matters mostly at `None`, which the
  opt-in hook avoids by analyzing at `Default`.
- **No float intervals.** Any index derived from a float conversion is
  unknown.
- **Memory model.** Only non-escaping scalar integer `Function` variables are
  promoted; aggregates in memory (structs, vectors, arrays of indices) are
  untracked. Slang's own promotion at `Default`+ makes this rare in practice.
- **Intervals only.** No relations between variables beyond dominating
  comparisons, no loop summation invariants, no cross-thread reasoning
  (`groupshared` contents are not modeled, only their indices).
- **Subgroup size.** `WaveGetLaneCount()` is `[1, 128]` per Vulkan. Shaders
  that assume a minimum lane count should state it (`max(lanes, 8)`).
- **SPIR-V only**, and only when the `vulkan` feature compiles shaders to
  SPIR-V.

## Upstream vs Goldy-owned

Decision for now: **Goldy-owned analysis stage over SPIR-V**, with an eye to
upstreaming the *generic* pieces.

- The pieces that are Goldy-specific and stay here: reading the lowered SPIR-V,
  mapping locations back through the virtual-main rewrite, the
  `GOLDY_VALIDATION` plumbing, seeding built-ins from `LocalSize`, and the
  corpus-tuned diagnostics (provenance notes, unnamed-table naming).
- The pieces that are generic and would serve every Slang target: interval
  propagation with edge-refined phis and narrowing, dominating-comparison
  refinement including the bool-phi `&&`/`||` rule and the `a >= b ⇒ a - b >= 0`
  relational rule, and the `GetElementPtr`/`OpAccessChain` check with a
  "cannot prove `0 <= index < length`" warning. Slang's IR already has the
  infrastructure (SSA, dominators, `IRIntegerRelation`, loop analysis), so this
  is a plausible upstream proposal once the false-positive profile above has
  been discussed with the Slang maintainers.
- What we would need from upstream regardless: a way to surface the analysis
  through the C API so Goldy can keep formatting locations through its own
  `#line` mapping.

## Running it

```bash
# Warn on every dynamic index the analysis cannot prove in bounds
GOLDY_VALIDATION=bounds cargo run --features examples --example metaballs
RUST_LOG=goldy::slang=debug GOLDY_VALIDATION=bounds cargo test --features vulkan

# Included in `all`
GOLDY_VALIDATION=all cargo test --features vulkan

# Keep the debug-info SPIR-V the analysis looked at
GOLDY_DUMP_SHADERS=/tmp/shaders GOLDY_VALIDATION=bounds cargo run ...   # {entry}_bounds_debug.spv
```

Findings are logged once per distinct compile (including shader disk-cache
hits) at `warn` under the `goldy::slang` target:

```text
shader bounds: possible out-of-bounds index into `links[256]`: index range [-2, 253] (may be negative) (depends on SV_GroupThreadID) at shader.slang:11:1 in `cs_main`
```

Programmatic access: `SlangCompiler::analyze_spirv_bounds` returns a
`BoundsReport` with `checked_accesses`, `proven_safe` and the list of
`BoundsDiagnostic`s.
