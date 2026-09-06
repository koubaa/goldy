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
cannot prove with the Slang source location and the call path that reaches it.

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
flow (a branch plus a block parameter) already in the front end, so it behaves
like the guarded form; only `select(...)` produces an eager `select`
instruction. The analysis operates on what the compiler emitted, not on the
surface syntax, so it follows whatever Slang decides.

## Integration point: Slang IR, in Goldy

The issue asked where in Slang's lowering pipeline such an analysis belongs.
The first prototype analyzed the SPIR-V Slang emits; it was then retargeted to
Slang's own IR and the SPIR-V implementation removed. The options, revisited
with what each representation actually offers:

| Option | Verdict |
|--------|---------|
| Slang IR pass inside a Slang fork | Slang has SSA, dominators and loop analysis, but its IR is not a public API — Goldy consumes `libslang.so` through the C API only. A fork would have to be rebuilt and re-validated on every Slang bump. Rejected. |
| Upstream Slang contribution | Most attractive long-term home for the *generic* parts. The prototype here is the evidence for that conversation; see [Upstream vs Goldy-owned](#upstream-vs-goldy-owned). |
| SPIR-V output, analyzed by Goldy | Already SSA with structured control flow and `OpAccessChain`; `NonSemantic.Shader.DebugInfo.100` maps back to Slang source. But the analysis only sees the shader *after* Slang's target lowering: every generic is monomorphized and every helper inlined (so a finding in a library function is reported once per inlined copy, with no call path), struct fields become anonymous member indices, names survive only as debug hints, and it only exists when the Vulkan feature compiles SPIR-V. The first prototype; superseded. |
| **Slang front-end IR, analyzed by Goldy** | The `.slang-module` container Slang serializes for a translation unit (`spSetOutputContainerFormat` + `spGetContainerCode`) is a stable, documented format (`docs/design/serialization.md`, "fossil") with stable opcode names, and with `DebugInfoLevel::Standard` every instruction carries a source location. The IR is what the front end produced after semantic checking and SSA construction, *before* target lowering: functions, generics, witness tables, struct keys, structured `ifElse`/`loop`/`switch`, `getElementPtr` with the array type intact, and name hints on everything the user named. Target-independent, so one analysis covers every backend. No compiler fork; a pure Rust reader (RIFF + fossil) with no new dependencies. **Chosen.** |

### What changed in the analysis when the input changed

Slang IR and SPIR-V expose different information, and the analysis was
revisited accordingly rather than ported one-to-one:

- **Calls are real.** Nothing is inlined in the front-end IR, so an
  intraprocedural analysis would know nothing about `searchPredecessor(...)`.
  The analysis became *interprocedural and context-sensitive*: each callee is
  analyzed with the argument intervals of the call site, summaries (return
  value plus what was written through `out`/`inout` parameters) are memoized
  per `(function, argument intervals, generic substitution)`, and a finding in
  a helper is reported once, with the union of the ranges over every calling
  context and the shortest call path from the entry point. Recursion, depth
  and the number of contexts per function are bounded.
- **Generics are still generic.** `workgroup_reduce<T : IMonoid, let N : int>`
  is one body; a call is `specialize(generic, uint, 64, witness)`. The
  analysis carries the substitution into the body: `N` folds to a constant
  (also inside `constexpr*` length expressions such as `1 << LG_N`), values of
  type `T` take the argument's integer shape (or struct layout), and
  `T.combine(...)` is a `lookupWitness(table, key)` resolved through the
  witness table to the concrete conformance — including one defined in the
  shader for its own struct.
- **Imported modules are separate containers.** Slang does not embed
  `goldy_exp` in the translation unit's IR; it references its declarations
  through `import`-decorated stubs carrying the mangled name. Goldy compiles
  each imported module (transitively, discovered from those names) to its own
  container, caches it, and *links* everything into one instruction space by
  redirecting each stub to the `export`-decorated definition. After linking,
  cross-module calls, generic arguments, witness tables and struct field keys
  are ordinary intra-module references. Unresolvable stubs stay opaque and are
  named in the diagnostic (`no body available`).
- **Locals are memory, not phis.** The front end keeps `var`/`store`/`load`
  for constructors and `out` parameters, and copies by-value aggregate
  parameters into a local. Non-escaping locals are read through their stores
  (field-precise, so struct-valued values keep per-field ranges); escaping
  ones are unknown.
- **No common-subexpression elimination.** `i + 1` in a guard and `i + 1` in
  the index are distinct instructions. A value-numbering pass canonicalizes
  structurally identical pure instructions so that dominating facts apply to
  every copy of the expression they mention.
- **Debug info is in-band.** With debug info Slang mirrors stores into by-value
  aggregate parameters onto `DebugVar` pseudo-variables; those `getElementPtr`s
  are not real accesses and are skipped, otherwise every such site would be
  counted twice.
- **Built-ins are still seeded from the module** (`[numthreads]` for
  `SV_GroupThreadID` / `SV_GroupIndex`, `[1, 128]` for `WaveGetLaneCount()`,
  the type range for everything else) and core-module intrinsics (`min`,
  `clamp`, `abs`, `countbits`, `firstbitlow`, wave queries, ...) are modeled
  by name; `firstbitlow(N)` on a constant is exact, which is how generic
  library code spells `log2(N)` in a loop bound.

The analysis compiles the shader a second time as a separate Slang request
with debug info and container output; the production bytecode handed to the
driver is untouched. Because the front-end IR precedes optimization, the
optimization level and target of the production compile do not affect the
analysis; it runs for every target.

## Analysis model

Implemented in `src/slang/bounds_analysis.rs` (`analyze_container`) with the
reader in `bounds_analysis/{riff,fossil,ir,source_loc}.rs` and the analysis
in `bounds_analysis/analysis.rs`; `SlangCompiler::analyze_bounds` drives the
compiles.

1. **Values.** Every integer scalar/vector gets an interval per lane; structs
   are tracked field by field; everything else is opaque. Entry-point
   parameters are seeded from their system-value semantic and `[numthreads]`.
2. **Interval propagation.** A flow-insensitive ascending fixpoint with
   widening followed by a narrowing pass over each function. Block parameters
   (Slang's phis) join their incoming values, each evaluated under the facts
   that hold on its edge, so the back edge of `for (i = 0; i < 8; i++)`
   contributes `[1, 8]` rather than a wrapped increment and `verts[i]` is
   proven. Arithmetic that may wrap collapses to the type range (sound, not
   precise).
3. **Path-sensitive refinement.** For each dynamic index the dominator tree is
   walked for `ifElse` / `conditionalBranch` / `switch` conditions that
   dominate the access (`index >= 0`, `index < N`, `a >= b`, the bool-phi
   lowering of `&&` / `||`, ...) and the index expression is re-evaluated
   under them, including the relational rule
   `a >= b ⇒ a - b ∈ [0, hi(a) - lo(b)]` that workgroup scans rely on
   (`scratch[i - stride]` under `if (i >= stride)`).
4. **Interprocedural.** Calls to functions with bodies (in this module or a
   linked one, generics and witness-table dispatch included) are analyzed in
   the calling context as described above.
5. **Check.** Every `getElementPtr` / `getElement` index into a fixed-length
   array, vector or matrix must satisfy `0 <= index <= length - 1`. Constant
   indices are skipped (Slang already validates them). Runtime arrays
   (buffers) are not checked: their length is a runtime property of the
   binding.

Every finding carries the array name (variable plus struct member path; a
by-value array parameter is named after the parameter), the array length, the
proven index range when it is narrower than the type range, the Slang source
location (through Goldy's `#line` mapping for `virtual_main` wrappers), the
call path, and a *provenance note* naming what the index depends on that the
analysis cannot bound (`SV_VertexID`, `WaveGetLaneCount()`, a buffer load,
groupshared memory, a float-to-int conversion, a function without a body, a
widened loop-carried value). Provenance follows arguments back through the
caller, so an index that is a field of a `ThreadId` parameter is attributed to
the `SV_DispatchThreadID` the wrapper built it from.

## What the prototype proves and reports

Covered by `src/slang/bounds_analysis/tests.rs` (every case compiles real
Slang):

| Pattern | Result |
|---------|--------|
| `select(link >= 0, links[link], link - 1)` with `searchPredecessor` as a real call | **reported**, range `[-2, 253] (may be negative)` |
| `if (link >= 0) parent = links[link];` | proven |
| `cond ? links[link] : x` (scalar ternary) | proven — lowered to control flow |
| `links[max(link, 0)]`, `links[i & 255]`, `links[min(i, 255)]`, `links[i % 256]` | proven |
| `if (i < 256)` alone on a signed index from a buffer | **reported** (lower bound missing) |
| `if (i >= 0 && i < 64)` on a signed index from a buffer | proven (bool-phi conjunction) |
| Workgroup inclusive scan `if (lid >= s) x += scratch[lid - s]`, reduce, XOR butterfly | proven |
| Same scan with the `lid >= s` guard removed | **reported** |
| Padded dispatch: `if (gid >= n) return;` then `scratch[gtid.x]` | proven |
| Padded dispatch: sentinel `uint idx = gid < n ? gid : 0xffffffff; scratch[idx]` | **reported**, `depends on SV_DispatchThreadID` |
| `for (int i = 0; i < 8; i++) verts[i]` (function-local array) | proven |
| `static const float2 quad[6]; quad[vertex_id]` | **reported**, `depends on SV_VertexID`; `quad[vertex_id % 6]` proven |
| `palette[int(hue * 6.0)]` | **reported**, `depends on a float-to-int conversion` |
| `totals[gtid.x / WaveGetLaneCount()]` with `totals[32]`, 256 threads | **reported**, `[0, 255]` — safe only if the subgroup has ≥ 8 lanes |
| Helper called from two sites, `sh[i]` for `i ∈ [0, 63]` and `i ∈ [0, 127]` | **reported once**, range `[0, 127]`, call path named |
| Struct argument `p.i` with a range, indexed in the callee | proven (fields keep their ranges) |
| `void f(out uint i)` result used as an index | analyzed through the summary |
| `wrap_read<64>` vs `wrap_read<128>` (`sh[i % N]`) | proven / **reported** — per specialization |
| `[goldy_compute]` wrapper building `GroupThreadId` through `goldy_exp` | `gtid.x` keeps its `[numthreads]` bound |
| `workgroup_reduce(gtid.x, gtid.x, scratch)` from `goldy_exp`, `T = uint` | proven (3 sites inside the library body) |
| Same with a shader-defined `struct Pair : IMonoid` whose `combine` indexes a table | **reported** in `Pair.combine`, path `cs_main -> _goldy_user_cs_main -> workgroup_reduce`, `depends on groupshared memory` |
| `pick<8>(i, uint t[1 << LG_N])` | length `256` evaluated from the generic argument |

## Corpus evaluation

All 33 shaders under `shaders/` (51 entry points; `triangle.slang` and the
`goldy_exp` library itself do not compile as stand-alone entry points) were
analyzed. Of 44 dynamic indices into fixed-length aggregates, 23 are proven and
21 are reported:

| Shader | Finding | Assessment |
|--------|---------|------------|
| `game_of_life_render`, `particle_render`, `rain_snow_render`, `starfield_render` (7 sites) | `static const` vertex table indexed by `SV_VertexID` | Unprovable from the shader: correctness depends on the draw's vertex count. The idiomatic `quad[vertex_id % 6]` is proven; documented as the recommended form. |
| `starfield_render` (2 sites) | `galaxyColors[int(hue)]`, `galaxyColors[int(hue) + 1]` | Safe if `hash1 < 1.0`; the analysis has no float intervals. Real dependency on a float invariant the code does not state. |
| `goldy_exp/collectives.slang` `workgroup_inclusive_scan_wave_uint_sum` (4 sites, reached from `test_collectives`) | `scratch_wave_totals[32]` indexed by `local_ix / lanes` and `i < 256 / lanes` | True precondition: the function's own comment says sizes `<= 32` cover 256 threads only for subgroups of 8+ lanes. Vulkan only guarantees `[1, 128]`. Encoding the assumption (`lanes = max(WaveGetLaneCount(), 8)`) would make it provable. |
| `goldy_exp/collectives.slang` `workgroup_upper_bound` (1 site) | `prefix_sums[probe - 1]` in a branchless binary search | **False positive.** `probe - 1 ∈ [0, N - 2]` holds by a summation invariant over the loop (`ix` accumulates halving strides) that interval analysis with widening cannot express. Reported with `depends on a loop-carried value the analysis could not bound`. |
| `test_goldy_exp_interlocked.slang` (7 sites) | `groupshared sh[64]` indexed by `ThreadId.x` (`SV_DispatchThreadID`) | Latent bug in a compile-only test: correct only when exactly one workgroup is dispatched. One source access per line; SPIR-V had merged them into one access chain. |

Barrier-heavy algorithms (`test_collectives`, `test_monoids`, `test_algebra`:
`workgroup_reduce`, `workgroup_inclusive_scan`, `mapped_workgroup_reduce`,
`workgroup_broadcast`, prefix sums — 20 dynamic indices inside `goldy_exp`
generics, all reached through `import goldy_exp` and specialized per call)
account for the 5 library findings above; the other 15 are proven. Every
finding in a library body is reported once with the call path, where the
SPIR-V analysis had reported each inlined copy.

The findings agree with the SPIR-V prototype's site for site; the counts
differ only because the front-end IR has one access per source expression
(no CSE) and one body per generic (no inlining). Analysis time is 60–150 ms
per entry point including the extra compiles.

Re-run with
`cargo test --lib bounds_analysis::tests::survey -- --ignored --nocapture`.

## Known limitations

- **No float intervals.** Any index derived from a float conversion is
  unknown.
- **Memory model.** Only non-escaping locals (and pointer parameters) are read
  through their stores; `groupshared` and buffer *contents* are unknown (only
  their indices are checked), and an address that escapes to an intrinsic
  makes its variable unknown.
- **Intervals only.** No relations between variables beyond dominating
  comparisons, no loop summation invariants, no cross-thread reasoning.
- **Bounded interprocedural analysis.** Call depth, contexts per function and
  total function analyses are capped; past the caps a callee is analyzed once
  with unconstrained parameters. Recursion is cut at the first repeated
  function. Functions whose bodies are unavailable (target intrinsics not
  modeled by name, modules whose source is not on a search path) are unknown
  and named in the diagnostic.
- **Generic witness tables** that are themselves generic (`extension<T> ...`)
  are not followed; the dispatched method is unknown.
- **Subgroup size.** `WaveGetLaneCount()` is `[1, 128]` per Vulkan. Shaders
  that assume a minimum lane count should state it (`max(lanes, 8)`).
- **Slang serialization format.** The reader follows Slang 2026.13's fossil
  layout and stable opcode names (`stable_names.rs`, generated from
  `slang-ir-insts-stable-names.lua`); a Slang bump that changes either fails
  the analysis with a `Malformed` error (logged at `debug`), never the
  compile.

## Upstream vs Goldy-owned

Decision for now: **Goldy-owned analysis stage over Slang's serialized IR**,
with an eye to upstreaming the *generic* pieces.

- The pieces that are Goldy-specific and stay here: reading the container,
  compiling and linking imported modules, mapping locations back through the
  virtual-main rewrite, the `GOLDY_VALIDATION` plumbing, seeding built-ins
  from `[numthreads]`, and the corpus-tuned diagnostics (provenance notes,
  parameter-source attribution, naming through by-value copies).
- The pieces that are generic and would serve every Slang user: interval
  propagation with edge-refined phis and narrowing, dominating-comparison
  refinement including the bool-phi `&&`/`||` rule and the
  `a >= b ⇒ a - b >= 0` relational rule, context-sensitive summaries over
  `specialize` / `lookupWitness`, and the `getElementPtr` check with a
  "cannot prove `0 <= index < length`" warning. Working on Slang IR now means
  the analysis is expressed in the terms an upstream pass would use (Slang has
  SSA, dominators, `IRIntegerRelation` and loop analysis of its own), so the
  false-positive profile above can be discussed with the Slang maintainers on
  equal footing.
- What we would need from upstream regardless: a way to surface the analysis
  through the C API so Goldy can keep formatting locations through its own
  `#line` mapping.

## Running it

```bash
# Warn on every dynamic index the analysis cannot prove in bounds
GOLDY_VALIDATION=bounds cargo run --features examples --example metaballs
RUST_LOG=goldy::slang=debug GOLDY_VALIDATION=bounds cargo test

# Included in `all`
GOLDY_VALIDATION=all cargo test

# Keep the IR container the analysis looked at
GOLDY_DUMP_SHADERS=/tmp/shaders GOLDY_VALIDATION=bounds cargo run ...   # {entry}_bounds.slang-module

# Print the linked IR as text / re-run the corpus survey
GOLDY_BOUNDS_DUMP=shaders/test_collectives.slang cargo test --lib bounds_analysis::tests::dump_ir -- --ignored --nocapture
cargo test --lib bounds_analysis::tests::survey -- --ignored --nocapture
```

Findings are logged once per distinct compile (including shader disk-cache
hits) at `warn` under the `goldy::slang` target:

```text
shader bounds: possible out-of-bounds index into `links[256]`: index range [-2, 253] (may be negative) (depends on SV_GroupThreadID) at shader.slang:11:1 in `cs_main`
shader bounds: possible out-of-bounds index into `scratch_wave_totals[32]`: index range [0, 255] (may exceed 31) (depends on the result of `WaveGetLaneCount()`) at shaders/goldy_exp/collectives.slang:123:28 in `workgroup_inclusive_scan_wave_uint_sum` (called from cs_main -> _goldy_user_cs_main)
```

Programmatic access: `SlangCompiler::analyze_bounds` compiles a shader (and
the modules it imports) and returns a `BoundsReport` with `checked_accesses`,
`proven_safe` and the list of `BoundsDiagnostic`s;
`goldy::slang::bounds_analysis::analyze_container` runs the analysis on
`.slang-module` bytes produced elsewhere.
