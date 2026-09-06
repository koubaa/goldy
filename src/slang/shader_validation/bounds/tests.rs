//! Tests for the Slang IR bounds analysis prototype.
//!
//! Each shader is compiled with the real Slang compiler to its front-end IR container and
//! analyzed. The cases mirror the "Done when" list of the tracking issue: eager-select form
//! warns, explicit guard is clean, plus lower-bound, upper-bound, padded-lane and
//! unknown-range coverage, and the interprocedural cases Slang IR makes possible.

use super::analysis::{reinterpret, wrap, Interval};
use super::*;
use crate::slang::ir::IntTy;
use crate::slang::{ShaderChecks, ShaderTarget, SlangCompiler, SlangStage};

const BOUNDS_ONLY: ShaderChecks = ShaderChecks { bounds: true };

fn analyze(source: &str) -> BoundsReport {
    analyze_stage(source, "cs_main", SlangStage::Compute)
}

fn analyze_stage(source: &str, entry: &str, stage: SlangStage) -> BoundsReport {
    let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
    compiler
        .validate_shader(source, &[(entry, stage)], &[], &[], ShaderTarget::Spirv, BOUNDS_ONLY)
        .expect("compile + analyze")
        .bounds
        .expect("bounds check ran")
}

/// Analyze with the repository's `shaders/` directory on the search path (for `goldy_exp`).
fn analyze_with_goldy_exp(source: &str) -> BoundsReport {
    let shaders = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .to_string_lossy()
        .into_owned();
    let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
    compiler
        .validate_shader(
            source,
            &[("cs_main", SlangStage::Compute)],
            &[shaders.as_str()],
            &[],
            ShaderTarget::Spirv,
            BOUNDS_ONLY,
        )
        .expect("compile + analyze")
        .bounds
        .expect("bounds check ran")
}

/// 1-based line of the first source line containing `needle`.
fn line_of(source: &str, needle: &str) -> u32 {
    source
        .lines()
        .position(|l| l.contains(needle))
        .map(|i| i as u32 + 1)
        .unwrap_or_else(|| panic!("needle `{needle}` not found"))
}

fn assert_clean(report: &BoundsReport) {
    assert!(
        report.is_clean(),
        "expected no diagnostics, got:\n{}",
        report
            .diagnostics
            .iter()
            .map(|d| format!("  {d}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        report.checked_accesses > 0,
        "expected at least one dynamic access to be checked"
    );
    assert_eq!(report.proven_safe, report.checked_accesses);
}

const PREAMBLE: &str = r#"
groupshared int links[256];

int searchPredecessor(uint id)
{
    // Fails (-1) for odd lanes; otherwise points two slots back (so -2 for lane 0).
    return (id & 1) == 0 ? int(id) - 2 : -1;
}
"#;

// ---------------------------------------------------------------------------
// The motivating example: eager select vs. explicit control flow
// ---------------------------------------------------------------------------

#[test]
fn eager_select_reports_possible_negative_index() {
    let source = format!(
        "{PREAMBLE}
[shader(\"compute\")]
[numthreads(256, 1, 1)]
void cs_main(uint3 localThreadId : SV_GroupThreadID, RWStructuredBuffer<int> out_buf)
{{
    links[localThreadId.x] = int(localThreadId.x);
    GroupMemoryBarrierWithGroupSync();
    int link = searchPredecessor(localThreadId.x);
    int parent = select(link >= 0, links[link], link - 1);
    out_buf[localThreadId.x] = parent;
}}
"
    );
    let report = analyze(&source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.array, "links");
    assert_eq!(d.array_length, 256);
    assert_eq!(d.function, "cs_main");
    assert!(d.call_path.is_empty());
    // The callee is analyzed in context: even ids give `int(id) - 2 in [-2, 253]`, odd give -1.
    assert_eq!(d.index_range, Some((-2, 253)), "{d}");
    let loc = d.location.as_ref().expect("debug info should give a location");
    assert_eq!(loc.file, "shader.slang");
    assert_eq!(loc.line, line_of(&source, "select(link >= 0"));
    // The store `links[localThreadId.x]` is proven safe by `numthreads`.
    assert_eq!(report.checked_accesses, 2);
    assert_eq!(report.proven_safe, 1);
}

#[test]
fn explicit_guard_is_clean() {
    let source = format!(
        "{PREAMBLE}
[shader(\"compute\")]
[numthreads(256, 1, 1)]
void cs_main(uint3 localThreadId : SV_GroupThreadID, RWStructuredBuffer<int> out_buf)
{{
    links[localThreadId.x] = int(localThreadId.x);
    GroupMemoryBarrierWithGroupSync();
    int link = searchPredecessor(localThreadId.x);
    int parent = link - 1;
    if (link >= 0) {{
        parent = links[link];
    }}
    out_buf[localThreadId.x] = parent;
}}
"
    );
    assert_clean(&analyze(&source));
}

/// A scalar `?:` is short-circuiting in Slang: the front end lowers it to control flow, so the
/// ternary form of the motivating example is a guard, not an eager select.
#[test]
fn scalar_ternary_is_lowered_to_control_flow_and_is_clean() {
    let source = format!(
        "{PREAMBLE}
[shader(\"compute\")]
[numthreads(256, 1, 1)]
void cs_main(uint3 localThreadId : SV_GroupThreadID, RWStructuredBuffer<int> out_buf)
{{
    links[localThreadId.x] = int(localThreadId.x);
    GroupMemoryBarrierWithGroupSync();
    int link = searchPredecessor(localThreadId.x);
    int parent = (link >= 0) ? links[link] : (link - 1);
    out_buf[localThreadId.x] = parent;
}}
"
    );
    assert_clean(&analyze(&source));
}

// ---------------------------------------------------------------------------
// Lower bound
// ---------------------------------------------------------------------------

#[test]
fn lower_bound_only_upper_guard_still_warns() {
    let source = format!(
        "{PREAMBLE}
[shader(\"compute\")]
[numthreads(256, 1, 1)]
void cs_main(uint3 localThreadId : SV_GroupThreadID, RWStructuredBuffer<int> out_buf)
{{
    int link = searchPredecessor(localThreadId.x);
    int parent = 0;
    if (link < 256) {{
        parent = links[link];
    }}
    out_buf[localThreadId.x] = parent;
}}
"
    );
    let report = analyze(&source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let (lo, _) = report.diagnostics[0].index_range.unwrap();
    assert!(lo < 0);
    assert!(report.diagnostics[0].to_string().contains("may be negative"));
}

#[test]
fn lower_bound_guard_alone_suffices_when_upper_bound_is_implied() {
    // searchPredecessor returns at most 253, so `link >= 0` is the only guard needed.
    let source = format!(
        "{PREAMBLE}
[shader(\"compute\")]
[numthreads(256, 1, 1)]
void cs_main(uint3 localThreadId : SV_GroupThreadID, RWStructuredBuffer<int> out_buf)
{{
    int link = searchPredecessor(localThreadId.x);
    int parent = 0;
    if (link >= 0) parent = links[link];
    out_buf[localThreadId.x] = parent;
}}
"
    );
    assert_clean(&analyze(&source));
}

#[test]
fn lower_bound_max_clamp_is_clean() {
    let source = format!(
        "{PREAMBLE}
[shader(\"compute\")]
[numthreads(256, 1, 1)]
void cs_main(uint3 localThreadId : SV_GroupThreadID, RWStructuredBuffer<int> out_buf)
{{
    int link = searchPredecessor(localThreadId.x);
    out_buf[localThreadId.x] = links[max(link, 0)];
}}
"
    );
    assert_clean(&analyze(&source));
}

// ---------------------------------------------------------------------------
// Upper bound
// ---------------------------------------------------------------------------

#[test]
fn upper_bound_off_by_one_warns() {
    let source = r#"
groupshared uint sh[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    out_buf[gtid.x] = sh[gtid.x + 1];
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.index_range, Some((1, 256)));
    assert!(d.to_string().contains("may exceed 255"), "{d}");
    assert_eq!(d.location.as_ref().unwrap().line, line_of(source, "sh[gtid.x + 1]"));
}

#[test]
fn upper_bound_guard_on_expression_is_clean() {
    // The guard is on the same expression as the index (`gtid.x + 1`).
    let source = r#"
groupshared uint sh[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    uint v = 0;
    if (gtid.x + 1 < 256) v = sh[gtid.x + 1];
    out_buf[gtid.x] = v;
}
"#;
    assert_clean(&analyze(source));
}

#[test]
fn upper_bound_mask_min_and_mod_are_clean() {
    let source = r#"
groupshared uint sh[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    uint a = sh[(gtid.x + 1) & 255];
    uint b = sh[min(gtid.x + 1, 255u)];
    uint c = sh[(gtid.x + 1) % 256];
    uint d = sh[(gtid.x * 3) >> 2];
    out_buf[gtid.x] = a + b + c + d;
}
"#;
    let report = analyze(source);
    assert_clean(&report);
    assert_eq!(report.checked_accesses, 5);
}

#[test]
fn two_dimensional_local_size_uses_each_component() {
    let source = r#"
groupshared uint tile[16][8];

[shader("compute")]
[numthreads(8, 16, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    tile[gtid.y][gtid.x] = gtid.x;      // in bounds: y < 16, x < 8
    GroupMemoryBarrierWithGroupSync();
    out_buf[gtid.x] = tile[gtid.x][gtid.y]; // swapped: y may reach 15 > 7
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.array_length, 8);
    assert_eq!(d.index_range, Some((0, 15)));
    assert_eq!(report.checked_accesses, 4);
    assert_eq!(report.proven_safe, 3);
}

// ---------------------------------------------------------------------------
// Padded dispatch: the last workgroup has inactive lanes
// ---------------------------------------------------------------------------

#[test]
fn padded_lane_sentinel_index_warns() {
    let source = r#"
groupshared uint sh[64];

struct Params { uint count; };
ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = dtid.x;
    GroupMemoryBarrierWithGroupSync();
    // Inactive (padded) lanes get a sentinel that must not be used as an index.
    int lane = select(dtid.x < params.count, int(gtid.x), -1);
    out_buf[dtid.x] = sh[lane];
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.index_range, Some((-1, 63)));
    assert_eq!(d.location.as_ref().unwrap().line, line_of(source, "sh[lane]"));
}

#[test]
fn padded_lane_sentinel_guarded_is_clean() {
    let source = r#"
groupshared uint sh[64];

struct Params { uint count; };
ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = dtid.x;
    GroupMemoryBarrierWithGroupSync();
    int lane = select(dtid.x < params.count, int(gtid.x), -1);
    uint v = 0;
    if (lane >= 0) v = sh[lane];
    out_buf[dtid.x] = v;
}
"#;
    assert_clean(&analyze(source));
}

#[test]
fn padded_lane_early_return_then_local_index_is_clean() {
    // The classic Goldy pattern: over-dispatch, early-out on the global id, index shared
    // memory by the group-local id (bounded by numthreads regardless of the early-out).
    let source = r#"
groupshared uint sh[64];

struct Params { uint count; };
ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    if (dtid.x >= params.count) return;
    sh[gtid.x] = dtid.x;
    GroupMemoryBarrierWithGroupSync();
    out_buf[dtid.x] = sh[63 - gtid.x];
}
"#;
    assert_clean(&analyze(source));
}

#[test]
fn padded_lane_global_id_as_local_index_warns_with_unknown_range() {
    // Using the *dispatch* thread id to index workgroup memory is never provable.
    let source = r#"
groupshared uint sh[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dtid : SV_DispatchThreadID, RWStructuredBuffer<uint> out_buf)
{
    sh[dtid.x] = dtid.x;
    GroupMemoryBarrierWithGroupSync();
    out_buf[dtid.x] = sh[0];
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.index_range, None);
    assert!(d.to_string().contains("index range unknown"), "{d}");
}

// ---------------------------------------------------------------------------
// Unknown range: index comes from memory
// ---------------------------------------------------------------------------

#[test]
fn unknown_range_from_buffer_warns() {
    let source = r#"
groupshared uint sh[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<uint> indices, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    uint i = indices[gtid.x];
    out_buf[gtid.x] = sh[i];
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    assert_eq!(report.diagnostics[0].index_range, None);
    assert_eq!(report.diagnostics[0].array, "sh");
    assert_eq!(report.diagnostics[0].depends_on, vec!["a buffer load".to_string()]);
}

#[test]
fn unknown_range_with_unsigned_guard_is_clean() {
    let source = r#"
groupshared uint sh[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<uint> indices, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    uint i = indices[gtid.x];
    uint v = 0;
    if (i < 64) v = sh[i];
    out_buf[gtid.x] = v;
}
"#;
    assert_clean(&analyze(source));
}

#[test]
fn unknown_signed_range_with_conjunction_guard_is_clean() {
    // `i >= 0 && i < 64` is short-circuit control flow with a bool block parameter.
    let source = r#"
groupshared uint sh[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<int> indices, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    int i = indices[gtid.x];
    uint v = 0;
    if (i >= 0 && i < 64) v = sh[i];
    out_buf[gtid.x] = v;
}
"#;
    assert_clean(&analyze(source));
}

#[test]
fn unknown_signed_range_with_only_one_conjunct_warns() {
    let source = r#"
groupshared uint sh[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<int> indices, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    int i = indices[gtid.x];
    uint v = 0;
    if (i < 64) v = sh[i];
    out_buf[gtid.x] = v;
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let (lo, hi) = report.diagnostics[0].index_range.unwrap();
    assert!(lo < 0 && hi == 63, "{:?}", report.diagnostics[0]);
}

#[test]
fn unknown_range_clamped_is_clean() {
    let source = r#"
groupshared uint sh[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<int> indices, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    int i = indices[gtid.x];
    out_buf[gtid.x] = sh[clamp(i, 0, 63)] + sh[uint(i) % 64u];
}
"#;
    assert_clean(&analyze(source));
}

// ---------------------------------------------------------------------------
// Real Goldy patterns: workgroup scan / reduce loops
// ---------------------------------------------------------------------------

#[test]
fn workgroup_inclusive_scan_loop_is_clean() {
    // `local_ix >= (1u << i)` justifies `local_ix - (1u << i)` (relational rule) and the
    // loop guard bounds `i` through its block parameter.
    let source = r#"
groupshared uint scratch[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID, RWStructuredBuffer<uint> out_buf)
{
    uint local_ix = gtid.x;
    uint val = dtid.x;
    scratch[local_ix] = val;
    for (uint i = 0; i < 8; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_ix >= (1u << i)) {
            uint other = scratch[local_ix - (1u << i)];
            val = other + val;
        }
        GroupMemoryBarrierWithGroupSync();
        scratch[local_ix] = val;
    }
    out_buf[dtid.x] = val;
}
"#;
    assert_clean(&analyze(source));
}

#[test]
fn workgroup_reduce_loop_is_clean() {
    let source = r#"
groupshared uint scratch[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID, RWStructuredBuffer<uint> out_buf)
{
    uint local_ix = gtid.x;
    uint val = dtid.x;
    scratch[local_ix] = val;
    for (uint i = 0; i < 8; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_ix + (1u << i) < 256)
            val = val + scratch[local_ix + (1u << i)];
        GroupMemoryBarrierWithGroupSync();
        scratch[local_ix] = val;
    }
    out_buf[dtid.x] = val;
}
"#;
    assert_clean(&analyze(source));
}

#[test]
fn xor_butterfly_is_clean() {
    let source = r#"
groupshared uint scratch[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    uint val = gtid.x;
    scratch[gtid.x] = val;
    for (uint i = 0; i < 8; i++) {
        GroupMemoryBarrierWithGroupSync();
        val += scratch[gtid.x ^ (1u << i)];
        GroupMemoryBarrierWithGroupSync();
        scratch[gtid.x] = val;
    }
    out_buf[gtid.x] = val;
}
"#;
    assert_clean(&analyze(source));
}

#[test]
fn scan_loop_missing_guard_warns() {
    let source = r#"
groupshared uint scratch[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    uint local_ix = gtid.x;
    uint val = gtid.x;
    scratch[local_ix] = val;
    for (uint i = 0; i < 8; i++) {
        GroupMemoryBarrierWithGroupSync();
        val += scratch[local_ix - (1u << i)]; // wraps for local_ix < (1u << i)
        GroupMemoryBarrierWithGroupSync();
        scratch[local_ix] = val;
    }
    out_buf[gtid.x] = val;
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    assert_eq!(
        report.diagnostics[0].location.as_ref().unwrap().line,
        line_of(source, "local_ix - (1u << i)")
    );
}

#[test]
fn wave_scan_totals_depend_on_subgroup_size() {
    // `totals[32]` indexed by `local_ix / WaveGetLaneCount()` is only safe when the subgroup
    // is at least 8 wide; the analysis knows subgroups can be as small as 1.
    let source = r#"
groupshared uint totals[32];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    uint lc = WaveGetLaneCount();
    uint wave_ix = gtid.x / lc;
    if (WaveIsFirstLane()) totals[wave_ix] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    out_buf[gtid.x] = totals[wave_ix];
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 2, "{report:?}");
    for d in &report.diagnostics {
        assert_eq!(d.index_range, Some((0, 255)), "{d}");
        assert_eq!(
            d.depends_on,
            vec!["the result of `WaveGetLaneCount()`".to_string()],
            "{d}"
        );
    }
}

// ---------------------------------------------------------------------------
// Other storage classes and names
// ---------------------------------------------------------------------------

/// The `spinning_cube` example: a counted loop over a function-local array. The header block
/// parameter is widened to `[0, MAX]` during the ascending fixpoint; evaluating the back edge
/// under the loop guard plus the narrowing phase must bring it back to `[0, 8]`.
#[test]
fn counted_loop_over_local_array_is_clean() {
    let source = r#"
[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<float3> out_buf)
{
    float3 verts[8];
    for (int i = 0; i < 8; i++) {
        verts[i] = float3(i, 0, 0);
    }
    for (int i = 0; i < 8; i++) {
        verts[i] = verts[i] * 2.0;
    }
    out_buf[gtid.x] = verts[gtid.x & 7];
}
"#;
    assert_clean(&analyze(source));
}

/// A loop whose trip count is itself only known as a range (`nw = 256 / lanes`).
#[test]
fn counted_loop_with_ranged_bound_is_clean() {
    let source = r#"
groupshared uint totals[64];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    uint nw = 256 / max(WaveGetLaneCount(), 4u);
    if (gtid.x == 0) {
        uint run = 0;
        for (uint i = 0; i < nw; i++) {
            uint s = totals[i];
            totals[i] = run;
            run += s;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    out_buf[gtid.x] = totals[gtid.x & 63];
}
"#;
    assert_clean(&analyze(source));
}

// ---------------------------------------------------------------------------
// Interprocedural: what Slang IR adds over SPIR-V
// ---------------------------------------------------------------------------

/// A helper indexing `groupshared` through its parameter is safe or not depending on the
/// caller; the access is reported once, with the call path, and only in the unsafe context.
#[test]
fn helper_is_checked_in_each_calling_context() {
    let source = r#"
groupshared uint sh[64];

uint read_slot(uint slot)
{
    return sh[slot];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<uint> indices, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    uint a = read_slot(gtid.x);            // in bounds
    uint b = read_slot(gtid.x ^ 63);       // in bounds
    out_buf[gtid.x] = a + b;
}
"#;
    let report = analyze(source);
    assert_clean(&report);
    assert_eq!(report.checked_accesses, 2);

    let unsafe_source = source.replace(
        "uint b = read_slot(gtid.x ^ 63);",
        "uint b = read_slot(indices[gtid.x]);",
    );
    let report = analyze(&unsafe_source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.function, "read_slot");
    assert_eq!(d.call_path, vec!["cs_main".to_string()]);
    assert_eq!(
        d.location.as_ref().unwrap().line,
        line_of(&unsafe_source, "return sh[slot];")
    );
    assert_eq!(d.index_range, None);
    assert!(d.to_string().ends_with("in `read_slot` (called from cs_main)"), "{d}");
}

/// Nested helpers: the call path names every frame, deepest last.
#[test]
fn call_path_is_reported_through_nested_helpers() {
    let source = r#"
groupshared uint sh[64];

uint inner(uint slot) { return sh[slot]; }
uint outer(uint slot) { return inner(slot + 1); }

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    out_buf[gtid.x] = outer(gtid.x);
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.function, "inner");
    assert_eq!(d.call_path, vec!["cs_main".to_string(), "outer".to_string()]);
    assert_eq!(d.index_range, Some((1, 64)));
}

/// Struct fields are tracked through constructors (`var` + field stores + `load`) and
/// `getField`, so a wrapped thread id keeps its `numthreads` bound.
#[test]
fn struct_fields_keep_their_ranges() {
    let source = r#"
groupshared uint sh[64];

struct Lane { uint ix; uint other; };

Lane make_lane(uint3 gtid) { Lane l; l.ix = gtid.x; l.other = gtid.x + 100; return l; }

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    Lane lane = make_lane(gtid);
    sh[lane.ix] = lane.other;
    GroupMemoryBarrierWithGroupSync();
    out_buf[gtid.x] = sh[lane.other];
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    assert_eq!(report.diagnostics[0].index_range, Some((100, 163)));
    assert_eq!(report.checked_accesses, 2);
}

/// Values returned through `out` parameters flow back to the caller.
#[test]
fn out_parameters_flow_back() {
    let source = r#"
groupshared uint sh[64];

void split(uint x, out uint lo, out uint hi) { lo = x & 63; hi = x >> 6; }

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<uint> indices, RWStructuredBuffer<uint> out_buf)
{
    uint lo, hi;
    split(indices[gtid.x], lo, hi);
    sh[lo] = hi;
    GroupMemoryBarrierWithGroupSync();
    out_buf[gtid.x] = sh[hi];
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    assert_eq!(report.diagnostics[0].array, "sh");
    assert_eq!(report.checked_accesses, 2);
}

/// Generic helpers are analyzed per specialization: `N` is substituted from the call site.
#[test]
fn generic_helper_is_specialized_per_call() {
    let source = r#"
groupshared uint sh[64];

uint wrap_read<let N : int>(uint i) { return sh[i % N]; }

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<uint> indices, RWStructuredBuffer<uint> out_buf)
{
    sh[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    out_buf[gtid.x] = wrap_read<64>(indices[gtid.x]) + wrap_read<128>(indices[gtid.x]);
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.function, "wrap_read");
    assert_eq!(d.index_range, Some((0, 127)), "{d}");
}

/// A `[goldy_compute]` kernel: the entry point is a generated `virtual_main` wrapper that
/// builds `GroupThreadId` through a `goldy_exp` constructor. The imported module is compiled
/// alongside so the wrapper's `gtid.x` keeps its `numthreads` bound, and `#line` directives
/// map the finding back to the user's line.
#[test]
fn goldy_exp_thread_id_wrapper_is_understood() {
    let source = r#"import goldy_exp;
groupshared uint scratch[256];

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(Scattered<uint> out_buf, GroupThreadId gtid)
{
    scratch[gtid.x] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    out_buf[gtid.x] = scratch[gtid.x - 1];
}
"#;
    let report = analyze_with_goldy_exp(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.array, "scratch");
    assert_eq!(d.index_range, None, "{d}");
    assert_eq!(
        d.location.as_ref().unwrap().line,
        line_of(source, "scratch[gtid.x - 1]")
    );
    assert_eq!(report.checked_accesses, 2);
    assert_eq!(report.proven_safe, 1);

    let fixed = source.replace("scratch[gtid.x - 1]", "scratch[gtid.x ^ 1]");
    assert_clean(&analyze_with_goldy_exp(&fixed));
}

/// A generic from an imported module: `workgroup_reduce<T : IMonoid, let N : int>` is linked
/// to its `goldy_exp` definition and analyzed with `T = uint`, `N = 64` from the call site,
/// so `scratch[N]` has a length, `firstbitlow(N)` bounds the loop, and `T.combine` resolves
/// through the witness table to `uint`'s conformance.
#[test]
fn imported_generic_is_linked_and_specialized() {
    let source = r#"import goldy_exp;
groupshared uint scratch[64];

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> out_buf, GroupThreadId gtid)
{
    out_buf[gtid.x] = workgroup_reduce(gtid.x, gtid.x, scratch);
}
"#;
    let report = analyze_with_goldy_exp(source);
    assert_clean(&report);
    // `scratch[local_ix] = val`, `scratch[local_ix + (1u << i)]`, `scratch[local_ix] = val`.
    assert_eq!(report.checked_accesses, 3, "{report:?}");

    // The same generic with a struct monoid defined in the translation unit: `T.combine`
    // dispatches through the shader's own witness table into `Pair.combine`, whose table
    // lookup on a value read back from groupshared memory is reported on the full call path.
    let source = r#"import goldy_exp;
static const uint lut[4] = { 1, 2, 4, 8 };
struct Pair { uint a; uint b; }
extension Pair : IMonoid {
    static Pair identity() { Pair p; p.a = 0; p.b = 0; return p; }
    Pair combine(Pair o) { Pair p; p.a = a + o.a; p.b = lut[o.b]; return p; }
}
groupshared Pair scratch[64];

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> out_buf, GroupThreadId gtid)
{
    Pair p = Pair.identity();
    p.a = gtid.x;
    p.b = gtid.x & 3;
    out_buf[gtid.x] = workgroup_reduce(p, gtid.x, scratch).a;
}
"#;
    let report = analyze_with_goldy_exp(source);
    assert_eq!(report.checked_accesses, 4, "{report:?}");
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.function, "Pair.combine", "{d}");
    assert_eq!(d.array, "lut", "{d}");
    assert_eq!(
        d.call_path,
        vec![
            "cs_main".to_string(),
            "_goldy_user_cs_main".to_string(),
            "workgroup_reduce".to_string()
        ],
        "{d}"
    );
    assert_eq!(d.depends_on, vec!["groupshared memory `scratch`".to_string()], "{d}");
    assert_eq!(d.location.as_ref().unwrap().line, line_of(source, "lut[o.b]"));
}

/// An array length that is a constant expression over a generic argument
/// (`uint prefix_sums[1 << LG_N]`) is evaluated under the substitution.
#[test]
fn generic_array_length_expression_is_evaluated() {
    let source = r#"
groupshared uint table[256];

uint pick<let LG_N : int>(uint i, groupshared uint t[1 << LG_N]) { return t[i]; }

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    out_buf[gtid.x] = pick<8>(gtid.x, table) + pick<8>(gtid.x * 8, table);
}
"#;
    let report = analyze(source);
    assert_eq!(report.checked_accesses, 1, "{report:?}");
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.array_length, 256, "{d}");
    assert_eq!(d.array, "t", "{d}");
    assert_eq!(d.index_range, Some((0, 504)), "{d}");
}

/// The provenance note follows an argument back through the caller: inside the user function
/// the index is a field of a `ThreadId` parameter, which the `virtual_main` wrapper built
/// from `SV_DispatchThreadID`.
#[test]
fn provenance_follows_wrapper_struct_argument() {
    let source = r#"import goldy_exp;
groupshared uint sh[64];

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> out_buf, ThreadId id)
{
    sh[id.x] = id.x;
    GroupMemoryBarrierWithGroupSync();
    out_buf[id.x] = sh[id.x ^ 1];
}
"#;
    let report = analyze_with_goldy_exp(source);
    assert_eq!(report.diagnostics.len(), 2, "{report:?}");
    for d in &report.diagnostics {
        assert_eq!(d.depends_on, vec!["SV_DispatchThreadID".to_string()], "{d}");
        assert_eq!(d.array, "sh", "{d}");
    }
}

// ---------------------------------------------------------------------------
// Provenance notes
// ---------------------------------------------------------------------------

/// `SV_VertexID` indexing a `static const` table: unprovable from the shader alone (the draw
/// call decides), reported with the system value named.
#[test]
fn vertex_id_into_static_table_names_the_system_value() {
    let source = r#"
static const float2 positions[3] = { float2(0, 0), float2(1, 0), float2(0, 1) };

[shader("vertex")]
float4 vs_main(uint vertex_id : SV_VertexID) : SV_Position
{
    return float4(positions[vertex_id], 0, 1);
}
"#;
    let report = analyze_stage(source, "vs_main", SlangStage::Vertex);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.array_length, 3);
    assert_eq!(d.array, "positions", "{d}");
    assert_eq!(d.depends_on, vec!["SV_VertexID".to_string()], "{d}");
    assert_eq!(d.index_range, None);

    // The idiomatic fix is provable.
    let fixed = source.replace("positions[vertex_id]", "positions[vertex_id % 3]");
    assert_clean(&analyze_stage(&fixed, "vs_main", SlangStage::Vertex));
}

#[test]
fn float_to_int_index_names_the_conversion() {
    let source = r#"
[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<float> hues, RWStructuredBuffer<float3> out_buf)
{
    float3 palette[7] = {
        float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1), float3(1, 1, 0),
        float3(0, 1, 1), float3(1, 0, 1), float3(1, 1, 1)
    };
    int idx = int(hues[gtid.x] * 6.0);
    out_buf[gtid.x] = palette[idx];
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert!(d.array.contains("palette"), "{d}");
    assert_eq!(d.depends_on, vec!["a float-to-int conversion".to_string()], "{d}");
}

#[test]
fn groupshared_indexed_by_dispatch_thread_id_names_the_system_value() {
    let source = r#"
groupshared uint sh[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID, RWStructuredBuffer<uint> out_buf)
{
    sh[id.x] = id.x;
    GroupMemoryBarrierWithGroupSync();
    out_buf[id.x] = sh[id.x ^ 1];
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 2, "{report:?}");
    for d in &report.diagnostics {
        assert_eq!(d.depends_on, vec!["SV_DispatchThreadID".to_string()], "{d}");
    }
}

#[test]
fn function_local_array_is_checked_and_named() {
    let source = r#"
[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<uint> indices, RWStructuredBuffer<uint> out_buf)
{
    uint table[4] = { 1, 2, 3, 4 };
    uint i = indices[gtid.x];
    out_buf[gtid.x] = table[i];
}
"#;
    let report = analyze(source);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.array_length, 4);
    assert_eq!(d.array, "table", "array name should come from the name hint: {d}");
}

#[test]
fn unguarded_vector_component_index_warns() {
    let source = r#"
[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<uint> indices, RWStructuredBuffer<float> out_buf)
{
    float4 v = float4(1, 2, 3, 4);
    uint i = indices[gtid.x];
    out_buf[gtid.x] = v[i];
}
"#;
    let report = analyze(source);
    assert!(
        report.diagnostics.iter().any(|d| d.array_length == 4),
        "dynamic vector component index should be flagged: {report:?}"
    );
}

#[test]
fn constant_indices_are_not_counted() {
    let source = r#"
groupshared uint sh[4];

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, RWStructuredBuffer<uint> out_buf)
{
    if (gtid.x == 0) { sh[0] = 1; sh[3] = 2; }
    GroupMemoryBarrierWithGroupSync();
    out_buf[gtid.x] = sh[0] + sh[3];
}
"#;
    let report = analyze(source);
    assert!(report.is_clean());
    assert_eq!(report.checked_accesses, 0);
}

#[test]
fn equality_guard_pins_index() {
    let source = r#"
groupshared uint sh[1];

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 gtid : SV_GroupThreadID, StructuredBuffer<uint> indices, RWStructuredBuffer<uint> out_buf)
{
    uint i = indices[gtid.x];
    if (i == 0) sh[i] = gtid.x;
    GroupMemoryBarrierWithGroupSync();
    out_buf[gtid.x] = sh[0];
}
"#;
    assert_clean(&analyze(source));
}

// ---------------------------------------------------------------------------
// Diagnostics, errors, helpers
// ---------------------------------------------------------------------------

#[test]
fn diagnostic_display_is_actionable() {
    let d = BoundsDiagnostic {
        function: "cs_main".into(),
        call_path: Vec::new(),
        array: "links".into(),
        array_length: 256,
        index_range: Some((-2, 253)),
        location: Some(SourceLocation {
            file: "shader.slang".into(),
            line: 16,
            column: 5,
        }),
        depends_on: Vec::new(),
    };
    assert_eq!(
        d.to_string(),
        "possible out-of-bounds index into `links[256]`: index range [-2, 253] (may be negative) \
         at shader.slang:16:5 in `cs_main`"
    );
    let unknown = BoundsDiagnostic {
        function: "read_slot".into(),
        call_path: vec!["cs_main".into(), "outer".into()],
        index_range: None,
        location: None,
        depends_on: vec!["SV_VertexID".into(), "a buffer load".into()],
        ..d
    };
    assert_eq!(
        unknown.to_string(),
        "possible out-of-bounds index into `links[256]`: index range unknown \
         (depends on SV_VertexID, a buffer load) in `read_slot` (called from cs_main -> outer)"
    );
}

#[test]
fn interval_reinterpretation() {
    let u32t = IntTy {
        bits: 32,
        signed: false,
    };
    let i32t = IntTy { bits: 32, signed: true };
    // Values in the shared non-negative range are unchanged.
    assert_eq!(reinterpret(Interval::new(0, 100), i32t, u32t), Interval::new(0, 100));
    // All-negative signed maps to the upper half unsigned.
    assert_eq!(
        reinterpret(Interval::new(-2, -1), i32t, u32t),
        Interval::new((1 << 32) - 2, (1 << 32) - 1)
    );
    // Straddling zero loses everything.
    assert_eq!(
        reinterpret(Interval::new(-1, 1), i32t, u32t),
        Interval::new(0, (1 << 32) - 1)
    );
    // Upper-half unsigned maps to negatives.
    assert_eq!(
        reinterpret(Interval::new(1 << 31, (1 << 32) - 1), u32t, i32t),
        Interval::new(-(1 << 31), -1)
    );
    // Wrap collapses to the type range.
    assert_eq!(wrap(Interval::new(-1, 5), u32t), Interval::new(0, (1 << 32) - 1));
    assert_eq!(wrap(Interval::new(0, 5), u32t), Interval::new(0, 5));
}

/// Survey of the repository's `shaders/` corpus: prints one line per entry point plus every
/// diagnostic. Run with `cargo test --lib shader_validation::bounds::tests::survey -- --ignored --nocapture`.
#[test]
#[ignore]
fn survey() {
    let shaders = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let search = shaders.to_string_lossy().into_owned();
    let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
    let mut files: Vec<_> = std::fs::read_dir(&shaders)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "slang"))
        .collect();
    files.sort();
    let (mut entries, mut checked, mut proven, mut warnings) = (0, 0, 0, 0);
    for path in files {
        let source = std::fs::read_to_string(&path).unwrap();
        let name = format!("shaders/{}", path.file_name().unwrap().to_string_lossy());
        for (entry, stage) in [
            ("cs_main", SlangStage::Compute),
            ("vs_main", SlangStage::Vertex),
            ("fs_main", SlangStage::Fragment),
        ] {
            if !source.contains(&format!(" {entry}(")) {
                continue;
            }
            let t0 = std::time::Instant::now();
            let result = compiler.validate_shader(
                &source,
                &[(entry, stage)],
                &[search.as_str()],
                &[],
                ShaderTarget::Spirv,
                BOUNDS_ONLY,
            );
            let ms = t0.elapsed().as_millis();
            match result.map(|r| r.bounds.expect("bounds check ran")) {
                Ok(r) => {
                    entries += 1;
                    checked += r.checked_accesses;
                    proven += r.proven_safe;
                    warnings += r.diagnostics.len();
                    println!(
                        "{name:<44} {entry:<8} checked={:<3} proven_safe={:<3} warnings={} ({ms} ms)",
                        r.checked_accesses,
                        r.proven_safe,
                        r.diagnostics.len()
                    );
                    for d in &r.diagnostics {
                        println!("    {d}");
                    }
                }
                Err(e) => println!(
                    "{name:<44} {entry:<8} ERROR: {}",
                    e.to_string().lines().take(6).collect::<Vec<_>>().join(" | ")
                ),
            }
        }
    }
    println!("\nTOTAL entry points={entries} checked={checked} proven_safe={proven} warnings={warnings}");
}
