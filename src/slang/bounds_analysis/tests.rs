//! Tests for the SPIR-V bounds analysis prototype.
//!
//! Each shader is compiled with the real Slang compiler (standard debug info) and the
//! resulting SPIR-V is analyzed. The cases mirror the "Done when" list of the tracking
//! issue: eager-select form warns, explicit guard is clean, plus lower-bound, upper-bound,
//! padded-lane and unknown-range coverage.

use super::*;
use crate::slang::{SlangCompiler, SlangStage};
use crate::types::OptimizationLevel;

fn analyze(source: &str) -> BoundsReport {
    analyze_opt(source, OptimizationLevel::Default)
}

fn analyze_opt(source: &str, opt: OptimizationLevel) -> BoundsReport {
    analyze_stage(source, "cs_main", SlangStage::Compute, opt)
}

fn analyze_stage(source: &str, entry: &str, stage: SlangStage, opt: OptimizationLevel) -> BoundsReport {
    let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
    compiler
        .analyze_spirv_bounds(source, &[(entry, stage)], &[], &[], opt)
        .expect("compile + analyze")
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
    let (lo, hi) = d.index_range.expect("range should be known");
    assert!(lo < 0, "lower bound should be negative: {d}");
    assert!(hi <= 255, "upper bound should be inside the array: {d}");
    let loc = d.location.as_ref().expect("debug info should give a location");
    assert_eq!(loc.file, "shader.slang");
    assert_eq!(loc.line, line_of(&source, "select(link >= 0"));
    // The store `links[localThreadId.x]` is proven safe by LocalSize.
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

/// Slang 2026.x lowers a scalar `?:` to short-circuiting control flow, so the ternary form of
/// the motivating example is *not* an eager select in current SPIR-V output. The analysis
/// reports what the compiler actually emitted.
///
/// `OptimizationLevel::None` is excluded: Slang then leaves `searchPredecessor` as an
/// `OpFunctionCall`, and the analysis is intraprocedural (call results are unknown), so the
/// guard `link >= 0` alone cannot bound the index from above.
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
    for opt in [OptimizationLevel::Default, OptimizationLevel::High] {
        assert_clean(&analyze_opt(&source, opt));
    }
}

/// Documents the intraprocedural limitation: with the callee not inlined, the result range
/// is unknown and only the guard's lower bound survives.
#[test]
fn uninlined_call_result_is_unknown_at_opt_none() {
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
    let report = analyze_opt(&source, OptimizationLevel::None);
    if report.is_clean() {
        // Slang inlined the helper even at -O0; nothing to document.
        return;
    }
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    assert_eq!(report.diagnostics[0].index_range, Some((0, i32::MAX as i128)));
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
    // The guard is on the same expression as the index (`gtid.x + 1`), matched by value numbering.
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
    // memory by the group-local id (bounded by LocalSize regardless of the early-out).
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
    // `i >= 0 && i < 64` — both conjuncts must be picked up from the LogicalAnd / select lowering.
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
    // loop guard bounds `i` once the loop counter is reconstructed as SSA.
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
    for opt in [OptimizationLevel::None, OptimizationLevel::Default] {
        assert_clean(&analyze_opt(source, opt));
    }
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
    // `scratch_wave_totals[32]` indexed by `local_ix / WaveGetLaneCount()` is only safe when
    // the subgroup is at least 8 wide; the analysis knows subgroups can be as small as 1.
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
    }
}

// ---------------------------------------------------------------------------
// Other storage classes and names
// ---------------------------------------------------------------------------

/// The `spinning_cube` example: a counted loop over a function-local array. The header phi
/// is widened to `[0, MAX]` during the ascending fixpoint; evaluating the back edge under the
/// loop guard plus the narrowing phase must bring it back to `[0, 8]` so `verts[i]` is proven.
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
    for opt in [
        OptimizationLevel::None,
        OptimizationLevel::Default,
        OptimizationLevel::High,
    ] {
        let report = analyze_opt(source, opt);
        assert_clean(&report);
    }
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
// Provenance notes
// ---------------------------------------------------------------------------

/// `SV_VertexID` indexing a `static const` table: unprovable from the shader alone (the draw
/// call decides), reported with the system value named and the folded table described by type.
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
    let report = analyze_stage(source, "vs_main", SlangStage::Vertex, OptimizationLevel::Default);
    assert_eq!(report.diagnostics.len(), 1, "{report:?}");
    let d = &report.diagnostics[0];
    assert_eq!(d.array_length, 3);
    assert_eq!(d.array, "<unnamed float2 array>", "{d}");
    assert_eq!(d.depends_on, vec!["SV_VertexID".to_string()], "{d}");
    assert_eq!(d.index_range, None);

    // The idiomatic fix is provable.
    let fixed = source.replace("positions[vertex_id]", "positions[vertex_id % 3]");
    assert_clean(&analyze_stage(
        &fixed,
        "vs_main",
        SlangStage::Vertex,
        OptimizationLevel::Default,
    ));
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
    assert!(d.array.contains("table"), "array name should come from OpName: {d}");
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
        index_range: None,
        location: None,
        depends_on: vec!["SV_VertexID".into(), "a buffer load `indices`".into()],
        ..d
    };
    assert_eq!(
        unknown.to_string(),
        "possible out-of-bounds index into `links[256]`: index range unknown \
         (depends on SV_VertexID, a buffer load `indices`) in `cs_main`"
    );
}

#[test]
fn malformed_input_is_an_error() {
    assert_eq!(analyze_spirv_bytes(&[1, 2, 3]), Err(BoundsAnalysisError::Misaligned));
    assert_eq!(
        analyze_spirv(&[0xdead_beef, 0, 0, 0, 0]),
        Err(BoundsAnalysisError::BadMagic(0xdead_beef))
    );
    assert!(matches!(
        analyze_spirv(&[0x0723_0203, 0x0001_0600, 0, 100, 0, 0x0003_0011]),
        Err(BoundsAnalysisError::Truncated(_))
    ));
    // Header only: valid, empty module.
    assert_eq!(
        analyze_spirv(&[0x0723_0203, 0x0001_0600, 0, 1, 0]),
        Ok(BoundsReport::default())
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
    assert_eq!(reinterpret(Interval::new(-1, 1), i32t, u32t), u32t.range());
    // Upper-half unsigned maps to negatives.
    assert_eq!(
        reinterpret(Interval::new(1 << 31, (1 << 32) - 1), u32t, i32t),
        Interval::new(-(1 << 31), -1)
    );
    // Wrap collapses to the type range.
    assert_eq!(wrap(Interval::new(-1, 5), u32t), u32t.range());
    assert_eq!(wrap(Interval::new(0, 5), u32t), Interval::new(0, 5));
}
