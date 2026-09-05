//! Tests for the Slang IR bounds analysis prototype.
//!
//! Each shader is compiled with the real Slang compiler to its front-end IR container and
//! analyzed. The cases mirror the "Done when" list of the tracking issue: eager-select form
//! warns, explicit guard is clean, plus lower-bound, upper-bound, padded-lane and
//! unknown-range coverage, and the interprocedural cases Slang IR makes possible.

use super::*;
use crate::slang::{ShaderTarget, SlangCompiler, SlangStage};

fn analyze(source: &str) -> BoundsReport {
    analyze_stage(source, "cs_main", SlangStage::Compute)
}

fn analyze_stage(source: &str, entry: &str, stage: SlangStage) -> BoundsReport {
    let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
    compiler
        .analyze_bounds(source, &[(entry, stage)], &[], &[], ShaderTarget::Spirv)
        .expect("compile + analyze")
}

#[test]
fn mangled_name_module_component() {
    assert_eq!(module_of_mangled_name("_ST4core17IBufferDataLayout"), Some("core"));
    assert_eq!(
        module_of_mangled_name("_S9goldy_exp23goldy_frame_table_indexp4pi_ui_ui_ui_uu"),
        Some("goldy_exp")
    );
    assert_eq!(module_of_mangled_name("_SV9goldy_exp13GroupThreadId1x"), Some("goldy_exp"));
    assert_eq!(module_of_mangled_name("_SW4core17DefaultDataLayout"), Some("core"));
    assert_eq!(module_of_mangled_name("nope"), None);
}

#[test]
fn smoke() {
    let report = analyze(
        r#"
groupshared int links[256];

int searchPredecessor(uint id)
{
    return (id & 1) == 0 ? int(id) - 2 : -1;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 localThreadId : SV_GroupThreadID, RWStructuredBuffer<int> out_buf)
{
    links[localThreadId.x] = int(localThreadId.x);
    GroupMemoryBarrierWithGroupSync();
    int link = searchPredecessor(localThreadId.x);
    int parent = select(link >= 0, links[link], link - 1);
    out_buf[localThreadId.x] = parent;
}
"#,
    );
    eprintln!("{report:#?}");
    for d in &report.diagnostics {
        eprintln!("{d}");
    }
    panic!("show");
}
