use super::*;
use crate::slang::{ShaderTarget, SlangCompiler, SlangStage};

#[test]
fn malformed_input_is_an_error() {
    assert!(matches!(link_containers(&[1, 2, 3], &[]), Err(IrError::Malformed(_))));
    // A RIFF file that is not a Slang module container.
    let mut riff = b"RIFF".to_vec();
    riff.extend_from_slice(&4u32.to_le_bytes());
    riff.extend_from_slice(b"WAVE");
    assert_eq!(
        link_containers(&riff, &[]).err(),
        Some(IrError::Malformed("not a Slang module container"))
    );
    assert!(matches!(imported_modules(&riff), Err(IrError::Malformed(_))));
}

#[test]
fn mangled_name_module_component() {
    assert_eq!(module_of_mangled_name("_ST4core17IBufferDataLayout"), Some("core"));
    assert_eq!(
        module_of_mangled_name("_S9goldy_exp23goldy_frame_table_indexp4pi_ui_ui_ui_uu"),
        Some("goldy_exp")
    );
    assert_eq!(
        module_of_mangled_name("_SV9goldy_exp13GroupThreadId1x"),
        Some("goldy_exp")
    );
    assert_eq!(module_of_mangled_name("_SW4core17DefaultDataLayout"), Some("core"));
    assert_eq!(module_of_mangled_name("nope"), None);
}

#[test]
fn imports_are_listed_and_linked() {
    let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
    let shaders = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .to_string_lossy()
        .into_owned();
    let source = r#"
import goldy_exp;
RWStructuredBuffer<uint> out_buf;
[numthreads(64, 1, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    out_buf[tid.x] = goldy_frame_table_index(0, 0, 0, 0);
}
"#;
    let defines = SlangCompiler::bindless_defines_for_target(ShaderTarget::Spirv);
    let container = compiler
        .compile_ir_container(
            source,
            &[("cs_main", SlangStage::Compute)],
            &[shaders.as_str()],
            &defines,
        )
        .expect("compile");
    assert_eq!(imported_modules(&container).unwrap(), vec!["goldy_exp".to_string()]);

    let libraries = compiler.imported_library_containers(&container, &[shaders.as_str()], &defines);
    assert_eq!(libraries.len(), 1);
    let refs: Vec<&[u8]> = libraries.iter().map(|l| l.as_slice()).collect();
    let linked = link_containers(&container, &refs).expect("link");
    // The translation unit's instructions come first; the library was appended.
    assert!(linked.tu_end > 0 && (linked.tu_end as usize) < linked.insts.len());
    let cfg = Cfg::build(&linked, linked.function_defs().next().expect("a function"));
    assert!(!cfg.blocks.is_empty());
    assert_eq!(cfg.idom[0], None);
}

/// Dump the linked Slang IR for one shader file. Run with
/// `GOLDY_IR_DUMP=path/to/shader.slang cargo test --lib slang::ir::tests::dump_ir -- --ignored --nocapture`
/// (compute entry point `cs_main` unless `GOLDY_IR_DUMP_ENTRY` says otherwise, `shaders/` on
/// the search path).
#[test]
#[ignore]
fn dump_ir() {
    let Ok(path) = std::env::var("GOLDY_IR_DUMP") else {
        eprintln!("set GOLDY_IR_DUMP to a .slang file");
        return;
    };
    let shaders = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .to_string_lossy()
        .into_owned();
    let entry = std::env::var("GOLDY_IR_DUMP_ENTRY").unwrap_or_else(|_| "cs_main".into());
    let stage = match entry.as_str() {
        "vs_main" => SlangStage::Vertex,
        "fs_main" => SlangStage::Fragment,
        _ => SlangStage::Compute,
    };
    let source = std::fs::read_to_string(&path).expect("read shader");
    let compiler = SlangCompiler::new().expect("Slang compiler unavailable");
    let effective = crate::slang::virtual_main::effective_slang_source_for_compile(&source);
    let defines = SlangCompiler::bindless_defines_for_target(ShaderTarget::Spirv);
    let container = compiler
        .compile_ir_container(
            effective.as_ref(),
            &[(entry.as_str(), stage)],
            &[shaders.as_str()],
            &defines,
        )
        .expect("compile");
    let libraries = compiler.imported_library_containers(&container, &[shaders.as_str()], &defines);
    let refs: Vec<&[u8]> = libraries.iter().map(|l| l.as_slice()).collect();
    println!("{}", link_containers(&container, &refs).expect("link").dump());
}
