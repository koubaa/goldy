//! Shader compilation and management logic.

use super::super::shared::{ShaderDesc, ShaderStageCompileDesc};
use super::super::{DeviceHandle, ShaderHandle};
use super::types::{MetalState, ShaderState};
use crate::slang::{ShaderTarget, SlangCompiler, SlangStage};
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{Device as MTLDevice, Library};
use std::collections::HashMap;

/// Patch vertex-shader MSL to inject the missing EntryPointParams [[buffer(1)]] binding.
///
/// Slang's Metal backend generates `EntryPointParams_0` inside `KernelContext_0` for
/// vertex shaders with `uniform` entry-point parameters, but — unlike compute shaders —
/// it never exposes the struct as a `[[buffer(1)]]` parameter in the vertex entry point.
/// The `entryPointParams_0` pointer in `KernelContext_0` therefore remains uninitialized,
/// causing `goldy_scattered` / `goldy_broadcast` to read from garbage memory.
///
/// This function patches the generated MSL to:
///   1. Add `EntryPointParams_0 constant* _goldy_ep [[buffer(1)]]` to the entry-point
///      parameter list (right before the closing `)` after `[[buffer(0)]]`).
///   2. Set `(&kernelContextN)->entryPointParams_0 = _goldy_ep;` immediately after the
///      `gGoldy_0` assignment that already exists in the entry-point body.
///
/// Only applied when `EntryPointParams_0` appears in the source but is not yet a
/// `[[buffer(1)]]` parameter (i.e., the Slang bug is present).
pub(super) fn patch_vertex_msl_entry_point_params(msl: &str) -> String {
    // Nothing to do if there are no entry-point params.
    if !msl.contains("EntryPointParams_0") {
        return msl.to_string();
    }
    // Already correctly bound (Slang fixed the bug, or it's a compute shader).
    if msl.contains("EntryPointParams_0 constant*") && msl.contains("[[buffer(1)]]") {
        return msl.to_string();
    }

    // Step 1 — inject the missing parameter into the entry-point signature.
    // The vertex entry point ends with `[[buffer(0)]])` (the closing paren is immediately
    // after the last attribute). We replace the first occurrence only.
    const SIG_NEEDLE: &str = "[[buffer(0)]])";
    const SIG_REPLACEMENT: &str =
        "[[buffer(0)]], EntryPointParams_0 constant* _goldy_ep [[buffer(1)]])";
    let patched = if let Some(pos) = msl.find(SIG_NEEDLE) {
        let mut s = String::with_capacity(msl.len() + 80);
        s.push_str(&msl[..pos]);
        s.push_str(SIG_REPLACEMENT);
        s.push_str(&msl[pos + SIG_NEEDLE.len()..]);
        s
    } else {
        // Pattern not found — leave unchanged (defensive).
        return msl.to_string();
    };

    // Step 2 — inject the assignment inside the entry-point body.
    // Find the ASSIGNMENT `->gGoldy_0 = ` (as opposed to member accesses `->gGoldy_0->`).
    // Extract the kernel-context variable name from the LHS: `(&KCTX)->gGoldy_0 = `.
    const ASSIGN_NEEDLE: &str = ")->gGoldy_0 = ";
    if let Some(arrow_pos) = patched.find(ASSIGN_NEEDLE) {
        // Walk backwards from arrow_pos to find the opening `(&`.
        let prefix = &patched[..arrow_pos];
        if let Some(amp_pos) = prefix.rfind("(&") {
            let kctx_name = &prefix[amp_pos + 2..]; // from after `(&` to arrow_pos
                                                    // Find the end of the assignment statement (the semicolon).
            let after_arrow = &patched[arrow_pos + ASSIGN_NEEDLE.len()..];
            if let Some(semi_rel) = after_arrow.find(';') {
                let semi_abs = arrow_pos + ASSIGN_NEEDLE.len() + semi_rel;
                let injection = format!("\n    (&{})->entryPointParams_0 = _goldy_ep;", kctx_name);
                let mut result = String::with_capacity(patched.len() + injection.len());
                result.push_str(&patched[..=semi_abs]);
                result.push_str(&injection);
                result.push_str(&patched[semi_abs + 1..]);
                return result;
            }
        }
    }

    patched
}

/// Compile a shader stage to MSL and create a Metal library.
fn compile_stage_with_reflection(
    slang_compiler: &SlangCompiler,
    device: &MTLDevice,
    desc: &ShaderStageCompileDesc<'_>,
) -> Result<(Library, Option<crate::slang::ShaderReflection>)> {
    let compile_outcome = slang_compiler.compile_bindless_with_reflection_and_defines(
        desc.slang_source,
        ShaderTarget::Metal,
        &[(desc.entry_point, desc.stage)],
        desc.search_paths,
        desc.extra_defines,
        desc.layout_checks,
        desc.optimization_level,
    );

    let result = compile_outcome
        .with_context(|| format!("Failed to compile {} shader stage", desc.entry_point))?;

    if !result.reflection.parameter_blocks.is_empty() {
        tracing::debug!(
            "Shader {} has {} ParameterBlock(s):",
            desc.entry_point,
            result.reflection.parameter_blocks.len()
        );
        for pb in &result.reflection.parameter_blocks {
            tracing::debug!(
                "  - {} at slot {} (size={}, alignment={}, fields={})",
                pb.name,
                pb.binding_slot,
                pb.size,
                pb.alignment,
                pb.fields.len()
            );
            for field in &pb.fields {
                tracing::debug!(
                    "    - {}: {:?} at offset {} (size={})",
                    field.name,
                    field.resource_kind,
                    field.offset,
                    field.size
                );
            }
        }
    }

    let raw_msl = result
        .shader
        .as_str()
        .context("Failed to get MSL source")?
        .to_string();

    // Apply vertex-shader patch for missing EntryPointParams [[buffer(1)]] binding.
    let msl_source = if desc.stage == SlangStage::Vertex {
        let patched = patch_vertex_msl_entry_point_params(&raw_msl);
        if patched != raw_msl {
            tracing::debug!(
                "Applied vertex EntryPointParams [[buffer(1)]] patch for {}",
                desc.entry_point
            );
        }
        patched
    } else {
        raw_msl
    };

    tracing::debug!(
        "Compiled MSL {} shader ({} bytes)",
        desc.entry_point,
        msl_source.len()
    );

    if let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") {
        use std::io::Write;
        use std::sync::atomic::{AtomicU32, Ordering};
        static DUMP_IDX: AtomicU32 = AtomicU32::new(0);
        let idx = DUMP_IDX.fetch_add(1, Ordering::Relaxed);
        let dir = std::path::Path::new(&dump_dir);
        let _ = std::fs::create_dir_all(dir);
        let filename = format!("{:03}_{}.metal", idx, desc.entry_point);
        if let Ok(mut f) = std::fs::File::create(dir.join(&filename)) {
            let _ = f.write_all(msl_source.as_bytes());
            tracing::info!("Dumped MSL to {}/{}", dump_dir, filename);
        }
    }

    let library = device
        .new_library_with_source(&msl_source, &mtl::CompileOptions::new())
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to create Metal library for {}: {}",
                desc.entry_point,
                e
            )
        })?;

    Ok((library, Some(result.reflection)))
}

/// Ensure a shader stage is compiled. Compiles on first access.
pub(super) fn ensure_stage_compiled(
    state: &mut MetalState,
    shader_handle: ShaderHandle,
    stage: SlangStage,
) -> Result<()> {
    struct CompileScratch {
        device_handle: DeviceHandle,
        slang_source: String,
        search_paths: Vec<String>,
        optimization_level: crate::types::OptimizationLevel,
        defines: Vec<(String, String)>,
        layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
        entry_point: &'static str,
    }

    let maybe_scratch: Option<CompileScratch> = {
        let shaders = &state.shaders;
        let shader = shaders
            .get(&shader_handle)
            .context("Invalid shader handle")?;
        let (entry_point, need_compile) = match stage {
            SlangStage::Vertex => ("vs_main", shader.vertex_library.is_none()),
            SlangStage::Fragment => ("fs_main", shader.fragment_library.is_none()),
            SlangStage::Compute => ("cs_main", shader.compute_library.is_none()),
            _ => anyhow::bail!("Metal backend only supports Vertex, Fragment, and Compute stages"),
        };
        if !need_compile {
            None
        } else {
            Some(CompileScratch {
                device_handle: shader.device_handle,
                slang_source: shader.slang_source.clone(),
                search_paths: shader.search_paths.clone(),
                optimization_level: shader.optimization_level,
                defines: shader.defines.clone(),
                layout_checks: shader.layout_checks.clone(),
                entry_point,
            })
        }
    };

    let Some(scratch) = maybe_scratch else {
        return Ok(());
    };

    let mtl_dev = state
        .devices
        .get(&scratch.device_handle)
        .context("Shader's device no longer valid")?
        .device
        .clone();

    let search_path_refs: Vec<&str> = scratch.search_paths.iter().map(|s| s.as_str()).collect();
    let extra_defines: Vec<(&str, &str)> = scratch
        .defines
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let compile_desc = ShaderStageCompileDesc {
        slang_source: &scratch.slang_source,
        search_paths: &search_path_refs,
        entry_point: scratch.entry_point,
        stage,
        extra_defines: &extra_defines,
        layout_checks: &scratch.layout_checks,
        optimization_level: scratch.optimization_level,
    };

    let compiler = state.slang_compiler_mut_or_init()?;
    let (library, reflection) = compile_stage_with_reflection(compiler, &mtl_dev, &compile_desc)?;

    let shader = state
        .shaders
        .get_mut(&shader_handle)
        .expect("shader handle must be valid after ensure_stage_compiled");
    match stage {
        SlangStage::Vertex => shader.vertex_library = Some(library),
        SlangStage::Fragment => shader.fragment_library = Some(library),
        SlangStage::Compute => shader.compute_library = Some(library),
        _ => unreachable!("stage already validated"),
    }

    if shader.reflection.is_none() {
        let reflection = reflection.map(|mut r| {
            if r.push_constant_categories.is_empty() {
                r.push_constant_categories =
                    crate::slang::virtual_main::extract_push_constant_categories(
                        &scratch.slang_source,
                    );
            }
            r
        });
        shader.reflection = reflection;
    }

    if !scratch.layout_checks.is_empty() {
        shader.layout_checks.clear();
    }

    Ok(())
}

/// Create a shader handle (compilation deferred to pipeline creation).
pub(super) fn create(
    devices: &HashMap<DeviceHandle, super::types::LogicalDevice>,
    shaders: &mut HashMap<ShaderHandle, ShaderState>,
    next_shader_handle: &mut ShaderHandle,
    desc: ShaderDesc<'_>,
) -> Result<ShaderHandle> {
    devices.get(&desc.device).context("Invalid device handle")?;

    let handle = *next_shader_handle;
    *next_shader_handle += 1;

    shaders.insert(
        handle,
        ShaderState {
            device_handle: desc.device,
            slang_source: desc.slang_source.to_string(),
            search_paths: desc.search_paths.iter().map(|s| s.to_string()).collect(),
            defines: desc
                .defines
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            optimization_level: desc.optimization_level,
            vertex_library: None,
            fragment_library: None,
            compute_library: None,
            reflection: None,
            layout_checks: desc.layout_checks,
        },
    );

    tracing::debug!("Created shader handle {} (compilation deferred)", handle);
    Ok(handle)
}

/// Destroy a shader.
pub(super) fn destroy(
    _devices: &HashMap<DeviceHandle, super::types::LogicalDevice>,
    shaders: &mut HashMap<ShaderHandle, ShaderState>,
    shader_handle: ShaderHandle,
) {
    shaders.remove(&shader_handle);
}
