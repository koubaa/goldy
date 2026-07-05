//! Shader management logic.

use super::types::{Dx12State, ShaderState};
use super::ShaderHandle;
use anyhow::{Context, Result};

/// Entry point name for a Slang shader stage, shared by the deferred (`ensure_stage_compiled`)
/// and pre-warm (`prepare_stage_precompile`) paths.
fn entry_point_name(stage: crate::slang::SlangStage) -> Result<&'static str> {
    Ok(match stage {
        crate::slang::SlangStage::Vertex => "vs_main",
        crate::slang::SlangStage::Fragment => "fs_main",
        crate::slang::SlangStage::Compute => "cs_main",
        _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
    })
}

fn cached_bytecode(state: &Dx12State, shader_handle: ShaderHandle, stage: crate::slang::SlangStage) -> Result<Option<Vec<u8>>> {
    let shaders_read = state.shaders.read().unwrap();
    let shader = shaders_read
        .entries
        .get(&shader_handle)
        .context("Invalid shader handle")?;
    Ok(match stage {
        crate::slang::SlangStage::Vertex => shader.vertex_bytecode.clone(),
        crate::slang::SlangStage::Fragment => shader.fragment_bytecode.clone(),
        crate::slang::SlangStage::Compute => shader.compute_bytecode.clone(),
        _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
    })
}

/// Snapshot compile inputs for `stage` without holding `Dx12State` exclusively — the caller
/// can drop this backend's per-device lock before running the returned prep's (slow) compile.
///
/// Returns `Ok(None)` when `stage`'s bytecode is already cached (nothing to precompile) so
/// the caller can just proceed straight to pipeline creation.
pub(super) fn prepare_stage_precompile(
    state: &Dx12State,
    shader_handle: ShaderHandle,
    stage: crate::slang::SlangStage,
) -> Result<Option<crate::backend::ShaderStagePrecompilePrep>> {
    if cached_bytecode(state, shader_handle, stage)?.is_some() {
        return Ok(None);
    }
    let entry_point = entry_point_name(stage)?;
    let (slang_source, search_paths, optimization_level, extra_defines, layout_checks_snapshot) = {
        let shaders_read = state.shaders.read().unwrap();
        let shader = shaders_read
            .entries
            .get(&shader_handle)
            .context("Invalid shader handle")?;
        (
            shader.slang_source.clone(),
            shader.search_paths.clone(),
            shader.optimization_level,
            shader.defines.clone(),
            shader.layout_checks.clone(),
        )
    };
    let mut defines: Vec<(String, String)> = vec![("__DX12__".to_string(), "1".to_string())];
    defines.extend(extra_defines);
    Ok(Some(crate::backend::ShaderStagePrecompilePrep::new(
        std::sync::Arc::clone(&state.slang_compiler),
        std::sync::Arc::clone(&state.shader_compile_lock),
        slang_source,
        search_paths,
        defines,
        optimization_level,
        layout_checks_snapshot,
        entry_point,
        crate::slang::ShaderTarget::Dxil,
        stage,
    )))
}

/// Store the result of running [`crate::backend::ShaderStagePrecompilePrep::compile`] for
/// `stage`, mirroring the tail of [`ensure_stage_compiled`].
pub(super) fn store_precompiled_stage(
    state: &mut Dx12State,
    shader_handle: ShaderHandle,
    stage: crate::slang::SlangStage,
    compiled: crate::slang::CompiledShaderWithReflection,
) -> Result<()> {
    let entry_point = entry_point_name(stage)?;
    let bytecode = compiled.shader.as_dxil().context("Invalid DXIL output")?.to_vec();
    let slang_source = {
        let shaders_read = state.shaders.read().unwrap();
        shaders_read
            .entries
            .get(&shader_handle)
            .context("Invalid shader handle")?
            .slang_source
            .clone()
    };
    let new_reflection = {
        let mut r = compiled.reflection;
        if r.push_constant_categories.is_empty() {
            r.push_constant_categories = crate::slang::virtual_main::extract_push_constant_categories(&slang_source);
        }
        if r.push_constant_slot_kinds.is_empty() {
            r.push_constant_slot_kinds = crate::slang::virtual_main::extract_push_constant_slot_kinds(&slang_source);
        }
        r
    };

    tracing::debug!("Compiled {} to DXIL ({} bytes)", entry_point, bytecode.len());

    if let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") {
        use std::io::Write;
        let dir = std::path::Path::new(&dump_dir);
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(format!("{}_h{}_dx12.dxil", entry_point, shader_handle));
        if let Ok(mut file) = std::fs::File::create(&path) {
            let _ = file.write_all(&bytecode);
            tracing::info!("Dumped DXIL bytecode to {}", path.display());
        }
    }

    let mut shaders_write = state.shaders.write().unwrap();
    let shader = shaders_write
        .entries
        .get_mut(&shader_handle)
        .context("Invalid shader handle")?;
    match stage {
        crate::slang::SlangStage::Vertex => shader.vertex_bytecode = Some(bytecode),
        crate::slang::SlangStage::Fragment => shader.fragment_bytecode = Some(bytecode),
        crate::slang::SlangStage::Compute => shader.compute_bytecode = Some(bytecode),
        _ => {}
    }

    if !shader.layout_checks.is_empty() {
        shader.layout_checks.clear();
    }

    if let Some(ref mut existing) = shader.reflection {
        for pb in &new_reflection.parameter_blocks {
            if !existing.parameter_blocks.iter().any(|p| p.name == pb.name) {
                existing.parameter_blocks.push(pb.clone());
            }
        }
        if existing.push_constant_categories.is_empty() {
            existing.push_constant_categories = new_reflection.push_constant_categories;
        }
        if existing.push_constant_slot_kinds.is_empty() {
            existing.push_constant_slot_kinds = new_reflection.push_constant_slot_kinds;
        }
        if existing.binding_element_strides.is_empty() {
            existing.binding_element_strides = new_reflection.binding_element_strides;
        }
    } else {
        shader.reflection = Some(new_reflection);
    }

    Ok(())
}

pub(super) fn create_with_checks(
    state: &mut Dx12State,
    desc: crate::backend::shared::ShaderDesc<'_>,
) -> Result<ShaderHandle> {
    let _ = state.devices.get(&desc.device).context("Invalid device handle")?;

    let handle = state.shaders.write().unwrap().alloc_handle();

    let stored_paths: Vec<String> = desc.search_paths.iter().map(|s| s.to_string()).collect();
    let stored_defines: Vec<(String, String)> = desc
        .defines
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

    state.shaders.write().unwrap().entries.insert(
        handle,
        ShaderState {
            device_handle: desc.device,
            slang_source: desc.slang_source.to_string(),
            search_paths: stored_paths,
            defines: stored_defines,
            optimization_level: desc.optimization_level,
            vertex_bytecode: None,
            fragment_bytecode: None,
            compute_bytecode: None,
            reflection: None,
            layout_checks: desc.layout_checks,
        },
    );

    tracing::debug!("Created shader handle {} (compilation deferred)", handle);
    Ok(handle)
}

/// Destroy a shader.
pub(super) fn destroy(state: &mut Dx12State, shader_handle: ShaderHandle) {
    state.shaders.write().unwrap().entries.remove(&shader_handle);
}

/// Compile a shader for a specific stage on demand.
/// Uses Slang to compile directly to DXIL (SM 6.6) for bindless support.
///
/// This is the fallback / non-pre-warmed path (used by graphics pipeline creation, which
/// doesn't pre-warm compiles yet): it compiles under whatever lock the caller already holds
/// on `state` (still serialized behind this backend's per-device lock via the caller). See
/// [`prepare_stage_precompile`] / [`store_precompiled_stage`] for the pre-warmed path used by
/// compute pipeline creation, which compiles under a dedicated shader-compilation lock
/// instead of the backend's exclusive per-device lock.
pub(super) fn ensure_stage_compiled(
    state: &mut Dx12State,
    shader_handle: ShaderHandle,
    stage: crate::slang::SlangStage,
) -> Result<Vec<u8>> {
    if let Some(bytecode) = cached_bytecode(state, shader_handle, stage)? {
        return Ok(bytecode);
    }

    let prep = prepare_stage_precompile(state, shader_handle, stage)?
        .context("shader entry disappeared while preparing to compile")?;

    let compiled = {
        let _tz = crate::tracy_zone!("goldy.ensure_stage_compiled.slang_cache");
        prep.compile()?
    };

    store_precompiled_stage(state, shader_handle, stage, compiled)?;

    cached_bytecode(state, shader_handle, stage)?.context("bytecode missing immediately after compile")
}
