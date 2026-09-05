//! Shader management logic.

use super::types::{Dx12State, ShaderState};
use super::ShaderHandle;
use anyhow::{Context, Result};

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
            extra_bytecode: std::collections::HashMap::new(),
            reflection: None,
            layout_checks: desc.layout_checks,
            stage_slot_remaps: std::collections::HashMap::new(),
            remapped_bytecode: std::collections::HashMap::new(),
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
pub(super) fn ensure_stage_compiled(
    state: &mut Dx12State,
    shader_handle: ShaderHandle,
    stage: crate::slang::SlangStage,
) -> Result<Vec<u8>> {
    let remap_fp = {
        let shaders_read = state.shaders.read().unwrap();
        let shader = shaders_read
            .entries
            .get(&shader_handle)
            .context("Invalid shader handle")?;
        let remap = shader.stage_slot_remaps.get(&stage);
        let fp = remap
            .map(crate::slang::graphics_link::slot_remap_fingerprint)
            .unwrap_or(0);
        if fp != 0 {
            if let Some(bytecode) = shader.remapped_bytecode.get(&(stage as u32, fp)) {
                return Ok(bytecode.clone());
            }
        } else {
            let cached_bytecode = match stage {
                crate::slang::SlangStage::Vertex => shader.vertex_bytecode.clone(),
                crate::slang::SlangStage::Fragment => shader.fragment_bytecode.clone(),
                crate::slang::SlangStage::Compute => shader.compute_bytecode.clone(),
                crate::slang::SlangStage::RayGeneration
                | crate::slang::SlangStage::Intersection
                | crate::slang::SlangStage::AnyHit
                | crate::slang::SlangStage::ClosestHit
                | crate::slang::SlangStage::Miss
                | crate::slang::SlangStage::Callable
                | crate::slang::SlangStage::Mesh
                | crate::slang::SlangStage::Amplification => shader.extra_bytecode.get(&stage).cloned(),
                other => anyhow::bail!("Unsupported shader stage: {:?}", other),
            };
            if let Some(bytecode) = cached_bytecode {
                return Ok(bytecode);
            }
        }
        fp
    };

    let entry_point_name = crate::slang::canonical_entry_point(stage)
        .ok_or_else(|| anyhow::anyhow!("Unsupported shader stage: {:?}", stage))?;

    let (slang_source, search_paths, optimization_level, extra_defines, layout_checks_snapshot) = {
        let shaders_read = state.shaders.read().unwrap();
        let shader = shaders_read
            .entries
            .get(&shader_handle)
            .context("Invalid shader handle")?;
        let remap = shader.stage_slot_remaps.get(&stage);
        let source = crate::backend::shared::shader_source_with_stage_remap(&shader.slang_source, stage, remap)
            .into_owned();
        (
            source,
            shader.search_paths.clone(),
            shader.optimization_level,
            shader.defines.clone(),
            shader.layout_checks.clone(),
        )
    };

    // Convert search_paths to &str references
    let search_path_refs: Vec<&str> = search_paths.iter().map(|s| s.as_str()).collect();

    // Merge target define with shader-specific defines
    let mut defines: Vec<(&str, &str)> = vec![("__DX12__", "1")];
    for (k, v) in &extra_defines {
        defines.push((k.as_str(), v.as_str()));
    }
    if extra_defines.iter().all(|(k, _)| k.as_str() != "GOLDY_RAY_QUERY") {
        if let Some(shader) = state.shaders.read().unwrap().entries.get(&shader_handle) {
            if let Some(ld) = state.devices.get(&shader.device_handle) {
                if state
                    .adapters
                    .iter()
                    .find(|a| a.adapter_id == ld.adapter_id)
                    .is_some_and(|a| a.ray_query || a.ray_tracing_pipelines)
                {
                    defines.push(("GOLDY_RAY_QUERY", "1"));
                }
            }
        }
    }

    // Compile Slang directly to DXIL (SM 6.6 bindless)
    let compile_result = {
        let _tz = crate::tracy_zone!("goldy.ensure_stage_compiled.slang_cache");
        let _st = crate::shader_timing::scope("dx12.ensure_stage_compiled.slang", entry_point_name);
        state.slang_compiler.compile_with_reflection(
            &slang_source,
            crate::slang::ShaderTarget::Dxil,
            &[(entry_point_name, stage)],
            &search_path_refs,
            &defines,
            &layout_checks_snapshot,
            optimization_level,
        )
    };

    let result =
        compile_result.with_context(|| format!("Failed to compile {} shader to DXIL (bindless)", entry_point_name))?;

    let bytecode = result.shader.as_dxil().context("Invalid DXIL output")?.to_vec();
    let new_reflection = {
        let mut r = result.reflection;
        if r.push_constant_categories.is_empty() {
            r.push_constant_categories = crate::slang::virtual_main::extract_push_constant_categories(&slang_source);
        }
        if r.push_constant_slot_kinds.is_empty() {
            r.push_constant_slot_kinds = crate::slang::virtual_main::extract_push_constant_slot_kinds(&slang_source);
        }
        r
    };

    tracing::debug!("Compiled {} to DXIL ({} bytes)", entry_point_name, bytecode.len(),);

    // Dump DXIL for debugging when GOLDY_DUMP_SHADERS is set
    if let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") {
        use std::io::Write;
        let dir = std::path::Path::new(&dump_dir);
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(format!("{}_h{}_dx12.dxil", entry_point_name, shader_handle));
        if let Ok(mut file) = std::fs::File::create(&path) {
            let _ = file.write_all(&bytecode);
            tracing::info!("Dumped DXIL bytecode to {}", path.display());
        }
    }

    {
        let mut shaders_write = state.shaders.write().unwrap();
        let shader = shaders_write.entries.get_mut(&shader_handle).unwrap();
        if remap_fp != 0 {
            shader.remapped_bytecode.insert((stage as u32, remap_fp), bytecode.clone());
        } else {
            match stage {
                crate::slang::SlangStage::Vertex => shader.vertex_bytecode = Some(bytecode.clone()),
                crate::slang::SlangStage::Fragment => shader.fragment_bytecode = Some(bytecode.clone()),
                crate::slang::SlangStage::Compute => shader.compute_bytecode = Some(bytecode.clone()),
                crate::slang::SlangStage::RayGeneration
                | crate::slang::SlangStage::Intersection
                | crate::slang::SlangStage::AnyHit
                | crate::slang::SlangStage::ClosestHit
                | crate::slang::SlangStage::Miss
                | crate::slang::SlangStage::Callable
                | crate::slang::SlangStage::Mesh
                | crate::slang::SlangStage::Amplification => {
                    shader.extra_bytecode.insert(stage, bytecode.clone());
                }
                _ => {}
            }
        }

        if !layout_checks_snapshot.is_empty() {
            shader.layout_checks.clear();
        }

        // Store reflection data (merge with existing if any)
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
            for iface in new_reflection.stage_interfaces {
                if !existing.stage_interfaces.iter().any(|s| s.entry_name == iface.entry_name && s.stage == iface.stage)
                {
                    existing.stage_interfaces.push(iface);
                }
            }
        } else {
            shader.reflection = Some(new_reflection);
        }
    }

    Ok(bytecode)
}
