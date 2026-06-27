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
pub(super) fn ensure_stage_compiled(
    state: &mut Dx12State,
    shader_handle: ShaderHandle,
    stage: crate::slang::SlangStage,
) -> Result<Vec<u8>> {
    {
        let shaders_read = state.shaders.read().unwrap();
        let shader = shaders_read
            .entries
            .get(&shader_handle)
            .context("Invalid shader handle")?;
        let cached_bytecode = match stage {
            crate::slang::SlangStage::Vertex => shader.vertex_bytecode.clone(),
            crate::slang::SlangStage::Fragment => shader.fragment_bytecode.clone(),
            crate::slang::SlangStage::Compute => shader.compute_bytecode.clone(),
            _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
        };
        if let Some(bytecode) = cached_bytecode {
            return Ok(bytecode);
        }
    }

    // Get the entry point name based on stage
    let entry_point_name = match stage {
        crate::slang::SlangStage::Vertex => "vs_main",
        crate::slang::SlangStage::Fragment => "fs_main",
        crate::slang::SlangStage::Compute => "cs_main",
        _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
    };

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

    // Convert search_paths to &str references
    let search_path_refs: Vec<&str> = search_paths.iter().map(|s| s.as_str()).collect();

    // Merge target define with shader-specific defines
    let mut defines: Vec<(&str, &str)> = vec![("__DX12__", "1")];
    for (k, v) in &extra_defines {
        defines.push((k.as_str(), v.as_str()));
    }

    // Compile Slang directly to DXIL (SM 6.6 bindless)
    let compile_result = {
        let _tz = crate::tracy_zone!("goldy.ensure_stage_compiled.slang_cache");
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
        match stage {
            crate::slang::SlangStage::Vertex => shader.vertex_bytecode = Some(bytecode.clone()),
            crate::slang::SlangStage::Fragment => shader.fragment_bytecode = Some(bytecode.clone()),
            crate::slang::SlangStage::Compute => shader.compute_bytecode = Some(bytecode.clone()),
            _ => {} // Already validated above
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
        } else {
            shader.reflection = Some(new_reflection);
        }
    }

    Ok(bytecode)
}
