//! Shader management logic.

use super::types::{Dx12State, ShaderState};
use super::{DeviceHandle, ShaderHandle};
use anyhow::{Context, Result};

/// Create a shader from Slang source code.
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    slang_source: &str,
) -> Result<ShaderHandle> {
    create_with_paths(state, device_handle, slang_source, &[])
}

/// Create a shader from Slang source code with custom search paths.
pub(super) fn create_with_paths(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    slang_source: &str,
    search_paths: &[&str],
) -> Result<ShaderHandle> {
    let _ = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let handle = state.next_shader_handle;
    state.next_shader_handle += 1;

    let stored_paths: Vec<String> = search_paths.iter().map(|s| s.to_string()).collect();

    state.shaders.insert(
        handle,
        ShaderState {
            device_handle,
            slang_source: slang_source.to_string(),
            search_paths: stored_paths,
            vertex_bytecode: None,
            fragment_bytecode: None,
            compute_bytecode: None,
            reflection: None,
        },
    );

    tracing::debug!("Created shader handle {} (compilation deferred)", handle);
    Ok(handle)
}

/// Destroy a shader.
pub(super) fn destroy(state: &mut Dx12State, shader_handle: ShaderHandle) {
    state.shaders.remove(&shader_handle);
}

/// Compile a shader for a specific stage on demand.
/// Uses Slang to compile directly to DXIL (SM 6.6) for bindless support.
pub(super) fn ensure_stage_compiled(
    state: &mut Dx12State,
    shader_handle: ShaderHandle,
    stage: crate::slang::SlangStage,
) -> Result<Vec<u8>> {
    let shader = state
        .shaders
        .get_mut(&shader_handle)
        .context("Invalid shader handle")?;

    // Check if already compiled for this stage
    let cached_bytecode = match stage {
        crate::slang::SlangStage::Vertex => shader.vertex_bytecode.clone(),
        crate::slang::SlangStage::Fragment => shader.fragment_bytecode.clone(),
        crate::slang::SlangStage::Compute => shader.compute_bytecode.clone(),
        _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
    };

    if let Some(bytecode) = cached_bytecode {
        return Ok(bytecode);
    }

    // Get the entry point name based on stage
    let entry_point_name = match stage {
        crate::slang::SlangStage::Vertex => "vs_main",
        crate::slang::SlangStage::Fragment => "fs_main",
        crate::slang::SlangStage::Compute => "cs_main",
        _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
    };

    // Clone source and search paths to avoid borrow issues
    let slang_source = shader.slang_source.clone();
    let search_paths = shader.search_paths.clone();
    let device_handle = shader.device_handle;

    // Convert search_paths to &str references
    let search_path_refs: Vec<&str> = search_paths.iter().map(|s| s.as_str()).collect();

    // Check if bindless is enabled on the device
    let bindless_enabled = state
        .devices
        .get(&device_handle)
        .map(|d| d.bindless_enabled)
        .unwrap_or(false);

    // Compile Slang directly to DXIL (SM 6.6 for bindless support)
    // This bypasses FXC entirely and uses Slang's built-in DXIL emission
    let (bytecode, reflection) = if bindless_enabled {
        // Define __DX12__ for DX12 ResourceDescriptorHeap pattern
        let compile_result = state.slang_compiler.compile_with_reflection(
            &slang_source,
            crate::slang::ShaderTarget::Dxil,
            &[(entry_point_name, stage)],
            &search_path_refs,
            &[("__DX12__", "1")],
        );

        let result = compile_result.with_context(|| {
            format!(
                "Failed to compile {} shader to DXIL (bindless)",
                entry_point_name
            )
        })?;

        let dxil = result
            .shader
            .as_dxil()
            .context("Invalid DXIL output")?
            .to_vec();
        (dxil, Some(result.reflection))
    } else {
        let dxil_compiled = state
            .slang_compiler
            .compile_with_options(
                &slang_source,
                crate::slang::ShaderTarget::Dxil,
                &[(entry_point_name, stage)],
                &search_path_refs,
            )
            .with_context(|| {
                format!("Failed to compile {} shader to DXIL", entry_point_name)
            })?;

        let dxil = dxil_compiled
            .as_dxil()
            .context("Invalid DXIL output")?
            .to_vec();
        (dxil, None)
    };

    tracing::debug!(
        "Compiled {} to DXIL ({} bytes, bindless={})",
        entry_point_name,
        bytecode.len(),
        bindless_enabled
    );

    // Dump DXIL for debugging when GOLDY_DUMP_SHADERS is set
    if let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") {
        use std::io::Write;
        let path =
            std::path::Path::new(&dump_dir).join(format!("{}_dx12.dxil", entry_point_name));
        if let Ok(mut file) = std::fs::File::create(&path) {
            let _ = file.write_all(&bytecode);
            tracing::info!("Dumped DXIL bytecode to {}", path.display());
        }
    }

    // Cache the bytecode and reflection
    let shader = state.shaders.get_mut(&shader_handle).unwrap();
    match stage {
        crate::slang::SlangStage::Vertex => shader.vertex_bytecode = Some(bytecode.clone()),
        crate::slang::SlangStage::Fragment => shader.fragment_bytecode = Some(bytecode.clone()),
        crate::slang::SlangStage::Compute => shader.compute_bytecode = Some(bytecode.clone()),
        _ => {} // Already validated above
    }

    // Store reflection data (merge with existing if any)
    if let Some(new_reflection) = reflection {
        if let Some(ref mut existing) = shader.reflection {
            // Merge parameter blocks
            for pb in &new_reflection.parameter_blocks {
                if !existing.parameter_blocks.iter().any(|p| p.name == pb.name) {
                    existing.parameter_blocks.push(pb.clone());
                }
            }
        } else {
            shader.reflection = Some(new_reflection);
        }
    }

    Ok(bytecode)
}
