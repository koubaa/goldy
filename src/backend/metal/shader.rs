//! Shader compilation and management logic.

use super::super::{DeviceHandle, ShaderHandle};
use super::types::ShaderState;
use crate::slang::{ShaderTarget, SlangCompiler, SlangStage};
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{Device as MTLDevice, Library};
use std::collections::HashMap;

/// Parse `[numthreads(x, y, z)]` from Slang shader source.
///
/// Returns `None` if the attribute is absent; the caller falls back to `[64, 1, 1]`.
pub(super) fn parse_numthreads(source: &str) -> Option<[u32; 3]> {
    let kw_pos = source.find("numthreads")?;
    let after_kw = source[kw_pos + "numthreads".len()..].trim_start();
    let args_str = after_kw.strip_prefix('(')?;
    let close = args_str.find(')')?;
    let parts: Vec<&str> = args_str[..close].split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return None;
    }
    let x = parts[0].parse().ok()?;
    let y = parts[1].parse().ok()?;
    let z = parts[2].parse().ok()?;
    Some([x, y, z])
}

/// Compile a shader stage to MSL and create a Metal library.
#[allow(clippy::too_many_arguments)] // Mirrors Slang compile API surface.
fn compile_stage_with_reflection(
    slang_compiler: &SlangCompiler,
    device: &MTLDevice,
    slang_source: &str,
    search_paths: &[String],
    entry_point: &str,
    stage: SlangStage,
    extra_defines: &[(&str, &str)],
    optimization_level: crate::types::OptimizationLevel,
) -> Result<(Library, Option<crate::slang::ShaderReflection>)> {
    let search_path_refs: Vec<&str> = search_paths.iter().map(|s| s.as_str()).collect();

    let result = slang_compiler
        .compile_bindless_with_reflection_and_defines(
            slang_source,
            ShaderTarget::Metal,
            &[(entry_point, stage)],
            &search_path_refs,
            extra_defines,
            optimization_level,
        )
        .with_context(|| format!("Failed to compile {} shader stage", entry_point))?;

    if !result.reflection.parameter_blocks.is_empty() {
        tracing::info!(
            "Shader {} has {} ParameterBlock(s):",
            entry_point,
            result.reflection.parameter_blocks.len()
        );
        for pb in &result.reflection.parameter_blocks {
            tracing::info!(
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

    let msl_source = result
        .shader
        .as_str()
        .context("Failed to get MSL source")?
        .to_string();

    tracing::debug!(
        "Compiled MSL {} shader ({} bytes)",
        entry_point,
        msl_source.len()
    );

    let library = device
        .new_library_with_source(&msl_source, &mtl::CompileOptions::new())
        .map_err(|e| {
            anyhow::anyhow!("Failed to create Metal library for {}: {}", entry_point, e)
        })?;

    Ok((library, Some(result.reflection)))
}

/// Ensure a shader stage is compiled. Compiles on first access.
pub(super) fn ensure_stage_compiled(
    slang_compiler: &SlangCompiler,
    devices: &HashMap<DeviceHandle, super::types::LogicalDevice>,
    shaders: &mut HashMap<ShaderHandle, ShaderState>,
    shader_handle: ShaderHandle,
    stage: SlangStage,
) -> Result<()> {
    let (entry_point, need_compile) = match stage {
        SlangStage::Vertex => ("vs_main", {
            let s = shaders
                .get(&shader_handle)
                .context("Invalid shader handle")?;
            s.vertex_library.is_none()
        }),
        SlangStage::Fragment => ("fs_main", {
            let s = shaders
                .get(&shader_handle)
                .context("Invalid shader handle")?;
            s.fragment_library.is_none()
        }),
        SlangStage::Compute => ("cs_main", {
            let s = shaders
                .get(&shader_handle)
                .context("Invalid shader handle")?;
            s.compute_library.is_none()
        }),
        _ => anyhow::bail!("Metal backend only supports Vertex, Fragment, and Compute stages"),
    };

    if !need_compile {
        return Ok(());
    }

    let shader = shaders
        .get(&shader_handle)
        .context("Invalid shader handle")?;
    let device_handle = shader.device_handle;
    let slang_source = shader.slang_source.clone();
    let search_paths = shader.search_paths.clone();
    let optimization_level = shader.optimization_level;
    let extra_defines: Vec<(&str, &str)> = shader
        .defines
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let logical_device = devices
        .get(&device_handle)
        .context("Shader's device no longer valid")?;

    let (library, reflection) = compile_stage_with_reflection(
        slang_compiler,
        &logical_device.device,
        &slang_source,
        &search_paths,
        entry_point,
        stage,
        &extra_defines,
        optimization_level,
    )?;

    let shader = shaders.get_mut(&shader_handle).unwrap();
    match stage {
        SlangStage::Vertex => shader.vertex_library = Some(library),
        SlangStage::Fragment => shader.fragment_library = Some(library),
        SlangStage::Compute => shader.compute_library = Some(library),
        _ => unreachable!("stage already validated above"),
    }
    if shader.reflection.is_none() {
        shader.reflection = reflection;
    }

    Ok(())
}

/// Create a shader handle (compilation deferred to pipeline creation).
#[allow(clippy::too_many_arguments)] // Backend entry point; parameters map 1:1 to GpuBackend::create_shader.
pub(super) fn create(
    devices: &HashMap<DeviceHandle, super::types::LogicalDevice>,
    shaders: &mut HashMap<ShaderHandle, ShaderState>,
    next_shader_handle: &mut ShaderHandle,
    device_handle: DeviceHandle,
    slang_source: &str,
    search_paths: &[&str],
    defines: &[(&str, &str)],
    optimization_level: crate::types::OptimizationLevel,
) -> Result<ShaderHandle> {
    devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let handle = *next_shader_handle;
    *next_shader_handle += 1;

    shaders.insert(
        handle,
        ShaderState {
            device_handle,
            slang_source: slang_source.to_string(),
            search_paths: search_paths.iter().map(|s| s.to_string()).collect(),
            defines: defines
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            optimization_level,
            vertex_library: None,
            fragment_library: None,
            compute_library: None,
            reflection: None,
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
