//! Shader management logic.

use super::types::{self, ShaderState};
use super::{DeviceHandle, ShaderHandle};
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

/// Create a shader from Slang source code.
pub(super) fn create(
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    shaders: &mut HashMap<ShaderHandle, ShaderState>,
    next_shader_handle: &mut ShaderHandle,
    desc: crate::backend::shared::ShaderDesc<'_>,
) -> Result<ShaderHandle> {
    // Just validate the device exists - actual compilation happens at pipeline creation
    let _ = devices.get(&desc.device).context("Invalid device handle")?;

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
            vertex_module: None,
            fragment_module: None,
            compute_module: None,
            reflection: None,
            layout_checks: desc.layout_checks,
        },
    );

    tracing::debug!("Created shader handle {} (compilation deferred)", handle);
    Ok(handle)
}

/// Destroy a shader and clean up compiled shader modules.
pub(super) fn destroy(
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    shaders: &mut HashMap<ShaderHandle, ShaderState>,
    shader_handle: ShaderHandle,
) {
    if let Some(shader) = shaders.remove(&shader_handle) {
        if let Some(device) = devices.get(&shader.device_handle) {
            unsafe {
                if let Some(module) = shader.vertex_module {
                    device.device.destroy_shader_module(module, None);
                }
                if let Some(module) = shader.fragment_module {
                    device.device.destroy_shader_module(module, None);
                }
                if let Some(module) = shader.compute_module {
                    device.device.destroy_shader_module(module, None);
                }
            }
        }
    }
}

/// Compile a shader for a specific stage on demand.
pub(super) fn ensure_stage_compiled(
    slang_compiler: &crate::slang::SlangCompiler,
    devices: &HashMap<DeviceHandle, types::LogicalDevice>,
    shaders: &mut HashMap<ShaderHandle, ShaderState>,
    shader_handle: ShaderHandle,
    stage: crate::slang::SlangStage,
) -> Result<vk::ShaderModule> {
    let shader = shaders
        .get_mut(&shader_handle)
        .context("Invalid shader handle")?;

    // Check if already compiled for this stage
    let cached_module = match stage {
        crate::slang::SlangStage::Vertex => shader.vertex_module,
        crate::slang::SlangStage::Fragment => shader.fragment_module,
        crate::slang::SlangStage::Compute => shader.compute_module,
        _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
    };

    if let Some(module) = cached_module {
        return Ok(module);
    }

    // Get the entry point name based on stage
    let entry_point_name = match stage {
        crate::slang::SlangStage::Vertex => "vs_main",
        crate::slang::SlangStage::Fragment => "fs_main",
        crate::slang::SlangStage::Compute => "cs_main",
        _ => anyhow::bail!("Unsupported shader stage: {:?}", stage),
    };

    // Clone source, search paths, and defines to avoid borrow issues
    let slang_source = shader.slang_source.clone();
    let search_paths: Vec<&str> = shader.search_paths.iter().map(|s| s.as_str()).collect();
    let extra_defines: Vec<(&str, &str)> = shader
        .defines
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let device_handle = shader.device_handle;
    let optimization_level = shader.optimization_level;
    let layout_checks_snapshot = shader.layout_checks.clone();

    // Compile shader with reflection data for resource binding
    let result = slang_compiler
        .compile_bindless_with_reflection_and_defines(
            &slang_source,
            crate::slang::ShaderTarget::Spirv,
            &[(entry_point_name, stage)],
            &search_paths,
            &extra_defines,
            &layout_checks_snapshot,
            optimization_level,
        )
        .with_context(|| format!("Failed to compile {} shader", entry_point_name))?;

    let spirv_data = result
        .shader
        .as_spirv()
        .context("Invalid SPIR-V output")?
        .to_vec();
    let reflection = {
        let mut r = result.reflection;
        if r.push_constant_categories.is_empty() {
            r.push_constant_categories =
                crate::slang::virtual_main::extract_push_constant_categories(&slang_source);
        }
        Some(r)
    };

    // Get device
    let logical_device = devices
        .get(&device_handle)
        .context("Shader's device no longer valid")?;

    // Create Vulkan shader module
    // Convert Vec<u8> to &[u32] for SPIR-V
    let spirv_u32: &[u32] = bytemuck::cast_slice(&spirv_data);
    let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32);
    let module = unsafe {
        logical_device
            .device
            .create_shader_module(&create_info, None)
    }
    .context("Failed to create Vulkan shader module")?;

    tracing::debug!(
        "Compiled {} ({} SPIR-V words)",
        entry_point_name,
        spirv_u32.len()
    );

    // Dump SPIR-V for debugging when GOLDY_DUMP_SHADERS is set
    if let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") {
        use std::io::Write;
        let dir = std::path::Path::new(&dump_dir);
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(format!(
            "{}_h{}_vulkan.spv",
            entry_point_name, shader_handle
        ));
        if let Ok(mut file) = std::fs::File::create(&path) {
            let spirv_bytes: &[u8] = bytemuck::cast_slice(spirv_u32);
            let _ = file.write_all(spirv_bytes);
            tracing::info!("Dumped SPIR-V bytecode to {}", path.display());
        }
    }
    // Cache the module and reflection data
    let shader = shaders.get_mut(&shader_handle).unwrap();
    match stage {
        crate::slang::SlangStage::Vertex => shader.vertex_module = Some(module),
        crate::slang::SlangStage::Fragment => shader.fragment_module = Some(module),
        crate::slang::SlangStage::Compute => shader.compute_module = Some(module),
        _ => {} // Already validated above, shouldn't reach here
    }

    if !layout_checks_snapshot.is_empty() {
        shader.layout_checks.clear();
    }

    // Store reflection data (merge with existing if any)
    if let Some(ref new_reflection) = reflection {
        if let Some(ref mut existing) = shader.reflection {
            // Merge parameter blocks
            for pb in &new_reflection.parameter_blocks {
                if !existing.parameter_blocks.iter().any(|p| p.name == pb.name) {
                    existing.parameter_blocks.push(pb.clone());
                }
            }
            if existing.push_constant_categories.is_empty() {
                existing.push_constant_categories = new_reflection.push_constant_categories.clone();
            }
            if existing.binding_element_strides.is_empty() {
                existing.binding_element_strides = new_reflection.binding_element_strides.clone();
            }
        } else {
            shader.reflection = reflection;
        }
    }

    Ok(module)
}
