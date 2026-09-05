//! Shader management logic.

use super::types::{self, ShaderState, SharedShaderTable};
use super::{DeviceHandle, ShaderHandle};
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

/// Create a shader from Slang source code.
pub(super) fn create(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    shaders: &SharedShaderTable,
    desc: crate::backend::shared::ShaderDesc<'_>,
) -> Result<ShaderHandle> {
    // Just validate the device exists - actual compilation happens at pipeline creation
    let _ = devices.get(&desc.device).context("Invalid device handle")?;

    let handle = shaders.write().unwrap().alloc_handle();

    shaders.write().unwrap().entries.insert(
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
            extra_modules: HashMap::new(),
            reflection: None,
            layout_checks: desc.layout_checks,
            stage_slot_remaps: HashMap::new(),
            remapped_modules: HashMap::new(),
        },
    );

    tracing::debug!("Created shader handle {} (compilation deferred)", handle);
    Ok(handle)
}

/// Destroy a shader and clean up compiled shader modules.
pub(super) fn destroy(
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    shaders: &SharedShaderTable,
    shader_handle: ShaderHandle,
) {
    if let Some(shader) = shaders.write().unwrap().entries.remove(&shader_handle) {
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
                for module in shader.extra_modules.into_values() {
                    device.device.destroy_shader_module(module, None);
                }
                for module in shader.remapped_modules.into_values() {
                    device.device.destroy_shader_module(module, None);
                }
            }
        }
    }
}

/// Compile a shader for a specific stage on demand.
pub(super) fn ensure_stage_compiled(
    slang_compiler: &crate::slang::SlangCompiler,
    devices: &HashMap<DeviceHandle, types::SharedLogicalDevice>,
    shaders: &SharedShaderTable,
    shader_handle: ShaderHandle,
    stage: crate::slang::SlangStage,
) -> Result<vk::ShaderModule> {
    let remap_fp = {
        let shaders_read = shaders.read().unwrap();
        let shader = shaders_read
            .entries
            .get(&shader_handle)
            .context("Invalid shader handle")?;
        let remap = shader.stage_slot_remaps.get(&stage);
        let fp = remap
            .map(crate::slang::graphics_link::slot_remap_fingerprint)
            .unwrap_or(0);
        if fp != 0 {
            if let Some(module) = shader.remapped_modules.get(&(stage as u32, fp)).copied() {
                return Ok(module);
            }
        } else {
            let cached_module = match stage {
                crate::slang::SlangStage::Vertex => shader.vertex_module,
                crate::slang::SlangStage::Fragment => shader.fragment_module,
                crate::slang::SlangStage::Compute => shader.compute_module,
                crate::slang::SlangStage::RayGeneration
                | crate::slang::SlangStage::Intersection
                | crate::slang::SlangStage::AnyHit
                | crate::slang::SlangStage::ClosestHit
                | crate::slang::SlangStage::Miss
                | crate::slang::SlangStage::Callable
                | crate::slang::SlangStage::Mesh
                | crate::slang::SlangStage::Amplification => shader.extra_modules.get(&stage).copied(),
                other => anyhow::bail!("Unsupported shader stage: {:?}", other),
            };
            if let Some(module) = cached_module {
                return Ok(module);
            }
        }
        fp
    };

    let entry_point_name = crate::slang::canonical_entry_point(stage)
        .ok_or_else(|| anyhow::anyhow!("Unsupported shader stage: {:?}", stage))?;

    let (slang_source, search_paths, extra_defines, device_handle, optimization_level, layout_checks_snapshot) = {
        let shaders_read = shaders.read().unwrap();
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
            shader.defines.clone(),
            shader.device_handle,
            shader.optimization_level,
            shader.layout_checks.clone(),
        )
    };

    let search_path_refs: Vec<&str> = search_paths.iter().map(|s| s.as_str()).collect();
    let mut extra_define_refs: Vec<(&str, &str)> =
        extra_defines.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    if (devices
        .get(&device_handle)
        .is_some_and(|ld| ld.ray_query || ld.ray_tracing_pipelines))
        && extra_define_refs.iter().all(|(k, _)| *k != "GOLDY_RAY_QUERY")
    {
        extra_define_refs.push(("GOLDY_RAY_QUERY", "1"));
    }

    // Compile shader with reflection data for resource binding
    let result = slang_compiler
        .compile_bindless_with_reflection_and_defines(
            &slang_source,
            crate::slang::ShaderTarget::Spirv,
            &[(entry_point_name, stage)],
            &search_path_refs,
            &extra_define_refs,
            &layout_checks_snapshot,
            optimization_level,
        )
        .with_context(|| format!("Failed to compile {} shader", entry_point_name))?;

    let spirv_data = result.shader.as_spirv().context("Invalid SPIR-V output")?.to_vec();
    let reflection = {
        let mut r = result.reflection;
        if r.push_constant_categories.is_empty() {
            r.push_constant_categories = crate::slang::virtual_main::extract_push_constant_categories(&slang_source);
        }
        Some(r)
    };

    // Get device
    let logical_device = devices.get(&device_handle).context("Shader's device no longer valid")?;

    // Create Vulkan shader module
    // Convert Vec<u8> to &[u32] for SPIR-V
    let spirv_u32: &[u32] = bytemuck::cast_slice(&spirv_data);
    let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_u32);
    let module = unsafe { logical_device.device.create_shader_module(&create_info, None) }
        .context("Failed to create Vulkan shader module")?;

    tracing::debug!("Compiled {} ({} SPIR-V words)", entry_point_name, spirv_u32.len());

    // Dump SPIR-V for debugging when GOLDY_DUMP_SHADERS is set
    if let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") {
        use std::io::Write;
        let dir = std::path::Path::new(&dump_dir);
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(format!("{}_h{}_vulkan.spv", entry_point_name, shader_handle));
        if let Ok(mut file) = std::fs::File::create(&path) {
            let spirv_bytes: &[u8] = bytemuck::cast_slice(spirv_u32);
            let _ = file.write_all(spirv_bytes);
            tracing::info!("Dumped SPIR-V bytecode to {}", path.display());
        }
    }

    {
        let mut shaders_write = shaders.write().unwrap();
        let shader = shaders_write.entries.get_mut(&shader_handle).unwrap();
        if remap_fp != 0 {
            shader.remapped_modules.insert((stage as u32, remap_fp), module);
        } else {
            match stage {
                crate::slang::SlangStage::Vertex => shader.vertex_module = Some(module),
                crate::slang::SlangStage::Fragment => shader.fragment_module = Some(module),
                crate::slang::SlangStage::Compute => shader.compute_module = Some(module),
                crate::slang::SlangStage::RayGeneration
                | crate::slang::SlangStage::Intersection
                | crate::slang::SlangStage::AnyHit
                | crate::slang::SlangStage::ClosestHit
                | crate::slang::SlangStage::Miss
                | crate::slang::SlangStage::Callable
                | crate::slang::SlangStage::Mesh
                | crate::slang::SlangStage::Amplification => {
                    shader.extra_modules.insert(stage, module);
                }
                _ => {}
            }
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
                for iface in &new_reflection.stage_interfaces {
                    if !existing
                        .stage_interfaces
                        .iter()
                        .any(|s| s.entry_name == iface.entry_name && s.stage == iface.stage)
                    {
                        existing.stage_interfaces.push(iface.clone());
                    }
                }
            } else {
                shader.reflection = reflection;
            }
        }
    }

    Ok(module)
}
