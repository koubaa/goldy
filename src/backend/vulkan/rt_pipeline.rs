//! Vulkan ray-tracing pipelines and shader-binding tables.

use super::types::{
    LogicalDevice, RayTracingPipelineState, SharedLogicalDevice, SharedRayTracingPipelineTable, VulkanState,
};
use super::{DeviceHandle, RayTracingPipelineHandle};
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;

fn align_up(value: u64, align: u64) -> u64 {
    let a = align.max(1);
    (value + a - 1) / a * a
}

fn create_sbt_buffer(
    instance: &ash::Instance,
    ld: &LogicalDevice,
    size: u64,
) -> Result<(vk::Buffer, vk::DeviceMemory, u64)> {
    let qf = ld.concurrent_queue_families();
    let usage = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::TRANSFER_DST;
    let buffer_info = super::utils::with_buffer_sharing(
        vk::BufferCreateInfo::default().size(size.max(1)).usage(usage),
        qf.as_ref(),
    );
    let buffer = unsafe { ld.device.create_buffer(&buffer_info, None) }.context("RT SBT create_buffer")?;
    let req = unsafe { ld.device.get_buffer_memory_requirements(buffer) };
    let memory_type = super::utils::find_memory_type(
        instance,
        ld.physical_device,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .context("RT SBT memory type")?;
    let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(memory_type)
        .push_next(&mut flags);
    let memory = unsafe { ld.device.allocate_memory(&alloc_info, None) }.context("RT SBT allocate_memory")?;
    unsafe { ld.device.bind_buffer_memory(buffer, memory, 0) }.context("RT SBT bind")?;
    let addr = {
        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        unsafe { ld.device.get_buffer_device_address(&info) }
    };
    Ok((buffer, memory, addr))
}

pub(super) fn create(
    instance: &ash::Instance,
    devices: &HashMap<DeviceHandle, SharedLogicalDevice>,
    rt_pipelines: &SharedRayTracingPipelineTable,
    device_handle: DeviceHandle,
    rgen: vk::ShaderModule,
    rmiss: vk::ShaderModule,
    rchit: vk::ShaderModule,
    shader_debug_name: String,
) -> Result<RayTracingPipelineHandle> {
    let logical_device = devices.get(&device_handle).context("Invalid device handle")?;
    anyhow::ensure!(
        logical_device.ray_tracing_pipelines && logical_device.rtp_khr.is_some(),
        "Vulkan device has no ray tracing pipelines"
    );
    let rtp = logical_device.rtp_khr.as_ref().unwrap();
    let layout = logical_device
        .bindless_pipeline_layout
        .context("Bindless pipeline layout required")?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .module(rgen)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(rmiss)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(rchit)
            .name(c"main"),
    ];
    let groups = [
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(1)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(2)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
    ];

    let create_info = vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(&stages)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(1)
        .layout(layout);

    let pipelines = unsafe { rtp.create_ray_tracing_pipelines(vk::DeferredOperationKHR::null(), logical_device.pipeline_cache, &[create_info], None) }
        .map_err(|e| anyhow::anyhow!("vkCreateRayTracingPipelinesKHR: {e:?}"))?;
    let pipeline = pipelines[0];

    let handle_size = logical_device.rt_shader_group_handle_size as usize;
    let handles = unsafe { rtp.get_ray_tracing_shader_group_handles(pipeline, 0, 3, handle_size * 3) }
        .context("vkGetRayTracingShaderGroupHandlesKHR")?;

    let rec = align_up(
        logical_device.rt_shader_group_handle_size as u64,
        logical_device.rt_shader_group_handle_alignment as u64,
    );
    let region = align_up(rec, logical_device.rt_shader_group_base_alignment as u64);
    let sbt_size = region * 3;
    let (sbt_buffer, sbt_memory, sbt_addr) = create_sbt_buffer(instance, logical_device, sbt_size)?;

    unsafe {
        let ptr = logical_device
            .device
            .map_memory(sbt_memory, 0, sbt_size, vk::MemoryMapFlags::empty())
            .context("map RT SBT")?;
        let bytes = std::slice::from_raw_parts_mut(ptr as *mut u8, sbt_size as usize);
        bytes.fill(0);
        bytes[..handle_size].copy_from_slice(&handles[..handle_size]);
        let miss_off = region as usize;
        bytes[miss_off..miss_off + handle_size].copy_from_slice(&handles[handle_size..handle_size * 2]);
        let hit_off = (region * 2) as usize;
        bytes[hit_off..hit_off + handle_size].copy_from_slice(&handles[handle_size * 2..]);
        logical_device.device.unmap_memory(sbt_memory);
    }

    let raygen = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_addr,
        stride: region,
        size: region,
    };
    let miss = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_addr + region,
        stride: rec,
        size: region,
    };
    let hit = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_addr + region * 2,
        stride: rec,
        size: region,
    };
    let callable = vk::StridedDeviceAddressRegionKHR::default();

    let handle = rt_pipelines.write().unwrap().alloc_handle();
    rt_pipelines.write().unwrap().entries.insert(
        handle,
        RayTracingPipelineState {
            device_handle,
            pipeline,
            layout,
            sbt_buffer,
            sbt_memory,
            raygen,
            miss,
            hit,
            callable,
            push_constant_categories: Vec::new(),
            binding_element_strides: Vec::new(),
            shader_debug_name,
        },
    );
    Ok(handle)
}

pub(super) fn destroy(
    devices: &HashMap<DeviceHandle, SharedLogicalDevice>,
    rt_pipelines: &SharedRayTracingPipelineTable,
    pipeline_handle: RayTracingPipelineHandle,
) {
    if let Some(ps) = rt_pipelines.write().unwrap().entries.remove(&pipeline_handle) {
        if let Some(ld) = devices.get(&ps.device_handle) {
            unsafe {
                let _ = ld.synchronized_device_wait_idle();
                ld.device.destroy_pipeline(ps.pipeline, None);
                ld.device.destroy_buffer(ps.sbt_buffer, None);
                ld.device.free_memory(ps.sbt_memory, None);
            }
        }
    }
}

pub(super) fn bind_pipeline(
    ld: &LogicalDevice,
    cmd: vk::CommandBuffer,
    ps: &RayTracingPipelineState,
) {
    unsafe {
        ld.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, ps.pipeline);
    }
}

pub(super) fn cmd_trace_rays(
    ld: &LogicalDevice,
    cmd: vk::CommandBuffer,
    ps: &RayTracingPipelineState,
    width: u32,
    height: u32,
    depth: u32,
) -> Result<()> {
    let rtp = ld.rtp_khr.as_ref().context("no ray tracing pipeline loader")?;
    unsafe {
        rtp.cmd_trace_rays(
            cmd,
            &ps.raygen,
            &ps.miss,
            &ps.hit,
            &ps.callable,
            width.max(1),
            height.max(1),
            depth.max(1),
        );
    }
    Ok(())
}

pub(super) fn bind_rt_descriptor_set(ld: &LogicalDevice, cmd: vk::CommandBuffer) {
    if ld.rtp_khr.is_none() {
        return;
    }
    if let (Some(bindless_set), Some(bindless_layout)) =
        (ld.bindless_descriptor_set, ld.bindless_pipeline_layout)
    {
        unsafe {
            ld.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                bindless_layout,
                0,
                std::slice::from_ref(&bindless_set),
                &[],
            );
        }
    }
}

/// Used only to keep `VulkanState` in the module graph for destroy-on-idle helpers.
#[allow(dead_code)]
pub(super) fn _mark_state(_: &VulkanState) {}
