//! Vulkan acceleration structures and `vkCmdBuildAccelerationStructuresKHR`.

use super::types::{self, AccelState, LogicalDevice, VulkanState};
use super::{AccelerationStructureHandle, DeviceHandle};
use crate::backend::{AccelBuildCommand, GpuAccelCreate};
use anyhow::{Context, Result};
use ash::vk;

fn buffer_device_address(device: &ash::Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    unsafe { device.get_buffer_device_address(&info) }
}

fn alloc_device_address_memory(
    instance: &ash::Instance,
    ld: &LogicalDevice,
    mem_requirements: vk::MemoryRequirements,
    property_flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory> {
    let memory_type = super::utils::find_memory_type(
        instance,
        ld.physical_device,
        mem_requirements.memory_type_bits,
        property_flags,
    )
    .context("accel: no memory type")?;
    let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type)
        .push_next(&mut flags);
    unsafe { ld.device.allocate_memory(&alloc_info, None) }.context("accel: allocate_memory")
}

fn create_gpu_buffer(
    instance: &ash::Instance,
    ld: &LogicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let qf = ld.concurrent_queue_families();
    let buffer_info = super::utils::with_buffer_sharing(
        vk::BufferCreateInfo::default().size(size.max(1)).usage(usage),
        qf.as_ref(),
    );
    let buffer = unsafe { ld.device.create_buffer(&buffer_info, None) }.context("accel: create_buffer")?;
    let req = unsafe { ld.device.get_buffer_memory_requirements(buffer) };
    let memory = alloc_device_address_memory(instance, ld, req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    unsafe { ld.device.bind_buffer_memory(buffer, memory, 0) }.context("accel: bind_buffer_memory")?;
    Ok((buffer, memory))
}

fn write_accel_descriptor(ld: &LogicalDevice, as_handle: vk::AccelerationStructureKHR, index: u32) {
    let Some(set) = ld.bindless_descriptor_set else {
        return;
    };
    let handles = [as_handle];
    let mut as_info = vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&handles);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(types::bindless_bindings::ACCEL)
        .dst_array_element(index)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .descriptor_count(1)
        .push_next(&mut as_info);
    unsafe {
        ld.device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
    }
}

pub(super) fn create(
    state: &mut VulkanState,
    device_handle: DeviceHandle,
    desc: &GpuAccelCreate,
) -> Result<AccelerationStructureHandle> {
    let ld = state.devices.get(&device_handle).context("Invalid device handle")?.clone();
    anyhow::ensure!(ld.accel_khr.is_some(), "Vulkan device has no acceleration structures");
    let accel_khr = ld.accel_khr.as_ref().unwrap();

    let (ty, geom, max_primitives) = match *desc {
        GpuAccelCreate::BlasTriangles {
            max_triangles,
            max_vertices,
            vertex_stride,
        } => {
            let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_stride(vertex_stride as u64)
                .max_vertex(max_vertices.saturating_sub(1))
                .index_type(vk::IndexType::UINT32);
            let geom = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                .flags(vk::GeometryFlagsKHR::OPAQUE);
            (vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, geom, max_triangles)
        }
        GpuAccelCreate::Tlas { max_instances } => {
            let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default().array_of_pointers(false);
            let geom = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { instances });
            (vk::AccelerationStructureTypeKHR::TOP_LEVEL, geom, max_instances)
        }
    };

    let geoms = [geom];
    let max_prim = [max_primitives];
    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geoms);

    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel_khr.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &max_prim,
            &mut size_info,
        );
    }

    let as_size = size_info.acceleration_structure_size.max(256);
    let scratch_size = size_info.build_scratch_size.max(256);
    let (buffer, memory) = create_gpu_buffer(
        &state.instance,
        &ld,
        as_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;

    let create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(buffer)
        .size(as_size)
        .ty(ty);
    let as_handle = unsafe { accel_khr.create_acceleration_structure(&create_info, None) }
        .context("vkCreateAccelerationStructureKHR")?;

    let handle = {
        let mut table = state.accels.write().unwrap();
        table.alloc_handle()
    };
    let bindless_index = {
        let index = ld.descriptors.lock().unwrap().resource_registry.register_accel(handle);
        write_accel_descriptor(&ld, as_handle, index);
        Some(index)
    };

    let device_address = {
        let info = vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(as_handle);
        unsafe { accel_khr.get_acceleration_structure_device_address(&info) }
    };

    state.accels.write().unwrap().entries.insert(
        handle,
        AccelState {
            device_handle,
            kind: ty,
            buffer,
            memory,
            as_handle,
            device_address,
            bindless_index,
            scratch_size,
            max_primitives,
        },
    );
    Ok(handle)
}

pub(super) fn destroy(state: &VulkanState, handle: AccelerationStructureHandle) {
    let Some(accel) = state.accels.write().unwrap().entries.remove(&handle) else {
        return;
    };
    let Some(ld) = state.devices.get(&accel.device_handle) else {
        return;
    };
    if let Some(_index) = accel.bindless_index {
        ld.descriptors.lock().unwrap().resource_registry.unregister_accel(handle);
    }
    if let Some(khr) = ld.accel_khr.as_ref() {
        unsafe {
            khr.destroy_acceleration_structure(accel.as_handle, None);
        }
    }
    unsafe {
        ld.device.destroy_buffer(accel.buffer, None);
        ld.device.free_memory(accel.memory, None);
    }
}

pub(super) fn bindless_index(state: &VulkanState, handle: AccelerationStructureHandle) -> Option<u32> {
    state
        .accels
        .read()
        .unwrap()
        .entries
        .get(&handle)
        .and_then(|a| a.bindless_index)
}

pub(super) fn record_build(
    view: &super::submit_session::VulkanSubmitView<'_>,
    cmd: vk::CommandBuffer,
    device_handle: DeviceHandle,
    build: &AccelBuildCommand,
) -> Result<()> {
    let ld = view.devices.get(&device_handle).context("Invalid device")?;
    let accel_khr = ld.accel_khr.as_ref().context("no acceleration_structure loader")?;
    let buffers = view.buffers.read().unwrap();
    let accels = view.accels.read().unwrap();

    match build {
        AccelBuildCommand::BlasTriangles {
            dest,
            vertex_buffer,
            vertex_offset,
            vertex_count,
            vertex_stride,
            index_buffer,
            index_offset,
            index_count,
        } => {
            let dest_as = accels.entries.get(dest).context("invalid BLAS handle")?;
            anyhow::ensure!(
                dest_as.kind == vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                "build_blas destination is not a BLAS"
            );
            let vb = buffers.entries.get(vertex_buffer).context("invalid vertex buffer")?;
            let vaddr = buffer_device_address(&ld.device, vb.buffer) + vertex_offset;
            let mut triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: vaddr,
                })
                .vertex_stride(*vertex_stride as u64)
                .max_vertex(vertex_count.saturating_sub(1));
            if let Some(ib) = index_buffer {
                let idx = buffers.entries.get(ib).context("invalid index buffer")?;
                let iaddr = buffer_device_address(&ld.device, idx.buffer) + index_offset;
                triangles = triangles
                    .index_type(vk::IndexType::UINT32)
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: iaddr,
                    });
            } else {
                triangles = triangles.index_type(vk::IndexType::NONE_KHR);
            }
            let geom = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                .flags(vk::GeometryFlagsKHR::OPAQUE);
            let primitive_count = if *index_count > 0 {
                *index_count / 3
            } else {
                *vertex_count / 3
            };
            record_build_inner(
                view.instance,
                ld,
                accel_khr,
                cmd,
                dest_as,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                geom,
                primitive_count,
            )?;
        }
        AccelBuildCommand::Tlas { dest, instances } => {
            let dest_as = accels.entries.get(dest).context("invalid TLAS handle")?;
            anyhow::ensure!(
                dest_as.kind == vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                "build_tlas destination is not a TLAS"
            );
            let mut vk_instances = Vec::with_capacity(instances.len());
            for inst in instances.iter() {
                let blas = accels.entries.get(&inst.blas).context("invalid instance BLAS")?;
                let mut transform = vk::TransformMatrixKHR { matrix: [0.0; 12] };
                transform.matrix.copy_from_slice(&inst.transform);
                let cull_disable = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8;
                vk_instances.push(vk::AccelerationStructureInstanceKHR {
                    transform,
                    instance_custom_index_and_mask: vk::Packed24_8::new(inst.custom_index, inst.mask),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, cull_disable),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: blas.device_address,
                    },
                });
            }
            let mut packed = Vec::with_capacity(vk_instances.len() * 64);
            for inst in &vk_instances {
                packed.extend_from_slice(&instance_bytes::instance_to_bytes(inst));
            }
            let (inst_buf, inst_mem) = create_host_upload(view.instance, ld, &packed)?;
            let inst_addr = buffer_device_address(&ld.device, inst_buf);
            let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: inst_addr,
                });
            let geom = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { instances: instances_data });
            record_build_inner(
                view.instance,
                ld,
                accel_khr,
                cmd,
                dest_as,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                geom,
                instances.len() as u32,
            )?;
            // Keep instance buffer alive until GPU idle by leaking into device deletion via immediate
            // destroy after wait is not available here — destroy at end of submit is unsafe.
            // Store on a thread-local isn't great. For MVP, attach to dest_as by not freeing
            // until AccelState drop... we leak inst_buf for now by stuffing into a mutex list.
            ld.accel_transient_buffers
                .lock()
                .unwrap()
                .push((inst_buf, inst_mem));
        }
    }
    Ok(())
}

fn create_host_upload(
    instance: &ash::Instance,
    ld: &LogicalDevice,
    bytes: &[u8],
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let size = bytes.len() as u64;
    let qf = ld.concurrent_queue_families();
    let buffer_info = super::utils::with_buffer_sharing(
        vk::BufferCreateInfo::default()
            .size(size.max(1))
            .usage(
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST,
            ),
        qf.as_ref(),
    );
    let buffer = unsafe { ld.device.create_buffer(&buffer_info, None) }?;
    let req = unsafe { ld.device.get_buffer_memory_requirements(buffer) };
    let memory = alloc_device_address_memory(
        instance,
        ld,
        req,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe { ld.device.bind_buffer_memory(buffer, memory, 0) }?;
    let ptr = unsafe { ld.map_memory2(memory, 0, size.max(1)) }?;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
        ld.unmap_memory2(memory)?;
    }
    Ok((buffer, memory))
}

fn record_build_inner(
    instance: &ash::Instance,
    ld: &LogicalDevice,
    accel_khr: &ash::khr::acceleration_structure::Device,
    cmd: vk::CommandBuffer,
    dest: &AccelState,
    ty: vk::AccelerationStructureTypeKHR,
    geom: vk::AccelerationStructureGeometryKHR<'_>,
    primitive_count: u32,
) -> Result<()> {
    anyhow::ensure!(
        dest.kind == ty,
        "acceleration structure build kind mismatch"
    );
    anyhow::ensure!(
        primitive_count <= dest.max_primitives,
        "AS build count {primitive_count} exceeds create-time max {}",
        dest.max_primitives
    );
    let (scratch, scratch_mem) = create_gpu_buffer(
        instance,
        ld,
        dest.scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    )?;
    let scratch_addr = buffer_device_address(&ld.device, scratch);
    // Host-visible instance buffers and vertex uploads must be visible to the
    // AS-build stage (not just TRANSFER / COMPUTE).
    let pre = vk::MemoryBarrier2::default()
        .src_stage_mask(
            vk::PipelineStageFlags2::HOST
                | vk::PipelineStageFlags2::TRANSFER
                | vk::PipelineStageFlags2::COMPUTE_SHADER
                | vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
        )
        .src_access_mask(
            vk::AccessFlags2::HOST_WRITE
                | vk::AccessFlags2::TRANSFER_WRITE
                | vk::AccessFlags2::SHADER_WRITE
                | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
        )
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .dst_access_mask(
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
                | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
                | vk::AccessFlags2::SHADER_READ,
        );
    let pre_dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&pre));
    unsafe {
        ld.device.cmd_pipeline_barrier2(cmd, &pre_dep);
    }
    let geoms = [geom];
    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .dst_acceleration_structure(dest.as_handle)
        .geometries(&geoms)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        });
    let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count);
    let ranges = [&[range][..]];
    let infos = [build_info];
    unsafe {
        accel_khr.cmd_build_acceleration_structures(cmd, &infos, &ranges);
    }
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_stage_mask(
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags2::COMPUTE_SHADER,
        )
        .dst_access_mask(
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ,
        );
    let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier));
    unsafe {
        ld.device.cmd_pipeline_barrier2(cmd, &dep);
    }
    ld.accel_transient_buffers.lock().unwrap().push((scratch, scratch_mem));
    Ok(())
}

/// `vk::AccelerationStructureInstanceKHR` is not `Pod`; convert via raw bytes.
mod instance_bytes {
    use super::*;
    pub fn instance_to_bytes(i: &vk::AccelerationStructureInstanceKHR) -> [u8; 64] {
        const _: () = assert!(std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() == 64);
        // SAFETY: Vulkan instance desc is exactly 64 bytes.
        unsafe { std::mem::transmute_copy(i) }
    }
}
