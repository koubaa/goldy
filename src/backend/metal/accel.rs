//! Metal primitive / instance acceleration structures and bindless encoding.

use super::types::{AccelState, MetalState, ResourceRegistry};
use super::{AccelerationStructureHandle, DeviceHandle};
use crate::backend::{AccelBuildCommand, GpuAccelCreate};
use anyhow::{Context, Result};
use ::metal as mtl;
use mtl::{DeviceRef, MTLResourceOptions};
use objc::{msg_send, sel, sel_impl};

fn write_bindless(ld: &super::types::LogicalDevice, accel: &mtl::AccelerationStructure, local: u32) {
    let offset = (ResourceRegistry::accel_global_index(local) as u64) * ld.accel_encoder.encoded_length();
    if offset + ld.accel_encoder.encoded_length() > super::types::ARGUMENT_BUFFER_SIZE {
        return;
    }
    ld.accel_encoder
        .set_argument_buffer(&ld.argument_buffer, offset);
    unsafe {
        let _: () = msg_send![
            ld.accel_encoder.as_ref(),
            setAccelerationStructure: accel.as_ref()
            atIndex: 0u64
        ];
    }
}

fn dummy_triangle_descriptor(max_triangles: u32, vertex_stride: u32) -> mtl::PrimitiveAccelerationStructureDescriptor {
    let geom = mtl::AccelerationStructureTriangleGeometryDescriptor::descriptor();
    geom.set_triangle_count(max_triangles as u64);
    geom.set_vertex_stride(vertex_stride as u64);
    geom.set_opaque(true);
    let prim = mtl::PrimitiveAccelerationStructureDescriptor::descriptor();
    prim.set_geometry_descriptors(&mtl::Array::from_slice(&[geom]));
    prim
}

pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    desc: &GpuAccelCreate,
) -> Result<AccelerationStructureHandle> {
    let ld = state.devices.get(&device_handle).context("Invalid device handle")?.clone();
    anyhow::ensure!(ld.device.supports_raytracing(), "Metal device has no ray tracing");

    let (is_tlas, as_desc, max_primitives, max_vertices, vertex_stride): (
        bool,
        mtl::AccelerationStructureDescriptor,
        u32,
        u32,
        u32,
    ) = match *desc {
        GpuAccelCreate::BlasTriangles {
            max_triangles,
            max_vertices,
            vertex_stride,
        } => {
            let prim = dummy_triangle_descriptor(max_triangles, vertex_stride);
            (false, prim.into(), max_triangles, max_vertices, vertex_stride)
        }
        GpuAccelCreate::Tlas { max_instances } => {
            let inst = mtl::InstanceAccelerationStructureDescriptor::descriptor();
            inst.set_instance_count(max_instances as u64);
            (true, inst.into(), max_instances, 0, 0)
        }
    };

    let sizes = ld.device.acceleration_structure_sizes_with_descriptor(&as_desc);
    let accel = ld.device.new_acceleration_structure_with_size(sizes.acceleration_structure_size);
    let scratch = ld
        .device
        .new_buffer(sizes.build_scratch_buffer_size.max(16), MTLResourceOptions::StorageModePrivate);

    let handle = state.next_accel_handle;
    state.next_accel_handle += 1;
    let local = ld.descriptors.lock().unwrap().resource_registry.register_accel(handle);
    write_bindless(&ld, &accel, local);

    state.accels.insert(
        handle,
        AccelState {
            device_handle,
            is_tlas,
            accel,
            scratch,
            max_primitives,
            max_vertices,
            vertex_stride,
            arg_buffer_index: local,
        },
    );
    let _ = max_vertices;
    Ok(handle)
}

pub(super) fn destroy(state: &mut MetalState, handle: AccelerationStructureHandle) {
    let Some(accel) = state.accels.remove(&handle) else {
        return;
    };
    if let Some(ld) = state.devices.get(&accel.device_handle) {
        ld.descriptors.lock().unwrap().resource_registry.unregister_accel(handle);
    }
}

pub(super) fn bindless_index(state: &MetalState, handle: AccelerationStructureHandle) -> Option<u32> {
    state.accels.get(&handle).map(|a| a.arg_buffer_index)
}

pub(super) fn encode_build(
    state: &MetalState,
    command_buffer: &mtl::CommandBufferRef,
    build: &AccelBuildCommand,
) -> Result<()> {
    let encoder = command_buffer.new_acceleration_structure_command_encoder();
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
            let dest_as = state.accels.get(dest).context("invalid BLAS")?;
            anyhow::ensure!(!dest_as.is_tlas, "build_blas destination is a TLAS");
            let vb = state.buffers.get(vertex_buffer).context("invalid vertex buffer")?;
            let geom = mtl::AccelerationStructureTriangleGeometryDescriptor::descriptor();
            geom.set_vertex_buffer(Some(&vb.buffer));
            geom.set_vertex_buffer_offset(*vertex_offset);
            geom.set_vertex_stride(*vertex_stride as u64);
            geom.set_opaque(true);
            let tri_count = if *index_count > 0 {
                *index_count / 3
            } else {
                *vertex_count / 3
            };
            geom.set_triangle_count(tri_count as u64);
            if let Some(ib) = index_buffer {
                let idx = state.buffers.get(ib).context("invalid index buffer")?;
                geom.set_index_buffer(Some(&idx.buffer));
                geom.set_index_buffer_offset(*index_offset);
                geom.set_index_type(mtl::MTLIndexType::UInt32);
            }
            let prim = mtl::PrimitiveAccelerationStructureDescriptor::descriptor();
            prim.set_geometry_descriptors(&mtl::Array::from_slice(&[geom]));
            encoder.build_acceleration_structure(&dest_as.accel, &prim, &dest_as.scratch, 0);
        }
        AccelBuildCommand::Tlas { dest, instances } => {
            let dest_as = state.accels.get(dest).context("invalid TLAS")?;
            anyhow::ensure!(dest_as.is_tlas, "build_tlas destination is a BLAS");
            let ld = state.devices.get(&dest_as.device_handle).context("invalid device")?;
            let mut packed = Vec::with_capacity(instances.len());
            let mut as_refs = Vec::with_capacity(instances.len());
            for inst in instances {
                let blas = state.accels.get(&inst.blas).context("invalid instance BLAS")?;
                as_refs.push(blas.accel.clone());
                let mut d = mtl::MTLAccelerationStructureUserIDInstanceDescriptor::default();
                // Metal stores a 4×3 column-major packed matrix.
                d.transformation_matrix = [
                    [inst.transform[0], inst.transform[4], inst.transform[8]],
                    [inst.transform[1], inst.transform[5], inst.transform[9]],
                    [inst.transform[2], inst.transform[6], inst.transform[10]],
                    [inst.transform[3], inst.transform[7], inst.transform[11]],
                ];
                d.options = mtl::MTLAccelerationStructureInstanceOptions::Opaque;
                d.mask = inst.mask as u32;
                d.user_id = inst.custom_index;
                d.acceleration_structure_index = (as_refs.len() - 1) as u32;
                packed.push(d);
            }
            let byte_len = (packed.len() * std::mem::size_of::<mtl::MTLAccelerationStructureUserIDInstanceDescriptor>()) as u64;
            let inst_buf = ld.device.new_buffer_with_data(
                packed.as_ptr() as *const std::ffi::c_void,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            );
            let tlas_desc = mtl::InstanceAccelerationStructureDescriptor::descriptor();
            tlas_desc.set_instance_count(instances.len() as u64);
            tlas_desc.set_instance_descriptor_type(mtl::MTLAccelerationStructureInstanceDescriptorType::UserID);
            tlas_desc.set_instance_descriptor_buffer(&inst_buf);
            tlas_desc.set_instance_descriptor_stride(
                std::mem::size_of::<mtl::MTLAccelerationStructureUserIDInstanceDescriptor>() as u64,
            );
            let arr = mtl::Array::from_owned_slice(&as_refs);
            tlas_desc.set_instanced_acceleration_structures(&arr);
            encoder.build_acceleration_structure(&dest_as.accel, &tlas_desc, &dest_as.scratch, 0);
        }
    }
    encoder.end_encoding();
    Ok(())
}

// Silence unused DeviceRef import if metal-rs sizes API takes DeviceRef.
#[allow(dead_code)]
fn _device_ref(_: &DeviceRef) {}
