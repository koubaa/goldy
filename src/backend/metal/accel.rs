//! Metal primitive / instance acceleration structures and bindless encoding.

use super::types::{AccelState, MetalState, ResourceRegistry};
use super::{AccelerationStructureHandle, DeviceHandle};
use crate::backend::{AccelBuildCommand, GpuAccelCreate};
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{DeviceRef, MTLResourceOptions};
use objc::{msg_send, sel, sel_impl};

fn write_bindless(ld: &super::types::LogicalDevice, accel: &mtl::AccelerationStructure, local: u32) {
    let offset = (ResourceRegistry::accel_global_index(local) as u64) * ld.accel_encoder.encoded_length();
    if offset + ld.accel_encoder.encoded_length() > super::types::ARGUMENT_BUFFER_SIZE {
        return;
    }
    ld.accel_encoder.set_argument_buffer(&ld.argument_buffer, offset);
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
    let geom: mtl::AccelerationStructureGeometryDescriptor = From::from(geom);
    let prim = mtl::PrimitiveAccelerationStructureDescriptor::descriptor();
    prim.set_geometry_descriptors(mtl::Array::from_owned_slice(&[geom]));
    prim
}

pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    desc: &GpuAccelCreate,
) -> Result<AccelerationStructureHandle> {
    let ld = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?
        .clone();
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
    let accel = ld
        .device
        .new_acceleration_structure_with_size(sizes.acceleration_structure_size);
    let scratch = ld.device.new_buffer(
        sizes.build_scratch_buffer_size.max(16),
        MTLResourceOptions::StorageModePrivate,
    );

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
    let gpu_idle = super::gpu_is_idle(state);
    let device_handle = accel.device_handle;
    let ctx_h = super::context::context_handle_for_thread(state, device_handle);
    let base_barrier = super::context::reclamation_barrier(state, device_handle, gpu_idle);
    let base = ctx_h
        .filter(|_| !gpu_idle && base_barrier > 0)
        .map(|h| vec![(h, base_barrier)])
        .unwrap_or_default();
    let barrier = if let Some(device) = state.devices.get(&device_handle) {
        let slots = {
            let registry = device.descriptors.lock().unwrap();
            registry.resource_registry.accel_slot_keys(handle)
        };
        super::compute::evict_retained_graphs_using_slots(state, device_handle, &slots);
        let mut registry = device.descriptors.lock().unwrap();
        let requirements = registry.bindless_retirement_requirements_for_accel(handle, base);
        let barrier = requirements.iter().map(|(_, seq)| *seq).max().unwrap_or(0);
        let _ = registry.reclaim_accel_slots(handle);
        barrier
    } else {
        base_barrier
    };
    let deletion = super::types::PendingDeletion::Accel {
        accel: accel.accel,
        scratch: accel.scratch,
    };
    if let Some(h) = ctx_h {
        if let Some(sc_arc) = state.contexts.get(&h) {
            sc_arc.lock().unwrap().deletion_queue.queue(barrier, deletion);
            return;
        }
    }
    if let Some(device) = state.devices.get(&device_handle) {
        device.deletion_queue.lock().unwrap().queue(barrier, deletion);
    }
}

pub(super) fn bindless_index(state: &MetalState, handle: AccelerationStructureHandle) -> Option<u32> {
    state.accels.get(&handle).map(|a| a.arg_buffer_index)
}

pub(super) fn encode_build(
    state: &MetalState,
    command_buffer: &mtl::CommandBufferRef,
    build: &AccelBuildCommand,
    uploads: &mut Vec<mtl::Buffer>,
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
            anyhow::ensure!(
                *vertex_count <= dest_as.max_vertices,
                "build_blas vertex_count {vertex_count} exceeds create-time max_vertices {}",
                dest_as.max_vertices
            );
            anyhow::ensure!(
                *vertex_stride == dest_as.vertex_stride,
                "build_blas vertex_stride {vertex_stride} does not match create-time stride {}",
                dest_as.vertex_stride
            );
            let vb = state.buffers.get(vertex_buffer).context("invalid vertex buffer")?;
            let vertex_end = vertex_offset.saturating_add(*vertex_count as u64 * *vertex_stride as u64);
            anyhow::ensure!(vertex_end <= vb.size, "build_blas vertex range exceeds buffer size");
            let geom = mtl::AccelerationStructureTriangleGeometryDescriptor::descriptor();
            geom.set_vertex_buffer(Some(&vb.buffer));
            geom.set_vertex_buffer_offset(*vertex_offset);
            geom.set_vertex_stride(*vertex_stride as u64);
            geom.set_opaque(true);
            let tri_count = if *index_count > 0 {
                anyhow::ensure!(
                    *index_count % 3 == 0,
                    "build_blas index_count {index_count} is not a multiple of 3"
                );
                *index_count / 3
            } else {
                *vertex_count / 3
            };
            anyhow::ensure!(
                tri_count <= dest_as.max_primitives,
                "build_blas triangle count {tri_count} exceeds create-time max {}",
                dest_as.max_primitives
            );
            geom.set_triangle_count(tri_count as u64);
            if let Some(ib) = index_buffer {
                let idx = state.buffers.get(ib).context("invalid index buffer")?;
                let index_end = index_offset.saturating_add(*index_count as u64 * 4);
                anyhow::ensure!(index_end <= idx.size, "build_blas index range exceeds buffer size");
                geom.set_index_buffer(Some(&idx.buffer));
                geom.set_index_buffer_offset(*index_offset);
                geom.set_index_type(mtl::MTLIndexType::UInt32);
            }
            let geom: mtl::AccelerationStructureGeometryDescriptor = From::from(geom);
            let prim = mtl::PrimitiveAccelerationStructureDescriptor::descriptor();
            prim.set_geometry_descriptors(mtl::Array::from_owned_slice(&[geom]));
            encoder.build_acceleration_structure(&dest_as.accel, &prim, &dest_as.scratch, 0);
        }
        AccelBuildCommand::Tlas { dest, instances } => {
            let dest_as = state.accels.get(dest).context("invalid TLAS")?;
            anyhow::ensure!(dest_as.is_tlas, "build_tlas destination is a BLAS");
            let ld = state.devices.get(&dest_as.device_handle).context("invalid device")?;
            let mut packed = Vec::with_capacity(instances.len());
            let mut as_refs = Vec::with_capacity(instances.len());
            for inst in instances.iter() {
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
            let byte_len =
                (packed.len() * std::mem::size_of::<mtl::MTLAccelerationStructureUserIDInstanceDescriptor>()) as u64;
            let inst_buf = ld.device.new_buffer_with_data(
                packed.as_ptr() as *const std::ffi::c_void,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            );
            let tlas_desc = mtl::InstanceAccelerationStructureDescriptor::descriptor();
            tlas_desc.set_instance_count(instances.len() as u64);
            tlas_desc.set_instance_descriptor_type(mtl::MTLAccelerationStructureInstanceDescriptorType::UserID);
            tlas_desc.set_instance_descriptor_buffer(&inst_buf);
            tlas_desc.set_instance_descriptor_stride(std::mem::size_of::<
                mtl::MTLAccelerationStructureUserIDInstanceDescriptor,
            >() as u64);
            let arr = mtl::Array::from_owned_slice(&as_refs);
            tlas_desc.set_instanced_acceleration_structures(&arr);
            encoder.build_acceleration_structure(&dest_as.accel, &tlas_desc, &dest_as.scratch, 0);
            uploads.push(inst_buf);
        }
    }
    encoder.end_encoding();
    Ok(())
}

// Silence unused DeviceRef import if metal-rs sizes API takes DeviceRef.
#[allow(dead_code)]
fn _device_ref(_: &DeviceRef) {}
