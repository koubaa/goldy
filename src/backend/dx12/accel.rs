//! DXR 1.1 acceleration structures and `BuildRaytracingAccelerationStructure`.

use super::barriers;
use super::submit_session::Dx12SubmitScope;
use super::types::{AccelState, Dx12State};
use super::{AccelerationStructureHandle, DeviceHandle};
use crate::backend::{AccelBuildCommand, GpuAccelCreate};
use anyhow::{Context, Result};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

fn default_heap() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    }
}

fn upload_heap() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 0,
        VisibleNodeMask: 0,
    }
}

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
    }
}

fn create_committed(
    device: &ID3D12Device10,
    heap: D3D12_HEAP_PROPERTIES,
    desc: D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource> {
    let mut resource: Option<ID3D12Resource> = None;
    unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut resource) }
        .context("CreateCommittedResource (accel)")?;
    resource.context("CreateCommittedResource returned null")
}

fn write_as_srv(ld: &super::types::LogicalDevice, gpu_va: u64, offset: u32) {
    let mut cpu = unsafe { ld.cbv_srv_uav_heap.GetCPUDescriptorHandleForHeapStart() };
    cpu.ptr += (offset * ld.cbv_srv_uav_descriptor_size) as usize;
    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV { Location: gpu_va },
        },
    };
    unsafe {
        ld.device.CreateShaderResourceView(None, Some(&srv), cpu);
    }
}

fn dummy_triangle_geom(
    max_vertices: u32,
    max_triangles: u32,
    vertex_stride: u32,
    indexed: bool,
) -> D3D12_RAYTRACING_GEOMETRY_DESC {
    D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: if indexed {
                    DXGI_FORMAT_R32_UINT
                } else {
                    DXGI_FORMAT_UNKNOWN
                },
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: if indexed { max_triangles.saturating_mul(3) } else { 0 },
                VertexCount: max_vertices,
                IndexBuffer: 0,
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: 0,
                    StrideInBytes: vertex_stride as u64,
                },
            },
        },
    }
}

fn prebuild_sizes(
    device: &ID3D12Device10,
    ty: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE,
    geom: Option<&D3D12_RAYTRACING_GEOMETRY_DESC>,
    instance_count: u32,
) -> Result<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO> {
    let mut inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: ty,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
        NumDescs: if ty == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL {
            instance_count
        } else {
            1
        },
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: Default::default(),
    };
    if let Some(geom) = geom {
        inputs.Anonymous.pGeometryDescs = geom;
    }
    let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    unsafe {
        device.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut info);
    }
    anyhow::ensure!(info.ResultDataMaxSizeInBytes > 0, "DX12 AS prebuild returned zero size");
    Ok(info)
}

pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    desc: &GpuAccelCreate,
) -> Result<AccelerationStructureHandle> {
    let ld = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?
        .clone();
    anyhow::ensure!(
        state
            .adapters
            .iter()
            .find(|a| a.adapter_id == ld.adapter_id)
            .is_some_and(|a| a.ray_query || a.ray_tracing_pipelines),
        "DX12 adapter has no DXR acceleration structures"
    );

    let (is_tlas, sizes, max_primitives, max_vertices, vertex_stride) = match *desc {
        GpuAccelCreate::BlasTriangles {
            max_triangles,
            max_vertices,
            vertex_stride,
        } => {
            // Scratch/result sizes can differ for indexed vs non-indexed builds —
            // allocate the max so either path is safe.
            let geom_indexed = dummy_triangle_geom(max_vertices, max_triangles, vertex_stride, true);
            let geom_list = dummy_triangle_geom(max_vertices, max_triangles, vertex_stride, false);
            let sizes_i = prebuild_sizes(
                &ld.device,
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Some(&geom_indexed),
                0,
            )?;
            let sizes_n = prebuild_sizes(
                &ld.device,
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Some(&geom_list),
                0,
            )?;
            let sizes = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO {
                ResultDataMaxSizeInBytes: sizes_i.ResultDataMaxSizeInBytes.max(sizes_n.ResultDataMaxSizeInBytes),
                ScratchDataSizeInBytes: sizes_i.ScratchDataSizeInBytes.max(sizes_n.ScratchDataSizeInBytes),
                UpdateScratchDataSizeInBytes: sizes_i
                    .UpdateScratchDataSizeInBytes
                    .max(sizes_n.UpdateScratchDataSizeInBytes),
            };
            (false, sizes, max_triangles, max_vertices, vertex_stride)
        }
        GpuAccelCreate::Tlas { max_instances } => {
            let sizes = prebuild_sizes(
                &ld.device,
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                None,
                max_instances,
            )?;
            (true, sizes, max_instances, 0, 0)
        }
    };

    // Enhanced barriers: create in COMMON; first build uses COMMON→AS_WRITE.
    let as_res = create_committed(
        &ld.device,
        default_heap(),
        buffer_desc(
            sizes.ResultDataMaxSizeInBytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE,
        ),
        D3D12_RESOURCE_STATE_COMMON,
    )?;
    let scratch = create_committed(
        &ld.device,
        default_heap(),
        buffer_desc(
            sizes.ScratchDataSizeInBytes.max(256),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ),
        D3D12_RESOURCE_STATE_COMMON,
    )?;
    let gpu_va = unsafe { as_res.GetGPUVirtualAddress() };

    let handle = state.accels.write().unwrap().alloc_handle();
    let bindless_offset = {
        let offset = ld.descriptors.lock().unwrap().resource_registry.register_accel(handle);
        write_as_srv(&ld, gpu_va, offset);
        Some(offset)
    };

    state.accels.write().unwrap().entries.insert(
        handle,
        AccelState {
            device_handle,
            is_tlas,
            resource: as_res,
            gpu_va,
            bindless_offset,
            scratch,
            max_primitives,
            max_vertices,
            vertex_stride,
            built: std::sync::atomic::AtomicBool::new(false),
        },
    );
    Ok(handle)
}

pub(super) fn destroy(state: &Dx12State, handle: AccelerationStructureHandle) {
    let Some(accel) = state.accels.write().unwrap().entries.remove(&handle) else {
        return;
    };
    let Some(ld) = state.devices.get(&accel.device_handle) else {
        return;
    };
    let slots = ld.descriptors.lock().unwrap().accel_slot_keys(handle);
    super::compute::evict_retained_graphs_using_slots(state, accel.device_handle, &slots);
    let ctx_h = super::context::destroy_attribution_context(state, accel.device_handle);
    let base = super::context::reclamation_requirements(state, accel.device_handle, ctx_h);
    let requirements = {
        let registry = ld.descriptors.lock().unwrap();
        registry.bindless_retirement_requirements_for_accel(handle, base)
    };
    ld.deletion_queue.lock().unwrap().queue(
        requirements,
        super::types::PendingDeletion::Accel {
            accel_handle: handle,
            resource: accel.resource,
            scratch: accel.scratch,
        },
    );
}

pub(super) fn bindless_index(state: &Dx12State, handle: AccelerationStructureHandle) -> Option<u32> {
    state
        .accels
        .read()
        .unwrap()
        .entries
        .get(&handle)
        .and_then(|a| a.bindless_offset)
}

/// Record an AS build onto an already-open compute/direct list.
///
/// `geom_keep` holds CPU `pGeometryDescs` until the command list is `Close()`d.
pub(super) fn record_build_list(
    scope: &Dx12SubmitScope<'_>,
    cl: &ID3D12GraphicsCommandList,
    cl7: &ID3D12GraphicsCommandList7,
    build: &AccelBuildCommand,
    geom_keep: &mut Vec<Box<D3D12_RAYTRACING_GEOMETRY_DESC>>,
    pending_deletions: &mut Vec<super::types::PendingDeletion>,
) -> Result<()> {
    let cl4: ID3D12GraphicsCommandList4 = cl.cast().context("BuildRaytracingAccelerationStructure needs CL4")?;
    let buffers = scope.buffers().read().unwrap();
    let accels = scope.accels().read().unwrap();
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
            let dest_as = accels.entries.get(dest).context("invalid BLAS")?;
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
            let vb = buffers.entries.get(vertex_buffer).context("invalid vertex buffer")?;
            let vaddr = unsafe { vb.resource.GetGPUVirtualAddress() } + vertex_offset;
            let primitive_count = if *index_count > 0 {
                anyhow::ensure!(
                    *index_count % 3 == 0,
                    "build_blas index_count {index_count} is not a multiple of 3"
                );
                *index_count / 3
            } else {
                *vertex_count / 3
            };
            anyhow::ensure!(
                primitive_count <= dest_as.max_primitives,
                "build_blas primitive count {primitive_count} exceeds create-time max_triangles {}",
                dest_as.max_primitives
            );

            // Per-resource barriers (global COMMON→typed is illegal per debug layer 1331).
            // Rebuilds must use AS_READ→AS_WRITE: post-barrier leaves AS_READ, and
            // claiming COMMON→AS_WRITE on a later build TDRs (RQ15 stress iter 3).
            let rebuild = dest_as.built.load(std::sync::atomic::Ordering::Relaxed);
            let as_access_before = if rebuild {
                D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ
            } else {
                D3D12_BARRIER_ACCESS_COMMON
            };
            // Geometry stays in SRV after the first build's pre-barrier; rebuilds must
            // not claim COMMON→SRV (enhanced barriers + RQ15 iter-2 TDR).
            let geom_access_before = if rebuild {
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE
            } else {
                D3D12_BARRIER_ACCESS_COMMON
            };
            let mut pre = vec![
                barriers::buffer_barrier_full(
                    &vb.resource,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    geom_access_before,
                    D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                ),
                barriers::buffer_barrier_full(
                    &dest_as.scratch,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_ACCESS_COMMON,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                ),
                barriers::buffer_barrier_full(
                    &dest_as.resource,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    as_access_before,
                    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,
                ),
            ];
            if let Some(ib) = index_buffer {
                let idx = buffers.entries.get(ib).context("invalid index buffer")?;
                pre.push(barriers::buffer_barrier_full(
                    &idx.resource,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    geom_access_before,
                    D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                ));
            }
            unsafe {
                barriers::barrier_buffers(cl7, &pre);
                barriers::drop_buffer_barriers(&mut pre);
            }

            let mut geom = dummy_triangle_geom(*vertex_count, dest_as.max_primitives, *vertex_stride, false);
            geom.Anonymous.Triangles.VertexBuffer.StartAddress = vaddr;
            geom.Anonymous.Triangles.VertexBuffer.StrideInBytes = *vertex_stride as u64;
            geom.Anonymous.Triangles.VertexCount = *vertex_count;
            if let Some(ib) = index_buffer {
                let idx = buffers.entries.get(ib).context("invalid index buffer")?;
                geom.Anonymous.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
                geom.Anonymous.Triangles.IndexCount = *index_count;
                geom.Anonymous.Triangles.IndexBuffer = unsafe { idx.resource.GetGPUVirtualAddress() } + index_offset;
            }
            geom_keep.push(Box::new(geom));
            let geom = geom_keep.last().expect("just pushed");
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: unsafe { dest_as.resource.GetGPUVirtualAddress() },
                Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
                    NumDescs: 1,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: geom.as_ref(),
                    },
                },
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: unsafe { dest_as.scratch.GetGPUVirtualAddress() },
            };
            unsafe {
                cl4.BuildRaytracingAccelerationStructure(&desc, None);
            }
            let mut post = [
                barriers::buffer_barrier_full(
                    &dest_as.resource,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_SYNC(
                        D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE.0
                            | D3D12_BARRIER_SYNC_COMPUTE_SHADING.0
                            | D3D12_BARRIER_SYNC_RAYTRACING.0,
                    ),
                    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,
                    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ,
                ),
                barriers::buffer_barrier_full(
                    &dest_as.scratch,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_COMMON,
                ),
            ];
            unsafe {
                barriers::barrier_buffers(cl7, &post);
                barriers::drop_buffer_barriers(&mut post);
            }
            dest_as.built.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        AccelBuildCommand::Tlas { dest, instances } => {
            let dest_as = accels.entries.get(dest).context("invalid TLAS")?;
            anyhow::ensure!(dest_as.is_tlas, "build_tlas destination is a BLAS");
            anyhow::ensure!(
                instances.len() as u32 <= dest_as.max_primitives,
                "build_tlas instance count {} exceeds create-time max {}",
                instances.len(),
                dest_as.max_primitives
            );
            let mut packed = Vec::with_capacity(instances.len());
            for inst in instances.iter() {
                let blas = accels.entries.get(&inst.blas).context("invalid instance BLAS")?;
                let mut d = D3D12_RAYTRACING_INSTANCE_DESC::default();
                d.Transform = inst.transform;
                d._bitfield1 = (inst.custom_index & 0x00ff_ffff) | (u32::from(inst.mask) << 24);
                // High 8 bits are instance flags; disable facing cull so a +Z ray
                // from -Z still hits a CCW triangle in the XY plane.
                d._bitfield2 = (D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE.0 as u32) << 24;
                d.AccelerationStructure = blas.gpu_va;
                packed.push(d);
            }
            let byte_len = (packed.len() * std::mem::size_of::<D3D12_RAYTRACING_INSTANCE_DESC>()) as u64;
            let inst_res = create_committed(
                &scope.ld().device,
                upload_heap(),
                buffer_desc(byte_len, D3D12_RESOURCE_FLAG_NONE),
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            unsafe {
                let mut mapped = std::ptr::null_mut();
                inst_res.Map(0, None, Some(&mut mapped))?;
                std::ptr::copy_nonoverlapping(packed.as_ptr() as *const u8, mapped as *mut u8, byte_len as usize);
                inst_res.Unmap(0, None);
            }
            let inst_va = unsafe { inst_res.GetGPUVirtualAddress() };

            let as_access_before = if dest_as.built.load(std::sync::atomic::Ordering::Relaxed) {
                D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ
            } else {
                D3D12_BARRIER_ACCESS_COMMON
            };
            let mut pre = vec![
                barriers::buffer_barrier_full(
                    &inst_res,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_ACCESS_COMMON,
                    D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                ),
                barriers::buffer_barrier_full(
                    &dest_as.scratch,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_ACCESS_COMMON,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                ),
                barriers::buffer_barrier_full(
                    &dest_as.resource,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    as_access_before,
                    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,
                ),
            ];
            for inst in instances.iter() {
                let blas = accels.entries.get(&inst.blas).context("invalid instance BLAS")?;
                pre.push(barriers::buffer_barrier_full(
                    &blas.resource,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ,
                    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ,
                ));
            }
            unsafe {
                barriers::barrier_buffers(cl7, &pre);
                barriers::drop_buffer_barriers(&mut pre);
            }

            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: unsafe { dest_as.resource.GetGPUVirtualAddress() },
                Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                    Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
                    NumDescs: instances.len() as u32,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { InstanceDescs: inst_va },
                },
                SourceAccelerationStructureData: 0,
                ScratchAccelerationStructureData: unsafe { dest_as.scratch.GetGPUVirtualAddress() },
            };
            unsafe {
                cl4.BuildRaytracingAccelerationStructure(&desc, None);
            }
            let mut post = [
                barriers::buffer_barrier_full(
                    &dest_as.resource,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_SYNC(
                        D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE.0
                            | D3D12_BARRIER_SYNC_COMPUTE_SHADING.0
                            | D3D12_BARRIER_SYNC_RAYTRACING.0,
                    ),
                    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,
                    D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ,
                ),
                barriers::buffer_barrier_full(
                    &dest_as.scratch,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_SYNC_ALL,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_COMMON,
                ),
            ];
            unsafe {
                barriers::barrier_buffers(cl7, &post);
                barriers::drop_buffer_barriers(&mut post);
            }
            dest_as.built.store(true, std::sync::atomic::Ordering::Relaxed);
            pending_deletions.push(super::types::PendingDeletion::StandaloneResource(inst_res));
        }
    }
    Ok(())
}
