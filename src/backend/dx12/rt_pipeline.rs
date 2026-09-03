//! DXR state objects, shader-binding tables, and `DispatchRays`.

use super::shader;
use super::types::{Dx12State, RayTracingPipelineState};
use super::{DeviceHandle, RayTracingPipelineHandle, ShaderHandle};
use anyhow::{Context, Result};
use windows::core::{Interface, PCWSTR};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn align_up(v: u64, a: u64) -> u64 {
    let a = a.max(1);
    (v + a - 1) / a * a
}

pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    raygen: ShaderHandle,
    miss: ShaderHandle,
    closest_hit: ShaderHandle,
    debug_name: Option<&str>,
) -> Result<RayTracingPipelineHandle> {
    anyhow::ensure!(
        state
            .devices
            .get(&device_handle)
            .and_then(|ld| state.adapters.iter().find(|a| a.adapter_id == ld.adapter_id))
            .is_some_and(|a| a.ray_tracing_pipelines),
        "DX12 adapter has no DXR ray tracing pipelines"
    );

    let rgen_dxil = shader::ensure_stage_compiled(state, raygen, crate::slang::SlangStage::RayGeneration)?;
    let miss_dxil = shader::ensure_stage_compiled(state, miss, crate::slang::SlangStage::Miss)?;
    let chit_dxil = shader::ensure_stage_compiled(state, closest_hit, crate::slang::SlangStage::ClosestHit)?;

    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;
    let root_signature = logical_device
        .bindless_root_signature
        .as_ref()
        .context("Bindless root signature not available")?
        .clone();

    let rgen_name = utf16("rgen_main");
    let miss_name = utf16("rmiss_main");
    let chit_name = utf16("rchit_main");
    let hit_group_name = utf16("HitGroup");

    let export_rgen = D3D12_EXPORT_DESC {
        Name: PCWSTR(rgen_name.as_ptr()),
        ExportToRename: PCWSTR::null(),
        Flags: D3D12_EXPORT_FLAGS(0),
    };
    let export_miss = D3D12_EXPORT_DESC {
        Name: PCWSTR(miss_name.as_ptr()),
        ExportToRename: PCWSTR::null(),
        Flags: D3D12_EXPORT_FLAGS(0),
    };
    let export_chit = D3D12_EXPORT_DESC {
        Name: PCWSTR(chit_name.as_ptr()),
        ExportToRename: PCWSTR::null(),
        Flags: D3D12_EXPORT_FLAGS(0),
    };

    let lib_rgen = D3D12_DXIL_LIBRARY_DESC {
        DXILLibrary: D3D12_SHADER_BYTECODE {
            pShaderBytecode: rgen_dxil.as_ptr() as *const _,
            BytecodeLength: rgen_dxil.len(),
        },
        NumExports: 1,
        pExports: &export_rgen,
    };
    let lib_miss = D3D12_DXIL_LIBRARY_DESC {
        DXILLibrary: D3D12_SHADER_BYTECODE {
            pShaderBytecode: miss_dxil.as_ptr() as *const _,
            BytecodeLength: miss_dxil.len(),
        },
        NumExports: 1,
        pExports: &export_miss,
    };
    let lib_chit = D3D12_DXIL_LIBRARY_DESC {
        DXILLibrary: D3D12_SHADER_BYTECODE {
            pShaderBytecode: chit_dxil.as_ptr() as *const _,
            BytecodeLength: chit_dxil.len(),
        },
        NumExports: 1,
        pExports: &export_chit,
    };

    let hit_group = D3D12_HIT_GROUP_DESC {
        HitGroupExport: PCWSTR(hit_group_name.as_ptr()),
        Type: D3D12_HIT_GROUP_TYPE_TRIANGLES,
        AnyHitShaderImport: PCWSTR::null(),
        ClosestHitShaderImport: PCWSTR(chit_name.as_ptr()),
        IntersectionShaderImport: PCWSTR::null(),
    };

    let shader_config = D3D12_RAYTRACING_SHADER_CONFIG {
        MaxPayloadSizeInBytes: crate::rt_pipeline::MAX_RAY_PAYLOAD_BYTES,
        MaxAttributeSizeInBytes: 8,
    };
    let pipeline_config = D3D12_RAYTRACING_PIPELINE_CONFIG {
        MaxTraceRecursionDepth: 1,
    };
    let global_rs = D3D12_GLOBAL_ROOT_SIGNATURE {
        pGlobalRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
    };

    let subobjects = [
        D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY,
            pDesc: &lib_rgen as *const _ as *const _,
        },
        D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY,
            pDesc: &lib_miss as *const _ as *const _,
        },
        D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY,
            pDesc: &lib_chit as *const _ as *const _,
        },
        D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP,
            pDesc: &hit_group as *const _ as *const _,
        },
        D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG,
            pDesc: &shader_config as *const _ as *const _,
        },
        D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG,
            pDesc: &pipeline_config as *const _ as *const _,
        },
        D3D12_STATE_SUBOBJECT {
            Type: D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE,
            pDesc: &global_rs as *const _ as *const _,
        },
    ];

    let desc = D3D12_STATE_OBJECT_DESC {
        Type: D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
        NumSubobjects: subobjects.len() as u32,
        pSubobjects: subobjects.as_ptr(),
    };

    let state_object: ID3D12StateObject = unsafe { logical_device.device.CreateStateObject(&desc) }
        .context("CreateStateObject (DXR pipeline)")?;

    let props: ID3D12StateObjectProperties = state_object.cast().context("ID3D12StateObjectProperties")?;
    let id_size = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES as usize;
    let copy_id = |name: &[u16]| -> Result<Vec<u8>> {
        let ptr = unsafe { props.GetShaderIdentifier(PCWSTR(name.as_ptr())) };
        anyhow::ensure!(!ptr.is_null(), "GetShaderIdentifier returned null");
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const u8, id_size) }.to_vec())
    };
    let rgen_id = copy_id(&rgen_name)?;
    let miss_id = copy_id(&miss_name)?;
    let hit_id = copy_id(&hit_group_name)?;

    let rec = align_up(id_size as u64, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT as u64);
    let region = align_up(rec, D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT as u64);
    let sbt_size = region * 3;
    let sbt = create_upload_buffer(&logical_device.device, sbt_size)?;

    unsafe {
        let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
        let no_read = D3D12_RANGE { Begin: 0, End: 0 };
        sbt.Map(0, Some(&no_read), Some(&mut mapped))
            .context("map DXR SBT")?;
        let bytes = std::slice::from_raw_parts_mut(mapped as *mut u8, sbt_size as usize);
        bytes.fill(0);
        bytes[..id_size].copy_from_slice(&rgen_id);
        let miss_off = region as usize;
        bytes[miss_off..miss_off + id_size].copy_from_slice(&miss_id);
        let hit_off = (region * 2) as usize;
        bytes[hit_off..hit_off + id_size].copy_from_slice(&hit_id);
        sbt.Unmap(0, None);
    }

    let base = unsafe { sbt.GetGPUVirtualAddress() };
    let shader_debug_name = debug_name
        .map(str::to_owned)
        .unwrap_or_else(|| format!("rt_pipeline#{raygen}"));

    let handle = state.rt_pipelines.write().unwrap().alloc_handle();
    state.rt_pipelines.write().unwrap().entries.insert(
        handle,
        RayTracingPipelineState {
            device_handle,
            state_object,
            root_signature,
            sbt,
            raygen_va: base,
            raygen_size: region,
            miss_va: base + region,
            miss_size: region,
            miss_stride: rec,
            hit_va: base + region * 2,
            hit_size: region,
            hit_stride: rec,
            push_constant_categories: Vec::new(),
            push_constant_slot_kinds: Vec::new(),
            binding_element_strides: Vec::new(),
            shader_debug_name,
        },
    );
    Ok(handle)
}

fn create_upload_buffer(device: &ID3D12Device10, size: u64) -> Result<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut resource,
        )
    }
    .context("DXR SBT CreateCommittedResource")?;
    resource.context("DXR SBT resource is None")
}

pub(super) fn destroy(state: &Dx12State, handle: RayTracingPipelineHandle) {
    state.rt_pipelines.write().unwrap().entries.remove(&handle);
}

pub(super) fn dispatch_rays(
    cl4: &ID3D12GraphicsCommandList4,
    ps: &RayTracingPipelineState,
    width: u32,
    height: u32,
    depth: u32,
) {
    let desc = D3D12_DISPATCH_RAYS_DESC {
        RayGenerationShaderRecord: D3D12_GPU_VIRTUAL_ADDRESS_RANGE {
            StartAddress: ps.raygen_va,
            SizeInBytes: ps.raygen_size,
        },
        MissShaderTable: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
            StartAddress: ps.miss_va,
            SizeInBytes: ps.miss_size,
            StrideInBytes: ps.miss_stride,
        },
        HitGroupTable: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE {
            StartAddress: ps.hit_va,
            SizeInBytes: ps.hit_size,
            StrideInBytes: ps.hit_stride,
        },
        CallableShaderTable: D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE::default(),
        Width: width.max(1),
        Height: height.max(1),
        Depth: depth.max(1),
    };
    unsafe {
        cl4.SetPipelineState1(&ps.state_object);
        cl4.DispatchRays(&desc);
    }
}
