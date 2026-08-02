//! First-slice CUDA + DX12 raster: offscreen color targets, graphics PSOs, and draws.
//!
//! Scope (intentionally narrow):
//! - Windows + `cuda` + `graphics` + `dx12`
//! - Color-only [`TextureFormat::Rgba32Float`] render targets
//! - [`PrimitiveTopology::TriangleList`] only
//! - Vertex buffers + draw (no indexed draw, no bindless resources, no depth)
//!
//! Vertex buffers are CUDA allocations; for IA they are mirrored into DX12 UPLOAD
//! heaps at `render_to_target` time and cached by buffer content epoch (static
//! geometry uploads once). Render-target color is a shared D3D12 resource imported
//! into CUDA so [`GpuCommand::CopyRenderTarget`] can array-copy into present scratch
//! / other CUDA textures after a companion-fence wait. When the copy destination is
//! surface scratch, the CUDA backend skips that copy and presents the RT on DX12.

use super::dx12_companion::{cuda_wait_fence, Dx12Companion};
use super::dx12_interop::{create_shared_texture, import_shared_texture, CudaImportedTexture};
use super::texture::{memcpy_array_to_array, CudaTextureResource};
use super::{CudaBackend, CudaBuffer, CudaShader};
use crate::backend::{BufferHandle, DeviceHandle, PipelineHandle, RenderCommand, RenderTargetHandle, TextureHandle};
use crate::types::{
    PrimitiveTopology, TargetLoad, TextureFlags, TextureFormat, TextureKind, VertexBufferLayout, VertexFormat,
};
use anyhow::{bail, Context as _, Result};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use windows::Win32::Foundation::{CloseHandle, GENERIC_ALL, HANDLE, RECT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// First-slice color format for CUDA/DX12 offscreen targets.
pub(super) const RASTER_COLOR_FORMAT: TextureFormat = TextureFormat::Rgba32Float;

pub(super) struct CudaGraphicsPipeline {
    pub device: DeviceHandle,
    pub pipeline_state: ID3D12PipelineState,
    pub root_signature: ID3D12RootSignature,
    pub vertex_stride: u32,
    pub topology: PrimitiveTopology,
}

// SAFETY: COM objects used under Goldy's backend lock.
unsafe impl Send for CudaGraphicsPipeline {}
unsafe impl Sync for CudaGraphicsPipeline {}

pub(super) struct CudaRenderTarget {
    pub device: DeviceHandle,
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub format: TextureFormat,
    /// Field order: CUDA views before import before D3D12 resource.
    pub cuda_texture: Arc<CudaTextureResource>,
    #[allow(dead_code)]
    pub import: CudaImportedTexture,
    pub d3d12_resource: ID3D12Resource,
    pub rtv_offset: u32,
    /// Companion fence value signaled after the last `render_to_target`.
    pub last_dx12_fence: u64,
}

// SAFETY: see [`CudaGraphicsPipeline`].
unsafe impl Send for CudaRenderTarget {}
unsafe impl Sync for CudaRenderTarget {}

/// DX12 UPLOAD-heap mirror of a CUDA buffer for IA vertex binding.
pub(super) struct Dx12VertexMirror {
    pub resource: ID3D12Resource,
    pub size: u64,
    /// [`CudaBuffer::content_epoch`] last uploaded into this mirror.
    pub content_epoch: u64,
}

#[derive(Clone, Copy)]
pub(super) struct RasterListCache {
    pub fingerprint: u64,
}

// SAFETY: see [`CudaGraphicsPipeline`].
unsafe impl Send for Dx12VertexMirror {}
unsafe impl Sync for Dx12VertexMirror {}

fn raster_fingerprint(
    backend: &CudaBackend,
    target: RenderTargetHandle,
    color_load: TargetLoad,
    commands: &[RenderCommand],
) -> Result<u64> {
    let rt = backend
        .render_targets
        .get(&target)
        .context("CUDA/DX12: invalid render target")?;
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    target.hash(&mut hash);
    rt.width.hash(&mut hash);
    rt.height.hash(&mut hash);
    match color_load {
        TargetLoad::Clear(color) => {
            0u8.hash(&mut hash);
            color.r.to_bits().hash(&mut hash);
            color.g.to_bits().hash(&mut hash);
            color.b.to_bits().hash(&mut hash);
            color.a.to_bits().hash(&mut hash);
        }
        TargetLoad::Load => 1u8.hash(&mut hash),
        TargetLoad::Discard => 2u8.hash(&mut hash),
    }
    for command in commands {
        std::mem::discriminant(command).hash(&mut hash);
        match command {
            RenderCommand::SetPipeline(handle) => {
                backend
                    .pipelines
                    .get(handle)
                    .context("CUDA/DX12: invalid graphics pipeline")?;
                handle.hash(&mut hash);
            }
            RenderCommand::SetVertexBuffer { slot, buffer, offset } => {
                let cuda_buf = backend
                    .buffers
                    .get(buffer)
                    .context("CUDA/DX12: invalid vertex buffer")?;
                slot.hash(&mut hash);
                buffer.hash(&mut hash);
                offset.hash(&mut hash);
                cuda_buf.content_epoch.hash(&mut hash);
            }
            RenderCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => {
                vertex_count.hash(&mut hash);
                instance_count.hash(&mut hash);
                first_vertex.hash(&mut hash);
                first_instance.hash(&mut hash);
            }
            RenderCommand::ClearDepth(depth) => depth.to_bits().hash(&mut hash),
            RenderCommand::SetIndexBuffer { buffer, offset, .. } => {
                buffer.hash(&mut hash);
                offset.hash(&mut hash);
            }
            RenderCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => {
                index_count.hash(&mut hash);
                instance_count.hash(&mut hash);
                first_index.hash(&mut hash);
                base_vertex.hash(&mut hash);
                first_instance.hash(&mut hash);
            }
            RenderCommand::BindResources { .. }
            | RenderCommand::BindResourcesRaw { .. }
            | RenderCommand::BindResourcesTyped { .. } => {}
        }
    }
    Ok(hash.finish())
}

fn companion<'a>(backend: &'a CudaBackend, device: DeviceHandle) -> Result<&'a Dx12Companion> {
    backend
        .devices
        .get(&device)
        .context("CUDA: invalid device")?
        .dx12
        .as_deref()
        .context("CUDA/DX12: companion required for raster")
}

fn format_to_dxgi(format: TextureFormat) -> Result<DXGI_FORMAT> {
    Ok(match format {
        TextureFormat::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        TextureFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        other => bail!("CUDA/DX12 raster: unsupported texture format {other:?}"),
    })
}

fn vertex_format_to_dxgi(format: VertexFormat) -> DXGI_FORMAT {
    match format {
        VertexFormat::Float32 => DXGI_FORMAT_R32_FLOAT,
        VertexFormat::Float32x2 => DXGI_FORMAT_R32G32_FLOAT,
        VertexFormat::Float32x3 => DXGI_FORMAT_R32G32B32_FLOAT,
        VertexFormat::Float32x4 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        VertexFormat::Uint32 => DXGI_FORMAT_R32_UINT,
        VertexFormat::Sint32 => DXGI_FORMAT_R32_SINT,
        VertexFormat::Uint8x4 => DXGI_FORMAT_R8G8B8A8_UINT,
        VertexFormat::Unorm8x4 => DXGI_FORMAT_R8G8B8A8_UNORM,
    }
}

fn topology_to_d3d12(topology: PrimitiveTopology) -> windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY {
    match topology {
        PrimitiveTopology::PointList => windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        PrimitiveTopology::LineList => windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        PrimitiveTopology::LineStrip => windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        PrimitiveTopology::TriangleList => windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        PrimitiveTopology::TriangleStrip => windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    }
}

fn topology_type_to_d3d12(topology: PrimitiveTopology) -> D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    match topology {
        PrimitiveTopology::PointList => D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
        PrimitiveTopology::LineList | PrimitiveTopology::LineStrip => D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
        PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
    }
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn compile_stage_dxil(shader: &CudaShader, stage: crate::slang::SlangStage) -> Result<Vec<u8>> {
    let entry = match stage {
        crate::slang::SlangStage::Vertex => "vs_main",
        crate::slang::SlangStage::Fragment => "fs_main",
        other => bail!("CUDA/DX12 raster: unsupported stage {other:?}"),
    };
    let compiler = crate::slang::SlangCompiler::new().context("CUDA/DX12: initialize Slang")?;
    let paths: Vec<&str> = shader.search_paths.iter().map(String::as_str).collect();
    let mut defines: Vec<(&str, &str)> = vec![("__DX12__", "1")];
    for (k, v) in &shader.defines {
        defines.push((k.as_str(), v.as_str()));
    }
    let compiled = compiler
        .compile_with_reflection(
            &shader.source,
            crate::slang::ShaderTarget::Dxil,
            &[(entry, stage)],
            &paths,
            &defines,
            &[],
            shader.optimization_level,
        )
        .with_context(|| format!("CUDA/DX12: compile {entry} to DXIL"))?;
    compiled
        .shader
        .as_dxil()
        .context("CUDA/DX12: expected DXIL bytecode")
        .map(|b| b.to_vec())
}

pub(super) fn create_pipeline(
    backend: &mut CudaBackend,
    device: DeviceHandle,
    vertex_shader: crate::backend::ShaderHandle,
    fragment_shader: crate::backend::ShaderHandle,
    vertex_layout: &VertexBufferLayout,
    topology: PrimitiveTopology,
    target_format: TextureFormat,
) -> Result<PipelineHandle> {
    if topology != PrimitiveTopology::TriangleList {
        bail!("CUDA/DX12 raster: only TriangleList is supported in the first slice (got {topology:?})");
    }
    if target_format != RASTER_COLOR_FORMAT {
        bail!("CUDA/DX12 raster: only {RASTER_COLOR_FORMAT:?} targets are supported (got {target_format:?})");
    }
    let companion = companion(backend, device)?;
    let device_com = companion.device.clone();
    let root_signature = companion.graphics_root_signature.clone();
    let vs = backend
        .shaders
        .get(&vertex_shader)
        .context("CUDA/DX12: invalid vertex shader")?;
    if vs.device != device {
        bail!("CUDA/DX12: vertex shader belongs to a different device");
    }
    let fs = backend
        .shaders
        .get(&fragment_shader)
        .context("CUDA/DX12: invalid fragment shader")?;
    if fs.device != device {
        bail!("CUDA/DX12: fragment shader belongs to a different device");
    }

    let vs_dxil = compile_stage_dxil(vs, crate::slang::SlangStage::Vertex)?;
    let fs_dxil = compile_stage_dxil(fs, crate::slang::SlangStage::Fragment)?;

    let mut texcoord_index = 0u32;
    let input_elements: Vec<D3D12_INPUT_ELEMENT_DESC> = vertex_layout
        .attributes
        .iter()
        .map(|attr| {
            let (semantic_name, semantic_index) = if attr.location == 0 {
                (c"POSITION".as_ptr() as *const u8, 0)
            } else {
                let is_color = matches!(
                    attr.format,
                    VertexFormat::Float32x3 | VertexFormat::Float32x4 | VertexFormat::Unorm8x4 | VertexFormat::Uint8x4
                ) && attr.location == 1;
                if is_color {
                    (c"COLOR".as_ptr() as *const u8, 0)
                } else {
                    let idx = texcoord_index;
                    texcoord_index += 1;
                    (c"TEXCOORD".as_ptr() as *const u8, idx)
                }
            };
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::PCSTR(semantic_name),
                SemanticIndex: semantic_index,
                Format: vertex_format_to_dxgi(attr.format),
                InputSlot: 0,
                AlignedByteOffset: attr.offset,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            }
        })
        .collect();

    let rtv_dxgi = format_to_dxgi(target_format)?;
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs_dxil.as_ptr() as *const _,
            BytecodeLength: vs_dxil.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: fs_dxil.as_ptr() as *const _,
            BytecodeLength: fs_dxil.len(),
        },
        BlendState: D3D12_BLEND_DESC {
            AlphaToCoverageEnable: false.into(),
            IndependentBlendEnable: false.into(),
            RenderTarget: [
                D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: false.into(),
                    LogicOpEnable: false.into(),
                    SrcBlend: D3D12_BLEND_ONE,
                    DestBlend: D3D12_BLEND_ZERO,
                    BlendOp: D3D12_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D12_BLEND_ONE,
                    DestBlendAlpha: D3D12_BLEND_ZERO,
                    BlendOpAlpha: D3D12_BLEND_OP_ADD,
                    LogicOp: D3D12_LOGIC_OP_NOOP,
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                },
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
            ],
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthBias: 0,
            DepthBiasClamp: 0.0,
            SlopeScaledDepthBias: 0.0,
            DepthClipEnable: true.into(),
            MultisampleEnable: false.into(),
            AntialiasedLineEnable: false.into(),
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: input_elements.as_ptr(),
            NumElements: input_elements.len() as u32,
        },
        PrimitiveTopologyType: topology_type_to_d3d12(topology),
        NumRenderTargets: 1,
        RTVFormats: [
            rtv_dxgi,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
        ],
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };

    let pipeline_state: ID3D12PipelineState = unsafe { device_com.CreateGraphicsPipelineState(&pso_desc) }
        .context("CUDA/DX12: CreateGraphicsPipelineState failed")?;

    let handle = backend.next_pipeline;
    backend.next_pipeline += 1;
    backend.pipelines.insert(
        handle,
        CudaGraphicsPipeline {
            device,
            pipeline_state,
            root_signature,
            vertex_stride: vertex_layout.stride,
            topology,
        },
    );
    tracing::debug!("CUDA/DX12: created graphics pipeline {handle}");
    Ok(handle)
}

pub(super) fn destroy_pipeline(backend: &mut CudaBackend, pipeline: PipelineHandle) {
    backend.pipelines.remove(&pipeline);
}

pub(super) fn create_render_target(
    backend: &mut CudaBackend,
    device: DeviceHandle,
    width: u32,
    height: u32,
    color_format: TextureFormat,
    depth_format: Option<crate::types::DepthFormat>,
) -> Result<RenderTargetHandle> {
    if depth_format.is_some() {
        bail!("CUDA/DX12 raster: depth buffers are not supported in the first slice");
    }
    if color_format != RASTER_COLOR_FORMAT {
        bail!("CUDA/DX12 raster: only {RASTER_COLOR_FORMAT:?} render targets are supported (got {color_format:?})");
    }
    if width == 0 || height == 0 {
        bail!("CUDA/DX12 raster: render target dimensions must be non-zero");
    }

    let companion = Arc::clone(
        backend
            .devices
            .get(&device)
            .context("CUDA: invalid device")?
            .dx12
            .as_ref()
            .context("CUDA/DX12: companion required for raster")?,
    );
    let cuda_ctx = Arc::clone(&backend.device(device)?.ctx);
    let dxgi = format_to_dxgi(color_format)?;

    let (d3d12_resource, allocation_size) = create_shared_texture(
        &companion.device,
        width,
        height,
        dxgi,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    )?;

    let handle_nt: HANDLE = unsafe {
        companion
            .device
            .CreateSharedHandle(&d3d12_resource, None, GENERIC_ALL.0, None)
    }
    .context("CUDA/DX12: CreateSharedHandle(render target) failed")?;

    let import = import_shared_texture(&cuda_ctx, handle_nt, allocation_size, width, height, color_format)?;
    unsafe {
        let _ = CloseHandle(handle_nt);
    }

    let storage_slot = backend.alloc_registry_slot();
    let cuda_texture = CudaTextureResource::from_imported_array(
        &cuda_ctx,
        import.level0(),
        width,
        height,
        color_format,
        TextureKind::Direct,
        TextureFlags::empty(),
        Some(storage_slot),
        None,
    )?;

    let rtv_offset = companion.alloc_rtv_offset();
    let rtv = unsafe {
        let mut h = companion.rtv_heap.GetCPUDescriptorHandleForHeapStart();
        h.ptr += (rtv_offset as usize) * companion.rtv_descriptor_size as usize;
        h
    };
    unsafe {
        companion.device.CreateRenderTargetView(&d3d12_resource, None, rtv);
    }

    let tex_handle = backend.next_texture;
    backend.next_texture += 1;
    backend.texture_slots.insert(storage_slot, tex_handle);
    backend.textures.insert(tex_handle, Arc::clone(&cuda_texture));

    let handle = backend.next_render_target;
    backend.next_render_target += 1;
    backend.render_targets.insert(
        handle,
        CudaRenderTarget {
            device,
            width,
            height,
            format: color_format,
            cuda_texture,
            import,
            d3d12_resource,
            rtv_offset,
            last_dx12_fence: 0,
        },
    );
    tracing::debug!("CUDA/DX12: created render target {handle} ({width}x{height})");
    Ok(handle)
}

fn ensure_vertex_mirror<'a>(
    companion: &Dx12Companion,
    mirrors: &'a mut std::collections::HashMap<BufferHandle, Dx12VertexMirror>,
    buffer: BufferHandle,
    size: u64,
) -> Result<&'a mut Dx12VertexMirror> {
    let needs_recreate = mirrors.get(&buffer).map(|m| m.size < size).unwrap_or(true);
    if needs_recreate {
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size.max(4),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            companion.device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut resource,
            )
        }
        .context("CUDA/DX12: CreateCommittedResource(UPLOAD VB) failed")?;
        let resource = resource.context("CUDA/DX12: null UPLOAD VB")?;
        mirrors.insert(
            buffer,
            Dx12VertexMirror {
                resource,
                size: size.max(4),
                content_epoch: u64::MAX, // force upload on first use
            },
        );
    }
    Ok(mirrors.get_mut(&buffer).unwrap())
}

fn upload_vertex_mirror(companion: &Dx12Companion, mirror: &Dx12VertexMirror, host: &[u8]) -> Result<()> {
    let _ = companion;
    let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe { mirror.resource.Map(0, None, Some(&mut mapped)) }.context("CUDA/DX12: Map UPLOAD VB failed")?;
    if mapped.is_null() {
        bail!("CUDA/DX12: Map UPLOAD VB returned null");
    }
    unsafe {
        std::ptr::copy_nonoverlapping(host.as_ptr(), mapped as *mut u8, host.len());
        mirror.resource.Unmap(0, None);
    }
    Ok(())
}

fn read_cuda_buffer_host(stream: &Arc<cudarc::driver::CudaStream>, buffer: &CudaBuffer) -> Result<Vec<u8>> {
    let memory = buffer.memory.lock().unwrap();
    let start = buffer.offset as usize;
    let end = (buffer.offset + buffer.size) as usize;
    let view = memory
        .try_slice(start..end)
        .context("CUDA/DX12: vertex buffer view out of range")?;
    let mut host = vec![0u8; buffer.size as usize];
    stream
        .memcpy_dtoh(&view, &mut host[..])
        .context("CUDA/DX12: DtoH vertex buffer failed")?;
    stream.synchronize().context("CUDA/DX12: synchronize after VB DtoH")?;
    Ok(host)
}

pub(super) fn render_to_target(
    backend: &mut CudaBackend,
    device: DeviceHandle,
    target: RenderTargetHandle,
    color_load: TargetLoad,
    commands: &[RenderCommand],
) -> Result<()> {
    // Validate handles and collect what we need under short borrows.
    {
        let rt = backend
            .render_targets
            .get(&target)
            .context("CUDA/DX12: invalid render target")?;
        if rt.device != device {
            bail!("CUDA/DX12: render target belongs to a different device");
        }
    }

    // Do not blanket-sync all CUDA streams here. Vertex mirrors are refreshed from
    // content_epoch (host writes bump the epoch). Shared RT consumers wait on
    // `last_dx12_fence` via the external-semaphore path.

    let companion = {
        Arc::clone(
            backend
                .devices
                .get(&device)
                .context("CUDA: invalid device")?
                .dx12
                .as_ref()
                .context("CUDA/DX12: companion required for raster")?,
        )
    };
    let fingerprint = raster_fingerprint(backend, target, color_load, commands)?;

    // Same DIRECT queue already orders this draw after prior raster/present submits;
    // no queue Wait on `last_dx12_fence` is required.

    // Prefer retained raster replay. Closed lists may be re-executed without a
    // CPU wait even while a prior execute is in flight; only Reset needs retirement.
    if backend
        .raster_list_cache
        .get(&target)
        .is_some_and(|cache| cache.fingerprint == fingerprint)
    {
        if let Some(signal) = companion.try_reuse_raster_for_fingerprint(fingerprint)? {
            backend.render_targets.get_mut(&target).unwrap().last_dx12_fence = signal;
            return Ok(());
        }
    }

    let (slot_idx, list, _slot_generation) = companion.begin_raster_list()?;

    let (width, height, rtv_offset, d3d12_resource) = {
        let rt = backend.render_targets.get(&target).unwrap();
        (rt.width, rt.height, rt.rtv_offset, rt.d3d12_resource.clone())
    };

    let to_rt = transition(
        &d3d12_resource,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );
    unsafe { list.ResourceBarrier(&[to_rt]) };

    let rtv = unsafe {
        let mut h = companion.rtv_heap.GetCPUDescriptorHandleForHeapStart();
        h.ptr += (rtv_offset as usize) * companion.rtv_descriptor_size as usize;
        h
    };
    unsafe {
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
    }

    match color_load {
        TargetLoad::Clear(color) => {
            let clear = [color.r, color.g, color.b, color.a];
            unsafe { list.ClearRenderTargetView(rtv, &clear, None) };
        }
        TargetLoad::Load | TargetLoad::Discard => {}
    }

    let viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: width as f32,
        Height: height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let scissor = RECT {
        left: 0,
        top: 0,
        right: width as i32,
        bottom: height as i32,
    };
    unsafe {
        list.RSSetViewports(&[viewport]);
        list.RSSetScissorRects(&[scissor]);
    }

    let mut current_stride = 24u32;
    let mut current_topology = PrimitiveTopology::TriangleList;
    let alloc_stream = Arc::clone(&backend.device(device)?.alloc_stream);

    for command in commands {
        match command {
            RenderCommand::ClearDepth(_) => {
                bail!("CUDA/DX12 raster: ClearDepth is not supported in the first slice");
            }
            RenderCommand::SetPipeline(pipeline_handle) => {
                let pipeline = backend
                    .pipelines
                    .get(pipeline_handle)
                    .context("CUDA/DX12: invalid graphics pipeline")?;
                current_stride = pipeline.vertex_stride;
                current_topology = pipeline.topology;
                unsafe {
                    list.SetGraphicsRootSignature(&pipeline.root_signature);
                    list.SetPipelineState(&pipeline.pipeline_state);
                    list.IASetPrimitiveTopology(topology_to_d3d12(pipeline.topology));
                }
            }
            RenderCommand::SetVertexBuffer { slot, buffer, offset } => {
                let cuda_buf = backend
                    .buffers
                    .get(buffer)
                    .context("CUDA/DX12: invalid vertex buffer")?
                    .clone_meta();
                if *offset >= cuda_buf.size {
                    bail!("CUDA/DX12: vertex buffer offset out of range");
                }
                let nbytes = cuda_buf.size - offset;
                let epoch = cuda_buf.content_epoch;
                let mirror = ensure_vertex_mirror(&companion, &mut backend.vb_mirrors, *buffer, nbytes)?;
                if mirror.content_epoch != epoch {
                    // Content changed (or first use): DtoH once into the UPLOAD mirror.
                    let host = read_cuda_buffer_host(&alloc_stream, &cuda_buf)?;
                    backend
                        .graph_stats
                        .dtoh_calls
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let host_slice = &host[*offset as usize..];
                    upload_vertex_mirror(&companion, mirror, host_slice)?;
                    backend
                        .graph_stats
                        .vb_mirror_uploads
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    mirror.content_epoch = epoch;
                }
                let view = D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: unsafe { mirror.resource.GetGPUVirtualAddress() },
                    SizeInBytes: nbytes as u32,
                    StrideInBytes: current_stride,
                };
                unsafe { list.IASetVertexBuffers(*slot, Some(&[view])) };
            }
            RenderCommand::SetIndexBuffer { .. } => {
                bail!("CUDA/DX12 raster: indexed draws are not supported in the first slice");
            }
            RenderCommand::BindResources { .. }
            | RenderCommand::BindResourcesRaw { .. }
            | RenderCommand::BindResourcesTyped { .. } => {
                // First slice has no bindless graphics root parameters; parcel
                // `with_parcel` still emits these for graph tracking — ignore.
            }
            RenderCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => {
                let _ = current_topology;
                unsafe {
                    list.DrawInstanced(*vertex_count, *instance_count, *first_vertex, *first_instance);
                }
            }
            RenderCommand::DrawIndexed { .. } => {
                bail!("CUDA/DX12 raster: DrawIndexed is not supported in the first slice");
            }
        }
    }

    let to_common = transition(
        &d3d12_resource,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COMMON,
    );
    unsafe { list.ResourceBarrier(&[to_common]) };

    let signal = companion.finish_raster_list(slot_idx, fingerprint)?;
    backend
        .graph_stats
        .raster_list_records
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Some(rt) = backend.render_targets.get_mut(&target) {
        rt.last_dx12_fence = signal;
    }
    backend
        .raster_list_cache
        .insert(target, RasterListCache { fingerprint });
    Ok(())
}

/// Copy render-target color into a CUDA texture after waiting on the DX12 fence.
#[allow(dead_code)] // available for tests / future sync helpers
pub(super) fn copy_render_target(
    backend: &CudaBackend,
    stream: &cudarc::driver::CudaStream,
    src: RenderTargetHandle,
    dst: TextureHandle,
) -> Result<()> {
    let (cuda_tex, fence, device) = {
        let rt = backend
            .render_targets
            .get(&src)
            .context("CUDA/DX12: invalid CopyRenderTarget source")?;
        (Arc::clone(&rt.cuda_texture), rt.last_dx12_fence, rt.device)
    };
    let dst_tex = backend
        .textures
        .get(&dst)
        .context("CUDA/DX12: invalid CopyRenderTarget destination")?;
    if cuda_tex.format != dst_tex.format {
        bail!(
            "CUDA/DX12: CopyRenderTarget format mismatch ({:?} → {:?})",
            cuda_tex.format,
            dst_tex.format
        );
    }
    if cuda_tex.width != dst_tex.width || cuda_tex.height != dst_tex.height {
        bail!(
            "CUDA/DX12: CopyRenderTarget size mismatch ({}x{} → {}x{})",
            cuda_tex.width,
            cuda_tex.height,
            dst_tex.width,
            dst_tex.height
        );
    }

    if fence > 0 {
        let companion = companion(backend, device)?;
        cuda_wait_fence(&companion.cuda_ctx, companion.cuda_semaphore, stream.cu_stream(), fence)?;
    }

    memcpy_array_to_array(stream, &cuda_tex, dst_tex)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_float_target_format_message() {
        let err = format_to_dxgi(TextureFormat::Bgra8Unorm).unwrap_err().to_string();
        assert!(err.contains("unsupported"), "{err}");
    }

    #[test]
    fn triangle_list_topology_maps() {
        assert_eq!(
            topology_type_to_d3d12(PrimitiveTopology::TriangleList),
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE
        );
    }
}
