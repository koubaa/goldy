//! First-slice CUDA + DX12 raster: offscreen color targets, graphics PSOs, and draws.
//!
//! Scope:
//! - Windows + `cuda` + `graphics` + `dx12`
//! - Color-only [`TextureFormat::Rgba32Float`] render targets
//! - Indexed and non-indexed draws (`PointList` / `LineList` / `LineStrip` /
//!   `TriangleList` / `TriangleStrip`)
//! - Bindless render resources via companion SM 6.6 heaps (CUDA registry slot =
//!   DX12 descriptor index); no depth yet
//!
//! Vertex / index buffers stay on native CUDA allocations for compute. A shareable
//! D3D12 twin is created lazily for IA; contents are refreshed with a device-to-device
//! copy (no host DtoH) before draws. Kernel writes directly into imported
//! `cuExternalMemoryGetMappedBuffer` pointers currently fault
//! (`CUDA_ERROR_ILLEGAL_ADDRESS`) on this driver stack. Render-target color is a
//! shared D3D12 resource imported into CUDA so [`GpuCommand::CopyRenderTarget`] can
//! array-copy into present scratch / other CUDA textures after a companion-fence
//! wait. When the copy destination is surface scratch, the CUDA backend skips that
//! copy and presents the RT on DX12.

use super::dx12_companion::Dx12Companion;
use super::dx12_interop::{create_shared_texture, import_shared_texture, CudaImportedTexture};
use super::texture::{memcpy_array_to_array, CudaTextureResource};
use super::{CudaBackend, CudaShader};
use crate::backend::shared::{fill_frame_table_dispatch, set_frame_table_slots, PushLayout, TOTAL_PUSH_BYTES};
use crate::backend::{BufferHandle, DeviceHandle, PipelineHandle, RenderCommand, RenderTargetHandle, TextureHandle};
use crate::frame_table::FrameTableStaging;
use crate::types::{
    BindlessSlotKind, IndexFormat, PrimitiveTopology, ResourceCategory, TargetLoad, TextureFlags, TextureFormat,
    TextureKind, VertexBufferLayout, VertexFormat,
};
use anyhow::{bail, Context as _, Result};
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
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
    pub push_constant_slot_kinds: Vec<Option<BindlessSlotKind>>,
    pub binding_element_strides: Vec<Option<u32>>,
    pub shader_debug_name: String,
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

#[derive(Clone, Copy)]
pub(super) struct RasterListCache {
    pub fingerprint: u64,
}

fn raster_fingerprint(
    backend: &CudaBackend,
    target: RenderTargetHandle,
    color_load: TargetLoad,
    commands: &[RenderCommand],
    staging_data: &[u32],
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
    // Graph lowering clears BindResourcesRaw.indices into the staging table; hash the
    // active row so bind identity changes bust retained list reuse.
    for word in staging_data
        .iter()
        .take(crate::frame_table::FRAME_TABLE_ROW_STRIDE as usize)
    {
        word.hash(&mut hash);
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
            RenderCommand::SetIndexBuffer { buffer, offset, format } => {
                let cuda_buf = backend
                    .buffers
                    .get(buffer)
                    .context("CUDA/DX12: invalid index buffer")?;
                buffer.hash(&mut hash);
                offset.hash(&mut hash);
                format.hash(&mut hash);
                cuda_buf.content_epoch.hash(&mut hash);
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
            RenderCommand::BindResources { buffers } => {
                for h in buffers {
                    h.hash(&mut hash);
                }
            }
            RenderCommand::BindResourcesRaw {
                indices,
                user,
                frame_table_base,
            } => {
                indices.hash(&mut hash);
                user.hash(&mut hash);
                frame_table_base.hash(&mut hash);
            }
            RenderCommand::BindResourcesTyped { handles } => {
                for h in handles {
                    h.index().hash(&mut hash);
                    std::mem::discriminant(&h.category()).hash(&mut hash);
                }
            }
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

fn index_format_to_dxgi(format: IndexFormat) -> DXGI_FORMAT {
    match format {
        IndexFormat::Uint16 => DXGI_FORMAT_R16_UINT,
        IndexFormat::Uint32 => DXGI_FORMAT_R32_UINT,
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

fn compile_stage_dxil(
    shader: &CudaShader,
    stage: crate::slang::SlangStage,
) -> Result<(Vec<u8>, crate::slang::ShaderReflection)> {
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
    let dxil = compiled
        .shader
        .as_dxil()
        .context("CUDA/DX12: expected DXIL bytecode")?
        .to_vec();
    let mut reflection = compiled.reflection;
    if reflection.push_constant_categories.is_empty() {
        reflection.push_constant_categories =
            crate::slang::virtual_main::extract_push_constant_categories(&shader.source);
    }
    if reflection.push_constant_slot_kinds.is_empty() {
        reflection.push_constant_slot_kinds =
            crate::slang::virtual_main::extract_push_constant_slot_kinds(&shader.source);
    }
    Ok((dxil, reflection))
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

    let (vs_dxil, _vs_refl) = compile_stage_dxil(vs, crate::slang::SlangStage::Vertex)?;
    let (fs_dxil, fs_refl) = compile_stage_dxil(fs, crate::slang::SlangStage::Fragment)?;
    let shader_debug_name = format!("shader(vs=#{vertex_shader}, fs=#{fragment_shader})");
    let push_constant_slot_kinds = fs_refl.push_constant_slot_kinds.clone();
    let binding_element_strides = fs_refl.binding_element_strides.clone();

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
            push_constant_slot_kinds,
            binding_element_strides,
            shader_debug_name,
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

    companion
        .bindless
        .write_texture_uav(&companion.device, storage_slot, &d3d12_resource, color_format)?;

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
    backend.texture_dx12.insert(tex_handle, d3d12_resource.clone());

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

pub(super) fn destroy_render_target(backend: &mut CudaBackend, target: RenderTargetHandle) {
    let _ = backend.render_targets.remove(&target);
}

/// Ensure VERTEX physicalization and return a companion fence DX12 must wait on
/// before IA (0 = already coherent / host-synced).
///
/// - **Shared**: no DtoD; wait on `last_cuda_fence` from CUDA writes into the import.
/// - **NativeAndTwin**: DtoD native→twin only when `content_epoch != shared_epoch`.
fn refresh_shared_vertex_backing(
    backend: &mut CudaBackend,
    device: DeviceHandle,
    buffer: BufferHandle,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> Result<u64> {
    use cudarc::driver::DevicePtr;
    use super::buffer_phys::{CudaBufferReq, CudaPhysKind};

    backend.ensure_buffer_requirements(buffer, CudaBufferReq::VERTEX)?;

    let phys_kind = backend
        .buffers
        .get(&buffer)
        .context("CUDA/DX12: invalid vertex buffer")?
        .phys_kind;

    match phys_kind {
        CudaPhysKind::Shared => {
            let shared = Arc::clone(
                backend
                    .buffers
                    .get(&buffer)
                    .unwrap()
                    .shared
                    .as_ref()
                    .context("CUDA/DX12: Shared VB missing backing")?,
            );
            // Deposit-only path: CUDA wrote the import without a companion Signal.
            // Flush/sync so DX12 IA observes the bytes (avoids Signal/DX12 fence races).
            if shared.pending_host_sync.swap(false, Ordering::AcqRel) {
                let worker = Arc::clone(&backend.device(device)?.submission_worker);
                worker
                    .flush()
                    .context("CUDA/DX12: flush before Shared VB host sync")?;
                backend
                    .graph_stats
                    .worker_flushes
                    .fetch_add(1, Ordering::Relaxed);
                for context in backend.contexts.values().filter(|context| context.device == device) {
                    context
                        .stream
                        .synchronize()
                        .context("CUDA/DX12: sync context stream before Shared VB bind")?;
                }
            }
            backend
                .graph_stats
                .shared_vb_binds
                .fetch_add(1, Ordering::Relaxed);
            Ok(shared.last_cuda_fence.load(Ordering::Acquire))
        }
        CudaPhysKind::NativeAndTwin => {
            let needs_create = {
                let buf = backend.buffers.get(&buffer).unwrap();
                match &buf.shared {
                    None => true,
                    Some(shared) => shared.size < buf.capacity.max(4),
                }
            };
            if needs_create {
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
                let capacity = backend.buffers.get(&buffer).unwrap().capacity.max(4);
                let backing = super::dx12_interop::create_shared_buffer_backing(
                    &companion,
                    &cuda_ctx,
                    stream,
                    capacity,
                )?;
                let old = backend
                    .buffers
                    .get_mut(&buffer)
                    .unwrap()
                    .shared
                    .replace(Arc::new(backing));
                if let Some(old) = old {
                    let fence = old.last_cuda_fence.load(Ordering::Acquire);
                    if fence > 0 {
                        let _ = companion.cpu_wait(fence);
                    }
                    drop(old);
                }
                backend.buffers.get_mut(&buffer).unwrap().shared_epoch = u64::MAX;
            }

            let (content_epoch, shared_epoch, offset, size, memory, shared) = {
                let buf = backend.buffers.get(&buffer).unwrap();
                (
                    buf.content_epoch,
                    buf.shared_epoch,
                    buf.offset,
                    buf.size,
                    Arc::clone(buf.memory_arc()?),
                    Arc::clone(buf.shared.as_ref().unwrap()),
                )
            };

            if size == 0 {
                backend.buffers.get_mut(&buffer).unwrap().shared_epoch = content_epoch;
                return Ok(0);
            }

            if content_epoch == shared_epoch {
                // Twin already matches native; still Wait if a prior DtoD fence is pending.
                return Ok(shared.last_cuda_fence.load(Ordering::Acquire));
            }

            let nbytes = size as usize;
            let src_ptr = {
                let guard = memory.lock().unwrap();
                let view = guard
                    .try_slice(offset as usize..(offset as usize + nbytes))
                    .context("CUDA/DX12: VB source view")?;
                let (ptr, _sync) = view.device_ptr(stream);
                ptr
            };
            let dst_ptr = shared.import.device_ptr;
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    dst_ptr,
                    src_ptr,
                    nbytes,
                    stream.cu_stream(),
                )
            }
            .context("CUDA/DX12: DtoD refresh into shared VB failed")?;

            let companion = companion(backend, device)?;
            let value = companion.next_fence_value();
            super::dx12_companion::cuda_signal_fence(
                &companion.cuda_ctx,
                companion.cuda_semaphore,
                stream.cu_stream(),
                value,
            )?;
            shared.last_cuda_fence.store(value, Ordering::Release);
            backend.buffers.get_mut(&buffer).unwrap().shared_epoch = content_epoch;
            backend
                .graph_stats
                .shared_vb_binds
                .fetch_add(1, Ordering::Relaxed);
            Ok(value)
        }
        other => bail!(
            "CUDA/DX12: vertex buffer phys_kind={other:?} after VERTEX ensure (expected Shared or NativeAndTwin)"
        ),
    }
}

/// Record and submit a companion DX12 raster pass for `target`.
///
/// `graph_staging` is the task-graph [`GpuCommand::FrameTableStaging`] payload.
/// When present, the companion prologue must upload that table — Scheme already
/// lowered binds to `BindResourcesRaw` with empty `indices`.
pub(super) fn render_to_target(
    backend: &mut CudaBackend,
    device: DeviceHandle,
    target: RenderTargetHandle,
    color_load: TargetLoad,
    commands: &[RenderCommand],
    graph_staging: Option<&[u32]>,
) -> Result<()> {
    {
        let rt = backend
            .render_targets
            .get(&target)
            .context("CUDA/DX12: invalid render target")?;
        if rt.device != device {
            bail!("CUDA/DX12: render target belongs to a different device");
        }
    }

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

    use super::buffer_phys::{CudaBufferReq, CudaPhysKind};

    // Lower typed/handle binds → frame-table routing before fingerprint / record.
    // Graph submit passes pre-built staging; standalone render_to_target rebuilds it.
    let (staging_data, lowered, has_bindings) =
        prepare_cuda_render_commands(backend, commands, graph_staging)?;

    // Pre-declare VERTEX / SHADER so deposit→Shared can win before provisional Native sticks.
    let mut ia_handles = Vec::new();
    let mut shader_buffers = Vec::new();
    for command in &lowered {
        match command {
            RenderCommand::SetVertexBuffer { buffer, .. }
            | RenderCommand::SetIndexBuffer { buffer, .. } => {
                backend.ensure_buffer_requirements(*buffer, CudaBufferReq::VERTEX)?;
                if !ia_handles.contains(buffer) {
                    ia_handles.push(*buffer);
                }
            }
            RenderCommand::BindResourcesRaw {
                indices,
                frame_table_base,
                ..
            } => {
                // Indices may be empty after lowering; staging row holds the real slots.
                let _ = (indices, frame_table_base);
            }
            RenderCommand::BindResourcesTyped { handles } => {
                for handle in handles {
                    if matches!(
                        handle.category(),
                        ResourceCategory::Scattered | ResourceCategory::Broadcast
                    ) {
                        // ResourceHandle.index() is the bindless slot; resolve to buffer.
                        if let Some(&buf) = backend.buffer_slots.get(&handle.index()) {
                            backend.ensure_buffer_requirements(buf, CudaBufferReq::SHADER)?;
                            if !shader_buffers.contains(&buf) {
                                shader_buffers.push(buf);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // Staging row may reference buffer slots without BindResourcesTyped surviving;
    // ensure any staging index that maps to a buffer gets SHADER phys.
    for &word in &staging_data {
        if let Some(&buf) = backend.buffer_slots.get(&word) {
            backend.ensure_buffer_requirements(buf, CudaBufferReq::SHADER)?;
            if !shader_buffers.contains(&buf) {
                shader_buffers.push(buf);
            }
        }
    }

    let fingerprint = raster_fingerprint(backend, target, color_load, &lowered, &staging_data)?;

    // Flush/sync when twin DtoD must observe in-flight native CUDA writes (IA or shader).
    let needs_twin_sync = ia_handles.iter().chain(shader_buffers.iter()).any(|handle| {
        backend.buffers.get(handle).is_some_and(|buf| {
            buf.phys_kind == CudaPhysKind::NativeAndTwin && buf.content_epoch != buf.shared_epoch
        })
    });
    if needs_twin_sync {
        let worker = Arc::clone(&backend.device(device)?.submission_worker);
        worker
            .flush()
            .context("CUDA/DX12: flush before twin VB refresh")?;
        backend
            .graph_stats
            .worker_flushes
            .fetch_add(1, Ordering::Relaxed);
        for context in backend.contexts.values().filter(|context| context.device == device) {
            context
                .stream
                .synchronize()
                .context("CUDA/DX12: sync context stream before twin VB refresh")?;
        }
    }

    let alloc_stream = Arc::clone(&backend.device(device)?.alloc_stream);
    let mut vb_wait = 0u64;
    let mut any_shared = false;
    for handle in ia_handles.iter().chain(shader_buffers.iter()) {
        if backend
            .buffers
            .get(handle)
            .is_some_and(|b| b.phys_kind == CudaPhysKind::Shared)
        {
            any_shared = true;
        }
        let fence = refresh_shared_vertex_backing(backend, device, *handle, &alloc_stream)?;
        vb_wait = vb_wait.max(fence);
    }
    // Immediate materialize/HtoD uses alloc_stream; make those bytes visible to DX12
    // SRV/IA without a submission-worker flush (which would break retained stats).
    if any_shared {
        alloc_stream
            .synchronize()
            .context("CUDA/DX12: sync alloc stream before Shared DX12 read")?;
    }
    if vb_wait > 0 {
        companion.wait_queue(vb_wait)?;
    }

    if backend
        .raster_list_cache
        .get(&target)
        .is_some_and(|cache| cache.fingerprint == fingerprint)
    {
        if let Some(signal) = companion.try_reuse_raster_for_fingerprint(fingerprint)? {
            backend.render_targets.get_mut(&target).unwrap().last_dx12_fence = signal;
            for handle in ia_handles.iter().chain(shader_buffers.iter()) {
                if let Some(shared) = backend.buffers.get(handle).and_then(|buf| buf.shared.as_ref()) {
                    shared.last_dx12_ia_fence.fetch_max(signal, Ordering::AcqRel);
                }
            }
            return Ok(());
        }
    }

    let (slot_idx, list, _slot_generation) = companion.begin_raster_list()?;

    companion.bindless.set_descriptor_heaps(&list);
    let mut frame_table_row = None;
    if has_bindings {
        let row = companion
            .frame_table
            .record_prologue(&companion, &list, &staging_data)?;
        frame_table_row = Some(row);
    }

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
    let mut current_pipeline: Option<PipelineHandle> = None;

    for command in &lowered {
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
                current_pipeline = Some(*pipeline_handle);
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
                    .context("CUDA/DX12: invalid vertex buffer")?;
                if *offset >= cuda_buf.size {
                    bail!("CUDA/DX12: vertex buffer offset out of range");
                }
                let shared = cuda_buf
                    .shared
                    .as_ref()
                    .context("CUDA/DX12: vertex buffer missing shared backing")?;
                let abs_offset = cuda_buf.offset + offset;
                let nbytes = (cuda_buf.size - offset) as u32;
                let gpu_va = unsafe { shared.d3d12_resource.GetGPUVirtualAddress() } + abs_offset;
                let view = D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: gpu_va,
                    SizeInBytes: nbytes,
                    StrideInBytes: current_stride,
                };
                unsafe { list.IASetVertexBuffers(*slot, Some(&[view])) };
            }
            RenderCommand::SetIndexBuffer { buffer, offset, format } => {
                let cuda_buf = backend
                    .buffers
                    .get(buffer)
                    .context("CUDA/DX12: invalid index buffer")?;
                if *offset >= cuda_buf.size {
                    bail!("CUDA/DX12: index buffer offset out of range");
                }
                let shared = cuda_buf
                    .shared
                    .as_ref()
                    .context("CUDA/DX12: index buffer missing shared backing")?;
                let abs_offset = cuda_buf.offset + offset;
                let nbytes = (cuda_buf.size - offset) as u32;
                let gpu_va = unsafe { shared.d3d12_resource.GetGPUVirtualAddress() } + abs_offset;
                let view = D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: gpu_va,
                    SizeInBytes: nbytes,
                    Format: index_format_to_dxgi(*format),
                };
                unsafe { list.IASetIndexBuffer(Some(&view)) };
            }
            RenderCommand::BindResources { .. } | RenderCommand::BindResourcesTyped { .. } => {
                bail!("CUDA/DX12 raster: BindResources must be lowered before record");
            }
            RenderCommand::BindResourcesRaw {
                indices: raw_indices,
                user: raw_user,
                frame_table_base,
            } => {
                if let Some(h) = current_pipeline {
                    let pipeline = backend
                        .pipelines
                        .get(&h)
                        .context("CUDA/DX12: invalid graphics pipeline")?;
                    // Graph lowering clears `indices` into the staging table; validate from there.
                    let staged: Vec<u32> = if raw_indices.is_empty() {
                        let n = pipeline.push_constant_slot_kinds.len();
                        let base = *frame_table_base as usize;
                        staging_data
                            .get(base..base.saturating_add(n))
                            .unwrap_or(&[])
                            .to_vec()
                    } else {
                        raw_indices.clone()
                    };
                    crate::backend::with_layout_validation(|| {
                        crate::backend::validate_bindless_slot_kinds(
                            &staged,
                            &pipeline.push_constant_slot_kinds,
                            |idx| companion.bindless.registry.lock().unwrap().slot_kind(idx),
                            &pipeline.shader_debug_name,
                        )
                    })?;
                }
                let mut layout = PushLayout::default();
                fill_frame_table_dispatch(&mut layout, *frame_table_base, raw_user);
                set_frame_table_slots(
                    &mut layout,
                    companion.frame_table.selector_slot,
                    companion.frame_table.table_slot,
                );
                unsafe {
                    list.SetGraphicsRoot32BitConstants(
                        0,
                        (TOTAL_PUSH_BYTES / 4) as u32,
                        &layout as *const _ as *const _,
                        0,
                    );
                }
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
            RenderCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => {
                let _ = current_topology;
                unsafe {
                    list.DrawIndexedInstanced(
                        *index_count,
                        *instance_count,
                        *first_index,
                        *base_vertex,
                        *first_instance,
                    );
                }
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
    if let Some(row) = frame_table_row {
        companion.frame_table.mark_row_submitted(row, signal);
    }
    companion.bindless.drain_reclaimed(unsafe { companion.fence.GetCompletedValue() });
    backend
        .graph_stats
        .raster_list_records
        .fetch_add(1, Ordering::Relaxed);
    if let Some(rt) = backend.render_targets.get_mut(&target) {
        rt.last_dx12_fence = signal;
    }
    for handle in ia_handles.iter().chain(shader_buffers.iter()) {
        if let Some(shared) = backend.buffers.get(handle).and_then(|buf| buf.shared.as_ref()) {
            shared.last_dx12_ia_fence.fetch_max(signal, Ordering::AcqRel);
        }
    }
    backend
        .raster_list_cache
        .insert(target, RasterListCache { fingerprint });
    Ok(())
}

fn prepare_cuda_render_commands(
    backend: &CudaBackend,
    commands: &[RenderCommand],
    graph_staging: Option<&[u32]>,
) -> Result<(Vec<u32>, Vec<RenderCommand>, bool)> {
    crate::backend::with_layout_validation(|| {
        crate::backend::validate_render_pass_bind_resources(
            commands,
            |h| {
                backend.pipelines.get(&h).map(|p| {
                    (
                        p.binding_element_strides.clone(),
                        p.shader_debug_name.clone(),
                    )
                })
            },
            |h| backend.buffers.get(&h).and_then(|b| b.element_stride),
        )
    })?;

    // Scheme/task-graph already lowered binds into FrameTableStaging + BindResourcesRaw
    // (empty indices, frame_table_base set). Prefer that staging for the companion prologue.
    if let Some(data) = graph_staging {
        let has_bindings = commands.iter().any(|c| {
            matches!(
                c,
                RenderCommand::BindResourcesRaw { .. }
                    | RenderCommand::BindResourcesTyped { .. }
                    | RenderCommand::BindResources { .. }
            )
        }) || data.iter().any(|&w| w != 0);
        return Ok((data.to_vec(), commands.to_vec(), has_bindings));
    }

    let mut staging = FrameTableStaging::new();
    let lowered = commands
        .iter()
        .map(|cmd| match cmd {
            RenderCommand::BindResources { buffers } => {
                let indices: Vec<u32> = buffers
                    .iter()
                    .map(|h| {
                        backend
                            .buffers
                            .get(h)
                            .and_then(|b| b.slot)
                            .with_context(|| format!("BindResources: buffer {h:?} has no registry slot"))
                    })
                    .collect::<Result<_>>()?;
                let frame_table_base = staging.alloc_dispatch(indices.len() as u32);
                staging.write_dispatch_indices(frame_table_base, &indices);
                Ok(RenderCommand::BindResourcesRaw {
                    indices: Vec::new(),
                    user: Vec::new(),
                    frame_table_base,
                })
            }
            other => {
                let batch =
                    crate::frame_table::lower_render_pass_commands(&mut staging, std::slice::from_ref(other));
                Ok(batch.into_iter().next().unwrap_or_else(|| other.clone()))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let has_bindings = staging.has_bindings();
    Ok((staging.data, lowered, has_bindings))
}

/// Copy render-target color into a CUDA texture after waiting on the DX12 fence.
#[allow(dead_code)]
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
        super::dx12_companion::cuda_wait_fence(
            &companion.cuda_ctx,
            companion.cuda_semaphore,
            stream.cu_stream(),
            fence,
        )?;
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

    #[test]
    fn line_topologies_map() {
        assert_eq!(
            topology_type_to_d3d12(PrimitiveTopology::LineList),
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
        );
        assert_eq!(
            topology_type_to_d3d12(PrimitiveTopology::LineStrip),
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
        );
    }
}
