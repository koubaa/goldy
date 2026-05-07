//! Pipeline creation and management.

use super::types::PipelineState;
use super::{shader, utils, DeviceHandle, Dx12State, PipelineHandle, ShaderHandle};
use crate::types::{DepthStencilState, PrimitiveTopology, TextureFormat, VertexBufferLayout};
use anyhow::{Context, Result};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

/// Create a graphics pipeline state object.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(super) fn create(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    vertex_shader: ShaderHandle,
    fragment_shader: ShaderHandle,
    vertex_layout: &VertexBufferLayout,
    topology: PrimitiveTopology,
    target_format: TextureFormat,
) -> Result<PipelineHandle> {
    // Compile shaders on-demand
    let vs_bytecode =
        shader::ensure_stage_compiled(state, vertex_shader, crate::slang::SlangStage::Vertex)?;
    let fs_bytecode =
        shader::ensure_stage_compiled(state, fragment_shader, crate::slang::SlangStage::Fragment)?;

    let shader_debug_name = format!("shader(vs=#{vertex_shader}, fs=#{fragment_shader})");

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    // Use the shared bindless root signature from the device
    let root_signature = logical_device
        .bindless_root_signature
        .as_ref()
        .context("Bindless root signature not available")?
        .clone();

    tracing::debug!("Using shared bindless root signature for graphics pipeline");

    // Build input layout
    // We use semantic conventions based on location and format:
    // - location 0 → POSITION (expected for all shaders)
    // - location 1 with 3-4 components → COLOR (for colored vertex shaders)
    // - location 1 with 1-2 components → TEXCOORD0 (for textured shaders)
    // - location 2+ → TEXCOORDn
    let mut texcoord_index = 0u32;
    let input_elements: Vec<D3D12_INPUT_ELEMENT_DESC> = vertex_layout
        .attributes
        .iter()
        .map(|attr| {
            let (semantic_name, semantic_index) = if attr.location == 0 {
                (c"POSITION".as_ptr() as *const u8, 0)
            } else {
                // Determine semantic based on format
                // 3-4 component formats at location 1 are likely COLOR
                // 1-2 component formats are likely TEXCOORD
                let is_color = match attr.format {
                    crate::types::VertexFormat::Float32x3
                    | crate::types::VertexFormat::Float32x4
                    | crate::types::VertexFormat::Unorm8x4
                    | crate::types::VertexFormat::Uint8x4 => attr.location == 1,
                    _ => false,
                };

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
                Format: utils::vertex_format_to_dxgi(attr.format),
                InputSlot: 0,
                AlignedByteOffset: attr.offset,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            }
        })
        .collect();

    // Create PSO
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs_bytecode.as_ptr() as *const _,
            BytecodeLength: vs_bytecode.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: fs_bytecode.as_ptr() as *const _,
            BytecodeLength: fs_bytecode.len(),
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
        PrimitiveTopologyType: utils::topology_type_to_d3d12(topology),
        NumRenderTargets: 1,
        RTVFormats: [
            utils::format_to_dxgi(target_format),
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
        ],
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };

    let pipeline_state: ID3D12PipelineState =
        unsafe { logical_device.device.CreateGraphicsPipelineState(&pso_desc) }
            .context("Failed to create pipeline state")?;

    let handle = state.next_pipeline_handle;
    state.next_pipeline_handle += 1;

    state.pipelines.insert(
        handle,
        PipelineState {
            device_handle,
            pipeline_state,
            root_signature,
            vertex_stride: vertex_layout.stride,
            topology,
            parameter_block_layouts: Vec::new(),
            push_constant_categories: Vec::new(),
            shader_debug_name,
        },
    );

    tracing::debug!("Created render pipeline {}", handle);
    Ok(handle)
}

/// Create a graphics pipeline with depth testing.
#[allow(clippy::too_many_arguments)]
pub(super) fn create_with_depth(
    state: &mut Dx12State,
    device_handle: DeviceHandle,
    vertex_shader: ShaderHandle,
    fragment_shader: ShaderHandle,
    vertex_layout: &VertexBufferLayout,
    topology: PrimitiveTopology,
    target_format: TextureFormat,
    depth_stencil: Option<&DepthStencilState>,
) -> Result<PipelineHandle> {
    let vs_bytecode =
        shader::ensure_stage_compiled(state, vertex_shader, crate::slang::SlangStage::Vertex)?;
    let fs_bytecode =
        shader::ensure_stage_compiled(state, fragment_shader, crate::slang::SlangStage::Fragment)?;

    let shader_debug_name = format!("shader(vs=#{vertex_shader}, fs=#{fragment_shader})");

    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let root_signature = logical_device
        .bindless_root_signature
        .as_ref()
        .context("Bindless root signature not available")?
        .clone();

    let mut texcoord_index = 0u32;
    let input_elements: Vec<D3D12_INPUT_ELEMENT_DESC> = vertex_layout
        .attributes
        .iter()
        .map(|attr| {
            let (semantic_name, semantic_index) = if attr.location == 0 {
                (c"POSITION".as_ptr() as *const u8, 0)
            } else {
                let is_color = match attr.format {
                    crate::types::VertexFormat::Float32x3
                    | crate::types::VertexFormat::Float32x4
                    | crate::types::VertexFormat::Unorm8x4
                    | crate::types::VertexFormat::Uint8x4 => attr.location == 1,
                    _ => false,
                };
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
                Format: utils::vertex_format_to_dxgi(attr.format),
                InputSlot: 0,
                AlignedByteOffset: attr.offset,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            }
        })
        .collect();

    // Build depth/stencil state and DSV format from the descriptor.
    let (depth_stencil_desc, dsv_format) = if let Some(ds) = depth_stencil {
        let desc = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: if ds.depth_write_enabled {
                D3D12_DEPTH_WRITE_MASK_ALL
            } else {
                D3D12_DEPTH_WRITE_MASK_ZERO
            },
            DepthFunc: utils::compare_to_d3d12(ds.depth_compare),
            StencilEnable: false.into(),
            ..Default::default()
        };
        (desc, utils::depth_format_to_dxgi(ds.format))
    } else {
        (
            D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: false.into(),
                ..Default::default()
            },
            DXGI_FORMAT_UNKNOWN,
        )
    };

    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs_bytecode.as_ptr() as *const _,
            BytecodeLength: vs_bytecode.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: fs_bytecode.as_ptr() as *const _,
            BytecodeLength: fs_bytecode.len(),
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
        DepthStencilState: depth_stencil_desc,
        DSVFormat: dsv_format,
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: input_elements.as_ptr(),
            NumElements: input_elements.len() as u32,
        },
        PrimitiveTopologyType: utils::topology_type_to_d3d12(topology),
        NumRenderTargets: 1,
        RTVFormats: [
            utils::format_to_dxgi(target_format),
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
        ],
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };

    let pipeline_state: ID3D12PipelineState =
        unsafe { logical_device.device.CreateGraphicsPipelineState(&pso_desc) }
            .context("Failed to create depth pipeline state")?;

    let handle = state.next_pipeline_handle;
    state.next_pipeline_handle += 1;

    state.pipelines.insert(
        handle,
        PipelineState {
            device_handle,
            pipeline_state,
            root_signature,
            vertex_stride: vertex_layout.stride,
            topology,
            parameter_block_layouts: Vec::new(),
            push_constant_categories: Vec::new(),
            shader_debug_name,
        },
    );

    tracing::debug!("Created depth pipeline {}", handle);
    Ok(handle)
}

/// Destroy a pipeline.
pub(super) fn destroy(state: &mut Dx12State, pipeline_handle: PipelineHandle) {
    state.pipelines.remove(&pipeline_handle);
}
