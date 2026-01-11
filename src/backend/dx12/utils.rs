//! DX12 utility functions.
//!
//! Format conversions and helpers.

use crate::types::{AddressMode, CompareFunction, DepthFormat, FilterMode, IndexFormat, TextureFormat, VertexFormat, PrimitiveTopology};
use windows::Win32::Graphics::{Direct3D, Direct3D12, Dxgi};

/// Convert RAG TextureFormat to DXGI format.
pub fn format_to_dxgi(format: TextureFormat) -> Dxgi::Common::DXGI_FORMAT {
    match format {
        TextureFormat::Rgba8UnormSrgb => Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        TextureFormat::Rgba8Unorm => Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        TextureFormat::Bgra8Unorm => Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
        TextureFormat::Rgba16Float => Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
        TextureFormat::Rgba32Float => Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT,
    }
}

/// Convert DXGI format to RAG TextureFormat.
pub fn dxgi_to_format(format: Dxgi::Common::DXGI_FORMAT) -> Option<TextureFormat> {
    match format {
        Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => Some(TextureFormat::Rgba8UnormSrgb),
        Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
        Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => Some(TextureFormat::Bgra8UnormSrgb),
        Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
        Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT => Some(TextureFormat::Rgba16Float),
        Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT => Some(TextureFormat::Rgba32Float),
        _ => None,
    }
}

/// Convert RAG VertexFormat to DXGI format.
pub fn vertex_format_to_dxgi(format: VertexFormat) -> Dxgi::Common::DXGI_FORMAT {
    match format {
        VertexFormat::Float32 => Dxgi::Common::DXGI_FORMAT_R32_FLOAT,
        VertexFormat::Float32x2 => Dxgi::Common::DXGI_FORMAT_R32G32_FLOAT,
        VertexFormat::Float32x3 => Dxgi::Common::DXGI_FORMAT_R32G32B32_FLOAT,
        VertexFormat::Float32x4 => Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT,
        VertexFormat::Uint32 => Dxgi::Common::DXGI_FORMAT_R32_UINT,
        VertexFormat::Sint32 => Dxgi::Common::DXGI_FORMAT_R32_SINT,
        VertexFormat::Uint8x4 => Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UINT,
        VertexFormat::Unorm8x4 => Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
    }
}

/// Convert RAG PrimitiveTopology to D3D12 topology.
pub fn topology_to_d3d12(topology: PrimitiveTopology) -> Direct3D::D3D_PRIMITIVE_TOPOLOGY {
    match topology {
        PrimitiveTopology::PointList => Direct3D::D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        PrimitiveTopology::LineList => Direct3D::D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        PrimitiveTopology::LineStrip => Direct3D::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        PrimitiveTopology::TriangleList => Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        PrimitiveTopology::TriangleStrip => Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    }
}

/// Convert RAG PrimitiveTopology to D3D12 topology type for PSO.
pub fn topology_type_to_d3d12(topology: PrimitiveTopology) -> Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    match topology {
        PrimitiveTopology::PointList => Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
        PrimitiveTopology::LineList | PrimitiveTopology::LineStrip => {
            Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE
        }
        PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip => {
            Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE
        }
    }
}

/// Convert RAG IndexFormat to DXGI index format.
pub fn index_format_to_dxgi(format: IndexFormat) -> Dxgi::Common::DXGI_FORMAT {
    match format {
        IndexFormat::Uint16 => Dxgi::Common::DXGI_FORMAT_R16_UINT,
        IndexFormat::Uint32 => Dxgi::Common::DXGI_FORMAT_R32_UINT,
    }
}

/// Map vendor ID to vendor name.
pub fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x1002 | 0x1022 => "AMD",
        0x10DE => "NVIDIA",
        0x8086 => "Intel",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x106B => "Apple",
        0x1414 => "Microsoft", // For WARP adapter
        _ => "Unknown",
    }
}

/// Map DXGI adapter flags to RAG DeviceType.
pub fn device_type_from_flags(flags: Dxgi::DXGI_ADAPTER_FLAG) -> crate::types::DeviceType {
    if flags.contains(Dxgi::DXGI_ADAPTER_FLAG_SOFTWARE) {
        crate::types::DeviceType::Cpu
    } else {
        // Can't easily distinguish discrete vs integrated from DXGI alone
        // Default to discrete for hardware adapters
        crate::types::DeviceType::DiscreteGpu
    }
}

/// Convert RAG DepthFormat to DXGI format.
pub fn depth_format_to_dxgi(format: DepthFormat) -> Dxgi::Common::DXGI_FORMAT {
    match format {
        DepthFormat::Depth16Unorm => Dxgi::Common::DXGI_FORMAT_D16_UNORM,
        DepthFormat::Depth24Plus => Dxgi::Common::DXGI_FORMAT_D32_FLOAT,
        DepthFormat::Depth24PlusStencil8 => Dxgi::Common::DXGI_FORMAT_D24_UNORM_S8_UINT,
        DepthFormat::Depth32Float => Dxgi::Common::DXGI_FORMAT_D32_FLOAT,
        DepthFormat::Depth32FloatStencil8 => Dxgi::Common::DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    }
}

/// Convert RAG CompareFunction to D3D12 comparison func.
pub fn compare_to_d3d12(compare: CompareFunction) -> Direct3D12::D3D12_COMPARISON_FUNC {
    match compare {
        CompareFunction::Never => Direct3D12::D3D12_COMPARISON_FUNC_NEVER,
        CompareFunction::Less => Direct3D12::D3D12_COMPARISON_FUNC_LESS,
        CompareFunction::Equal => Direct3D12::D3D12_COMPARISON_FUNC_EQUAL,
        CompareFunction::LessEqual => Direct3D12::D3D12_COMPARISON_FUNC_LESS_EQUAL,
        CompareFunction::Greater => Direct3D12::D3D12_COMPARISON_FUNC_GREATER,
        CompareFunction::NotEqual => Direct3D12::D3D12_COMPARISON_FUNC_NOT_EQUAL,
        CompareFunction::GreaterEqual => Direct3D12::D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        CompareFunction::Always => Direct3D12::D3D12_COMPARISON_FUNC_ALWAYS,
    }
}

/// Convert RAG FilterMode to D3D12 filter.
pub fn filter_to_d3d12(min: FilterMode, mag: FilterMode, mip: FilterMode) -> Direct3D12::D3D12_FILTER {
    match (min, mag, mip) {
        (FilterMode::Nearest, FilterMode::Nearest, FilterMode::Nearest) => Direct3D12::D3D12_FILTER_MIN_MAG_MIP_POINT,
        (FilterMode::Nearest, FilterMode::Nearest, FilterMode::Linear) => Direct3D12::D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR,
        (FilterMode::Nearest, FilterMode::Linear, FilterMode::Nearest) => Direct3D12::D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT,
        (FilterMode::Nearest, FilterMode::Linear, FilterMode::Linear) => Direct3D12::D3D12_FILTER_MIN_POINT_MAG_MIP_LINEAR,
        (FilterMode::Linear, FilterMode::Nearest, FilterMode::Nearest) => Direct3D12::D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT,
        (FilterMode::Linear, FilterMode::Nearest, FilterMode::Linear) => Direct3D12::D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
        (FilterMode::Linear, FilterMode::Linear, FilterMode::Nearest) => Direct3D12::D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
        (FilterMode::Linear, FilterMode::Linear, FilterMode::Linear) => Direct3D12::D3D12_FILTER_MIN_MAG_MIP_LINEAR,
    }
}

/// Convert RAG AddressMode to D3D12 texture address mode.
pub fn address_mode_to_d3d12(mode: AddressMode) -> Direct3D12::D3D12_TEXTURE_ADDRESS_MODE {
    match mode {
        AddressMode::ClampToEdge => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressMode::Repeat => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressMode::MirrorRepeat => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
    }
}

