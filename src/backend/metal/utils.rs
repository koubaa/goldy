//! Metal backend utility functions.
//!
//! Format conversion and helper functions for Metal types.

#![allow(dead_code)] // Some utils are for future use

use crate::types::{
    AddressMode, CompareFunction, DepthFormat, FilterMode, IndexFormat, PrimitiveTopology,
    TextureFormat, VertexFormat,
};
// Use explicit crate path to avoid collision with our module name
use ::metal as mtl;
use mtl::{
    MTLCompareFunction, MTLIndexType, MTLPixelFormat, MTLPrimitiveType, MTLSamplerAddressMode,
    MTLSamplerMinMagFilter, MTLVertexFormat,
};

/// Convert goldy TextureFormat to Metal MTLPixelFormat.
pub fn format_to_mtl(format: TextureFormat) -> MTLPixelFormat {
    match format {
        TextureFormat::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        TextureFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        TextureFormat::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        TextureFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        TextureFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        TextureFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
    }
}

/// Convert Metal MTLPixelFormat to goldy TextureFormat.
pub fn mtl_to_format(format: MTLPixelFormat) -> TextureFormat {
    match format {
        MTLPixelFormat::RGBA8Unorm_sRGB => TextureFormat::Rgba8UnormSrgb,
        MTLPixelFormat::RGBA8Unorm => TextureFormat::Rgba8Unorm,
        MTLPixelFormat::BGRA8Unorm_sRGB => TextureFormat::Bgra8UnormSrgb,
        MTLPixelFormat::BGRA8Unorm => TextureFormat::Bgra8Unorm,
        MTLPixelFormat::RGBA16Float => TextureFormat::Rgba16Float,
        MTLPixelFormat::RGBA32Float => TextureFormat::Rgba32Float,
        _ => TextureFormat::Bgra8Unorm, // Default for unknown formats
    }
}

/// Convert goldy DepthFormat to Metal MTLPixelFormat.
pub fn depth_format_to_mtl(format: DepthFormat) -> MTLPixelFormat {
    match format {
        DepthFormat::Depth16Unorm => MTLPixelFormat::Depth16Unorm,
        DepthFormat::Depth24Plus => MTLPixelFormat::Depth32Float, // macOS doesn't have 24-bit, use 32
        DepthFormat::Depth24PlusStencil8 => MTLPixelFormat::Depth32Float_Stencil8,
        DepthFormat::Depth32Float => MTLPixelFormat::Depth32Float,
        DepthFormat::Depth32FloatStencil8 => MTLPixelFormat::Depth32Float_Stencil8,
    }
}

/// Convert goldy VertexFormat to Metal MTLVertexFormat.
pub fn vertex_format_to_mtl(format: VertexFormat) -> MTLVertexFormat {
    match format {
        VertexFormat::Float32 => MTLVertexFormat::Float,
        VertexFormat::Float32x2 => MTLVertexFormat::Float2,
        VertexFormat::Float32x3 => MTLVertexFormat::Float3,
        VertexFormat::Float32x4 => MTLVertexFormat::Float4,
        VertexFormat::Uint32 => MTLVertexFormat::UInt,
        VertexFormat::Sint32 => MTLVertexFormat::Int,
        VertexFormat::Uint8x4 => MTLVertexFormat::UChar4,
        VertexFormat::Unorm8x4 => MTLVertexFormat::UChar4Normalized,
    }
}

/// Convert goldy PrimitiveTopology to Metal MTLPrimitiveType.
pub fn topology_to_mtl(topology: PrimitiveTopology) -> MTLPrimitiveType {
    match topology {
        PrimitiveTopology::PointList => MTLPrimitiveType::Point,
        PrimitiveTopology::LineList => MTLPrimitiveType::Line,
        PrimitiveTopology::LineStrip => MTLPrimitiveType::LineStrip,
        PrimitiveTopology::TriangleList => MTLPrimitiveType::Triangle,
        PrimitiveTopology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
    }
}

/// Convert goldy IndexFormat to Metal MTLIndexType.
pub fn index_format_to_mtl(format: IndexFormat) -> MTLIndexType {
    match format {
        IndexFormat::Uint16 => MTLIndexType::UInt16,
        IndexFormat::Uint32 => MTLIndexType::UInt32,
    }
}

/// Convert goldy CompareFunction to Metal MTLCompareFunction.
pub fn compare_to_mtl(compare: CompareFunction) -> MTLCompareFunction {
    match compare {
        CompareFunction::Never => MTLCompareFunction::Never,
        CompareFunction::Less => MTLCompareFunction::Less,
        CompareFunction::Equal => MTLCompareFunction::Equal,
        CompareFunction::LessEqual => MTLCompareFunction::LessEqual,
        CompareFunction::Greater => MTLCompareFunction::Greater,
        CompareFunction::NotEqual => MTLCompareFunction::NotEqual,
        CompareFunction::GreaterEqual => MTLCompareFunction::GreaterEqual,
        CompareFunction::Always => MTLCompareFunction::Always,
    }
}

/// Convert goldy AddressMode to Metal MTLSamplerAddressMode.
pub fn address_mode_to_mtl(mode: AddressMode) -> MTLSamplerAddressMode {
    match mode {
        AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
        AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
    }
}

/// Convert goldy FilterMode to Metal MTLSamplerMinMagFilter.
pub fn filter_to_mtl(mode: FilterMode) -> MTLSamplerMinMagFilter {
    match mode {
        FilterMode::Nearest => MTLSamplerMinMagFilter::Nearest,
        FilterMode::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

/// Convert goldy FilterMode to Metal mipmap filter.
pub fn mipmap_mode_to_mtl(mode: FilterMode) -> mtl::MTLSamplerMipFilter {
    match mode {
        FilterMode::Nearest => mtl::MTLSamplerMipFilter::Nearest,
        FilterMode::Linear => mtl::MTLSamplerMipFilter::Linear,
    }
}
