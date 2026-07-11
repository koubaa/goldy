//! Vulkan utility functions.
//!
//! Format conversions and memory helpers.

// Allow manual find loops for Vulkan memory type selection (common pattern)
#![allow(clippy::manual_find)]

use crate::types::{
    AddressMode, CompareFunction, DepthFormat, FilterMode, IndexFormat, PrimitiveTopology, TextureFormat, VertexFormat,
};
use ash::vk;

/// Convert Goldy TextureFormat to Vulkan format.
pub fn format_to_vk(format: TextureFormat) -> vk::Format {
    match format {
        TextureFormat::R8Unorm => vk::Format::R8_UNORM,
        TextureFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
    }
}

/// Convert Goldy VertexFormat to Vulkan format.
pub fn vertex_format_to_vk(format: VertexFormat) -> vk::Format {
    match format {
        VertexFormat::Float32 => vk::Format::R32_SFLOAT,
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Uint32 => vk::Format::R32_UINT,
        VertexFormat::Sint32 => vk::Format::R32_SINT,
        VertexFormat::Uint8x4 => vk::Format::R8G8B8A8_UINT,
        VertexFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
    }
}

/// Convert Goldy PrimitiveTopology to Vulkan topology.
pub fn topology_to_vk(topology: PrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

/// Convert Goldy IndexFormat to Vulkan index type.
pub fn index_format_to_vk(format: IndexFormat) -> vk::IndexType {
    match format {
        IndexFormat::Uint16 => vk::IndexType::UINT16,
        IndexFormat::Uint32 => vk::IndexType::UINT32,
    }
}

/// Convert Vulkan format to Goldy TextureFormat.
/// Returns None for unsupported formats.
pub fn vk_to_format(format: vk::Format) -> Option<TextureFormat> {
    match format {
        vk::Format::R8_UNORM => Some(TextureFormat::R8Unorm),
        vk::Format::R8G8_UNORM => Some(TextureFormat::Rg8Unorm),
        vk::Format::R8G8B8A8_SRGB => Some(TextureFormat::Rgba8UnormSrgb),
        vk::Format::R8G8B8A8_UNORM => Some(TextureFormat::Rgba8Unorm),
        vk::Format::B8G8R8A8_SRGB => Some(TextureFormat::Bgra8UnormSrgb),
        vk::Format::B8G8R8A8_UNORM => Some(TextureFormat::Bgra8Unorm),
        vk::Format::R16G16B16A16_SFLOAT => Some(TextureFormat::Rgba16Float),
        vk::Format::R32G32B32A32_SFLOAT => Some(TextureFormat::Rgba32Float),
        _ => None,
    }
}

/// Apply buffer sharing mode for resources that may be used from both the
/// graphics/present queue and a dedicated compute-family context queue.
///
/// When `families` is `Some([graphics, compute])`, uses [`vk::SharingMode::CONCURRENT`]
/// so timeline waits alone are sufficient (no queue-family ownership transfers).
/// When `None` (same family), keeps [`vk::SharingMode::EXCLUSIVE`].
pub fn with_buffer_sharing<'a>(
    info: vk::BufferCreateInfo<'a>,
    families: Option<&'a [u32; 2]>,
) -> vk::BufferCreateInfo<'a> {
    match families {
        Some(f) => info
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(f),
        None => info.sharing_mode(vk::SharingMode::EXCLUSIVE),
    }
}

/// Like [`with_buffer_sharing`] for images.
pub fn with_image_sharing<'a>(
    info: vk::ImageCreateInfo<'a>,
    families: Option<&'a [u32; 2]>,
) -> vk::ImageCreateInfo<'a> {
    match families {
        Some(f) => info
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(f),
        None => info.sharing_mode(vk::SharingMode::EXCLUSIVE),
    }
}

/// Find a suitable memory type for allocation.
pub fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && (mem_props.memory_types[i as usize].property_flags & properties) == properties
        {
            return Some(i);
        }
    }
    None
}

/// Convert Goldy DepthFormat to Vulkan format.
pub fn depth_format_to_vk(format: DepthFormat) -> vk::Format {
    match format {
        DepthFormat::Depth16Unorm => vk::Format::D16_UNORM,
        DepthFormat::Depth24Plus => vk::Format::D32_SFLOAT, // Use D32 for better precision
        DepthFormat::Depth24PlusStencil8 => vk::Format::D24_UNORM_S8_UINT,
        DepthFormat::Depth32Float => vk::Format::D32_SFLOAT,
        DepthFormat::Depth32FloatStencil8 => vk::Format::D32_SFLOAT_S8_UINT,
    }
}

/// Get the aspect mask for a depth format.
pub fn depth_aspect_mask(format: DepthFormat) -> vk::ImageAspectFlags {
    if format.has_stencil() {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::DEPTH
    }
}

/// Convert Goldy CompareFunction to Vulkan compare op.
pub fn compare_to_vk(compare: CompareFunction) -> vk::CompareOp {
    match compare {
        CompareFunction::Never => vk::CompareOp::NEVER,
        CompareFunction::Less => vk::CompareOp::LESS,
        CompareFunction::Equal => vk::CompareOp::EQUAL,
        CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareFunction::Greater => vk::CompareOp::GREATER,
        CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}

/// Convert Goldy AddressMode to Vulkan sampler address mode.
pub fn address_mode_to_vk(mode: AddressMode) -> vk::SamplerAddressMode {
    match mode {
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

/// Convert Goldy FilterMode to Vulkan filter.
pub fn filter_to_vk(mode: FilterMode) -> vk::Filter {
    match mode {
        FilterMode::Nearest => vk::Filter::NEAREST,
        FilterMode::Linear => vk::Filter::LINEAR,
    }
}

/// Convert Goldy FilterMode to Vulkan sampler mipmap mode.
pub fn mipmap_mode_to_vk(mode: FilterMode) -> vk::SamplerMipmapMode {
    match mode {
        FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}
