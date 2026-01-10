//! Vulkan utility functions.
//!
//! Format conversions and memory helpers.

use crate::types::{IndexFormat, TextureFormat, VertexFormat, PrimitiveTopology};
use ash::vk;

/// Convert RAG TextureFormat to Vulkan format.
pub fn format_to_vk(format: TextureFormat) -> vk::Format {
    match format {
        TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
    }
}

/// Convert RAG VertexFormat to Vulkan format.
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

/// Convert RAG PrimitiveTopology to Vulkan topology.
pub fn topology_to_vk(topology: PrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

/// Convert RAG IndexFormat to Vulkan index type.
pub fn index_format_to_vk(format: IndexFormat) -> vk::IndexType {
    match format {
        IndexFormat::Uint16 => vk::IndexType::UINT16,
        IndexFormat::Uint32 => vk::IndexType::UINT32,
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

/// Map vendor ID to vendor name.
pub fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x1002 | 0x1022 => "AMD",
        0x10DE => "NVIDIA",
        0x8086 => "Intel",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x106B => "Apple",
        _ => "Unknown",
    }
}

/// Map Vulkan physical device type to RAG DeviceType.
pub fn device_type_from_vk(vk_type: vk::PhysicalDeviceType) -> crate::types::DeviceType {
    match vk_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => crate::types::DeviceType::DiscreteGpu,
        vk::PhysicalDeviceType::INTEGRATED_GPU => crate::types::DeviceType::IntegratedGpu,
        vk::PhysicalDeviceType::CPU => crate::types::DeviceType::Cpu,
        _ => crate::types::DeviceType::Other,
    }
}

