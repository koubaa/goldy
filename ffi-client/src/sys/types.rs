//! C ABI types mirroring `goldy.h`.

use std::ffi::c_char;

pub enum GoldyBuffer {}
pub enum GoldyComputeEncoder {}
pub enum GoldyComputePipeline {}
pub enum GoldyDevice {}
pub enum GoldyInstance {}
pub enum GoldyRenderPipeline {}
pub enum GoldyRenderTarget {}
pub enum GoldySampler {}
pub enum GoldyShaderModule {}
pub enum GoldySurface {}
pub enum GoldySurfaceFrame {}
pub enum GoldyTaskGraph {}
pub enum GoldyTexture {}

#[repr(C)]
pub struct GoldySwapchainOutput {
    pub _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyBufferKind {
    GOLDY_BUFFER_KIND_SCATTERED = 0,
    GOLDY_BUFFER_KIND_BROADCAST = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyResult {
    GOLDY_RESULT_OK = 0,
    GOLDY_RESULT_INVALID_ARGUMENT = 1,
    GOLDY_RESULT_NULL_POINTER = 2,
    GOLDY_RESULT_GPU_ERROR = 3,
    GOLDY_RESULT_SHADER_ERROR = 4,
    GOLDY_RESULT_RESOURCE_ERROR = 5,
    GOLDY_RESULT_INTERNAL_ERROR = 6,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyBackendType {
    GOLDY_BACKEND_TYPE_VULKAN = 0,
    GOLDY_BACKEND_TYPE_METAL = 1,
    GOLDY_BACKEND_TYPE_DX12 = 2,
    GOLDY_BACKEND_TYPE_WEB_GPU = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyDeviceType {
    GOLDY_DEVICE_TYPE_DISCRETE_GPU = 0,
    GOLDY_DEVICE_TYPE_INTEGRATED_GPU = 1,
    GOLDY_DEVICE_TYPE_CPU = 2,
    GOLDY_DEVICE_TYPE_OTHER = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyVertexFormat {
    GOLDY_VERTEX_FORMAT_FLOAT32 = 0,
    GOLDY_VERTEX_FORMAT_FLOAT32X2 = 1,
    GOLDY_VERTEX_FORMAT_FLOAT32X3 = 2,
    GOLDY_VERTEX_FORMAT_FLOAT32X4 = 3,
    GOLDY_VERTEX_FORMAT_UINT32 = 4,
    GOLDY_VERTEX_FORMAT_SINT32 = 5,
    GOLDY_VERTEX_FORMAT_UINT8X4 = 6,
    GOLDY_VERTEX_FORMAT_UNORM8X4 = 7,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyPrimitiveTopology {
    GOLDY_PRIMITIVE_TOPOLOGY_POINT_LIST = 0,
    GOLDY_PRIMITIVE_TOPOLOGY_LINE_LIST = 1,
    GOLDY_PRIMITIVE_TOPOLOGY_LINE_STRIP = 2,
    GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3,
    GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyTextureFormat {
    GOLDY_TEXTURE_FORMAT_RGBA8_UNORM_SRGB = 0,
    GOLDY_TEXTURE_FORMAT_RGBA8_UNORM = 1,
    GOLDY_TEXTURE_FORMAT_BGRA8_UNORM_SRGB = 2,
    GOLDY_TEXTURE_FORMAT_BGRA8_UNORM = 3,
    GOLDY_TEXTURE_FORMAT_RGBA16_FLOAT = 4,
    GOLDY_TEXTURE_FORMAT_RGBA32_FLOAT = 5,
    GOLDY_TEXTURE_FORMAT_R8_UNORM = 6,
    GOLDY_TEXTURE_FORMAT_RG8_UNORM = 7,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyDepthFormat {
    GOLDY_DEPTH_FORMAT_DEPTH16_UNORM = 0,
    GOLDY_DEPTH_FORMAT_DEPTH24_PLUS = 1,
    GOLDY_DEPTH_FORMAT_DEPTH24_PLUS_STENCIL8 = 2,
    GOLDY_DEPTH_FORMAT_DEPTH32_FLOAT = 3,
    GOLDY_DEPTH_FORMAT_DEPTH32_FLOAT_STENCIL8 = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyCompareFunction {
    GOLDY_COMPARE_FUNCTION_NEVER = 0,
    GOLDY_COMPARE_FUNCTION_LESS = 1,
    GOLDY_COMPARE_FUNCTION_EQUAL = 2,
    GOLDY_COMPARE_FUNCTION_LESS_EQUAL = 3,
    GOLDY_COMPARE_FUNCTION_GREATER = 4,
    GOLDY_COMPARE_FUNCTION_NOT_EQUAL = 5,
    GOLDY_COMPARE_FUNCTION_GREATER_EQUAL = 6,
    GOLDY_COMPARE_FUNCTION_ALWAYS = 7,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyAddressMode {
    GOLDY_ADDRESS_MODE_CLAMP_TO_EDGE = 0,
    GOLDY_ADDRESS_MODE_REPEAT = 1,
    GOLDY_ADDRESS_MODE_MIRROR_REPEAT = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyFilterMode {
    GOLDY_FILTER_MODE_NEAREST = 0,
    GOLDY_FILTER_MODE_LINEAR = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyNodeAccess {
    GOLDY_NODE_ACCESS_READ = 0,
    GOLDY_NODE_ACCESS_WRITE = 1,
    GOLDY_NODE_ACCESS_READ_WRITE = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyResourceAccess {
    GOLDY_RESOURCE_ACCESS_READ = 0,
    GOLDY_RESOURCE_ACCESS_WRITE = 1,
    GOLDY_RESOURCE_ACCESS_READ_WRITE = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyIndexFormat {
    GOLDY_INDEX_FORMAT_UINT16 = 0,
    GOLDY_INDEX_FORMAT_UINT32 = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyTextureKind {
    GOLDY_TEXTURE_KIND_INTERPOLATED = 0,
    GOLDY_TEXTURE_KIND_DIRECT = 1,
    GOLDY_TEXTURE_KIND_DIRECT_INTERPOLATED = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyAdapterInfo {
    pub id: u32,
    pub device_type: GoldyDeviceType,
    pub name: [c_char; 256],
    pub vendor: [c_char; 64],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyVertexAttribute {
    pub location: u32,
    pub format: GoldyVertexFormat,
    pub offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyRenderPipelineDesc {
    pub vertex_attributes: *const GoldyVertexAttribute,
    pub vertex_attribute_count: u32,
    pub vertex_stride: u32,
    pub topology: GoldyPrimitiveTopology,
    pub target_format: GoldyTextureFormat,
    pub depth_enabled: bool,
    pub depth_format: GoldyDepthFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: GoldyCompareFunction,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldySamplerDesc {
    pub address_mode_u: GoldyAddressMode,
    pub address_mode_v: GoldyAddressMode,
    pub address_mode_w: GoldyAddressMode,
    pub mag_filter: GoldyFilterMode,
    pub min_filter: GoldyFilterMode,
    pub mipmap_filter: GoldyFilterMode,
    pub max_anisotropy: f32,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldyTextureFlags {
    pub _0: u32,
}
