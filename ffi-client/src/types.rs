//! Rust-friendly types mirroring the native Goldy API.

use crate::sys::{
    self, GoldyBufferKind, GoldyCompareFunction, GoldyDepthFormat, GoldyIndexFormat, GoldyNodeAccess,
    GoldyPrimitiveTopology, GoldyTextureFormat, GoldyVertexFormat,
};
use bytemuck::{Pod, Zeroable};

/// RGBA color with floating point components (0.0 - 1.0).
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const RED: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const GREEN: Color = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    pub const BLUE: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
}

impl From<Color> for sys::GoldyColor {
    fn from(c: Color) -> Self {
        Self {
            r: c.r,
            g: c.g,
            b: c.b,
            a: c.a,
        }
    }
}

/// Texture format for render targets and surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextureFormat {
    R8Unorm,
    Rg8Unorm,
    Rgba8UnormSrgb,
    #[default]
    Rgba8Unorm,
    Bgra8UnormSrgb,
    Bgra8Unorm,
    Rgba16Float,
    Rgba32Float,
}

impl From<TextureFormat> for GoldyTextureFormat {
    fn from(f: TextureFormat) -> Self {
        match f {
            TextureFormat::Rgba8UnormSrgb => GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_RGBA8_UNORM_SRGB,
            TextureFormat::Rgba8Unorm => GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_RGBA8_UNORM,
            TextureFormat::Bgra8UnormSrgb => GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_BGRA8_UNORM_SRGB,
            TextureFormat::Bgra8Unorm => GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_BGRA8_UNORM,
            TextureFormat::Rgba16Float => GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_RGBA16_FLOAT,
            TextureFormat::Rgba32Float => GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_RGBA32_FLOAT,
            TextureFormat::R8Unorm => GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_R8_UNORM,
            TextureFormat::Rg8Unorm => GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_RG8_UNORM,
        }
    }
}

impl From<GoldyTextureFormat> for TextureFormat {
    fn from(f: GoldyTextureFormat) -> Self {
        match f {
            GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_RGBA8_UNORM_SRGB => TextureFormat::Rgba8UnormSrgb,
            GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_RGBA8_UNORM => TextureFormat::Rgba8Unorm,
            GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_BGRA8_UNORM_SRGB => TextureFormat::Bgra8UnormSrgb,
            GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_BGRA8_UNORM => TextureFormat::Bgra8Unorm,
            GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_RGBA16_FLOAT => TextureFormat::Rgba16Float,
            GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_RGBA32_FLOAT => TextureFormat::Rgba32Float,
            GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_R8_UNORM => TextureFormat::R8Unorm,
            GoldyTextureFormat::GOLDY_TEXTURE_FORMAT_RG8_UNORM => TextureFormat::Rg8Unorm,
        }
    }
}

/// Spatial access pattern for textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureKind {
    Interpolated,
    Direct,
    DirectInterpolated,
}

impl From<TextureKind> for sys::GoldyTextureKind {
    fn from(k: TextureKind) -> Self {
        match k {
            TextureKind::Interpolated => sys::GoldyTextureKind::GOLDY_TEXTURE_KIND_INTERPOLATED,
            TextureKind::Direct => sys::GoldyTextureKind::GOLDY_TEXTURE_KIND_DIRECT,
            TextureKind::DirectInterpolated => sys::GoldyTextureKind::GOLDY_TEXTURE_KIND_DIRECT_INTERPOLATED,
        }
    }
}

/// Texture flags for copy and render operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureFlags(pub u32);

impl TextureFlags {
    pub const NONE: Self = Self(0);
    pub const COPY_SRC: Self = Self(1 << 0);
    pub const COPY_DST: Self = Self(1 << 1);
    pub const RENDER_TARGET: Self = Self(1 << 2);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl From<TextureFlags> for sys::GoldyTextureFlags {
    fn from(f: TextureFlags) -> Self {
        sys::GoldyTextureFlags { _0: f.0 }
    }
}

/// Buffer data access pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BufferKind {
    #[default]
    Scattered,
    Broadcast,
}

impl From<BufferKind> for GoldyBufferKind {
    fn from(k: BufferKind) -> Self {
        match k {
            BufferKind::Scattered => GoldyBufferKind::GOLDY_BUFFER_KIND_SCATTERED,
            BufferKind::Broadcast => GoldyBufferKind::GOLDY_BUFFER_KIND_BROADCAST,
        }
    }
}

/// Index buffer element format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

impl From<IndexFormat> for GoldyIndexFormat {
    fn from(f: IndexFormat) -> Self {
        match f {
            IndexFormat::Uint16 => GoldyIndexFormat::GOLDY_INDEX_FORMAT_UINT16,
            IndexFormat::Uint32 => GoldyIndexFormat::GOLDY_INDEX_FORMAT_UINT32,
        }
    }
}

/// Per-node resource access for task graph bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeAccess {
    Read,
    Write,
    ReadWrite,
    Overwrite,
}

impl From<NodeAccess> for GoldyNodeAccess {
    fn from(a: NodeAccess) -> Self {
        match a {
            NodeAccess::Read => GoldyNodeAccess::GOLDY_NODE_ACCESS_READ,
            NodeAccess::Write => GoldyNodeAccess::GOLDY_NODE_ACCESS_WRITE,
            NodeAccess::ReadWrite => GoldyNodeAccess::GOLDY_NODE_ACCESS_READ_WRITE,
            NodeAccess::Overwrite => GoldyNodeAccess::GOLDY_NODE_ACCESS_OVERWRITE,
        }
    }
}

/// Bindless resource category for [`ResourceHandle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceCategory {
    Scattered = 0,
    Broadcast = 1,
    StorageImage = 2,
    Texture = 3,
    Sampler = 4,
}

/// Typed bindless handle (category + slot index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceHandle {
    pub category: ResourceCategory,
    pub index: u32,
}

pub enum ResourceAccess {
    Read,
    Write,
    ReadWrite,
}

impl From<ResourceAccess> for sys::GoldyResourceAccess {
    fn from(a: ResourceAccess) -> Self {
        match a {
            ResourceAccess::Read => sys::GoldyResourceAccess::GOLDY_RESOURCE_ACCESS_READ,
            ResourceAccess::Write => sys::GoldyResourceAccess::GOLDY_RESOURCE_ACCESS_WRITE,
            ResourceAccess::ReadWrite => sys::GoldyResourceAccess::GOLDY_RESOURCE_ACCESS_READ_WRITE,
        }
    }
}

/// GPU device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    DiscreteGpu,
    IntegratedGpu,
    Cpu,
    Other,
}

impl From<sys::GoldyDeviceType> for DeviceType {
    fn from(t: sys::GoldyDeviceType) -> Self {
        match t {
            sys::GoldyDeviceType::GOLDY_DEVICE_TYPE_DISCRETE_GPU => DeviceType::DiscreteGpu,
            sys::GoldyDeviceType::GOLDY_DEVICE_TYPE_INTEGRATED_GPU => DeviceType::IntegratedGpu,
            sys::GoldyDeviceType::GOLDY_DEVICE_TYPE_CPU => DeviceType::Cpu,
            _ => DeviceType::Other,
        }
    }
}

/// Power preference for adapter selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PowerPreference {
    #[default]
    None,
    LowPower,
    HighPerformance,
}

/// Options for [`crate::Instance::request_adapter`].
#[derive(Debug, Clone, Default)]
pub struct RequestAdapterOptions {
    pub power_preference: PowerPreference,
    pub force_fallback_adapter: bool,
}

/// Descriptor for [`crate::Adapter::request_device`].
#[derive(Debug, Clone, Default)]
pub struct DeviceDescriptor {
    pub label: Option<String>,
}

/// Vertex format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Sint32,
    Uint8x4,
    Unorm8x4,
}

impl VertexFormat {
    pub fn size(self) -> u32 {
        match self {
            VertexFormat::Float32 => 4,
            VertexFormat::Float32x2 => 8,
            VertexFormat::Float32x3 => 12,
            VertexFormat::Float32x4 => 16,
            VertexFormat::Uint32 | VertexFormat::Sint32 => 4,
            VertexFormat::Uint8x4 | VertexFormat::Unorm8x4 => 4,
        }
    }
}

impl From<VertexFormat> for GoldyVertexFormat {
    fn from(f: VertexFormat) -> Self {
        match f {
            VertexFormat::Float32 => GoldyVertexFormat::GOLDY_VERTEX_FORMAT_FLOAT32,
            VertexFormat::Float32x2 => GoldyVertexFormat::GOLDY_VERTEX_FORMAT_FLOAT32X2,
            VertexFormat::Float32x3 => GoldyVertexFormat::GOLDY_VERTEX_FORMAT_FLOAT32X3,
            VertexFormat::Float32x4 => GoldyVertexFormat::GOLDY_VERTEX_FORMAT_FLOAT32X4,
            VertexFormat::Uint32 => GoldyVertexFormat::GOLDY_VERTEX_FORMAT_UINT32,
            VertexFormat::Sint32 => GoldyVertexFormat::GOLDY_VERTEX_FORMAT_SINT32,
            VertexFormat::Uint8x4 => GoldyVertexFormat::GOLDY_VERTEX_FORMAT_UINT8X4,
            VertexFormat::Unorm8x4 => GoldyVertexFormat::GOLDY_VERTEX_FORMAT_UNORM8X4,
        }
    }
}

/// Vertex attribute description.
#[derive(Debug, Clone)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: VertexFormat,
    pub offset: u32,
}

/// Vertex buffer layout.
#[derive(Debug, Clone)]
pub struct VertexBufferLayout {
    pub stride: u32,
    pub attributes: Vec<VertexAttribute>,
}

impl VertexBufferLayout {
    pub fn from_formats<T>(formats: &[VertexFormat]) -> Self {
        let mut offset = 0u32;
        let attributes: Vec<VertexAttribute> = formats
            .iter()
            .enumerate()
            .map(|(i, fmt)| {
                let attr = VertexAttribute {
                    location: i as u32,
                    offset,
                    format: *fmt,
                };
                offset += fmt.size();
                attr
            })
            .collect();
        let expected_stride = std::mem::size_of::<T>() as u32;
        assert_eq!(
            offset,
            expected_stride,
            "VertexBufferLayout::from_formats: sum of format sizes ({offset}) != \
             size_of::<{}>() ({expected_stride})",
            std::any::type_name::<T>(),
        );
        Self {
            stride: expected_stride,
            attributes,
        }
    }
}

/// Primitive topology for drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    #[default]
    TriangleList,
    TriangleStrip,
}

impl From<PrimitiveTopology> for GoldyPrimitiveTopology {
    fn from(t: PrimitiveTopology) -> Self {
        match t {
            PrimitiveTopology::PointList => GoldyPrimitiveTopology::GOLDY_PRIMITIVE_TOPOLOGY_POINT_LIST,
            PrimitiveTopology::LineList => GoldyPrimitiveTopology::GOLDY_PRIMITIVE_TOPOLOGY_LINE_LIST,
            PrimitiveTopology::LineStrip => GoldyPrimitiveTopology::GOLDY_PRIMITIVE_TOPOLOGY_LINE_STRIP,
            PrimitiveTopology::TriangleList => GoldyPrimitiveTopology::GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            PrimitiveTopology::TriangleStrip => GoldyPrimitiveTopology::GOLDY_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        }
    }
}

/// Depth format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DepthFormat {
    Depth16Unorm,
    #[default]
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
    Depth32FloatStencil8,
}

impl From<DepthFormat> for GoldyDepthFormat {
    fn from(f: DepthFormat) -> Self {
        match f {
            DepthFormat::Depth16Unorm => GoldyDepthFormat::GOLDY_DEPTH_FORMAT_DEPTH16_UNORM,
            DepthFormat::Depth24Plus => GoldyDepthFormat::GOLDY_DEPTH_FORMAT_DEPTH24_PLUS,
            DepthFormat::Depth24PlusStencil8 => GoldyDepthFormat::GOLDY_DEPTH_FORMAT_DEPTH24_PLUS_STENCIL8,
            DepthFormat::Depth32Float => GoldyDepthFormat::GOLDY_DEPTH_FORMAT_DEPTH32_FLOAT,
            DepthFormat::Depth32FloatStencil8 => GoldyDepthFormat::GOLDY_DEPTH_FORMAT_DEPTH32_FLOAT_STENCIL8,
        }
    }
}

/// Comparison function for depth testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompareFunction {
    Never,
    #[default]
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl From<CompareFunction> for GoldyCompareFunction {
    fn from(f: CompareFunction) -> Self {
        match f {
            CompareFunction::Never => GoldyCompareFunction::GOLDY_COMPARE_FUNCTION_NEVER,
            CompareFunction::Less => GoldyCompareFunction::GOLDY_COMPARE_FUNCTION_LESS,
            CompareFunction::Equal => GoldyCompareFunction::GOLDY_COMPARE_FUNCTION_EQUAL,
            CompareFunction::LessEqual => GoldyCompareFunction::GOLDY_COMPARE_FUNCTION_LESS_EQUAL,
            CompareFunction::Greater => GoldyCompareFunction::GOLDY_COMPARE_FUNCTION_GREATER,
            CompareFunction::NotEqual => GoldyCompareFunction::GOLDY_COMPARE_FUNCTION_NOT_EQUAL,
            CompareFunction::GreaterEqual => GoldyCompareFunction::GOLDY_COMPARE_FUNCTION_GREATER_EQUAL,
            CompareFunction::Always => GoldyCompareFunction::GOLDY_COMPARE_FUNCTION_ALWAYS,
        }
    }
}

/// Depth/stencil state for render pipelines.
#[derive(Debug, Clone, Default)]
pub struct DepthStencilState {
    pub format: DepthFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: CompareFunction,
}

/// Description for creating a render pipeline.
#[derive(Clone, Debug)]
pub struct RenderPipelineDesc {
    pub vertex_layout: VertexBufferLayout,
    pub topology: PrimitiveTopology,
    pub target_format: TextureFormat,
    pub depth_stencil: Option<DepthStencilState>,
}

impl Default for RenderPipelineDesc {
    fn default() -> Self {
        Self {
            vertex_layout: VertexBufferLayout {
                stride: 0,
                attributes: Vec::new(),
            },
            topology: PrimitiveTopology::TriangleList,
            target_format: TextureFormat::Rgba8Unorm,
            depth_stencil: None,
        }
    }
}

/// A 2D vertex with position and color.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex2D {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex2D {
    pub const fn new(x: f32, y: f32, color: Color) -> Self {
        Self {
            position: [x, y],
            color: [color.r, color.g, color.b, color.a],
        }
    }

    pub fn layout() -> VertexBufferLayout {
        VertexBufferLayout::from_formats::<Self>(&[VertexFormat::Float32x2, VertexFormat::Float32x4])
    }
}

pub(crate) fn render_pipeline_desc_to_ffi(
    desc: &RenderPipelineDesc,
) -> (sys::GoldyRenderPipelineDesc, Vec<sys::GoldyVertexAttribute>) {
    let attributes: Vec<sys::GoldyVertexAttribute> = desc
        .vertex_layout
        .attributes
        .iter()
        .map(|a| sys::GoldyVertexAttribute {
            location: a.location,
            format: a.format.into(),
            offset: a.offset,
        })
        .collect();

    let (depth_enabled, depth_format, depth_write_enabled, depth_compare) = if let Some(ds) = &desc.depth_stencil {
        (true, ds.format.into(), ds.depth_write_enabled, ds.depth_compare.into())
    } else {
        (
            false,
            GoldyDepthFormat::GOLDY_DEPTH_FORMAT_DEPTH24_PLUS,
            true,
            GoldyCompareFunction::GOLDY_COMPARE_FUNCTION_LESS,
        )
    };

    let ffi = sys::GoldyRenderPipelineDesc {
        vertex_attributes: attributes.as_ptr(),
        vertex_attribute_count: attributes.len() as u32,
        vertex_stride: desc.vertex_layout.stride,
        topology: desc.topology.into(),
        target_format: desc.target_format.into(),
        depth_enabled,
        depth_format,
        depth_write_enabled,
        depth_compare,
    };

    (ffi, attributes)
}
