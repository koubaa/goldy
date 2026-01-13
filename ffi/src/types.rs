//! FFI type definitions with repr(C).

use std::ffi::c_char;

/// RGBA color with floating point components (0.0 - 1.0).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl From<GoldyColor> for goldy::Color {
    fn from(c: GoldyColor) -> Self {
        goldy::Color {
            r: c.r,
            g: c.g,
            b: c.b,
            a: c.a,
        }
    }
}

impl From<goldy::Color> for GoldyColor {
    fn from(c: goldy::Color) -> Self {
        GoldyColor {
            r: c.r,
            g: c.g,
            b: c.b,
            a: c.a,
        }
    }
}

/// GPU device type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyDeviceType {
    DiscreteGpu = 0,
    IntegratedGpu = 1,
    Cpu = 2,
    Other = 3,
}

impl From<GoldyDeviceType> for goldy::DeviceType {
    fn from(t: GoldyDeviceType) -> Self {
        match t {
            GoldyDeviceType::DiscreteGpu => goldy::DeviceType::DiscreteGpu,
            GoldyDeviceType::IntegratedGpu => goldy::DeviceType::IntegratedGpu,
            GoldyDeviceType::Cpu => goldy::DeviceType::Cpu,
            GoldyDeviceType::Other => goldy::DeviceType::Other,
        }
    }
}

impl From<goldy::DeviceType> for GoldyDeviceType {
    fn from(t: goldy::DeviceType) -> Self {
        match t {
            goldy::DeviceType::DiscreteGpu => GoldyDeviceType::DiscreteGpu,
            goldy::DeviceType::IntegratedGpu => GoldyDeviceType::IntegratedGpu,
            goldy::DeviceType::Cpu => GoldyDeviceType::Cpu,
            goldy::DeviceType::Other => GoldyDeviceType::Other,
        }
    }
}

/// Graphics backend type.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyBackendType {
    Vulkan = 0,
    Metal = 1,
    Dx12 = 2,
    WebGpu = 3,
}

impl From<goldy::BackendType> for GoldyBackendType {
    fn from(t: goldy::BackendType) -> Self {
        match t {
            goldy::BackendType::Vulkan => GoldyBackendType::Vulkan,
            goldy::BackendType::Metal => GoldyBackendType::Metal,
            goldy::BackendType::Dx12 => GoldyBackendType::Dx12,
            goldy::BackendType::WebGPU => GoldyBackendType::WebGpu,
        }
    }
}

/// Texture format.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyTextureFormat {
    Rgba8UnormSrgb = 0,
    Rgba8Unorm = 1,
    Bgra8UnormSrgb = 2,
    Bgra8Unorm = 3,
    Rgba16Float = 4,
    Rgba32Float = 5,
}

impl From<GoldyTextureFormat> for goldy::TextureFormat {
    fn from(f: GoldyTextureFormat) -> Self {
        match f {
            GoldyTextureFormat::Rgba8UnormSrgb => goldy::TextureFormat::Rgba8UnormSrgb,
            GoldyTextureFormat::Rgba8Unorm => goldy::TextureFormat::Rgba8Unorm,
            GoldyTextureFormat::Bgra8UnormSrgb => goldy::TextureFormat::Bgra8UnormSrgb,
            GoldyTextureFormat::Bgra8Unorm => goldy::TextureFormat::Bgra8Unorm,
            GoldyTextureFormat::Rgba16Float => goldy::TextureFormat::Rgba16Float,
            GoldyTextureFormat::Rgba32Float => goldy::TextureFormat::Rgba32Float,
        }
    }
}

impl From<goldy::TextureFormat> for GoldyTextureFormat {
    fn from(f: goldy::TextureFormat) -> Self {
        match f {
            goldy::TextureFormat::Rgba8UnormSrgb => GoldyTextureFormat::Rgba8UnormSrgb,
            goldy::TextureFormat::Rgba8Unorm => GoldyTextureFormat::Rgba8Unorm,
            goldy::TextureFormat::Bgra8UnormSrgb => GoldyTextureFormat::Bgra8UnormSrgb,
            goldy::TextureFormat::Bgra8Unorm => GoldyTextureFormat::Bgra8Unorm,
            goldy::TextureFormat::Rgba16Float => GoldyTextureFormat::Rgba16Float,
            goldy::TextureFormat::Rgba32Float => GoldyTextureFormat::Rgba32Float,
        }
    }
}

/// Buffer usage flags.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldyBufferUsage(pub u32);

impl GoldyBufferUsage {
    pub const VERTEX: GoldyBufferUsage = GoldyBufferUsage(1 << 0);
    pub const INDEX: GoldyBufferUsage = GoldyBufferUsage(1 << 1);
    pub const UNIFORM: GoldyBufferUsage = GoldyBufferUsage(1 << 2);
    pub const STORAGE: GoldyBufferUsage = GoldyBufferUsage(1 << 3);
    pub const COPY_SRC: GoldyBufferUsage = GoldyBufferUsage(1 << 4);
    pub const COPY_DST: GoldyBufferUsage = GoldyBufferUsage(1 << 5);
}

impl From<GoldyBufferUsage> for goldy::BufferUsage {
    fn from(u: GoldyBufferUsage) -> Self {
        goldy::BufferUsage::from_bits_truncate(u.0)
    }
}

impl From<goldy::BufferUsage> for GoldyBufferUsage {
    fn from(u: goldy::BufferUsage) -> Self {
        GoldyBufferUsage(u.bits())
    }
}

/// Vertex format.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyVertexFormat {
    Float32 = 0,
    Float32x2 = 1,
    Float32x3 = 2,
    Float32x4 = 3,
    Uint32 = 4,
    Sint32 = 5,
    Uint8x4 = 6,
    Unorm8x4 = 7,
}

impl From<GoldyVertexFormat> for goldy::VertexFormat {
    fn from(f: GoldyVertexFormat) -> Self {
        match f {
            GoldyVertexFormat::Float32 => goldy::VertexFormat::Float32,
            GoldyVertexFormat::Float32x2 => goldy::VertexFormat::Float32x2,
            GoldyVertexFormat::Float32x3 => goldy::VertexFormat::Float32x3,
            GoldyVertexFormat::Float32x4 => goldy::VertexFormat::Float32x4,
            GoldyVertexFormat::Uint32 => goldy::VertexFormat::Uint32,
            GoldyVertexFormat::Sint32 => goldy::VertexFormat::Sint32,
            GoldyVertexFormat::Uint8x4 => goldy::VertexFormat::Uint8x4,
            GoldyVertexFormat::Unorm8x4 => goldy::VertexFormat::Unorm8x4,
        }
    }
}

/// Primitive topology.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyPrimitiveTopology {
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    TriangleList = 3,
    TriangleStrip = 4,
}

impl From<GoldyPrimitiveTopology> for goldy::PrimitiveTopology {
    fn from(t: GoldyPrimitiveTopology) -> Self {
        match t {
            GoldyPrimitiveTopology::PointList => goldy::PrimitiveTopology::PointList,
            GoldyPrimitiveTopology::LineList => goldy::PrimitiveTopology::LineList,
            GoldyPrimitiveTopology::LineStrip => goldy::PrimitiveTopology::LineStrip,
            GoldyPrimitiveTopology::TriangleList => goldy::PrimitiveTopology::TriangleList,
            GoldyPrimitiveTopology::TriangleStrip => goldy::PrimitiveTopology::TriangleStrip,
        }
    }
}

/// Index format.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyIndexFormat {
    Uint16 = 0,
    Uint32 = 1,
}

impl From<GoldyIndexFormat> for goldy::IndexFormat {
    fn from(f: GoldyIndexFormat) -> Self {
        match f {
            GoldyIndexFormat::Uint16 => goldy::IndexFormat::Uint16,
            GoldyIndexFormat::Uint32 => goldy::IndexFormat::Uint32,
        }
    }
}

/// Depth format.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyDepthFormat {
    Depth16Unorm = 0,
    Depth24Plus = 1,
    Depth24PlusStencil8 = 2,
    Depth32Float = 3,
    Depth32FloatStencil8 = 4,
}

impl From<GoldyDepthFormat> for goldy::DepthFormat {
    fn from(f: GoldyDepthFormat) -> Self {
        match f {
            GoldyDepthFormat::Depth16Unorm => goldy::DepthFormat::Depth16Unorm,
            GoldyDepthFormat::Depth24Plus => goldy::DepthFormat::Depth24Plus,
            GoldyDepthFormat::Depth24PlusStencil8 => goldy::DepthFormat::Depth24PlusStencil8,
            GoldyDepthFormat::Depth32Float => goldy::DepthFormat::Depth32Float,
            GoldyDepthFormat::Depth32FloatStencil8 => goldy::DepthFormat::Depth32FloatStencil8,
        }
    }
}

/// Comparison function for depth testing.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyCompareFunction {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Always = 7,
}

impl From<GoldyCompareFunction> for goldy::CompareFunction {
    fn from(f: GoldyCompareFunction) -> Self {
        match f {
            GoldyCompareFunction::Never => goldy::CompareFunction::Never,
            GoldyCompareFunction::Less => goldy::CompareFunction::Less,
            GoldyCompareFunction::Equal => goldy::CompareFunction::Equal,
            GoldyCompareFunction::LessEqual => goldy::CompareFunction::LessEqual,
            GoldyCompareFunction::Greater => goldy::CompareFunction::Greater,
            GoldyCompareFunction::NotEqual => goldy::CompareFunction::NotEqual,
            GoldyCompareFunction::GreaterEqual => goldy::CompareFunction::GreaterEqual,
            GoldyCompareFunction::Always => goldy::CompareFunction::Always,
        }
    }
}

/// Vertex attribute description.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyVertexAttribute {
    pub location: u32,
    pub format: GoldyVertexFormat,
    pub offset: u32,
}

impl From<GoldyVertexAttribute> for goldy::VertexAttribute {
    fn from(a: GoldyVertexAttribute) -> Self {
        goldy::VertexAttribute {
            location: a.location,
            format: a.format.into(),
            offset: a.offset,
        }
    }
}

/// Depth/stencil state.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyDepthStencilState {
    pub format: GoldyDepthFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: GoldyCompareFunction,
}

impl From<GoldyDepthStencilState> for goldy::DepthStencilState {
    fn from(s: GoldyDepthStencilState) -> Self {
        goldy::DepthStencilState {
            format: s.format.into(),
            depth_write_enabled: s.depth_write_enabled,
            depth_compare: s.depth_compare.into(),
        }
    }
}

/// Adapter info.
#[repr(C)]
#[derive(Debug)]
pub struct GoldyAdapterInfo {
    pub id: u32,
    pub device_type: GoldyDeviceType,
    pub name: [c_char; 256],
    pub vendor: [c_char; 64],
}

impl GoldyAdapterInfo {
    pub fn from_adapter(adapter: &goldy::Adapter) -> Self {
        let mut info = GoldyAdapterInfo {
            id: adapter.id(),
            device_type: adapter.device_type().into(),
            name: [0; 256],
            vendor: [0; 64],
        };
        
        // Copy name
        let name_bytes = adapter.name().as_bytes();
        let name_len = name_bytes.len().min(255);
        for (i, &b) in name_bytes[..name_len].iter().enumerate() {
            info.name[i] = b as c_char;
        }
        
        // Copy vendor
        let vendor_bytes = adapter.vendor().as_bytes();
        let vendor_len = vendor_bytes.len().min(63);
        for (i, &b) in vendor_bytes[..vendor_len].iter().enumerate() {
            info.vendor[i] = b as c_char;
        }
        
        info
    }
}

/// Texture usage flags.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldyTextureUsage(pub u32);

impl GoldyTextureUsage {
    pub const COPY_SRC: GoldyTextureUsage = GoldyTextureUsage(1 << 0);
    pub const COPY_DST: GoldyTextureUsage = GoldyTextureUsage(1 << 1);
    pub const SAMPLED: GoldyTextureUsage = GoldyTextureUsage(1 << 2);
    pub const STORAGE: GoldyTextureUsage = GoldyTextureUsage(1 << 3);
    pub const RENDER_TARGET: GoldyTextureUsage = GoldyTextureUsage(1 << 4);
}

impl From<GoldyTextureUsage> for goldy::TextureUsage {
    fn from(u: GoldyTextureUsage) -> Self {
        goldy::TextureUsage::from_bits_truncate(u.0)
    }
}

/// Texture addressing mode.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyAddressMode {
    ClampToEdge = 0,
    Repeat = 1,
    MirrorRepeat = 2,
}

impl From<GoldyAddressMode> for goldy::AddressMode {
    fn from(m: GoldyAddressMode) -> Self {
        match m {
            GoldyAddressMode::ClampToEdge => goldy::AddressMode::ClampToEdge,
            GoldyAddressMode::Repeat => goldy::AddressMode::Repeat,
            GoldyAddressMode::MirrorRepeat => goldy::AddressMode::MirrorRepeat,
        }
    }
}

/// Texture filter mode.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyFilterMode {
    Nearest = 0,
    Linear = 1,
}

impl From<GoldyFilterMode> for goldy::FilterMode {
    fn from(m: GoldyFilterMode) -> Self {
        match m {
            GoldyFilterMode::Nearest => goldy::FilterMode::Nearest,
            GoldyFilterMode::Linear => goldy::FilterMode::Linear,
        }
    }
}

/// Sampler descriptor.
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

impl Default for GoldySamplerDesc {
    fn default() -> Self {
        GoldySamplerDesc {
            address_mode_u: GoldyAddressMode::ClampToEdge,
            address_mode_v: GoldyAddressMode::ClampToEdge,
            address_mode_w: GoldyAddressMode::ClampToEdge,
            mag_filter: GoldyFilterMode::Nearest,
            min_filter: GoldyFilterMode::Nearest,
            mipmap_filter: GoldyFilterMode::Nearest,
            max_anisotropy: 1.0,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
        }
    }
}

impl From<GoldySamplerDesc> for goldy::SamplerDesc {
    fn from(d: GoldySamplerDesc) -> Self {
        goldy::SamplerDesc {
            address_mode_u: d.address_mode_u.into(),
            address_mode_v: d.address_mode_v.into(),
            address_mode_w: d.address_mode_w.into(),
            mag_filter: d.mag_filter.into(),
            min_filter: d.min_filter.into(),
            mipmap_filter: d.mipmap_filter.into(),
            max_anisotropy: d.max_anisotropy,
            compare: None,
            lod_min_clamp: d.lod_min_clamp,
            lod_max_clamp: d.lod_max_clamp,
        }
    }
}

/// Shader stages flags.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldyShaderStages(pub u32);

impl GoldyShaderStages {
    pub const VERTEX: GoldyShaderStages = GoldyShaderStages(1);
    pub const FRAGMENT: GoldyShaderStages = GoldyShaderStages(2);
    pub const COMPUTE: GoldyShaderStages = GoldyShaderStages(4);
    pub const ALL: GoldyShaderStages = GoldyShaderStages(7);
}

impl From<GoldyShaderStages> for goldy::ShaderStages {
    fn from(s: GoldyShaderStages) -> Self {
        goldy::ShaderStages(s.0)
    }
}

/// Binding type for bind groups.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoldyBindingType {
    UniformBuffer = 0,
    StorageBufferReadOnly = 1,
    StorageBufferReadWrite = 2,
    Texture = 3,
    Sampler = 4,
    StorageTexture = 5,
}

impl From<GoldyBindingType> for goldy::BindingType {
    fn from(t: GoldyBindingType) -> Self {
        match t {
            GoldyBindingType::UniformBuffer => goldy::BindingType::UniformBuffer,
            GoldyBindingType::StorageBufferReadOnly => goldy::BindingType::StorageBuffer { read_only: true },
            GoldyBindingType::StorageBufferReadWrite => goldy::BindingType::StorageBuffer { read_only: false },
            GoldyBindingType::Texture => goldy::BindingType::Texture,
            GoldyBindingType::Sampler => goldy::BindingType::Sampler,
            GoldyBindingType::StorageTexture => goldy::BindingType::StorageTexture,
        }
    }
}

