//! Python wrappers for Goldy types and enums.

#![allow(non_camel_case_types)]

use pyo3::prelude::*;

// =============================================================================
// DeviceType
// =============================================================================

/// Type of GPU device.
#[pyclass(name = "DeviceType", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyDeviceType {
    /// Discrete GPU (dedicated graphics card).
    DISCRETE_GPU = 0,
    /// Integrated GPU (part of CPU).
    INTEGRATED_GPU = 1,
    /// Software renderer (CPU).
    CPU = 2,
    /// Other/unknown.
    OTHER = 3,
}

impl From<PyDeviceType> for goldy::DeviceType {
    fn from(dt: PyDeviceType) -> Self {
        match dt {
            PyDeviceType::DISCRETE_GPU => goldy::DeviceType::DiscreteGpu,
            PyDeviceType::INTEGRATED_GPU => goldy::DeviceType::IntegratedGpu,
            PyDeviceType::CPU => goldy::DeviceType::Cpu,
            PyDeviceType::OTHER => goldy::DeviceType::Other,
        }
    }
}

impl From<goldy::DeviceType> for PyDeviceType {
    fn from(dt: goldy::DeviceType) -> Self {
        match dt {
            goldy::DeviceType::DiscreteGpu => PyDeviceType::DISCRETE_GPU,
            goldy::DeviceType::IntegratedGpu => PyDeviceType::INTEGRATED_GPU,
            goldy::DeviceType::Cpu => PyDeviceType::CPU,
            goldy::DeviceType::Other => PyDeviceType::OTHER,
        }
    }
}

// =============================================================================
// BackendType
// =============================================================================

/// Graphics backend type.
#[pyclass(name = "BackendType", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyBackendType {
    VULKAN = 0,
    METAL = 1,
    DX12 = 2,
    WEBGPU = 3,
}

impl From<goldy::BackendType> for PyBackendType {
    fn from(bt: goldy::BackendType) -> Self {
        match bt {
            goldy::BackendType::Vulkan => PyBackendType::VULKAN,
            goldy::BackendType::Metal => PyBackendType::METAL,
            goldy::BackendType::Dx12 => PyBackendType::DX12,
            goldy::BackendType::WebGPU => PyBackendType::WEBGPU,
        }
    }
}

// =============================================================================
// TextureFormat
// =============================================================================

/// Texture format for render targets.
#[pyclass(name = "TextureFormat", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub enum PyTextureFormat {
    /// 8-bit RGBA, sRGB color space.
    RGBA8_UNORM_SRGB = 0,
    /// 8-bit RGBA, linear color space.
    #[default]
    RGBA8_UNORM = 1,
    /// 8-bit BGRA, sRGB color space.
    BGRA8_UNORM_SRGB = 2,
    /// 8-bit BGRA, linear color space.
    BGRA8_UNORM = 3,
    /// 16-bit RGBA float.
    RGBA16_FLOAT = 4,
    /// 32-bit RGBA float.
    RGBA32_FLOAT = 5,
    /// Single-channel 8-bit unsigned normalized.
    R8_UNORM = 6,
    /// Two-channel 8-bit unsigned normalized.
    RG8_UNORM = 7,
}

impl From<PyTextureFormat> for goldy::TextureFormat {
    fn from(tf: PyTextureFormat) -> Self {
        match tf {
            PyTextureFormat::R8_UNORM => goldy::TextureFormat::R8Unorm,
            PyTextureFormat::RG8_UNORM => goldy::TextureFormat::Rg8Unorm,
            PyTextureFormat::RGBA8_UNORM_SRGB => goldy::TextureFormat::Rgba8UnormSrgb,
            PyTextureFormat::RGBA8_UNORM => goldy::TextureFormat::Rgba8Unorm,
            PyTextureFormat::BGRA8_UNORM_SRGB => goldy::TextureFormat::Bgra8UnormSrgb,
            PyTextureFormat::BGRA8_UNORM => goldy::TextureFormat::Bgra8Unorm,
            PyTextureFormat::RGBA16_FLOAT => goldy::TextureFormat::Rgba16Float,
            PyTextureFormat::RGBA32_FLOAT => goldy::TextureFormat::Rgba32Float,
        }
    }
}

impl From<goldy::TextureFormat> for PyTextureFormat {
    fn from(tf: goldy::TextureFormat) -> Self {
        match tf {
            goldy::TextureFormat::R8Unorm => PyTextureFormat::R8_UNORM,
            goldy::TextureFormat::Rg8Unorm => PyTextureFormat::RG8_UNORM,
            goldy::TextureFormat::Rgba8UnormSrgb => PyTextureFormat::RGBA8_UNORM_SRGB,
            goldy::TextureFormat::Rgba8Unorm => PyTextureFormat::RGBA8_UNORM,
            goldy::TextureFormat::Bgra8UnormSrgb => PyTextureFormat::BGRA8_UNORM_SRGB,
            goldy::TextureFormat::Bgra8Unorm => PyTextureFormat::BGRA8_UNORM,
            goldy::TextureFormat::Rgba16Float => PyTextureFormat::RGBA16_FLOAT,
            goldy::TextureFormat::Rgba32Float => PyTextureFormat::RGBA32_FLOAT,
        }
    }
}

// =============================================================================
// BufferKind
// =============================================================================

/// Data access pattern for buffers.
///
/// Describes how threads will access the buffer, which determines hardware optimization strategies:
/// - SCATTERED: Any thread can access any address (read/write). No coherence assumptions.
/// - BROADCAST: All threads read same address. Hardware broadcast optimization.
#[pyclass(name = "BufferKind", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PyBufferKind {
    /// Any thread, any address, read/write (StructuredBuffer, RWStructuredBuffer).
    #[default]
    SCATTERED = 0,
    /// All threads same address, broadcast optimized (ConstantBuffer).
    BROADCAST = 1,
}

impl From<PyBufferKind> for goldy::BufferKind {
    fn from(access: PyBufferKind) -> Self {
        match access {
            PyBufferKind::SCATTERED => goldy::BufferKind::Scattered,
            PyBufferKind::BROADCAST => goldy::BufferKind::Broadcast,
        }
    }
}

impl From<goldy::BufferKind> for PyBufferKind {
    fn from(access: goldy::BufferKind) -> Self {
        match access {
            goldy::BufferKind::Scattered => PyBufferKind::SCATTERED,
            goldy::BufferKind::Broadcast => PyBufferKind::BROADCAST,
        }
    }
}

// =============================================================================
// TextureKind
// =============================================================================

/// Spatial access pattern for textures.
///
/// Describes how the texture will be accessed, which determines hardware optimization strategies:
/// - INTERPOLATED: Hardware filtering between neighbors (texture units).
/// - DIRECT: Direct 2D/3D indexing, no filtering, read/write.
/// - DIRECT_INTERPOLATED: Both storage (UAV) and sampled (SRV) access on the same texture.
#[pyclass(name = "TextureKind", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PyTextureKind {
    /// Hardware filtering between neighbors (Texture2D with sampler).
    #[default]
    INTERPOLATED = 0,
    /// Direct 2D/3D indexing, no filtering (RWTexture2D).
    DIRECT = 1,
    /// Both UAV (storage/write) and SRV (sampled/read) access on the same texture.
    DIRECT_INTERPOLATED = 2,
}

impl From<PyTextureKind> for goldy::TextureKind {
    fn from(access: PyTextureKind) -> Self {
        match access {
            PyTextureKind::INTERPOLATED => goldy::TextureKind::Interpolated,
            PyTextureKind::DIRECT => goldy::TextureKind::Direct,
            PyTextureKind::DIRECT_INTERPOLATED => goldy::TextureKind::DirectInterpolated,
        }
    }
}

impl From<goldy::TextureKind> for PyTextureKind {
    fn from(access: goldy::TextureKind) -> Self {
        match access {
            goldy::TextureKind::Interpolated => PyTextureKind::INTERPOLATED,
            goldy::TextureKind::Direct => PyTextureKind::DIRECT,
            goldy::TextureKind::DirectInterpolated => PyTextureKind::DIRECT_INTERPOLATED,
        }
    }
}

// =============================================================================
// VertexFormat
// =============================================================================

/// Vertex attribute format.
#[pyclass(name = "VertexFormat", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyVertexFormat {
    FLOAT32 = 0,
    FLOAT32X2 = 1,
    FLOAT32X3 = 2,
    FLOAT32X4 = 3,
    UINT32 = 4,
    SINT32 = 5,
    UINT8X4 = 6,
    UNORM8X4 = 7,
}

impl From<PyVertexFormat> for goldy::VertexFormat {
    fn from(vf: PyVertexFormat) -> Self {
        match vf {
            PyVertexFormat::FLOAT32 => goldy::VertexFormat::Float32,
            PyVertexFormat::FLOAT32X2 => goldy::VertexFormat::Float32x2,
            PyVertexFormat::FLOAT32X3 => goldy::VertexFormat::Float32x3,
            PyVertexFormat::FLOAT32X4 => goldy::VertexFormat::Float32x4,
            PyVertexFormat::UINT32 => goldy::VertexFormat::Uint32,
            PyVertexFormat::SINT32 => goldy::VertexFormat::Sint32,
            PyVertexFormat::UINT8X4 => goldy::VertexFormat::Uint8x4,
            PyVertexFormat::UNORM8X4 => goldy::VertexFormat::Unorm8x4,
        }
    }
}

// =============================================================================
// PrimitiveTopology
// =============================================================================

/// Primitive topology for drawing.
#[pyclass(name = "PrimitiveTopology", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub enum PyPrimitiveTopology {
    POINT_LIST = 0,
    LINE_LIST = 1,
    LINE_STRIP = 2,
    #[default]
    TRIANGLE_LIST = 3,
    TRIANGLE_STRIP = 4,
}

impl From<PyPrimitiveTopology> for goldy::PrimitiveTopology {
    fn from(pt: PyPrimitiveTopology) -> Self {
        match pt {
            PyPrimitiveTopology::POINT_LIST => goldy::PrimitiveTopology::PointList,
            PyPrimitiveTopology::LINE_LIST => goldy::PrimitiveTopology::LineList,
            PyPrimitiveTopology::LINE_STRIP => goldy::PrimitiveTopology::LineStrip,
            PyPrimitiveTopology::TRIANGLE_LIST => goldy::PrimitiveTopology::TriangleList,
            PyPrimitiveTopology::TRIANGLE_STRIP => goldy::PrimitiveTopology::TriangleStrip,
        }
    }
}

// =============================================================================
// IndexFormat
// =============================================================================

/// Index buffer format.
#[pyclass(name = "IndexFormat", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PyIndexFormat {
    /// 16-bit unsigned indices (0-65535).
    #[default]
    UINT16 = 0,
    /// 32-bit unsigned indices (0-4 billion).
    UINT32 = 1,
}

impl From<PyIndexFormat> for goldy::IndexFormat {
    fn from(if_: PyIndexFormat) -> Self {
        match if_ {
            PyIndexFormat::UINT16 => goldy::IndexFormat::Uint16,
            PyIndexFormat::UINT32 => goldy::IndexFormat::Uint32,
        }
    }
}

// =============================================================================
// DepthFormat
// =============================================================================

/// Depth/stencil texture format.
#[pyclass(name = "DepthFormat", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyDepthFormat {
    /// 16-bit depth, no stencil.
    DEPTH16_UNORM = 0,
    /// 24-bit depth, no stencil.
    DEPTH24_PLUS = 1,
    /// 24-bit depth + 8-bit stencil.
    DEPTH24_PLUS_STENCIL8 = 2,
    /// 32-bit floating point depth, no stencil.
    DEPTH32_FLOAT = 3,
    /// 32-bit floating point depth + 8-bit stencil.
    DEPTH32_FLOAT_STENCIL8 = 4,
}

impl From<PyDepthFormat> for goldy::DepthFormat {
    fn from(df: PyDepthFormat) -> Self {
        match df {
            PyDepthFormat::DEPTH16_UNORM => goldy::DepthFormat::Depth16Unorm,
            PyDepthFormat::DEPTH24_PLUS => goldy::DepthFormat::Depth24Plus,
            PyDepthFormat::DEPTH24_PLUS_STENCIL8 => goldy::DepthFormat::Depth24PlusStencil8,
            PyDepthFormat::DEPTH32_FLOAT => goldy::DepthFormat::Depth32Float,
            PyDepthFormat::DEPTH32_FLOAT_STENCIL8 => goldy::DepthFormat::Depth32FloatStencil8,
        }
    }
}

// =============================================================================
// CompareFunction
// =============================================================================

/// Depth comparison function for depth testing.
#[pyclass(name = "CompareFunction", module = "goldy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PyCompareFunction {
    /// Never passes.
    NEVER = 0,
    /// Passes if new < current.
    #[default]
    LESS = 1,
    /// Passes if new == current.
    EQUAL = 2,
    /// Passes if new <= current.
    LESS_EQUAL = 3,
    /// Passes if new > current.
    GREATER = 4,
    /// Passes if new != current.
    NOT_EQUAL = 5,
    /// Passes if new >= current.
    GREATER_EQUAL = 6,
    /// Always passes.
    ALWAYS = 7,
}

impl From<PyCompareFunction> for goldy::CompareFunction {
    fn from(cf: PyCompareFunction) -> Self {
        match cf {
            PyCompareFunction::NEVER => goldy::CompareFunction::Never,
            PyCompareFunction::LESS => goldy::CompareFunction::Less,
            PyCompareFunction::EQUAL => goldy::CompareFunction::Equal,
            PyCompareFunction::LESS_EQUAL => goldy::CompareFunction::LessEqual,
            PyCompareFunction::GREATER => goldy::CompareFunction::Greater,
            PyCompareFunction::NOT_EQUAL => goldy::CompareFunction::NotEqual,
            PyCompareFunction::GREATER_EQUAL => goldy::CompareFunction::GreaterEqual,
            PyCompareFunction::ALWAYS => goldy::CompareFunction::Always,
        }
    }
}

// =============================================================================
// Color
// =============================================================================

/// RGBA color with floating point components (0.0 - 1.0).
#[pyclass(name = "Color", module = "goldy")]
#[derive(Clone, Copy)]
pub struct PyColor {
    pub inner: goldy::Color,
}

#[pymethods]
impl PyColor {
    /// Create a new color from RGBA float values (0.0 - 1.0).
    #[new]
    #[pyo3(signature = (r, g, b, a=1.0))]
    fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        PyColor {
            inner: goldy::Color { r, g, b, a },
        }
    }

    /// Create a color from RGB byte values (0-255).
    #[staticmethod]
    fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        PyColor {
            inner: goldy::Color::from_rgb(r, g, b),
        }
    }

    /// Create a color from RGBA byte values (0-255).
    #[staticmethod]
    fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        PyColor {
            inner: goldy::Color::from_rgba(r, g, b, a),
        }
    }

    /// Red component (0.0 - 1.0).
    #[getter]
    fn r(&self) -> f32 {
        self.inner.r
    }

    /// Green component (0.0 - 1.0).
    #[getter]
    fn g(&self) -> f32 {
        self.inner.g
    }

    /// Blue component (0.0 - 1.0).
    #[getter]
    fn b(&self) -> f32 {
        self.inner.b
    }

    /// Alpha component (0.0 - 1.0).
    #[getter]
    fn a(&self) -> f32 {
        self.inner.a
    }

    /// Convert to RGBA tuple.
    fn to_tuple(&self) -> (f32, f32, f32, f32) {
        (self.inner.r, self.inner.g, self.inner.b, self.inner.a)
    }

    /// Convert to RGBA u8 list.
    fn to_rgba8(&self) -> [u8; 4] {
        self.inner.to_rgba8()
    }

    fn __repr__(&self) -> String {
        format!(
            "Color({}, {}, {}, {})",
            self.inner.r, self.inner.g, self.inner.b, self.inner.a
        )
    }

    // Predefined colors as class attributes
    #[classattr]
    #[allow(non_snake_case)]
    fn BLACK() -> PyColor {
        PyColor {
            inner: goldy::Color::BLACK,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn WHITE() -> PyColor {
        PyColor {
            inner: goldy::Color::WHITE,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn RED() -> PyColor {
        PyColor {
            inner: goldy::Color::RED,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn GREEN() -> PyColor {
        PyColor {
            inner: goldy::Color::GREEN,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn BLUE() -> PyColor {
        PyColor {
            inner: goldy::Color::BLUE,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn CORNFLOWER_BLUE() -> PyColor {
        PyColor {
            inner: goldy::Color::CORNFLOWER_BLUE,
        }
    }
}

// =============================================================================
// VertexAttribute
// =============================================================================

/// Vertex attribute description.
#[pyclass(name = "VertexAttribute", module = "goldy")]
#[derive(Clone)]
pub struct PyVertexAttribute {
    pub inner: goldy::VertexAttribute,
}

#[pymethods]
impl PyVertexAttribute {
    /// Create a new vertex attribute.
    #[new]
    fn new(location: u32, format: PyVertexFormat, offset: u32) -> Self {
        PyVertexAttribute {
            inner: goldy::VertexAttribute {
                location,
                format: format.into(),
                offset,
            },
        }
    }

    /// Shader location.
    #[getter]
    fn location(&self) -> u32 {
        self.inner.location
    }

    /// Byte offset within the vertex.
    #[getter]
    fn offset(&self) -> u32 {
        self.inner.offset
    }

    fn __repr__(&self) -> String {
        format!(
            "VertexAttribute(location={}, offset={})",
            self.inner.location, self.inner.offset
        )
    }
}

// =============================================================================
// VertexBufferLayout
// =============================================================================

/// Vertex buffer layout.
#[pyclass(name = "VertexBufferLayout", module = "goldy")]
#[derive(Clone)]
pub struct PyVertexBufferLayout {
    pub inner: goldy::VertexBufferLayout,
}

#[pymethods]
impl PyVertexBufferLayout {
    /// Create a new vertex buffer layout.
    #[new]
    fn new(stride: u32, attributes: Vec<PyVertexAttribute>) -> Self {
        PyVertexBufferLayout {
            inner: goldy::VertexBufferLayout {
                stride,
                attributes: attributes.into_iter().map(|a| a.inner).collect(),
            },
        }
    }

    /// Create layout for Vertex2D (position + color).
    #[staticmethod]
    fn vertex_2d() -> Self {
        PyVertexBufferLayout {
            inner: goldy::Vertex2D::layout(),
        }
    }

    /// Create layout for Vertex2DUv (position + uv).
    #[staticmethod]
    fn vertex_2d_uv() -> Self {
        PyVertexBufferLayout {
            inner: goldy::Vertex2DUv::layout(),
        }
    }

    /// Create an empty layout (for shaders that generate vertices procedurally).
    ///
    /// Use this for fullscreen triangle shaders that use SV_VertexID.
    #[staticmethod]
    fn empty() -> Self {
        PyVertexBufferLayout {
            inner: goldy::VertexBufferLayout::default(),
        }
    }

    /// Stride in bytes between vertices.
    #[getter]
    fn stride(&self) -> u32 {
        self.inner.stride
    }

    fn __repr__(&self) -> String {
        format!(
            "VertexBufferLayout(stride={}, attributes={})",
            self.inner.stride,
            self.inner.attributes.len()
        )
    }
}

// =============================================================================
// DepthStencilState
// =============================================================================

/// Depth/stencil state for render pipelines.
#[pyclass(name = "DepthStencilState", module = "goldy")]
#[derive(Clone)]
pub struct PyDepthStencilState {
    pub inner: goldy::DepthStencilState,
}

#[pymethods]
impl PyDepthStencilState {
    /// Create a new depth stencil state.
    #[new]
    #[pyo3(signature = (format=PyDepthFormat::DEPTH24_PLUS, depth_write_enabled=true, depth_compare=PyCompareFunction::LESS))]
    fn new(
        format: PyDepthFormat,
        depth_write_enabled: bool,
        depth_compare: PyCompareFunction,
    ) -> Self {
        PyDepthStencilState {
            inner: goldy::DepthStencilState {
                format: format.into(),
                depth_write_enabled,
                depth_compare: depth_compare.into(),
            },
        }
    }

    /// Whether to write depth values.
    #[getter]
    fn depth_write_enabled(&self) -> bool {
        self.inner.depth_write_enabled
    }

    fn __repr__(&self) -> String {
        format!(
            "DepthStencilState(depth_write_enabled={})",
            self.inner.depth_write_enabled
        )
    }
}
