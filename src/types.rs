//! Common types used throughout RAG.

use bitflags::bitflags;
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
    pub const BLACK: Color = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const WHITE: Color = Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const RED: Color = Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Color = Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Color = Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const CORNFLOWER_BLUE: Color = Color { r: 0.392, g: 0.584, b: 0.929, a: 1.0 };

    /// Create a color from RGB values (0-255).
    pub const fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: 1.0,
        }
    }

    /// Create a color from RGBA values (0-255).
    pub const fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    /// Convert to RGBA u8 array.
    pub fn to_rgba8(&self) -> [u8; 4] {
        [
            (self.r * 255.0) as u8,
            (self.g * 255.0) as u8,
            (self.b * 255.0) as u8,
            (self.a * 255.0) as u8,
        ]
    }
}

/// Texture format for render targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// 8-bit RGBA, sRGB color space
    Rgba8UnormSrgb,
    /// 8-bit RGBA, linear color space
    Rgba8Unorm,
    /// 8-bit BGRA, sRGB color space
    Bgra8UnormSrgb,
    /// 8-bit BGRA, linear color space
    Bgra8Unorm,
    /// 16-bit RGBA float
    Rgba16Float,
    /// 32-bit RGBA float
    Rgba32Float,
}

impl TextureFormat {
    /// Get the number of bytes per pixel.
    pub fn bytes_per_pixel(&self) -> u32 {
        match self {
            TextureFormat::Rgba8UnormSrgb => 4,
            TextureFormat::Rgba8Unorm => 4,
            TextureFormat::Bgra8UnormSrgb => 4,
            TextureFormat::Bgra8Unorm => 4,
            TextureFormat::Rgba16Float => 8,
            TextureFormat::Rgba32Float => 16,
        }
    }
}

bitflags! {
    /// Buffer usage flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferUsage: u32 {
        /// Can be used as a vertex buffer.
        const VERTEX = 1 << 0;
        /// Can be used as an index buffer.
        const INDEX = 1 << 1;
        /// Can be used as a uniform buffer.
        const UNIFORM = 1 << 2;
        /// Can be used as a storage buffer.
        const STORAGE = 1 << 3;
        /// Can be used as a copy source.
        const COPY_SRC = 1 << 4;
        /// Can be used as a copy destination.
        const COPY_DST = 1 << 5;
    }
}

/// Vertex attribute format.
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
    /// Get the size in bytes of this format.
    pub fn size(&self) -> u32 {
        match self {
            VertexFormat::Float32 => 4,
            VertexFormat::Float32x2 => 8,
            VertexFormat::Float32x3 => 12,
            VertexFormat::Float32x4 => 16,
            VertexFormat::Uint32 => 4,
            VertexFormat::Sint32 => 4,
            VertexFormat::Uint8x4 => 4,
            VertexFormat::Unorm8x4 => 4,
        }
    }
}

/// Vertex attribute description.
#[derive(Debug, Clone)]
pub struct VertexAttribute {
    /// Shader location (e.g., `layout(location = 0)`).
    pub location: u32,
    /// Data format.
    pub format: VertexFormat,
    /// Byte offset within the vertex.
    pub offset: u32,
}

/// Vertex buffer layout.
#[derive(Debug, Clone)]
pub struct VertexBufferLayout {
    /// Stride in bytes between vertices.
    pub stride: u32,
    /// Attribute descriptions.
    pub attributes: Vec<VertexAttribute>,
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

/// Index buffer format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IndexFormat {
    /// 16-bit unsigned indices (0-65535).
    #[default]
    Uint16,
    /// 32-bit unsigned indices (0-4 billion).
    Uint32,
}

impl IndexFormat {
    /// Get the size in bytes of one index.
    pub fn size(&self) -> u32 {
        match self {
            IndexFormat::Uint16 => 2,
            IndexFormat::Uint32 => 4,
        }
    }
}

/// Type of GPU device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// Discrete GPU (dedicated graphics card).
    DiscreteGpu,
    /// Integrated GPU (part of CPU).
    IntegratedGpu,
    /// Software renderer (CPU).
    Cpu,
    /// Other/unknown.
    Other,
}

/// Graphics backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    Vulkan,
    Metal,
    Dx12,
    WebGPU,
}

/// A simple 2D vertex with position and color.
/// Use for colored primitives (triangle, particles, etc.)
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

    /// Get the vertex buffer layout for this vertex type.
    pub fn layout() -> VertexBufferLayout {
        VertexBufferLayout {
            stride: std::mem::size_of::<Self>() as u32,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    format: VertexFormat::Float32x2,
                    offset: 0,
                },
                VertexAttribute {
                    location: 1,
                    format: VertexFormat::Float32x4,
                    offset: 8,
                },
            ],
        }
    }
}

/// A 2D vertex with position and UV coordinates.
/// Use for textured/shader effects (plasma, gradient, etc.)
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex2DUv {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

impl Vertex2DUv {
    pub const fn new(x: f32, y: f32, u: f32, v: f32) -> Self {
        Self {
            position: [x, y],
            uv: [u, v],
        }
    }

    /// Get the vertex buffer layout for this vertex type.
    pub fn layout() -> VertexBufferLayout {
        VertexBufferLayout {
            stride: std::mem::size_of::<Self>() as u32,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    format: VertexFormat::Float32x2,
                    offset: 0,
                },
                VertexAttribute {
                    location: 1,
                    format: VertexFormat::Float32x2,
                    offset: 8,
                },
            ],
        }
    }
}

/// Fullscreen quad vertices using Vertex2DUv (position + UV)
pub const FULLSCREEN_QUAD: [Vertex2DUv; 6] = [
    Vertex2DUv { position: [-1.0, -1.0], uv: [0.0, 1.0] },
    Vertex2DUv { position: [1.0, -1.0], uv: [1.0, 1.0] },
    Vertex2DUv { position: [1.0, 1.0], uv: [1.0, 0.0] },
    Vertex2DUv { position: [-1.0, -1.0], uv: [0.0, 1.0] },
    Vertex2DUv { position: [1.0, 1.0], uv: [1.0, 0.0] },
    Vertex2DUv { position: [-1.0, 1.0], uv: [0.0, 0.0] },
];

// ============================================================================
// Depth Buffer Types
// ============================================================================

/// Depth/stencil texture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthFormat {
    /// 16-bit depth, no stencil
    Depth16Unorm,
    /// 24-bit depth, no stencil (platform may use 32-bit internally)
    Depth24Plus,
    /// 24-bit depth + 8-bit stencil
    Depth24PlusStencil8,
    /// 32-bit floating point depth, no stencil
    Depth32Float,
    /// 32-bit floating point depth + 8-bit stencil
    Depth32FloatStencil8,
}

impl DepthFormat {
    /// Returns true if this format includes a stencil component.
    pub fn has_stencil(&self) -> bool {
        matches!(self, DepthFormat::Depth24PlusStencil8 | DepthFormat::Depth32FloatStencil8)
    }
}

/// Depth comparison function for depth testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompareFunction {
    /// Never passes.
    Never,
    /// Passes if new < current.
    #[default]
    Less,
    /// Passes if new == current.
    Equal,
    /// Passes if new <= current.
    LessEqual,
    /// Passes if new > current.
    Greater,
    /// Passes if new != current.
    NotEqual,
    /// Passes if new >= current.
    GreaterEqual,
    /// Always passes.
    Always,
}

/// Depth/stencil state for render pipelines.
#[derive(Debug, Clone)]
pub struct DepthStencilState {
    /// Depth format to use. Must match the render target's depth format.
    pub format: DepthFormat,
    /// Whether to write depth values.
    pub depth_write_enabled: bool,
    /// Comparison function for depth test.
    pub depth_compare: CompareFunction,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            format: DepthFormat::Depth24Plus,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
        }
    }
}

// ============================================================================
// Texture Types
// ============================================================================

bitflags! {
    /// Texture usage flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TextureUsage: u32 {
        /// Can be used as a copy source.
        const COPY_SRC = 1 << 0;
        /// Can be used as a copy destination.
        const COPY_DST = 1 << 1;
        /// Can be sampled in a shader (e.g., texture2D).
        const SAMPLED = 1 << 2;
        /// Can be used as a storage texture.
        const STORAGE = 1 << 3;
        /// Can be used as a render attachment.
        const RENDER_TARGET = 1 << 4;
    }
}

/// Texture addressing mode for coordinates outside [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AddressMode {
    /// Clamp to edge color.
    #[default]
    ClampToEdge,
    /// Repeat the texture.
    Repeat,
    /// Mirror and repeat the texture.
    MirrorRepeat,
}

/// Texture filtering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FilterMode {
    /// Nearest-neighbor sampling (blocky).
    #[default]
    Nearest,
    /// Linear interpolation (smooth).
    Linear,
}

/// Sampler descriptor for texture sampling.
#[derive(Debug, Clone)]
pub struct SamplerDesc {
    /// Addressing mode for U (horizontal) coordinate.
    pub address_mode_u: AddressMode,
    /// Addressing mode for V (vertical) coordinate.
    pub address_mode_v: AddressMode,
    /// Addressing mode for W (depth) coordinate.
    pub address_mode_w: AddressMode,
    /// Magnification filter (when texture is enlarged).
    pub mag_filter: FilterMode,
    /// Minification filter (when texture is shrunk).
    pub min_filter: FilterMode,
    /// Mipmap filter mode.
    pub mipmap_filter: FilterMode,
    /// Maximum anisotropic filtering level (1.0 = disabled).
    pub max_anisotropy: f32,
    /// Compare function for depth textures (None = no comparison).
    pub compare: Option<CompareFunction>,
    /// Minimum LOD clamp.
    pub lod_min_clamp: f32,
    /// Maximum LOD clamp.
    pub lod_max_clamp: f32,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            max_anisotropy: 1.0,
            compare: None,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_rgb() {
        let color = Color::from_rgb(255, 128, 0);
        assert!((color.r - 1.0).abs() < 0.01);
        assert!((color.g - 0.502).abs() < 0.01);
        assert!((color.b - 0.0).abs() < 0.01);
        assert!((color.a - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_color_to_rgba8() {
        let color = Color { r: 1.0, g: 0.5, b: 0.0, a: 1.0 };
        let rgba = color.to_rgba8();
        assert_eq!(rgba[0], 255);
        assert_eq!(rgba[1], 127);
        assert_eq!(rgba[2], 0);
        assert_eq!(rgba[3], 255);
    }

    #[test]
    fn test_color_constants() {
        assert_eq!(Color::BLACK.r, 0.0);
        assert_eq!(Color::WHITE.r, 1.0);
        assert_eq!(Color::RED.r, 1.0);
        assert_eq!(Color::RED.g, 0.0);
    }

    #[test]
    fn test_texture_format_bytes_per_pixel() {
        assert_eq!(TextureFormat::Rgba8Unorm.bytes_per_pixel(), 4);
        assert_eq!(TextureFormat::Rgba16Float.bytes_per_pixel(), 8);
        assert_eq!(TextureFormat::Rgba32Float.bytes_per_pixel(), 16);
    }

    #[test]
    fn test_vertex_format_size() {
        assert_eq!(VertexFormat::Float32.size(), 4);
        assert_eq!(VertexFormat::Float32x2.size(), 8);
        assert_eq!(VertexFormat::Float32x3.size(), 12);
        assert_eq!(VertexFormat::Float32x4.size(), 16);
    }

    #[test]
    fn test_vertex2d_layout() {
        let layout = Vertex2D::layout();
        assert_eq!(layout.stride, 24); // 2 floats + 4 floats = 6 * 4 = 24
        assert_eq!(layout.attributes.len(), 2);
        assert_eq!(layout.attributes[0].location, 0);
        assert_eq!(layout.attributes[1].location, 1);
    }

    #[test]
    fn test_vertex2duv_layout() {
        let layout = Vertex2DUv::layout();
        assert_eq!(layout.stride, 16); // 2 floats + 2 floats = 4 * 4 = 16
        assert_eq!(layout.attributes.len(), 2);
    }

    #[test]
    fn test_fullscreen_quad_vertices() {
        // Check we have 6 vertices (2 triangles)
        assert_eq!(FULLSCREEN_QUAD.len(), 6);
        
        // Check positions span -1 to 1
        for v in &FULLSCREEN_QUAD {
            assert!(v.position[0] >= -1.0 && v.position[0] <= 1.0);
            assert!(v.position[1] >= -1.0 && v.position[1] <= 1.0);
        }
        
        // Check UVs span 0 to 1
        for v in &FULLSCREEN_QUAD {
            assert!(v.uv[0] >= 0.0 && v.uv[0] <= 1.0);
            assert!(v.uv[1] >= 0.0 && v.uv[1] <= 1.0);
        }
    }

    #[test]
    fn test_buffer_usage_flags() {
        let usage = BufferUsage::VERTEX | BufferUsage::COPY_DST;
        assert!(usage.contains(BufferUsage::VERTEX));
        assert!(usage.contains(BufferUsage::COPY_DST));
        assert!(!usage.contains(BufferUsage::INDEX));
    }

    #[test]
    fn test_index_format_size() {
        assert_eq!(IndexFormat::Uint16.size(), 2);
        assert_eq!(IndexFormat::Uint32.size(), 4);
    }

    #[test]
    fn test_index_format_default() {
        assert_eq!(IndexFormat::default(), IndexFormat::Uint16);
    }

    // Depth buffer tests
    #[test]
    fn test_depth_format_has_stencil() {
        assert!(!DepthFormat::Depth16Unorm.has_stencil());
        assert!(!DepthFormat::Depth24Plus.has_stencil());
        assert!(DepthFormat::Depth24PlusStencil8.has_stencil());
        assert!(!DepthFormat::Depth32Float.has_stencil());
        assert!(DepthFormat::Depth32FloatStencil8.has_stencil());
    }

    #[test]
    fn test_compare_function_default() {
        assert_eq!(CompareFunction::default(), CompareFunction::Less);
    }

    #[test]
    fn test_depth_stencil_state_default() {
        let state = DepthStencilState::default();
        assert_eq!(state.format, DepthFormat::Depth24Plus);
        assert!(state.depth_write_enabled);
        assert_eq!(state.depth_compare, CompareFunction::Less);
    }

    // Texture types tests
    #[test]
    fn test_texture_usage_flags() {
        let usage = TextureUsage::SAMPLED | TextureUsage::COPY_DST;
        assert!(usage.contains(TextureUsage::SAMPLED));
        assert!(usage.contains(TextureUsage::COPY_DST));
        assert!(!usage.contains(TextureUsage::STORAGE));
    }

    #[test]
    fn test_address_mode_default() {
        assert_eq!(AddressMode::default(), AddressMode::ClampToEdge);
    }

    #[test]
    fn test_filter_mode_default() {
        assert_eq!(FilterMode::default(), FilterMode::Nearest);
    }

    #[test]
    fn test_sampler_desc_default() {
        let desc = SamplerDesc::default();
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
        assert_eq!(desc.max_anisotropy, 1.0);
        assert!(desc.compare.is_none());
        assert_eq!(desc.lod_min_clamp, 0.0);
        assert_eq!(desc.lod_max_clamp, 32.0);
    }
}

