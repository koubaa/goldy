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

