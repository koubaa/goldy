//! Common types for rag-web
//!
//! These types mirror rag::types but include wgpu-specific implementations.
//! The data layouts match, allowing interop when needed.

use bytemuck::{Pod, Zeroable};

/// RGBA color with float components (0.0 - 1.0)
/// Mirrors rag::types::Color
#[derive(Clone, Copy, Debug)]
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
}

impl From<Color> for wgpu::Color {
    fn from(c: Color) -> Self {
        wgpu::Color {
            r: c.r as f64,
            g: c.g as f64,
            b: c.b as f64,
            a: c.a as f64,
        }
    }
}

/// 2D vertex with position and UV coordinates.
/// Mirrors rag::types::Vertex2DUv
///
/// Used for fullscreen quad effects (plasma, gradient, tunnel, etc.)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex2D {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

impl Vertex2D {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex2D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Alias for Vertex2D for clarity (matches rag::types::Vertex2DUv)
pub type Vertex2DUv = Vertex2D;

/// Full-screen quad vertices (two triangles, CCW winding)
/// Matches rag::types::FULLSCREEN_QUAD
pub const FULLSCREEN_QUAD: [Vertex2D; 6] = [
    Vertex2D { position: [-1.0, -1.0], uv: [0.0, 1.0] },
    Vertex2D { position: [1.0, -1.0], uv: [1.0, 1.0] },
    Vertex2D { position: [1.0, 1.0], uv: [1.0, 0.0] },
    Vertex2D { position: [-1.0, -1.0], uv: [0.0, 1.0] },
    Vertex2D { position: [1.0, 1.0], uv: [1.0, 0.0] },
    Vertex2D { position: [-1.0, 1.0], uv: [0.0, 0.0] },
];
