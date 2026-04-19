//! Common utilities shared between native and web examples.
//!
//! This module provides shared types and utilities that work across
//! both native (Vulkan) and web (WebGPU) platforms.

use crate::buffer::StructuredBufferElement;
use crate::types::{VertexBufferLayout, VertexFormat};
use bytemuck::{Pod, Zeroable};

/// Vertex with 2D position and UV coordinates.
/// Used for fullscreen quad effects (plasma, gradient, tunnel, etc.)
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct VertexUv {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

impl VertexUv {
    pub const fn new(x: f32, y: f32, u: f32, v: f32) -> Self {
        Self {
            position: [x, y],
            uv: [u, v],
        }
    }

    /// Get the vertex buffer layout for this vertex type.
    pub fn layout() -> VertexBufferLayout {
        VertexBufferLayout::from_formats::<Self>(&[
            VertexFormat::Float32x2, // position
            VertexFormat::Float32x2, // uv
        ])
    }
}

/// Vertex with 2D position, UV, and time.
/// Used for effects that pass time via vertex attributes.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct VertexUvTime {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub time: f32,
}

impl VertexUvTime {
    pub const fn new(x: f32, y: f32, u: f32, v: f32, time: f32) -> Self {
        Self {
            position: [x, y],
            uv: [u, v],
            time,
        }
    }

    /// Get the vertex buffer layout for this vertex type.
    pub fn layout() -> VertexBufferLayout {
        VertexBufferLayout::from_formats::<Self>(&[
            VertexFormat::Float32x2, // position
            VertexFormat::Float32x2, // uv
            VertexFormat::Float32,   // time
        ])
    }
}

impl StructuredBufferElement for VertexUv {}
impl StructuredBufferElement for VertexUvTime {}

/// Create a fullscreen quad with time baked into vertices.
pub fn create_fullscreen_quad_with_time(time: f32) -> [VertexUvTime; 6] {
    [
        VertexUvTime::new(-1.0, -1.0, 0.0, 1.0, time),
        VertexUvTime::new(1.0, -1.0, 1.0, 1.0, time),
        VertexUvTime::new(1.0, 1.0, 1.0, 0.0, time),
        VertexUvTime::new(-1.0, -1.0, 0.0, 1.0, time),
        VertexUvTime::new(1.0, 1.0, 1.0, 0.0, time),
        VertexUvTime::new(-1.0, 1.0, 0.0, 0.0, time),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_uv_layout() {
        let layout = VertexUv::layout();
        assert_eq!(layout.stride, 16); // 2 floats + 2 floats
        assert_eq!(layout.attributes.len(), 2);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[1].offset, 8);
    }

    #[test]
    fn test_vertex_uv_time_layout() {
        let layout = VertexUvTime::layout();
        assert_eq!(layout.stride, 20); // 2 floats + 2 floats + 1 float
        assert_eq!(layout.attributes.len(), 3);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[1].offset, 8);
        assert_eq!(layout.attributes[2].offset, 16);
    }

    #[test]
    fn test_create_fullscreen_quad_with_time() {
        let time = 1.5;
        let quad = create_fullscreen_quad_with_time(time);

        assert_eq!(quad.len(), 6);

        // All vertices should have the same time
        for v in &quad {
            assert_eq!(v.time, time);
        }
    }
}
