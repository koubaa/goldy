//! Common types shared between Rust and shaders.
//!
//! These types have matching definitions in the `goldy_exp.types` shader module,
//! enabling zero-copy buffer sharing between CPU and GPU.

use crate::buffer::StructuredBufferElement;
use bytemuck::{Pod, Zeroable};

/// Instance data for 2D instanced rendering.
///
/// Matches `goldy_exp.types.Instance2D` in shaders.
/// Useful for particle systems and sprite batching.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct Instance2D {
    /// Position in 2D space
    pub position: [f32; 2],
    /// Rotation in radians
    pub rotation: f32,
    /// Uniform scale
    pub scale: f32,
    /// RGBA color
    pub color: [f32; 4],
}

impl Instance2D {
    /// Create a new instance.
    pub const fn new(x: f32, y: f32, rotation: f32, scale: f32, color: [f32; 4]) -> Self {
        Self {
            position: [x, y],
            rotation,
            scale,
            color,
        }
    }
}

impl StructuredBufferElement for Instance2D {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance2d_size() {
        assert_eq!(std::mem::size_of::<Instance2D>(), 32);
    }
}
