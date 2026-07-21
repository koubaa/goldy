//! Per-instance data for the instancing example.
//!
//! Layout matches `QuadInstance` in `instancing_update.slang` / `instancing_render.slang`.

use goldy::StructuredBufferElement;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct Instance2D {
    pub position: [f32; 2],
    pub rotation: f32,
    pub scale: f32,
    pub color: [f32; 4],
}

impl Instance2D {
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
