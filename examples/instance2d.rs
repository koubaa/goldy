//! Per-instance data for the instancing example.

#[goldy::gpu]
#[derive(Debug, Default)]
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
