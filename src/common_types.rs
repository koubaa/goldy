//! Common types shared between Rust and shaders.
//!
//! These types have matching definitions in the `goldy_exp.types` shader module,
//! enabling zero-copy buffer sharing between CPU and GPU.
//!
//! # Example
//!
//! ```rust,ignore
//! use goldy::{Buffer, DataAccess, Particle2D};
//!
//! // Create particles on CPU
//! let particles = vec![
//!     Particle2D { position: [0.0, 0.0], velocity: [1.0, 0.0] },
//!     Particle2D { position: [0.5, 0.5], velocity: [0.0, -1.0] },
//! ];
//!
//! // Upload directly to GPU - layout matches shader struct exactly
//! let buffer = Buffer::with_data(&device, &particles, DataAccess::Scattered)?;
//! ```

use bytemuck::{Pod, Zeroable};

// ============================================================================
// Particle System Types
// ============================================================================

/// 2D particle with position and velocity.
///
/// Matches `goldy_exp.types.Particle2D` in shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct Particle2D {
    /// Position in 2D space
    pub position: [f32; 2],
    /// Velocity vector
    pub velocity: [f32; 2],
}

impl Particle2D {
    /// Create a new particle at the given position with the given velocity.
    pub const fn new(x: f32, y: f32, vx: f32, vy: f32) -> Self {
        Self {
            position: [x, y],
            velocity: [vx, vy],
        }
    }

    /// Create a stationary particle at the given position.
    pub const fn at(x: f32, y: f32) -> Self {
        Self::new(x, y, 0.0, 0.0)
    }
}

/// 3D particle with position, velocity, and lifetime.
///
/// Matches `goldy_exp.types.Particle3D` in shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct Particle3D {
    /// Position in 3D space
    pub position: [f32; 3],
    /// Velocity vector
    pub velocity: [f32; 3],
    /// Current age of the particle (seconds)
    pub age: f32,
    /// Maximum lifetime (seconds)
    pub lifetime: f32,
}

impl Particle3D {
    /// Create a new particle.
    pub const fn new(position: [f32; 3], velocity: [f32; 3], lifetime: f32) -> Self {
        Self {
            position,
            velocity,
            age: 0.0,
            lifetime,
        }
    }
}

// ============================================================================
// Time/Frame Uniforms
// ============================================================================

/// Standard time/frame uniform block.
///
/// Matches `goldy_exp.types.FrameUniforms` in shaders.
/// This is the most common uniform pattern for animated effects.
///
/// # Example
///
/// ```rust,ignore
/// let uniforms = FrameUniforms::new(elapsed_time, delta_time, frame_count);
/// uniform_buffer.write_data(0, &[uniforms])?;
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct FrameUniforms {
    /// Elapsed time in seconds since start
    pub time: f32,
    /// Time since last frame (seconds)
    pub delta_time: f32,
    /// Frame counter
    pub frame: u32,
    /// Padding for 16-byte alignment
    pub _pad: u32,
}

impl FrameUniforms {
    /// Create new frame uniforms.
    pub const fn new(time: f32, delta_time: f32, frame: u32) -> Self {
        Self {
            time,
            delta_time,
            frame,
            _pad: 0,
        }
    }
}

// ============================================================================
// Transform Types
// ============================================================================

/// 2D transform (position, rotation, scale).
///
/// Matches `goldy_exp.types.Transform2D` in shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct Transform2D {
    /// Position in 2D space
    pub position: [f32; 2],
    /// Rotation in radians
    pub rotation: f32,
    /// Scale (x, y)
    pub scale: [f32; 2],
    /// Padding for alignment
    pub _pad: f32,
}

impl Transform2D {
    /// Create a new transform.
    pub const fn new(x: f32, y: f32, rotation: f32, scale_x: f32, scale_y: f32) -> Self {
        Self {
            position: [x, y],
            rotation,
            scale: [scale_x, scale_y],
            _pad: 0.0,
        }
    }

    /// Create a transform at position with uniform scale.
    pub const fn at(x: f32, y: f32) -> Self {
        Self::new(x, y, 0.0, 1.0, 1.0)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle2d_size() {
        assert_eq!(std::mem::size_of::<Particle2D>(), 16);
    }

    #[test]
    fn test_particle3d_size() {
        assert_eq!(std::mem::size_of::<Particle3D>(), 32);
    }

    #[test]
    fn test_frame_uniforms_size() {
        assert_eq!(std::mem::size_of::<FrameUniforms>(), 16);
    }

    #[test]
    fn test_transform2d_size() {
        assert_eq!(std::mem::size_of::<Transform2D>(), 24);
    }

    #[test]
    fn test_instance2d_size() {
        assert_eq!(std::mem::size_of::<Instance2D>(), 32);
    }
}
