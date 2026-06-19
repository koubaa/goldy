// Common data types for Goldy shaders (EXPERIMENTAL)
// Part of the goldy_exp module
//
// These types have matching Rust definitions in goldy::common_types
// for zero-copy buffer sharing between CPU and GPU.

implementing goldy_exp;

// ============================================================================
// Dispatch Types
// ============================================================================

/// Workgroup-count triple for compute dispatch (direct or indirect).
/// Rust equivalent: goldy::DispatchShape
public struct DispatchShape {
    public uint x;
    public uint y;
    public uint z;
};

// ============================================================================
// Particle System Types
// Used by: compute_particles, starfield, bouncing_lines, game_of_life
// ============================================================================

/// 2D particle with position and velocity
/// Rust equivalent: goldy::Particle2D
public struct Particle2D {
    public float2 position;
    public float2 velocity;
};

/// 3D particle with position, velocity, and lifetime
/// Rust equivalent: goldy::Particle3D
public struct Particle3D {
    public float3 position;
    public float3 velocity;
    public float age;
    public float lifetime;
};

// ============================================================================
// Time/Frame Uniforms
// The most common uniform pattern - time and frame count
// ============================================================================

/// Standard time/frame uniform block
/// Rust equivalent: goldy::FrameUniforms
public struct FrameUniforms {
    public float time;        // Elapsed time in seconds
    public float delta_time;  // Time since last frame
    public uint frame;        // Frame counter
    public uint _pad;         // Padding for 16-byte alignment
};

// ============================================================================
// Transform Types
// Used for instanced rendering where each instance has a transform
// ============================================================================

/// 2D transform (position, rotation, scale)
/// Rust equivalent: goldy::Transform2D
public struct Transform2D {
    public float2 position;
    public float rotation;    // Radians
    public float2 scale;
    public float _pad;        // Padding for alignment
};

/// Instance data for 2D instanced rendering (transform + color)
/// Useful for instancing.rs-style effects
public struct Instance2D {
    public float2 position;
    public float rotation;
    public float scale;
    public float4 color;
};
