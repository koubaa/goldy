// Procedural geometry helpers for Goldy shaders (EXPERIMENTAL)
// Part of the goldy_exp module
//
// These functions generate geometry from vertex IDs, eliminating the need
// for vertex buffers in many common cases. This is the "Goldy way" - 
// let the GPU generate geometry procedurally.

implementing goldy_exp;

// ============================================================================
// Quad Geometry (6 vertices per quad, 2 triangles)
// Used for instanced rendering of particles, sprites, UI elements, etc.
// ============================================================================

/// Quad vertices in local space [-1, 1]
/// Triangle 1: bottom-left, bottom-right, top-right
/// Triangle 2: bottom-left, top-right, top-left
static const float2 QUAD_VERTS[6] = {
    float2(-1.0, -1.0),  // 0: bottom-left
    float2( 1.0, -1.0),  // 1: bottom-right
    float2( 1.0,  1.0),  // 2: top-right
    float2(-1.0, -1.0),  // 3: bottom-left
    float2( 1.0,  1.0),  // 4: top-right
    float2(-1.0,  1.0),  // 5: top-left
};

/// UV coordinates for quad vertices (0,0) at bottom-left, (1,1) at top-right
static const float2 QUAD_UVS[6] = {
    float2(0.0, 0.0),  // 0: bottom-left
    float2(1.0, 0.0),  // 1: bottom-right
    float2(1.0, 1.0),  // 2: top-right
    float2(0.0, 0.0),  // 3: bottom-left
    float2(1.0, 1.0),  // 4: top-right
    float2(0.0, 1.0),  // 5: top-left
};

/// Get quad vertex position in [-1, 1] local space
/// Use with: draw(0..6, 0..instance_count) or draw_quads(instance_count)
public float2 quad_local(uint vertex_id) {
    return QUAD_VERTS[vertex_id % 6];
}

/// Get quad UV coordinate
public float2 quad_uv(uint vertex_id) {
    return QUAD_UVS[vertex_id % 6];
}

/// Get quad vertex at world position with uniform size
public float2 quad_position(uint vertex_id, float2 center, float size) {
    return center + QUAD_VERTS[vertex_id % 6] * size;
}

/// Get quad vertex at world position with non-uniform size
public float2 quad_position_sized(uint vertex_id, float2 center, float2 size) {
    return center + QUAD_VERTS[vertex_id % 6] * size;
}

/// Get rotated quad vertex at world position
/// rotation is in radians
public float2 quad_position_rotated(uint vertex_id, float2 center, float size, float rotation) {
    float2 local = QUAD_VERTS[vertex_id % 6] * size;
    float c = cos(rotation);
    float s = sin(rotation);
    float2 rotated = float2(
        local.x * c - local.y * s,
        local.x * s + local.y * c
    );
    return center + rotated;
}

// ============================================================================
// Billboard Geometry (camera-facing quads in 3D)
// Used for particle effects, sprites in 3D space, etc.
// ============================================================================

/// Get billboard vertex position in world space
/// The quad always faces the camera.
/// cam_right and cam_up should be normalized camera basis vectors.
public float3 billboard_position(uint vertex_id, float3 center, float2 size, 
                                  float3 cam_right, float3 cam_up) {
    float2 local = QUAD_VERTS[vertex_id % 6];
    return center 
         + cam_right * local.x * size.x 
         + cam_up * local.y * size.y;
}

/// Get billboard vertex position with uniform size
public float3 billboard_position_uniform(uint vertex_id, float3 center, float size,
                                          float3 cam_right, float3 cam_up) {
    return billboard_position(vertex_id, center, float2(size, size), cam_right, cam_up);
}

/// Extract the camera/modelview right vector (column 0) from a 4x4 matrix.
/// Used for cylindrical billboards where sprites expand horizontally only.
public float3 modelview_right(float4x4 m) {
    return float3(m[0][0], m[1][0], m[2][0]);
}

/// Cylindrical billboard offset — for vertical sprites that expand only along
/// camera-right. Returns center + cam_right * horizontal_offset.
public float3 billboard_cylindrical_offset(float3 center, float3 cam_right, float horizontal_offset) {
    return center + cam_right * horizontal_offset;
}

// ============================================================================
// Fullscreen Triangle (3 vertices, covers entire screen)
// More efficient than fullscreen quad - use draw(0..3, 0..1) or draw_fullscreen()
// ============================================================================

/// Get fullscreen triangle position in clip space
/// Generates a triangle that extends beyond [-1,1] to cover entire screen
public float4 fullscreen_position(uint vertex_id) {
    float2 uv = float2(
        float((vertex_id << 1u) & 2u),
        float(vertex_id & 2u)
    );
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

/// Get fullscreen triangle UV (0,0 at bottom-left, 1,1 at top-right)
public float2 fullscreen_uv(uint vertex_id) {
    return float2(
        float((vertex_id << 1u) & 2u),
        float(vertex_id & 2u)
    );
}

// ============================================================================
// Line Geometry (2 vertices per line)
// Used for debug visualization, wireframe effects
// ============================================================================

/// Get line vertex position (0 = start, 1 = end)
public float2 line_position(uint vertex_id, float2 start, float2 end) {
    return (vertex_id % 2 == 0) ? start : end;
}

/// Get line vertex position in 3D
public float3 line_position_3d(uint vertex_id, float3 start, float3 end) {
    return (vertex_id % 2 == 0) ? start : end;
}
