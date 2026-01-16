// Spinning 3D wireframe cube
// Rendered entirely in fragment shader using SDF techniques
//
// ## Bindless Support
//
// When __BINDLESS__ is defined (Metal Argument Buffers Tier 2):
// - Uniforms are accessed via ParameterBlock for argument buffer support
// - Traditional binding syntax works as fallback

struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Uniform structure
struct TimeUniforms {
    float time;
};

#ifdef __BINDLESS__
// Bindless mode: Use ParameterBlock for Metal argument buffer support
struct SpinningCubeResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<SpinningCubeResources> gResources;
#define TIME gResources.uniforms.time
#else
// Traditional binding mode
[[vk::binding(0, 0)]]
ConstantBuffer<TimeUniforms> uniforms;
#define TIME uniforms.time
#endif

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

float3 rotate_y(float3 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

float3 rotate_x(float3 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float3(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

float2 project(float3 p) {
    float z = p.z + 4.0;
    float scale = 2.0 / z;
    return float2(p.x * scale, p.y * scale);
}

float line_dist(float2 p, float2 a, float2 b) {
    float2 pa = p - a;
    float2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = (input.uv - 0.5) * 2.0;
    float t = TIME;
    
    // Cube vertices (8 corners)
    float3 verts[8];
    verts[0] = float3(-1.0, -1.0, -1.0);
    verts[1] = float3(1.0, -1.0, -1.0);
    verts[2] = float3(1.0, 1.0, -1.0);
    verts[3] = float3(-1.0, 1.0, -1.0);
    verts[4] = float3(-1.0, -1.0, 1.0);
    verts[5] = float3(1.0, -1.0, 1.0);
    verts[6] = float3(1.0, 1.0, 1.0);
    verts[7] = float3(-1.0, 1.0, 1.0);
    
    // Rotate vertices
    for (int i = 0; i < 8; i++) {
        verts[i] = rotate_y(verts[i], t * 0.7);
        verts[i] = rotate_x(verts[i], t * 0.5);
    }
    
    // Project to 2D
    float2 proj[8];
    for (int i = 0; i < 8; i++) {
        proj[i] = project(verts[i]);
    }
    
    // Calculate distance to cube edges
    float min_dist = 1000.0;
    
    // Back face
    min_dist = min(min_dist, line_dist(uv, proj[0], proj[1]));
    min_dist = min(min_dist, line_dist(uv, proj[1], proj[2]));
    min_dist = min(min_dist, line_dist(uv, proj[2], proj[3]));
    min_dist = min(min_dist, line_dist(uv, proj[3], proj[0]));
    
    // Front face
    min_dist = min(min_dist, line_dist(uv, proj[4], proj[5]));
    min_dist = min(min_dist, line_dist(uv, proj[5], proj[6]));
    min_dist = min(min_dist, line_dist(uv, proj[6], proj[7]));
    min_dist = min(min_dist, line_dist(uv, proj[7], proj[4]));
    
    // Connecting edges
    min_dist = min(min_dist, line_dist(uv, proj[0], proj[4]));
    min_dist = min(min_dist, line_dist(uv, proj[1], proj[5]));
    min_dist = min(min_dist, line_dist(uv, proj[2], proj[6]));
    min_dist = min(min_dist, line_dist(uv, proj[3], proj[7]));
    
    // Draw edges with glow
    float edge = smoothstep(0.02, 0.005, min_dist);
    float glow = 0.01 / (min_dist + 0.01);
    
    float3 edge_color = float3(0.3, 0.7, 1.0);
    float3 glow_color = float3(0.1, 0.3, 0.6);
    
    float3 color = edge_color * edge + glow_color * glow * 0.3;
    color += float3(0.02, 0.02, 0.04);  // Background
    
    return float4(color, 1.0);
}

