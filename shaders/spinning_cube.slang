// Spinning 3D wireframe cube
// Rendered entirely in fragment shader using SDF techniques
// Uses goldy_exp module for vertex format utilities
//
// ## Bindless Support
//
// This shader demonstrates cross-platform bindless resource access:
//
// - Metal (__BINDLESS__ + __METAL__): Uses ParameterBlock for argument buffers
// - Vulkan (__BINDLESS__ + __SPIRV__): Uses push constants + descriptor arrays
// - DX12 (__BINDLESS__ + __HLSL__): Uses root constants + ResourceDescriptorHeap
// - Traditional mode (no __BINDLESS__): Standard binding syntax

import goldy_exp;

// Uniform structure
struct TimeUniforms {
    float time;
};

#ifdef __BINDLESS__

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer support
struct SpinningCubeResources {
    ConstantBuffer<TimeUniforms> uniforms;
};
ParameterBlock<SpinningCubeResources> gResources;
#define TIME gResources.uniforms.time

#elif defined(__SPIRV__)
// Vulkan: Use push constants for indices + descriptor array
import goldy_exp.bindless_indices;

// Global descriptor array of uniform buffers
[[vk::binding(1, 0)]] ConstantBuffer<TimeUniforms> g_UniformBuffers[];
#define TIME g_UniformBuffers[BINDLESS_INDEX(0)].time

#else
// DX12: Bindless via root constants + ResourceDescriptorHeap
cbuffer BindlessIndices : register(b0) {
    uint uniformsIndex;
};
#define TIME (*DescriptorHandle<ConstantBuffer<TimeUniforms>>(uint2(uniformsIndex, 0))).time
#endif

#else
// Traditional binding mode
[[vk::binding(0, 0)]]
ConstantBuffer<TimeUniforms> uniforms;
#define TIME uniforms.time
#endif

[shader("vertex")]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
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
float4 fs_main(FullscreenVarying input) : SV_Target {
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

