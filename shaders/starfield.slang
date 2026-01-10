// 3D Starfield - flying forward through space
// Rendered entirely in fragment shader using time uniform

struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]]
cbuffer TimeBuffer {
    float time;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

float hash(float p) {
    return frac(sin(p * 127.1) * 43758.5453);
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = (input.uv - 0.5) * 2.0;
    float3 color = float3(0.0, 0.0, 0.02);
    
    int num_stars = 200;
    float speed = 0.3;  // Single speed for all stars
    
    for (int i = 0; i < num_stars; i++) {
        float fi = float(i);
        
        // Random angle for each star
        float angle = hash(fi) * 6.28318;
        // Random max distance (how far from center it can go)
        float max_dist = 0.3 + hash(fi + 50.0) * 1.2;
        
        // All stars cycle at same speed, just different phases
        float phase = hash(fi + 100.0);
        float cycle = frac(time * speed + phase);
        
        // Distance from center increases as cycle goes 0->1
        float dist = max_dist * cycle;
        
        float star_x = cos(angle) * dist;
        float star_y = sin(angle) * dist;
        
        float pixel_dist = length(uv - float2(star_x, star_y));
        
        // Size increases with distance (closer = bigger)
        float size = 0.002 + cycle * 0.015;
        
        // Brightness increases with distance
        float brightness = cycle * smoothstep(size, 0.0, pixel_dist);
        
        // Fade in at spawn, no fade out (stars just disappear at edge)
        float fade = smoothstep(0.0, 0.1, cycle);
        
        color += float3(0.9, 0.95, 1.0) * brightness * fade;
    }
    
    return float4(clamp(color, float3(0.0, 0.0, 0.0), float3(1.0, 1.0, 1.0)), 1.0);
}

