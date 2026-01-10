// Animated color gradient
// Uses UV coordinates for smooth color transitions

struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    // Create gradient based on UV coordinates
    float2 uv = input.uv;
    
    // Multiple gradient layers
    float3 c1 = float3(uv.x, 0.2, 1.0 - uv.x);
    float3 c2 = float3(1.0 - uv.y, uv.y * 0.5, uv.x * uv.y);
    
    float3 color = lerp(c1, c2, uv.y);
    return float4(color, 1.0);
}

