// Simple colored triangle
// Uses vertex index to generate positions and colors procedurally

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

[goldy_vertex]
VertexOutput vs_main(VertexId idx) {
    // Hardcoded triangle vertices
    float2 positions[3];
    positions[0] = float2(0.0, 0.5);
    positions[1] = float2(-0.5, -0.5);
    positions[2] = float2(0.5, -0.5);
    
    float3 colors[3];
    colors[0] = float3(1.0, 0.0, 0.0); // Red
    colors[1] = float3(0.0, 1.0, 0.0); // Green
    colors[2] = float3(0.0, 0.0, 1.0); // Blue
    
    VertexOutput output;
    output.position = float4(positions[idx.value], 0.0, 1.0);
    output.color = colors[idx.value];
    return output;
}

[goldy_fragment]
float4 fs_main(VertexOutput input) : SV_Target {
    return float4(input.color, 1.0);
}
