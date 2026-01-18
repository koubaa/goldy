// Bouncing lines rendering shader
// Visualizes lines using instancing with LineList topology

struct Line {
    float2 p1;
    float2 v1;
    float2 p2;
    float2 v2;
    uint color_index;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

#ifdef __BINDLESS__

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer
struct RenderResources {
    StructuredBuffer<Line> lines;
};
ParameterBlock<RenderResources> gResources;
#define LINES gResources.lines

#elif defined(__SPIRV__)
// Vulkan: Push constants for indices + global descriptor arrays
import goldy_exp.bindless_indices;
[[vk::binding(0, 0)]] StructuredBuffer<Line> g_StorageBuffers[];
#define LINES g_StorageBuffers[getBindlessIndex(0)]

#elif defined(__HLSL__) || defined(__DX12__)
// DX12: Bindless via root constants + ResourceDescriptorHeap
cbuffer BindlessIndices : register(b0, space0) {
    uint lineBufferIndex;
};
#define LINES (*DescriptorHandle<StructuredBuffer<Line>>(uint2(lineBufferIndex, 0)))

#endif

#else
// Traditional binding mode
[[vk::binding(0, 0)]] StructuredBuffer<Line> lines;

#define LINES lines
#endif

struct VSOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

// Color palette matching the original example
static const float4 colors[6] = {
    float4(1.0, 0.0, 0.0, 1.0),  // Red
    float4(0.0, 1.0, 0.0, 1.0),  // Green
    float4(0.0, 0.0, 1.0, 1.0),  // Blue
    float4(1.0, 1.0, 0.0, 1.0),  // Yellow
    float4(1.0, 0.0, 1.0, 1.0),  // Magenta
    float4(0.0, 1.0, 1.0, 1.0),  // Cyan
};

[shader("vertex")]
VSOutput vs_main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID) {
    VSOutput output;
    
    Line line = LINES[instanceID];
    
    // vertexID is 0 for p1, 1 for p2
    float2 pos = (vertexID == 0) ? line.p1 : line.p2;
    
    output.position = float4(pos, 0.0, 1.0);
    output.color = colors[line.color_index % 6];
    
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    return input.color;
}
