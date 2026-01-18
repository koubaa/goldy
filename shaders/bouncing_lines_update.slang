// Bouncing lines compute shader
// Updates line endpoint positions with wall bouncing physics

struct Line {
    float2 p1;        // First endpoint position
    float2 v1;        // First endpoint velocity
    float2 p2;        // Second endpoint position
    float2 v2;        // Second endpoint velocity
    uint color_index; // Color lookup index
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

#ifdef __BINDLESS__
#ifdef __DX12__
cbuffer BindlessIndices : register(b0, space0) {
    uint lineBufferIndex;
};

#define LINES (*DescriptorHandle<RWStructuredBuffer<Line>>(uint2(lineBufferIndex, 0)))
#else
// Metal/Vulkan bindless
struct ComputeResources {
    RWStructuredBuffer<Line> lines;
};

ParameterBlock<ComputeResources> gResources;

#define LINES gResources.lines
#endif
#else
// Traditional binding mode
[[vk::binding(0, 0)]] RWStructuredBuffer<Line> lines;

#define LINES lines
#endif

static const uint NUM_LINES = 20;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint idx = id.x;
    
    if (idx >= NUM_LINES) return;
    
    Line line = LINES[idx];
    
    // Update positions
    line.p1 += line.v1;
    line.p2 += line.v2;
    
    // Bounce endpoint 1 off walls
    if (line.p1.x < -1.0 || line.p1.x > 1.0) {
        line.v1.x = -line.v1.x;
        line.p1.x = clamp(line.p1.x, -1.0, 1.0);
    }
    if (line.p1.y < -1.0 || line.p1.y > 1.0) {
        line.v1.y = -line.v1.y;
        line.p1.y = clamp(line.p1.y, -1.0, 1.0);
    }
    
    // Bounce endpoint 2 off walls
    if (line.p2.x < -1.0 || line.p2.x > 1.0) {
        line.v2.x = -line.v2.x;
        line.p2.x = clamp(line.p2.x, -1.0, 1.0);
    }
    if (line.p2.y < -1.0 || line.p2.y > 1.0) {
        line.v2.y = -line.v2.y;
        line.p2.y = clamp(line.p2.y, -1.0, 1.0);
    }
    
    LINES[idx] = line;
}
