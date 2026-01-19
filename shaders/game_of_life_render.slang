// Game of Life rendering shader
// Renders the grid as a fullscreen quad with cell colors

static const uint GRID_WIDTH = 128;
static const uint GRID_HEIGHT = 128;

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer
struct GameOfLifeRenderResources {
    StructuredBuffer<uint> cells;
};
ParameterBlock<GameOfLifeRenderResources> gResources;
#define CELLS gResources.cells

#elif defined(__SPIRV__)
// Vulkan: Push constants for indices + global descriptor arrays
import goldy_exp.buffer_indices;
[[vk::binding(0, 0)]] StructuredBuffer<uint> g_StorageBuffers[];
#define CELLS g_StorageBuffers[getBufferIndex(0)]

#elif defined(__DX12__)
// DX12: Root constants + ResourceDescriptorHeap
cbuffer BufferIndices : register(b0, space0) {
    uint cellsBufferIndex;
};
#define CELLS (*DescriptorHandle<StructuredBuffer<uint>>(uint2(cellsBufferIndex, 0)))

#endif

struct VSOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Fullscreen triangle (covers entire viewport with one triangle)
static const float2 positions[3] = {
    float2(-1, -1),
    float2( 3, -1),
    float2(-1,  3)
};

static const float2 uvs[3] = {
    float2(0, 1),
    float2(2, 1),
    float2(0, -1)
};

[shader("vertex")]
VSOutput vs_main(uint vertexID : SV_VertexID) {
    VSOutput output;
    output.position = float4(positions[vertexID], 0.0, 1.0);
    output.uv = uvs[vertexID];
    return output;
}

[shader("fragment")]
float4 fs_main(VSOutput input) : SV_Target {
    // Get grid coordinates from UV
    float2 uv = input.uv;
    
    // Flip Y so origin is top-left
    uv.y = 1.0 - uv.y;
    
    int x = int(uv.x * GRID_WIDTH);
    int y = int(uv.y * GRID_HEIGHT);
    
    // Clamp to grid bounds
    x = clamp(x, 0, int(GRID_WIDTH) - 1);
    y = clamp(y, 0, int(GRID_HEIGHT) - 1);
    
    uint idx = y * GRID_WIDTH + x;
    uint cell = CELLS[idx];
    
    // Grid lines for visual clarity
    float2 cellUV = frac(float2(uv.x * GRID_WIDTH, uv.y * GRID_HEIGHT));
    float gridLine = (cellUV.x < 0.05 || cellUV.y < 0.05) ? 0.15 : 0.0;
    
    if (cell == 1) {
        // Living cell - bright green with slight variation
        float3 alive = float3(0.2, 0.9, 0.3);
        return float4(alive + gridLine, 1.0);
    } else {
        // Dead cell - dark background
        float3 dead = float3(0.05, 0.08, 0.1);
        return float4(dead + gridLine, 1.0);
    }
}
