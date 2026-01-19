// Cross-platform buffer index access
// Part of the goldy_exp module
//
// This module provides unified access to buffer indices across backends:
// - Vulkan: Push constants
// - DX12: Root constants (with ResourceDescriptorHeap/SamplerDescriptorHeap)
// - Metal: Uses ParameterBlock approach (see bindless.slang)
//
// == Usage Pattern ==
//
// In your shader:
//
//   import goldy_exp.buffer_indices;
//   
//   // Declare descriptor arrays (Vulkan)
//   [[vk::binding(0, 0)]] StructuredBuffer<MyData> g_Buffers[];
//   [[vk::binding(2, 0)]] Texture2D g_Textures[];
//   [[vk::binding(3, 0)]] SamplerState g_Samplers[];
//   
//   // Access via indices from push/root constants
//   MyData data = g_Buffers[getBufferIndex(0)][elementIdx];
//   float4 color = g_Textures[getBufferIndex(1)].Sample(g_Samplers[getBufferIndex(2)], uv);

module buffer_indices;

// Maximum number of indices passed via push/root constants
static const uint MAX_BUFFER_INDICES = 16;

#ifdef __SPIRV__
// Vulkan: Use push constants
// Note: Must use struct + [[vk::push_constant]] ConstantBuffer syntax, NOT cbuffer
struct BufferIndicesBlock {
    uint indices[MAX_BUFFER_INDICES];
};
[[vk::push_constant]] ConstantBuffer<BufferIndicesBlock> g_BufferIndicesPC;
#define g_BufferIndices g_BufferIndicesPC.indices
#else
// DX12: Use root constants at b0
cbuffer BufferIndices : register(b0, space0) {
    uint g_BufferIndices[MAX_BUFFER_INDICES];
};
#endif

// Get buffer index at the specified slot
public uint getBufferIndex(uint slot) {
    return g_BufferIndices[slot];
}
