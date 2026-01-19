// Cross-platform bindless resource index access
// Part of the goldy_exp module
//
// This module provides unified access to bindless resource indices across backends:
// - Vulkan: Push constants
// - DX12: Root constants (with ResourceDescriptorHeap/SamplerDescriptorHeap)
// - Metal: Uses ParameterBlock approach (see bindless.slang)
//
// == Usage Pattern ==
//
// In your shader:
//
//   #ifdef __BINDLESS__
//   import goldy_exp.bindless_indices;
//   
//   // Declare descriptor arrays (Vulkan)
//   [[vk::binding(0, 0)]] StructuredBuffer<MyData> g_Buffers[];
//   [[vk::binding(2, 0)]] Texture2D g_Textures[];
//   [[vk::binding(3, 0)]] SamplerState g_Samplers[];
//   
//   // Access via indices from push/root constants
//   MyData data = g_Buffers[BINDLESS_INDEX(0)][elementIdx];
//   float4 color = g_Textures[BINDLESS_INDEX(1)].Sample(g_Samplers[BINDLESS_INDEX(2)], uv);
//   #endif

// This module is imported directly, not __included into goldy_exp
// NOTE: No #ifdef __BINDLESS__ guard here - this module is ONLY imported when bindless is used
// and preprocessor defines don't propagate to imported modules.
module bindless_indices;

// Maximum number of indices passed via push/root constants
static const uint MAX_BINDLESS_INDICES = 16;

#ifdef __SPIRV__
// Vulkan: Use push constants
// Note: Must use struct + [[vk::push_constant]] ConstantBuffer syntax, NOT cbuffer
struct BindlessIndicesBlock {
    uint indices[MAX_BINDLESS_INDICES];
};
[[vk::push_constant]] ConstantBuffer<BindlessIndicesBlock> g_BindlessIndicesPC;
#define g_BindlessIndices g_BindlessIndicesPC.indices
#else
// DX12/HLSL: Use root constants at b0
cbuffer BindlessIndices : register(b0, space0) {
    uint g_BindlessIndices[MAX_BINDLESS_INDICES];
};
#endif

// Get bindless index at the specified slot
public uint getBindlessIndex(uint slot) {
    return g_BindlessIndices[slot];
}

// Convenience macro for accessing bindless indices
#define BINDLESS_INDEX(slot) g_BindlessIndices[slot]

// DX12 SM 6.6+ heap access helpers
// These require Shader Model 6.6 and the CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED flag
#ifdef __HLSL__
#define BINDLESS_BUFFER(type, slot) ResourceDescriptorHeap[BINDLESS_INDEX(slot)]
#define BINDLESS_TEXTURE(slot) ResourceDescriptorHeap[BINDLESS_INDEX(slot)]
#define BINDLESS_SAMPLER(slot) SamplerDescriptorHeap[BINDLESS_INDEX(slot)]
#endif

