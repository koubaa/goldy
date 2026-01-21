// Unified resource access for Goldy shaders
// Part of the goldy_exp module
//
// This module provides cross-platform buffer index access.
//
// == Vulkan Usage ==
//
//   import goldy_exp.resources;
//   [[vk::binding(0, 0)]] StructuredBuffer<MyData> g_Storage[];
//   [[vk::binding(1, 0)]] ConstantBuffer<MyUniforms> g_Uniforms[];
//   
//   #define DATA g_Storage[getBufferIndex(0)]
//   #define PARAMS g_Uniforms[getBufferIndex(1)]
//
// == DX12 Usage ==
//
//   DX12 requires direct named indices (array access has issues):
//
//   cbuffer BufferIndices : register(b0, space0) {
//       uint dataBufferIndex;
//       uint paramsBufferIndex;
//   };
//   #define DATA (*DescriptorHandle<StructuredBuffer<MyData>>(uint2(dataBufferIndex, 0)))
//   #define PARAMS (*DescriptorHandle<ConstantBuffer<MyUniforms>>(uint2(paramsBufferIndex, 0)))
//
// == Metal ==
//
//   Use ParameterBlock pattern (see goldy_exp.bindless documentation)

module resources;

static const uint MAX_BUFFER_INDICES = 16;

// ============================================================================
// Vulkan: Push constants with array access (works correctly)
// ============================================================================

#if defined(__SPIRV__)

struct BufferIndicesBlock {
    uint indices[MAX_BUFFER_INDICES];
};
[[vk::push_constant]] ConstantBuffer<BufferIndicesBlock> g_BufferIndicesPC;

public uint getBufferIndex(uint slot) {
    return g_BufferIndicesPC.indices[slot];
}

#endif // __SPIRV__

// ============================================================================
// DX12: NOTE - Use direct named indices in your shader instead of this module
// The array-based getBufferIndex() has issues with DX12/DXIL compilation.
// See usage example in the header comment.
// ============================================================================

#if defined(__DX12__)

// Provided for reference but NOT recommended for DX12 - use direct named indices
cbuffer BufferIndices : register(b0, space0) {
    uint g_BufferIndices[MAX_BUFFER_INDICES];
};

// WARNING: This may not work correctly on all DX12 configurations
// Prefer direct named indices: uint myBufferIndex; instead of array access
public uint getBufferIndex(uint slot) {
    return g_BufferIndices[slot];
}

#endif // __DX12__
