// Cross-platform DescriptorHandle<T> support for Goldy
// Auto-included when you import goldy_exp.
//
// Slang's DescriptorHandle<T> provides cross-platform bindless resource access:
//   - DXIL: ResourceDescriptorHeap[index]
//   - Metal: Argument buffer pointer
//   - SPIRV: Custom routing via Goldy's binding layout
//
// Usage:
//   uniform ConstantBuffer<MyData>.Handle data;
//   float x = (*data).value;
//
// Or equivalently:
//   uniform DescriptorHandle<ConstantBuffer<MyData>> data;
//   float x = (*data).value;

implementing goldy_exp;

// =============================================================================
// SPIRV-specific: Custom getDescriptorFromHandle for Goldy's binding layout
//
// Organized by ACCESS PATTERN (what the hardware actually optimizes for):
//
//   0: SCATTERED      - Any thread, any address, read/write. No coherence.
//   1: BROADCAST      - All threads same address. Hardware broadcast cache.
//   2: INTERPOLATED   - Hardware filtering between neighbors (texture units).
//   3: DIRECT_SPATIAL - 2D/3D indexing without filtering. Read/write.
//   4: FILTER_CONFIG  - Not data. Settings for interpolated access.
//
// Must match Goldy's Vulkan backend (see vulkan/types.rs::bindless_bindings)
// =============================================================================

#ifdef __SPIRV__

// Goldy's descriptor heaps - organized by access pattern
// These use __DynamicResource which is Slang's generic bindless resource array type

// Binding 0: SCATTERED - Storage buffers (StructuredBuffer, RWStructuredBuffer)
[[vk::binding(0, 0)]]
__DynamicResource<__DynamicResourceKind.General> g_GoldyStorageHeap[];

// Binding 1: BROADCAST - Uniform/constant buffers (ConstantBuffer)
[[vk::binding(1, 0)]]
__DynamicResource<__DynamicResourceKind.General> g_GoldyUniformHeap[];

// Binding 2: INTERPOLATED - Sampled images (Texture2D with sampler)
[[vk::binding(2, 0)]]
__DynamicResource<__DynamicResourceKind.General> g_GoldyTextureHeap[];

// Binding 3: DIRECT_SPATIAL - Storage images (RWTexture2D)
[[vk::binding(3, 0)]]
__DynamicResource<__DynamicResourceKind.General> g_GoldyStorageImageHeap[];

// Binding 4: FILTER_CONFIG - Samplers (SamplerState)
[[vk::binding(4, 0)]]
__DynamicResource<__DynamicResourceKind.Sampler> g_GoldySamplerHeap[];

// Custom getDescriptorFromHandle for SPIRV that routes to Goldy's binding layout
export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle) where T : IOpaqueDescriptor
{
    let idx = ((uint2)handle).x;
    
    switch (T.kind)
    {
    case DescriptorKind.Sampler:
        // Binding 4: FILTER_CONFIG
        return g_GoldySamplerHeap[idx].asOpaqueDescriptor<T>();
        
    case DescriptorKind.CombinedTextureSampler:
        // Combined texture+sampler - use texture heap
        return g_GoldyTextureHeap[idx].asOpaqueDescriptor<T>();
        
    case DescriptorKind.Texture:
        if (T.descriptorAccess == DescriptorAccess.Read)
            // Binding 2: INTERPOLATED - read-only textures
            return g_GoldyTextureHeap[idx].asOpaqueDescriptor<T>();
        else
            // Binding 3: DIRECT_SPATIAL - read-write textures
            return g_GoldyStorageImageHeap[idx].asOpaqueDescriptor<T>();
            
    case DescriptorKind.ConstantBuffer:
        // Binding 1: BROADCAST - uniform buffers
        return g_GoldyUniformHeap[idx].asOpaqueDescriptor<T>();
        
    case DescriptorKind.StorageBuffer:
        // Binding 0: SCATTERED - storage buffers (both read and read-write)
        return g_GoldyStorageHeap[idx].asOpaqueDescriptor<T>();
        
    case DescriptorKind.UniformTexelBuffer:
        // Uniform texel buffers (read-only) - use storage heap
        return g_GoldyStorageHeap[idx].asOpaqueDescriptor<T>();
        
    case DescriptorKind.StorageTexelBuffer:
        // Storage texel buffers (read-write) - use storage heap
        return g_GoldyStorageHeap[idx].asOpaqueDescriptor<T>();
        
    default:
        // Unknown types default to storage heap
        return g_GoldyStorageHeap[idx].asOpaqueDescriptor<T>();
    }
}

#else

// =============================================================================
// Non-SPIRV targets: Use Slang's default behavior
// - HLSL/DXIL uses ResourceDescriptorHeap/SamplerDescriptorHeap  
// - Metal uses argument buffer pointers
// =============================================================================

export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle) where T : IOpaqueDescriptor
{
    return defaultGetDescriptorFromHandle(handle);
}

#endif
