// Cross-platform DescriptorHandle<T> support for Goldy
// Auto-included when you import goldy_exp.
//
// Slang's DescriptorHandle<T> provides cross-platform bindless resource access:
//   - DXIL: ResourceDescriptorHeap[index]
//   - Metal: Argument buffer pointer
//   - SPIRV: Custom routing (below) to match Goldy's access-pattern bindings
//
// Usage:
//   uniform DescriptorHandle<ConstantBuffer<MyData>> data;
//   float x = (*data).value;

implementing goldy_exp;

// =============================================================================
// SPIRV-specific: Vulkan descriptor heap declarations
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
[[vk::binding(0, 0)]] __DynamicResource<__DynamicResourceKind.General> __scattered_heap[];
[[vk::binding(1, 0)]] __DynamicResource<__DynamicResourceKind.General> __broadcast_heap[];
[[vk::binding(2, 0)]] __DynamicResource<__DynamicResourceKind.General> __interpolated_heap[];
[[vk::binding(3, 0)]] __DynamicResource<__DynamicResourceKind.General> __direct_spatial_heap[];
[[vk::binding(4, 0)]] __DynamicResource<__DynamicResourceKind.Sampler> __filter_config_heap[];

// =============================================================================
// Cross-platform descriptor dereference
// Slang calls this when you write (*handle) to access the resource.
// =============================================================================
export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle) where T : IOpaqueDescriptor
{
    __target_switch
    {
    case spirv:
        // Route to correct heap based on access pattern
        uint index = ((uint2)handle).x;
        
        if (T.kind == DescriptorKind.Sampler)
            // Filter configuration (not data)
            return __filter_config_heap[index].asOpaqueDescriptor<T>();
        
        else if (T.kind == DescriptorKind.ConstantBuffer)
            // Broadcast access (all threads same address)
            return __broadcast_heap[index].asOpaqueDescriptor<T>();
        
        else if (T.kind == DescriptorKind.Texture)
        {
            // Spatial data - route by read vs write
            if (T.descriptorAccess == DescriptorAccess.Read)
                // Interpolated access (hardware filtering)
                return __interpolated_heap[index].asOpaqueDescriptor<T>();
            else
                // Direct spatial access (no filtering)
                return __direct_spatial_heap[index].asOpaqueDescriptor<T>();
        }
        else
            // Scattered access (buffers - any thread, any address)
            return __scattered_heap[index].asOpaqueDescriptor<T>();
    
    default:
        // DXIL/Metal: Use Slang's built-in
        return defaultGetDescriptorFromHandle(handle);
    }
}
