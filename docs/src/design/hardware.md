# Target Hardware

Goldy targets modern GPUs exclusively. This is a deliberate design choice — by requiring hardware from roughly 2018 onward, Goldy can use bindless descriptors, dynamic rendering, and coherent caches as baseline assumptions rather than optional features.

## Backend Requirements

### Vulkan 1.4+

Goldy requires Vulkan 1.4, which promotes several extensions that were optional in earlier versions to core:

| Feature | Vulkan history | Goldy usage |
|---------|---------------|-------------|
| Dynamic rendering | `VK_KHR_dynamic_rendering` (1.3) | No render pass objects |
| Descriptor indexing | `VK_EXT_descriptor_indexing` (1.2) | Bindless resource access |
| Buffer device address | `VK_KHR_buffer_device_address` (1.2) | 64-bit GPU pointers |
| Synchronization2 | `VK_KHR_synchronization2` (1.3) | Simplified barrier model |
| Push descriptors | Core in 1.4 | Efficient uniform updates |

**Supported hardware:**

- NVIDIA: Turing and later (RTX 2000 / GTX 1600 series, 2018+)
- AMD: RDNA 1 and later (RX 5000 series, 2019+)
- Intel: Xe architecture and later (Arc, 2022+)
- Qualcomm: Adreno 650+ (2019+, driver dependent)

### DX12

Goldy's DX12 backend requires:

| Requirement | Details |
|-------------|---------|
| D3D12 Enhanced Barriers | Windows 11 + WDDM 3.0+ driver |
| `ResourceDescriptorHeap` | SM 6.6 bindless (Shader Model 6.6) |
| Root constants | Push constants equivalent |

Enhanced Barriers are mandatory — Goldy does not fall back to legacy resource state transitions. This effectively requires Windows 11 with a modern driver.

For software rendering and CI, Goldy supports the **WARP** software rasterizer via `GOLDY_DX12_FORCE_WARP=1`.

### Metal Tier 2+

Goldy's Metal backend is native (no MoltenVK) and requires **Argument Buffers Tier 2** for bindless resource access:

| Requirement | Details |
|-------------|---------|
| Argument Buffers Tier 2 | Bindless via `ParameterBlock` |
| MSL (via Slang) | Slang compiles directly to Metal Shading Language |

**Supported hardware:**

- Apple Silicon: All models (M1/M2/M3/M4, A14+)
- Intel Macs: 2017+ (different iGPUs; some very early Intel UHD may not qualify)
- AMD discrete GPUs in Macs: 2015+

Older Intel integrated GPUs (pre-2017 Macs) are **not supported** — they lack Argument Buffers Tier 2.

## What "Modern GPU" Means for Goldy

Goldy's hardware floor is defined by a set of architectural capabilities, not specific product names:

| Capability | Why Goldy needs it |
|------------|-------------------|
| **Coherent L2 cache** | No manual cache flush/invalidate logic |
| **Bindless descriptors** | Single global descriptor model, no set layouts |
| **Dynamic rendering** | No render pass objects or framebuffer compatibility |
| **64-bit buffer addresses** | Direct pointer access in shaders |
| **Unified or REBAR memory** | Simplified CPU-GPU data transfer |

GPUs from roughly 2018 onward universally support these features. The specific API version requirements (Vulkan 1.4, DX12 Enhanced Barriers, Metal Tier 2) are the mechanism by which Goldy enforces this floor.

## What This Excludes

| Excluded | Reason |
|----------|--------|
| NVIDIA GTX 900 series (Maxwell) | No Vulkan 1.4 support |
| AMD GCN (RX 400/500) | Driver support ended; limited bindless |
| Intel Gen9 (HD 500/600) | Incomplete Vulkan feature coverage |
| Intel integrated GPUs pre-2017 (Mac) | No Argument Buffers Tier 2 |
| Pre-Windows 11 DX12 | No Enhanced Barriers |

## Checking Compatibility

Goldy reports unsupported devices at initialization:

```rust
let instance = Instance::new()?;

for adapter in instance.enumerate_adapters() {
    println!("{}: {:?}", adapter.name, adapter.device_type);
}

// create_device returns an error on unsupported hardware
let device = instance.create_device(DeviceType::DiscreteGpu)?;
```

## The Tradeoff

By drawing a line at modern hardware, Goldy avoids the fallback paths, compatibility checks, and feature-level negotiation that dominate traditional GPU libraries. Every code path in Goldy assumes the full feature set is available. This keeps the implementation small and the API surface predictable.

The cost is clear: Goldy cannot run on the long tail of older hardware. For applications that need broad device support, [wgpu](./comparison.md) is the better choice.
