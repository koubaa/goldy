# Target Hardware

RAG targets **modern GPUs only**. This is a deliberate choice that enables significant API simplification.

## Minimum Requirements

| Platform | Minimum | Year | Notes |
|----------|---------|------|-------|
| **NVIDIA** | RTX 2000 / GTX 1600 | 2018 | Vulkan 1.4, mesh shaders |
| **AMD** | RDNA 1 (RX 5000) | 2019 | Vulkan 1.4, modern cache |
| **Intel** | Xe / Alchemist | 2022 | Vulkan 1.4, DX12 |
| **Apple** | M1 / A14 | 2020 | Metal 2+, unified memory |
| **Qualcomm** | Adreno 650+ | 2019 | Vulkan 1.2+ with extensions |

## What "Modern" Means

These hardware generations share key architectural features:

### 1. Coherent Cache Hierarchies

Modern GPUs have coherent L2 caches. This means:
- No manual cache flush/invalidate
- Simplified synchronization
- Better CPU-GPU data sharing

### 2. Bindless Resource Access

Descriptors live in GPU-visible memory:
- No descriptor set limits
- Index-based resource access in shaders
- Simpler binding model

### 3. Unified Memory (where available)

Apple Silicon and integrated GPUs share memory with CPU:
- Zero-copy data transfer
- Simplified allocation
- Lower latency

### 4. 64-bit Buffer Addresses

Shaders can use direct pointers:
- Buffer device address support
- Simpler data access patterns
- Pointer arithmetic in shaders

### 5. Dynamic Rendering

No need for render pass objects:
- Render targets specified at draw time
- Simpler API
- No render pass compatibility issues

## What This Excludes

RAG does **not** support:

| Excluded | Reason |
|----------|--------|
| GTX 900 series | No Vulkan 1.2 bindless |
| AMD GCN (RX 400/500) | Driver support ended 2021 |
| Intel Gen9 (HD 500/600) | Limited Vulkan support |
| iPhone 11 / A13 | No Metal 4.0 |
| Integrated Intel < Xe | Feature gaps |

## The Tradeoff

By requiring modern hardware, RAG can:

✅ **Assume modern features** as baseline (bindless, dynamic rendering)  
✅ **Skip compatibility layers** that add complexity  
✅ **Use native backend idioms** without translation  
✅ **Provide simpler API** with fewer edge cases  

But it cannot:

❌ Run on older hardware  
❌ Support the long tail of legacy systems  
❌ Deploy to low-end current devices  

## Checking Compatibility

RAG will report unsupported devices at initialization:

```rust
let instance = Instance::new()?;

// List available adapters
for adapter in instance.enumerate_adapters() {
    println!("{}: {:?}", adapter.name, adapter.device_type);
}

// create_device will fail on unsupported hardware
let device = instance.create_device(DeviceType::DiscreteGpu)?;
```

## Future Hardware

As GPUs evolve, RAG can adopt new features:

- **Automatic hazard detection** → Simpler barriers
- **Hardware descriptor management** → Simpler binding
- **Universal unified memory** → Simpler allocation

RAG's non-standard status means it can adopt these simplifications as hardware supports them, without waiting for committee consensus.

