# WASI Integration

RAG is designed to be exposed to WebAssembly guests through WASI (WebAssembly System Interface).

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     WASM Guest App                           │
│               (uses rag-guest crate)                         │
│                                                              │
│   Compiled to wasm32-wasip2, runs in Wasmtime               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          │ WIT interface (rag:gpu)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       rag-host                               │
│            (implements WIT, uses RAG internally)             │
│                                                              │
│   • Resource handle management (ResourceTable)              │
│   • Memory transfer (guest ↔ GPU)                           │
│   • Capability exposure                                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                          RAG                                 │
│                   (same core library)                        │
└─────────────────────────────────────────────────────────────┘
```

## WIT Interface

The WASI interface is defined using WIT (WebAssembly Interface Types):

```wit
// rag:gpu/types
package rag:gpu;

interface types {
    record color {
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    }
    
    enum texture-format {
        rgba8-unorm,
        rgba8-srgb,
        // ...
    }
    
    flags buffer-usage {
        vertex,
        index,
        uniform,
        storage,
    }
}
```

```wit
// rag:gpu/device
interface device {
    use types.{buffer-usage, texture-format};
    
    resource device {
        constructor(device-type: device-type);
        adapter-info: func() -> adapter-info;
    }
    
    resource buffer {
        constructor(device: borrow<device>, size: u64, usage: buffer-usage);
        write: func(data: list<u8>);
        size: func() -> u64;
    }
}
```

## Host Implementation

The host side implements the WIT interfaces:

```rust
use wasmtime::component::ResourceTable;

pub struct GpuHostState {
    backend: Box<dyn GpuBackend>,
    devices: ResourceTable<DeviceId>,
    buffers: ResourceTable<BufferId>,
    // ...
}

impl device::Host for GpuHostState {
    fn create_device(&mut self, device_type: DeviceType) -> Result<Resource<Device>> {
        let id = self.backend.create_device(...)?;
        Ok(self.devices.push(id)?)
    }
}

impl device::HostBuffer for GpuHostState {
    fn new(&mut self, device: Resource<Device>, size: u64, usage: BufferUsage) -> Result<Resource<Buffer>> {
        let device_id = self.devices.get(device)?;
        let buffer_id = self.backend.create_buffer(device_id, size, usage)?;
        Ok(self.buffers.push(buffer_id)?)
    }
    
    fn write(&mut self, buffer: Resource<Buffer>, data: Vec<u8>) -> Result<()> {
        let buffer_id = self.buffers.get(buffer)?;
        self.backend.write_buffer(buffer_id, &data)?;
        Ok(())
    }
}
```

## Guest Usage

From a WASM guest application:

```rust
// In guest crate (compiled to wasm32-wasip2)
use rag_guest::{Device, Buffer, BufferUsage, Color};

fn render() -> Vec<u8> {
    let device = Device::new(DeviceType::DiscreteGpu);
    
    let vertices = vec![/* ... */];
    let buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX);
    
    // ... render commands ...
    
    frame.render(encoder)
}
```

## Security Model

WASI provides sandboxing:

| Capability | Host Controls |
|------------|---------------|
| GPU access | Which adapters are visible |
| Memory | Maximum allocation size |
| Execution | Timeout on GPU operations |

The guest cannot:
- Access GPU memory directly
- Escape the sandbox
- Crash the host

## Resource Management

Resources (buffers, textures, pipelines) are managed through Wasmtime's `ResourceTable`:

```rust
// Host side
pub struct ResourceTable<T> {
    entries: Vec<Option<T>>,
    free_list: Vec<u32>,
}

impl<T> ResourceTable<T> {
    pub fn push(&mut self, value: T) -> Resource<T>;
    pub fn get(&self, handle: Resource<T>) -> Result<&T>;
    pub fn remove(&mut self, handle: Resource<T>) -> Result<T>;
}
```

When a guest drops a resource handle, the host cleans up the GPU resource.

## Memory Transfer

Data crosses the WASM boundary through WIT types:

```wit
// list<u8> becomes Vec<u8> in Rust
write: func(data: list<u8>);
```

The host copies data between:
1. Guest WASM linear memory
2. Host memory
3. GPU memory

## Status

WASI integration is designed but not fully implemented:

| Component | Status |
|-----------|--------|
| WIT definitions | 🔜 Planned |
| Host implementation | 🔜 Planned |
| Guest crate | 🔜 Planned |
| Wasmtime integration | 🔜 Planned |

Any WASI runtime can implement the `rag:gpu` interface.

