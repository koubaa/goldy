# Devices and Instances

The `Instance` and `Device` are the foundation of Goldy.

## Instance

The `Instance` represents a connection to the GPU subsystem and is used to discover available hardware.

```rust
use goldy::Instance;

let instance = Instance::new()?;
```

### Enumerating Adapters

```rust
for adapter in instance.enumerate_adapters() {
    println!("Name: {}", adapter.name);
    println!("Type: {:?}", adapter.device_type);
    println!("Backend: {:?}", adapter.backend);
}
```

Output:
```
Name: NVIDIA GeForce RTX 4060 Ti
Type: DiscreteGpu
Backend: Vulkan

Name: Intel(R) UHD Graphics 770
Type: IntegratedGpu
Backend: Vulkan
```

### Adapter Info

```rust
pub struct AdapterInfo {
    pub name: String,
    pub device_type: DeviceType,
    pub backend: BackendType,
}
```

## Device

A `Device` represents an opened connection to a specific GPU.

```rust
use goldy::{Instance, DeviceType};

let instance = Instance::new()?;
let device = instance.create_device(DeviceType::DiscreteGpu)?;
```

### Device Types

```rust
pub enum DeviceType {
    DiscreteGpu,    // Dedicated graphics card (NVIDIA, AMD)
    IntegratedGpu,  // Integrated graphics (Intel UHD, AMD APU)
    Cpu,            // Software rendering (fallback)
    Other,          // Unknown type
}
```

### Device Selection

Goldy will select the first matching adapter:

```rust
// Prefer discrete GPU (gaming/workstation)
let device = instance.create_device(DeviceType::DiscreteGpu)?;

// Or integrated (laptop power saving)
let device = instance.create_device(DeviceType::IntegratedGpu)?;
```

If no adapter matches, an error is returned.

### Adapter Info from Device

```rust
let device = instance.create_device(DeviceType::DiscreteGpu)?;
let info = device.adapter_info();
println!("Using: {}", info.name);
```

## Lifetime

The `Instance` must outlive all `Devices` created from it. Typically you keep both alive for the program's duration:

```rust
struct App {
    instance: Instance,
    device: Device,
    // other resources...
}

impl App {
    fn new() -> anyhow::Result<Self> {
        let instance = Instance::new()?;
        let device = instance.create_device(DeviceType::DiscreteGpu)?;
        Ok(Self { instance, device })
    }
}
```

## Error Handling

```rust
let instance = Instance::new()?;  // Fails if no Vulkan driver

// Fails if no discrete GPU
let device = instance.create_device(DeviceType::DiscreteGpu)?;
```

Common errors:
- No Vulkan driver installed
- No GPU of requested type
- GPU doesn't meet minimum requirements

## Multiple Devices

You can create multiple devices (useful for multi-GPU setups):

```rust
let instance = Instance::new()?;
let adapters = instance.enumerate_adapters();

// Open all discrete GPUs
let devices: Vec<Device> = adapters
    .iter()
    .filter(|a| a.device_type == DeviceType::DiscreteGpu)
    .map(|a| instance.create_device_for_adapter(a))
    .collect::<Result<_, _>>()?;
```

## Backend Type

```rust
pub enum BackendType {
    Vulkan,  // Currently the only backend
    Metal,   // Planned
    Dx12,    // Planned
}
```

Currently Goldy only supports Vulkan. Metal and DX12 backends are planned.

