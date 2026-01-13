# Goldy - Modern GPU Library for .NET

Goldy is a modern GPU library targeting Vulkan 1.3+, DX12, and Metal, with C# bindings using .NET 7+ source-generated interop.

## Quick Start

```csharp
using Goldy;

// Create instance and device
using var instance = new Instance();
using var device = instance.CreateDevice(DeviceType.DiscreteGpu);

// Create a render target
using var target = new RenderTarget(device, 800, 600, TextureFormat.Rgba8Unorm);

// Create and record commands
var encoder = new CommandEncoder();
encoder.Clear(new Color(0.2f, 0.3f, 0.8f, 1.0f));

// Render
target.Render(encoder);

// Read pixels back to CPU
byte[] pixels = target.ReadToCpu();
```

## Features

- Modern GPU API targeting Vulkan 1.3+, DX12, and Metal
- Slang shader compiler integration
- Zero-allocation render loop design
- SafeHandle-based resource management
- Compute shader support
- Windowed rendering via Surface API

## Requirements

- .NET 8.0 or later
- Windows x64, Linux x64, or macOS (x64/arm64)
- A Vulkan 1.3+, DX12, or Metal compatible GPU

## License

MIT License

