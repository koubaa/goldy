# Goldy - Modern GPU Library for .NET

Goldy is a modern GPU library targeting Vulkan 1.4+, DX12, and Metal, with C# bindings using .NET 7+ source-generated interop.

## Quick Start

```csharp
using Goldy;

// Create instance and device
using var instance = new Instance();
using var device = instance.RequestAdapter().RequestDevice();

// Compute example (graphics uses TaskGraph in Rust — see goldy/examples/)
using var pipeline = new ComputePipeline(device, computeShader);
using var encoder = new ComputeEncoder();
encoder.SetPipeline(pipeline);
encoder.Dispatch(1, 1, 1);
encoder.Execute(device);
```

## Features

- Modern GPU API targeting Vulkan 1.4+, DX12, and Metal
- Slang shader compiler integration
- Zero-allocation render loop design
- SafeHandle-based resource management
- Compute shader support
- Windowed rendering via Surface API

## Requirements

- .NET 8.0 or later
- Windows x64, Linux x64, or macOS (x64/arm64)
- A Vulkan 1.4+, DX12, or Metal compatible GPU

## License

LGPL-2.1-or-later. A commercial license is also available; contact [koubaa on github](permament email tbd) for terms.

