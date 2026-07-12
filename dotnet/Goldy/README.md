# Goldy - Modern GPU Library for .NET

Goldy is a modern GPU library targeting Vulkan 1.4+, DX12, and Metal, with C# bindings using .NET 7+ source-generated interop.

## Quick Start

```csharp
using Goldy;

// Create instance and device
using var instance = new Instance();
using var device = instance.RequestAdapter().RequestDevice();

// Headless triangle via TaskGraph
using var shader = new ShaderModule(device, ShaderModule.BuiltinVertexColor2D);
using var pipeline = new RenderPipeline(device, shader, shader, new RenderPipelineDesc
{
    VertexAttributes = VertexLayouts.Vertex2D,
    TargetFormat = TextureFormat.Rgba8Unorm,
});
using var target = new RenderTarget(device, 100, 100, TextureFormat.Rgba8Unorm);
using var graph = new TaskGraph();
using (var pass = graph.RenderPass("clear", target))
    pass.Clear(Color.CornflowerBlue);
graph.Dispatch(device);
byte[] pixels = target.ReadToCpu();
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

MIT License. See the [goldy repository](https://github.com/koubaa/goldy/blob/main/LICENSE) for the full text.

