# Goldy - Modern GPU Library for .NET

Goldy is a modern GPU library targeting Vulkan 1.4+, DX12, and Metal, with C# bindings using .NET 7+ source-generated interop.

## Quick Start

```csharp
using Goldy;

// Create instance and device
using var instance = new Instance();
using var device = instance.RequestAdapter().RequestDevice();
using var ctx = device.CreateContext();

// Headless triangle via Scheme
using var shader = new ShaderModule(device, ShaderModule.BuiltinVertexColor2D);
using var pipeline = new RenderPipeline(device, shader, shader, new RenderPipelineDesc
{
    VertexAttributes = VertexLayouts.Vertex2D,
    TargetFormat = TextureFormat.Rgba8Unorm,
});
using var retainedPool = new RetainedPool(device);
using var readback = retainedPool.AcquireTexture(
    100, 100, TextureFormat.Rgba8Unorm, TextureKind.Direct,
    TextureFlags.CopySrc | TextureFlags.CopyDst);

using var scheme = new Scheme(ctx);
using var rt = scheme.LeaseRenderTarget(100, 100, TextureFormat.Rgba8Unorm);
using (var pass = scheme.RenderPassClear("clear", rt, Color.CornflowerBlue)) { }

scheme.CopyToTexture(rt, readback);
using var memory = new MemoryExchange(ctx);
using var withdraw = memory.BindWithdrawTexture(scheme, readback);
using var submission = scheme.Submit();
using var claim = withdraw.Claim(submission);
using var pixels = claim.Consume();
```

## Features

- Modern GPU API targeting Vulkan 1.4+, DX12, and Metal
- Slang shader compiler integration
- Retained scheme recording API
- SafeHandle-based resource management
- Compute shader support
- Windowed rendering via SurfaceExchange + transaction claims

## Requirements

- .NET 8.0 or later
- Windows x64, Linux x64, or macOS (x64/arm64)
- A Vulkan 1.4+, DX12, or Metal compatible GPU

## License

MIT License. See the [goldy repository](https://github.com/koubaa/goldy/blob/main/LICENSE) for the full text.
