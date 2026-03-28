# .NET Quick Start

This guide walks through rendering to an off-screen texture and reading pixels back to the CPU — the simplest complete Goldy program in C#.

## Headless Rendering

```csharp
using Goldy;

// 1. Create instance and device
using var instance = new Instance();
using var device = instance.CreateDevice(DeviceType.DiscreteGpu);

// 2. Create an off-screen render target
using var target = new RenderTarget(device, 800, 600, TextureFormat.Rgba8Unorm);

// 3. Record render commands
var encoder = new CommandEncoder();
encoder.Clear(new Color(0.2f, 0.3f, 0.8f, 1.0f));

// 4. Render to GPU texture
target.Render(encoder);

// 5. Read pixels back to CPU
byte[] pixels = target.ReadToCpu();
Console.WriteLine($"Rendered {pixels.Length} bytes ({target.Width}x{target.Height})");
```

## Windowed Rendering

For interactive applications, use `Surface` with a `winit`-compatible window handle:

```csharp
using Goldy;

// device and surface created from your windowing library
using var surface = new Surface(device, windowHandle);

// Game loop
while (running)
{
    // Acquire next swapchain frame
    using var frame = surface.Acquire();

    var encoder = new CommandEncoder();
    encoder.Clear(Color.CornflowerBlue);
    // ... draw calls ...

    frame.Render(encoder);
    surface.Present(frame);
}
```

## Shaders (Slang)

Goldy uses [Slang](https://shader-slang.org/) as its shader language across all backends:

```csharp
var source = """
    [shader("vertex")]
    float4 vs_main(float2 pos : POSITION) : SV_Position {
        return float4(pos, 0.0, 1.0);
    }

    [shader("fragment")]
    float4 fs_main() : SV_Target {
        return float4(1.0, 0.5, 0.0, 1.0);
    }
    """;

using var shader = new ShaderModule(device, source);
using var pipeline = new RenderPipeline(device, shader, new RenderPipelineDesc
{
    TargetFormat = TextureFormat.Rgba8Unorm,
    Topology = PrimitiveTopology.TriangleList,
});
```

## Resource Management

All Goldy objects implement `IDisposable`. Use `using` declarations or `using` blocks to ensure GPU resources are released promptly:

```csharp
// Preferred: using declaration (C# 8+)
using var device = instance.CreateDevice(DeviceType.DiscreteGpu);

// Also valid: explicit using block
using (var target = new RenderTarget(device, 512, 512, TextureFormat.Rgba8Unorm))
{
    // target is released when the block exits
}
```

## Next Steps

- [Compute Shaders](./compute.md) — GPU-accelerated computing
- [API Reference](./api.md) — Full .NET API
