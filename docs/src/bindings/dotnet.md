# .NET Bindings

Goldy provides first-class C# bindings via P/Invoke interop over the native Rust FFI layer.

## Installation

### NuGet Package

```bash
dotnet add package Goldy
```

Or add to your `.csproj` directly:

```xml
<PackageReference Include="Goldy" Version="0.1.*" />
```

The NuGet package bundles native Goldy + Slang libraries for all supported platforms — no separate native installation is needed.

### Building from Source

```bash
cargo build --package goldy-ffi --release
dotnet add reference path/to/goldy/dotnet/Goldy/Goldy.csproj
```

### Requirements

- .NET 8.0 or later
- Windows x64, Linux x64, or macOS (x64 / arm64)
- A GPU with Vulkan 1.4+, DX12, or Metal Tier 2+ support

## Quick Start

### Headless Rendering

```csharp
using Goldy;

using var instance = new Instance();
using var device = instance.CreateDevice(DeviceType.DiscreteGpu);
using var target = new RenderTarget(device, 800, 600, TextureFormat.Rgba8Unorm);
using var graph = new TaskGraph();

using (var pass = graph.RenderPass("clear", target))
    pass.Clear(Color.CornflowerBlue);

graph.Dispatch(device);
byte[] pixels = target.ReadToCpu();
```

See `Goldy.Examples/TriangleHeadless.cs` for a full triangle readback demo.

### Windowed Rendering

Build a task graph each frame, blit an offscreen `RenderTarget` to the swapchain, then present:

```csharp
using var graph = new TaskGraph();
graph.Clear();

using (var pass = graph.RenderPass("main", sceneRt))
{
    pass.BindBuffer(vertexBuffer, NodeAccess.Read);
    pass.Clear(Color.CornflowerBlue);
    pass.SetPipeline(pipeline);
    pass.SetVertexBuffer(0, vertexBuffer);
    pass.Draw(3);
}

var swapchain = graph.DeclareSwapchainOutput();
graph.CopyRenderTargetToSwapchain(sceneRt, swapchain);

using var frame = surface.Acquire();
surface.SubmitGraphToFrame(graph, frame);
surface.Present(frame);
```

See `Goldy.Examples/TriangleWindow.cs` and `GameOfLifeWindow.cs`.

### Shaders (Slang)

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

## Key Differences from Rust

| Aspect | Rust | C# |
|--------|------|----|
| Instance creation | `Instance::new()?` | `new Instance()` |
| Error handling | `Result<T, GoldyError>` | Exceptions |
| Device lifetime | `Arc<Device>` | `IDisposable` / `using` |
| Buffer creation | `device.alloc_buffer_with_data( &[T], access)` | `Buffer.WithData<T>(device, data, access)` |
| Pixel readback | `Vec<u8>` | `byte[]` |
| Enums | `DeviceType::DiscreteGpu` | `DeviceType.DiscreteGpu` |

## API Reference

### Instance

```csharp
public sealed class Instance : IDisposable
{
    public Instance();
    public IEnumerable<AdapterInfo> EnumerateAdapters();
    public Device CreateDevice(DeviceType deviceType);
    public Device CreateDeviceById(uint adapterId);
}
```

### Device

```csharp
public sealed class Device : IDisposable
{
    public uint AdapterId { get; }
    public bool IsValid { get; }
    public ulong GpuProgress { get; }
    public void WaitUntil(ulong value);
    public bool WaitUntilTimeout(ulong value, uint timeoutMs);
    public bool HasLibrary(string name);
}
```

### Buffer

```csharp
public sealed class Buffer : IDisposable
{
    public static Buffer New(Device device, ulong size, BufferKind access);
    public static Buffer WithData<T>(Device device, T[] data, BufferKind access)
        where T : unmanaged;
    public void Write<T>(T[] data) where T : unmanaged;
    public void Write<T>(ulong offset, T[] data) where T : unmanaged;
    public ulong Size { get; }
}
```

### ShaderModule

```csharp
public sealed class ShaderModule : IDisposable
{
    public ShaderModule(Device device, string slangSource);
}
```

### RenderPipeline / RenderPipelineDesc

```csharp
public sealed class RenderPipeline : IDisposable
{
    public RenderPipeline(Device device, ShaderModule shader, RenderPipelineDesc desc);
}

public sealed class RenderPipelineDesc
{
    public TextureFormat TargetFormat { get; set; }
    public PrimitiveTopology Topology { get; set; }
    // ... vertex layout, depth state
}
```

### TaskGraph / RenderPass / ComputeNode

```csharp
public sealed class TaskGraph : IDisposable
{
    public TaskGraph();
    public void Clear();
    public void WriteBuffer(Buffer buffer, ulong offset, ReadOnlySpan<byte> data);
    public RenderPassScope RenderPass(string label, RenderTarget target);
    public ComputeNodeScope ComputeNode(string label, ComputePipeline pipeline,
        uint workgroupsX = 1, uint workgroupsY = 1, uint workgroupsZ = 1);
    public SwapchainOutput DeclareSwapchainOutput();
    public void CopyRenderTargetToSwapchain(RenderTarget source, SwapchainOutput swapchain);
    public void Dispatch(Device device);  // headless: submit and block
}

public sealed class RenderPassScope : IDisposable
{
    public void BindBuffer(Buffer buffer, NodeAccess access);
    public void Clear(Color color);
    public void SetPipeline(RenderPipeline pipeline);
    public void SetVertexBuffer(uint slot, Buffer buffer);
    public void Draw(uint vertexCount, uint instanceCount = 1);
    public void DrawFullscreen();
}

public sealed class ComputeNodeScope : IDisposable
{
    public ComputeNodeScope BindBuffer(Buffer buffer, NodeAccess access);
    public ComputeNodeScope BindResourcesRaw(ReadOnlySpan<uint> indices);
}
```

Imperative graphics recording (`RenderPass`, `SurfaceFrame.Render`) was removed. Graphics goes through `TaskGraph`; standalone compute still uses `ComputeEncoder`.

### RenderTarget

```csharp
public sealed class RenderTarget : IDisposable
{
    public RenderTarget(Device device, uint width, uint height, TextureFormat format);
    public byte[] ReadToCpu();
    public void ReadToBuffer(byte[] output);
    public uint Width { get; }
    public uint Height { get; }
    public TextureFormat Format { get; }
    public int BufferSize { get; }
}
```

### Surface / SurfaceFrame

```csharp
public sealed class Surface : IDisposable
{
    public Surface(Device device, nint windowHandle);
    public SurfaceFrame Acquire();
    public void SubmitGraphToFrame(TaskGraph graph, SurfaceFrame frame);
    public void Present(SurfaceFrame frame);
    public void Resize(uint width, uint height);
    public uint Width { get; }
    public uint Height { get; }
}

public sealed class SurfaceFrame : IDisposable
{
    public uint Width { get; }
    public uint Height { get; }
}
```

### Compute

```csharp
public sealed class ComputePipeline : IDisposable
{
    public ComputePipeline(Device device, ShaderModule computeShader);
}

public sealed class ComputeEncoder
{
    public ComputeEncoder();
    public void SetPipeline(ComputePipeline pipeline);
    public void BindResources(params Buffer[] buffers);
    public void BindResourcesRaw(uint[] indices);
    public void Dispatch(uint x, uint y, uint z);
    public void DispatchIndirect(Buffer buffer, ulong offset);
    public void ClearBuffer(Buffer buffer, ulong offset, ulong size);
    public void Dispatch(Device device);     // dispatch and block
    public ulong Submit(Device device);      // submit, return timeline value
}
```

### Texture / Sampler

```csharp
public sealed class Texture : IDisposable
{
    public Texture(Device device, uint width, uint height, TextureFormat format,
                   TextureKind access, TextureFlags flags = TextureFlags.None);
    public void Write(byte[] data);
    public uint Width { get; }
    public uint Height { get; }
    public TextureFormat Format { get; }
}

public sealed class Sampler : IDisposable
{
    public Sampler(Device device, SamplerDesc desc);
}

public struct SamplerDesc
{
    public FilterMode MagFilter { get; set; }
    public FilterMode MinFilter { get; set; }
    public AddressMode AddressModeU { get; set; }
    public AddressMode AddressModeV { get; set; }
}
```

### Enums

```csharp
public enum DeviceType   { DiscreteGpu, IntegratedGpu, Cpu, Other }
public enum BackendType  { Vulkan, Metal, Dx12 }
public enum BufferKind   { Scattered, Broadcast }
public enum TextureKind { Interpolated, Direct }
public enum FilterMode   { Nearest, Linear }
public enum AddressMode  { Repeat, MirrorRepeat, ClampToEdge, ClampToBorder }

public enum TextureFormat
{
    Rgba8Unorm, Rgba8Srgb, Bgra8Unorm,
    Rgba16Float, Rgba32Float, Depth32Float,
}

public struct Color
{
    public float R, G, B, A;
    public Color(float r, float g, float b, float a);
    public static Color CornflowerBlue { get; }
    public static Color Black { get; }
    public static Color White { get; }
}
```

### Non-Blocking Submissions

`ComputeEncoder.Submit` returns a `ulong` device timeline value. Poll or wait on it via `Device.GpuProgress` and `Device.WaitUntil`:

```csharp
ulong ticket = computeEncoder.Submit(device);

// ... do other work ...

device.WaitUntil(ticket);   // block until the GPU catches up
```
