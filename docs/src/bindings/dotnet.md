# .NET Bindings

Goldy provides first-class C# bindings via P/Invoke interop over the native Rust FFI layer.

## Installation

### NuGet Package

```bash
dotnet add package Goldy
```

Or add to your `.csproj` directly:

```xml
<PackageReference Include="Goldy" Version="0.2.*" />
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
- A GPU with Vulkan 1.4+, DX12, or Metal Tier 2+ support (CUDA and WebGPU backends are in progress; Tenstorrent is planned)

## Quick Start

### Headless Rendering

```csharp
using Goldy;

using var instance = new Instance();
using var device = instance.RequestAdapter().RequestDevice();
using var ctx = device.CreateContext();
using var retainedPool = new RetainedPool(device);
using var readback = retainedPool.AcquireTexture(
    100, 100, TextureFormat.Rgba8Unorm, TextureKind.Direct,
    TextureFlags.CopySrc | TextureFlags.CopyDst);

using var scheme = new Scheme(ctx);
using var rt = scheme.LeaseRenderTarget(100, 100, TextureFormat.Rgba8Unorm);
using (var pass = scheme.RenderPass("clear", rt))
    pass.Clear(Color.CornflowerBlue);

scheme.CopyToTexture(rt, readback);
using var memory = new MemoryExchange(ctx);
using var withdraw = memory.BindWithdrawTexture(scheme, readback);
using var submission = scheme.Submit();
using var claim = withdraw.Claim(submission);
byte[] pixels = claim.Consume();
```

See `Goldy.Examples/TriangleHeadless.cs` for a full triangle readback demo.

### Windowed Rendering

Record a retained scheme once, submit each frame, consume the present grant:

```csharp
using var scheme = new Scheme(ctx);
var (sceneRt, present) = RecordScheme(scheme, swapchain, pipeline, vertexParcel, screen, bg);

using var submission = scheme.Submit();
present.Consume(submission);
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
using var device = instance.RequestAdapter().RequestDevice();
using var ctx = device.CreateContext();
using var scheme = new Scheme(ctx);
```

## Key Differences from Rust

| Aspect | Rust | C# |
|--------|------|----|
| Instance creation | `Instance::new()?` | `new Instance()` |
| Error handling | `Result<T, GoldyError>` | Exceptions |
| Device lifetime | `Arc<Device>` | `IDisposable` / `using` |
| Retained buffer | `retained_pool.acquire_buffer_with_data(&data, access)` | `retainedPool.AcquireBuffer<T>(data, access)` → `Parcel` |
| Submission | `scheme.submit()?` | `scheme.Submit()` → `SchemeSubmission` |
| Enums | `DeviceType::DiscreteGpu` | `DeviceType.DiscreteGpu` |

## API Reference

### Scheme

```csharp
public sealed class Scheme : IDisposable
{
    public Scheme(Context ctx);
    public SchemeComputeNodeScope ComputeNode(string label, ComputePipeline pipeline);
    public SchemeRenderTargetLease LeaseRenderTarget(uint width, uint height, TextureFormat format, ...);
    public SchemeRenderPassScope RenderPass(string label, SchemeRenderTargetLease lease);
    public void CopyToTexture(SchemeRenderTargetLease src, Texture dst);
    public void CopyToPresent(SchemeRenderTargetLease src, PresentLease dst);
    public SchemeSubmission Submit();
}

public sealed class MemoryExchange : IDisposable
{
    public MemoryExchange(Context ctx);
    public WithdrawTransaction BindWithdraw(Scheme scheme, Parcel parcel);
    public WithdrawTransaction BindWithdrawTexture(Scheme scheme, Texture texture);
    public DepositTransaction BindDepositBuffer(Scheme scheme, Parcel destination, ulong capacity);
}

public sealed class WithdrawTransaction
{
    public WithdrawClaim Claim(SchemeSubmission submission);
}

public sealed class WithdrawClaim : IDisposable
{
    public byte[] Consume();
    public void Discard();
}
```

### SchemeRenderPassScope / SchemeComputeNodeScope

```csharp
using (var pass = scheme.RenderPass("main", rt))
{
    pass.WithParcel(vertexParcel, NodeAccess.Read);
    pass.Clear(Color.CornflowerBlue);
    pass.SetPipeline(pipeline);
    pass.SetVertexBuffer(0, vertexParcel);
    pass.Draw(3);
}

using (var node = scheme.ComputeNode("update", computePipeline))
{
    node.WithParcel(stateBuf, NodeAccess.ReadWrite);
    node.Dispatch(wgX, wgY, 1);
}
```

### SurfaceExchange / Transaction / Claim

```csharp
using var surface = GlfwSurfaceExchange.Create(ctx, window);
var present = surface.BindRenderTarget(scheme, sceneRt);
// each frame:
using var submission = scheme.Submit();
present.Claim(submission).Consume();
```

Graphics and compute both go through `Scheme`.

### Enums

```csharp
public enum DeviceType   { DiscreteGpu, IntegratedGpu, Cpu, Other }
public enum BackendType  { Vulkan, Metal, Dx12 }  // CUDA and WebGPU in progress in core Goldy
public enum BufferKind   { Scattered, Broadcast }
public enum NodeAccess   { Read, Write, ReadWrite, Overwrite }
```

### Headless vs windowed submission

Headless: record a scheme, bind a `MemoryExchange` withdraw, `Submit()`, then `withdraw.Claim(submission).Consume()`.

Windowed: record once with `SurfaceExchange.BindRenderTarget` (or `BindDestination` for compute-to-surface); each frame call `Submit()`, then `transaction.Claim(submission).Consume()`.
