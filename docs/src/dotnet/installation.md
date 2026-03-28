# .NET Installation

Goldy provides first-class C# bindings via P/Invoke interop over the native Rust FFI layer.

## Requirements

- **.NET 8.0** or later
- Windows x64, Linux x64, or macOS (x64 / arm64)
- A Vulkan 1.4+, DX12, or Metal compatible GPU

## NuGet Package

```bash
dotnet add package Goldy
```

Or in your `.csproj`:

```xml
<PackageReference Include="Goldy" Version="0.1.*" />
```

The NuGet package bundles the native Goldy + Slang libraries for all supported platforms, so no separate native installation is needed.

## Building from Source

If you want to build from source, build the Rust FFI library first, then reference the C# project:

```bash
# Build native library
cargo build --package goldy-ffi --release

# Add C# project reference
dotnet add reference path/to/goldy/dotnet/Goldy/Goldy.csproj
```

## Verifying Installation

```csharp
using Goldy;

using var instance = new Instance();
Console.WriteLine("Available GPUs:");
foreach (var adapter in instance.EnumerateAdapters())
{
    Console.WriteLine($"  {adapter.Name} ({adapter.DeviceType})");
}

using var device = instance.CreateDevice(DeviceType.DiscreteGpu);
Console.WriteLine($"\nUsing: {device.AdapterId}");
```

## Next Steps

- [Quick Start](./quickstart.md) — Render your first frame
- [Compute Shaders](./compute.md) — GPGPU with C#
