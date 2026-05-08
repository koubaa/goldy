# .NET API Reference

Full API reference for the Goldy .NET bindings (`Goldy` namespace).

## Instance

```csharp
public sealed class Instance : IDisposable
{
    /// Create a new Goldy instance, discovering available GPU backends.
    public Instance();

    /// List all available GPU adapters.
    public IEnumerable<AdapterInfo> EnumerateAdapters();

    /// Create a device for the preferred adapter type.
    public Device CreateDevice(DeviceType deviceType);

    /// Create a device for a specific adapter by ID.
    public Device CreateDeviceById(uint adapterId);
}
```

## Device

```csharp
public sealed class Device : IDisposable
{
    /// Adapter ID this device was created on.
    public uint AdapterId { get; }

    /// Whether the device is still valid (not lost).
    public bool IsValid { get; }

    /// Latest completed value on the device timeline (see `Submit` / `WaitUntil`).
    public ulong GpuProgress { get; }

    /// Block until the device timeline reaches at least `value`.
    public void WaitUntil(ulong value);

    /// Like `WaitUntil` but returns false if `timeoutMs` elapses first.
    public bool WaitUntilTimeout(ulong value, uint timeoutMs);

    /// Check if a named shader library is registered.
    public bool HasLibrary(string name);
}
```

## AdapterInfo

```csharp
public struct AdapterInfo
{
    public string Name { get; }
    public DeviceType DeviceType { get; }
    public BackendType Backend { get; }
}
```

## DeviceType / BackendType

```csharp
public enum DeviceType { DiscreteGpu, IntegratedGpu, Cpu, Other }
public enum BackendType { Vulkan, Metal, Dx12 }
```

## Buffer

```csharp
public sealed class Buffer : IDisposable
{
    /// Create an empty buffer.
    public static Buffer New(Device device, ulong size, DataAccess access);

    /// Create a buffer pre-filled with data.
    public static Buffer WithData<T>(Device device, T[] data, DataAccess access)
        where T : unmanaged;

    /// Write data into the buffer.
    public void Write<T>(T[] data) where T : unmanaged;
    public void Write<T>(ulong offset, T[] data) where T : unmanaged;

    /// Size in bytes.
    public ulong Size { get; }
}

public enum DataAccess
{
    /// Any thread, any address — maps to StructuredBuffer / RWStructuredBuffer.
    Scattered,
    /// All threads read same address — maps to ConstantBuffer.
    Broadcast,
}
```

## ShaderModule

```csharp
public sealed class ShaderModule : IDisposable
{
    /// Compile a shader from Slang source.
    public ShaderModule(Device device, string slangSource);
}
```

## RenderPipeline / RenderPipelineDesc

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

## CommandEncoder / RenderPass

```csharp
public sealed class CommandEncoder
{
    public CommandEncoder();

    /// Record a clear-color command.
    public void Clear(Color color);

    /// Begin a render pass.
    public RenderPass BeginRenderPass();
}

public sealed class RenderPass : IDisposable
{
    public void SetPipeline(RenderPipeline pipeline);
    public void SetVertexBuffer(uint slot, Buffer buffer);
    public void Draw(uint vertexStart, uint vertexCount, uint instanceStart = 0, uint instanceCount = 1);
    public void DrawIndexed(uint indexCount, uint instanceCount = 1);
}
```

## Surface / SurfaceFrame

```csharp
public sealed class Surface : IDisposable
{
    /// Create a surface from a raw window handle.
    public Surface(Device device, nint windowHandle);

    /// Acquire the next swapchain frame for rendering.
    public SurfaceFrame Acquire();

    /// Present a rendered frame to the display.
    public void Present(SurfaceFrame frame);

    /// Resize the swapchain after a window resize event.
    public void Resize(uint width, uint height);

    public uint Width { get; }
    public uint Height { get; }
}

public sealed class SurfaceFrame : IDisposable
{
    /// Render commands to this frame.
    public void Render(CommandEncoder encoder);
}
```

## RenderTarget (Headless)

```csharp
public sealed class RenderTarget : IDisposable
{
    /// Create an off-screen render target.
    public RenderTarget(Device device, uint width, uint height, TextureFormat format);

    /// Render commands to the GPU texture.
    public void Render(CommandEncoder encoder);

    /// Read rendered pixels back to CPU memory.
    public byte[] ReadToCpu();
    public void ReadToBuffer(byte[] output);

    public uint Width { get; }
    public uint Height { get; }
    public TextureFormat Format { get; }
    public int BufferSize { get; }
}
```

## Compute

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

    /// Dispatch and block until complete.
    public void Dispatch(Device device);

    /// Submit without blocking; returns a device timeline value (`ulong`).
    public ulong Submit(Device device);
}
```

## `TimelineValue`

Non-blocking submissions return a `ulong` device timeline counter. Use `Device.WaitUntil` or compare against `Device.GpuProgress`.

## Texture / Sampler

```csharp
public sealed class Texture : IDisposable
{
    public Texture(Device device, uint width, uint height, TextureFormat format,
                   SpatialAccess access, TextureFlags flags = TextureFlags.None);

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

public enum FilterMode  { Nearest, Linear }
public enum AddressMode { Repeat, MirrorRepeat, ClampToEdge, ClampToBorder }
public enum SpatialAccess { Interpolated, Direct }
```

## Color / TextureFormat

```csharp
public struct Color
{
    public float R, G, B, A;
    public Color(float r, float g, float b, float a);

    public static Color CornflowerBlue { get; }
    public static Color Black { get; }
    public static Color White { get; }
}

public enum TextureFormat
{
    Rgba8Unorm, Rgba8Srgb, Bgra8Unorm,
    Rgba16Float, Rgba32Float, Depth32Float,
}
```
