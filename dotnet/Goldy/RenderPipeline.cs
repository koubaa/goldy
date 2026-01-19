using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// A render pipeline.
/// </summary>
public sealed class RenderPipeline : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    /// <summary>
    /// Create a new render pipeline.
    /// </summary>
    public RenderPipeline(
        Device device,
        ShaderModule vertexShader,
        ShaderModule fragmentShader,
        RenderPipelineDesc desc)
    {
        device.ThrowIfDisposed();
        
        var nativeDesc = desc.ToNative();
        Handle = NativeMethods.RenderPipelineCreate(device.Handle, vertexShader.Handle, fragmentShader.Handle, in nativeDesc);
        
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("RenderPipeline creation");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.RenderPipelineDestroy(Handle);
            _disposed = true;
        }
    }
}

/// <summary>
/// Description for creating a render pipeline.
/// </summary>
public class RenderPipelineDesc
{
    /// <summary>
    /// Vertex attributes.
    /// </summary>
    public VertexAttribute[] VertexAttributes { get; set; } = [];

    /// <summary>
    /// Stride in bytes between vertices.
    /// </summary>
    public uint VertexStride { get; set; } = 24; // Default Vertex2D stride

    /// <summary>
    /// Primitive topology.
    /// </summary>
    public PrimitiveTopology Topology { get; set; } = PrimitiveTopology.TriangleList;

    /// <summary>
    /// Target texture format.
    /// </summary>
    public TextureFormat TargetFormat { get; set; } = TextureFormat.Rgba8Unorm;

    /// <summary>
    /// Depth stencil state (optional).
    /// </summary>
    public DepthStencilState? DepthStencil { get; set; }

    internal RenderPipelineDescNative ToNative()
    {
        return new RenderPipelineDescNative
        {
            VertexAttributes = nint.Zero,
            VertexAttributeCount = 0,
            VertexStride = VertexStride,
            Topology = Topology,
            TargetFormat = TargetFormat,
            DepthEnabled = DepthStencil.HasValue,
            DepthFormat = DepthStencil?.Format ?? Goldy.DepthFormat.Depth24Plus,
            DepthWriteEnabled = DepthStencil?.DepthWriteEnabled ?? true,
            DepthCompare = DepthStencil?.DepthCompare ?? CompareFunction.Less,
        };
    }
}

/// <summary>
/// Vertex attribute description.
/// </summary>
public readonly record struct VertexAttribute(uint Location, VertexFormat Format, uint Offset);

/// <summary>
/// Depth/stencil state for render pipelines.
/// </summary>
public readonly record struct DepthStencilState(
    DepthFormat Format,
    bool DepthWriteEnabled = true,
    CompareFunction DepthCompare = CompareFunction.Less);

