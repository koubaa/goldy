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

        var nativeAttrs = desc.ToNativeAttributes();
        unsafe
        {
            fixed (VertexAttributeNative* pAttrs = nativeAttrs)
            {
                var nativeDesc = desc.ToNative(pAttrs, (uint)nativeAttrs.Length);
                Handle = NativeMethods.RenderPipelineCreate(
                    device.Handle, vertexShader.Handle, fragmentShader.Handle, in nativeDesc);
            }
        }

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

    internal unsafe RenderPipelineDescNative ToNative(VertexAttributeNative* attributes, uint attributeCount)
    {
        return new RenderPipelineDescNative
        {
            VertexAttributes = attributeCount > 0 ? (nint)attributes : nint.Zero,
            VertexAttributeCount = attributeCount,
            VertexStride = VertexStride,
            Topology = Topology,
            TargetFormat = TargetFormat,
            DepthEnabled = DepthStencil.HasValue,
            DepthFormat = DepthStencil?.Format ?? Goldy.DepthFormat.Depth24Plus,
            DepthWriteEnabled = DepthStencil?.DepthWriteEnabled ?? true,
            DepthCompare = DepthStencil?.DepthCompare ?? CompareFunction.Less,
        };
    }

    internal VertexAttributeNative[] ToNativeAttributes() =>
        VertexAttributes.Select(a => new VertexAttributeNative
        {
            Location = a.Location,
            Format = a.Format,
            Offset = a.Offset,
        }).ToArray();
}

/// <summary>
/// Common vertex buffer layouts.
/// </summary>
public static class VertexLayouts
{
    /// <summary>Position (float2) + color (float4), 24-byte stride.</summary>
    public static VertexAttribute[] Vertex2D { get; } =
    [
        new(0, VertexFormat.Float32x2, 0),
        new(1, VertexFormat.Float32x4, 8),
    ];
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

