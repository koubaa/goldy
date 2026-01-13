using System.Linq;
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
        
        // Pin and pass bind group layout handles
        unsafe
        {
            if (desc._layoutHandles != null && desc._layoutHandles.Length > 0)
            {
                fixed (nint* layoutPtr = desc._layoutHandles)
                {
                    nativeDesc.BindGroupLayouts = (nint)layoutPtr;
                    Handle = NativeMethods.RenderPipelineCreate(device.Handle, vertexShader.Handle, fragmentShader.Handle, in nativeDesc);
                }
            }
            else
            {
                Handle = NativeMethods.RenderPipelineCreate(device.Handle, vertexShader.Handle, fragmentShader.Handle, in nativeDesc);
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
    /// Bind group layouts (optional).
    /// </summary>
    public BindGroupLayout[] BindGroupLayouts { get; set; } = [];

    /// <summary>
    /// Depth stencil state (optional).
    /// </summary>
    public DepthStencilState? DepthStencil { get; set; }

    internal nint[]? _layoutHandles; // Keep alive during pipeline creation

    internal RenderPipelineDescNative ToNative()
    {
        // Marshal bind group layouts
        _layoutHandles = BindGroupLayouts.Length > 0 
            ? BindGroupLayouts.Select(l => l.Handle).ToArray() 
            : null;

        return new RenderPipelineDescNative
        {
            VertexAttributes = nint.Zero,
            VertexAttributeCount = 0,
            VertexStride = VertexStride,
            Topology = Topology,
            TargetFormat = TargetFormat,
            BindGroupLayouts = nint.Zero, // Will be set in Create
            BindGroupLayoutCount = (uint)BindGroupLayouts.Length,
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

