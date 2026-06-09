using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// A GPU render target that stays on the GPU until explicitly read.
/// </summary>
public sealed class RenderTarget : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    /// <summary>
    /// Create a new render target without a depth buffer.
    /// </summary>
    public RenderTarget(Device device, uint width, uint height, TextureFormat format)
    {
        device.ThrowIfDisposed();
        
        Handle = NativeMethods.RenderTargetCreate(device.Handle, width, height, format);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("RenderTarget creation");
        
        Width = width;
        Height = height;
        Format = format;
        DepthFormat = null;
    }

    /// <summary>
    /// Create a new render target with a depth buffer.
    /// </summary>
    public RenderTarget(Device device, uint width, uint height, TextureFormat colorFormat, DepthFormat depthFormat)
    {
        device.ThrowIfDisposed();
        
        Handle = NativeMethods.RenderTargetCreateWithDepth(device.Handle, width, height, colorFormat, depthFormat);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("RenderTarget creation");
        
        Width = width;
        Height = height;
        Format = colorFormat;
        DepthFormat = depthFormat;
    }

    /// <summary>
    /// Get the width in pixels.
    /// </summary>
    public uint Width { get; }

    /// <summary>
    /// Get the height in pixels.
    /// </summary>
    public uint Height { get; }

    /// <summary>
    /// Get the color texture format.
    /// </summary>
    public TextureFormat Format { get; }

    /// <summary>
    /// Get the depth buffer format, if any.
    /// </summary>
    public DepthFormat? DepthFormat { get; }

    /// <summary>
    /// Returns true if this render target has a depth buffer.
    /// </summary>
    public bool HasDepth => DepthFormat.HasValue;

    /// <summary>
    /// Get the size of the pixel data in bytes.
    /// </summary>
    public nuint BufferSize => NativeMethods.RenderTargetBufferSize(Handle);

    /// <summary>
    /// Read the rendered pixels to a CPU buffer.
    /// </summary>
    public byte[] ReadToCpu()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        var size = (int)BufferSize;
        var buffer = new byte[size];
        ReadToBuffer(buffer);
        return buffer;
    }

    /// <summary>
    /// Read the rendered pixels into an existing buffer.
    /// </summary>
    public void ReadToBuffer(Span<byte> output)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        
        var requiredSize = BufferSize;
        if ((nuint)output.Length < requiredSize)
            throw new ArgumentException($"Output buffer too small: {output.Length} < {requiredSize}");
        
        unsafe
        {
            fixed (byte* ptr = output)
            {
                var result = NativeMethods.RenderTargetReadToBuffer(Handle, (nint)ptr, (nuint)output.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("RenderTarget read");
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.RenderTargetDestroy(Handle);
            _disposed = true;
        }
    }
}

