using Goldy.Native;

namespace Goldy;

/// <summary>
/// A GPU texture.
/// </summary>
public sealed class Texture : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    /// <summary>
    /// Create a new texture.
    /// </summary>
    public Texture(Device device, uint width, uint height, TextureFormat format, TextureUsage usage)
    {
        device.ThrowIfDisposed();
        
        Handle = NativeMethods.TextureCreate(device.Handle, width, height, format, usage);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("Texture creation");
        
        Width = width;
        Height = height;
        Format = format;
    }

    /// <summary>
    /// Get the texture width.
    /// </summary>
    public uint Width { get; }

    /// <summary>
    /// Get the texture height.
    /// </summary>
    public uint Height { get; }

    /// <summary>
    /// Get the texture format.
    /// </summary>
    public TextureFormat Format { get; }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.TextureDestroy(Handle);
            _disposed = true;
        }
    }
}

