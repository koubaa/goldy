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
    /// Create a new texture with the specified spatial access pattern.
    /// </summary>
    /// <param name="device">The GPU device.</param>
    /// <param name="width">Texture width in pixels.</param>
    /// <param name="height">Texture height in pixels.</param>
    /// <param name="format">Pixel format.</param>
    /// <param name="access">Spatial access pattern (Interpolated for filtering, Direct for indexing).</param>
    /// <param name="flags">Texture flags for copy and render operations.</param>
    public Texture(Device device, uint width, uint height, TextureFormat format, TextureKind access, TextureFlags flags = TextureFlags.None)
    {
        device.ThrowIfDisposed();
        
        Handle = NativeMethods.TextureCreate(device.Handle, width, height, format, access, flags);
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

