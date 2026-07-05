using Goldy.Native;

namespace Goldy;

/// <summary>
/// Acquired retained GPU texture.
/// </summary>
public sealed class Texture : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    internal Texture(nint handle)
    {
        Handle = handle;
    }

    /// <summary>
    /// Approximate committed byte size of this texture.
    /// </summary>
    public ulong ByteSize
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.TextureByteSize(Handle);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.TextureDestroy(Handle);
            _disposed = true;
        }
    }
}
