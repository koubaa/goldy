using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// Opaque retained GPU parcel (buffer or texture).
/// </summary>
public sealed class Parcel : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    internal Parcel(nint handle, ulong byteSize)
    {
        Handle = handle;
        ByteSize = byteSize;
    }

    /// <summary>
    /// Approximate committed byte size of this parcel.
    /// </summary>
    public ulong ByteSize { get; }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.ParcelDestroy(Handle);
            _disposed = true;
        }
    }
}
