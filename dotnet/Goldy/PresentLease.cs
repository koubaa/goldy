using Goldy.Native;

namespace Goldy;

/// <summary>
/// Stable present lease from a <see cref="SurfaceExchange"/>.
/// </summary>
public sealed class PresentLease : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal PresentLease(nint handle) => Handle = handle;

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.PresentLeaseDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}
