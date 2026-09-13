using Goldy.Native;

namespace Goldy;

/// <summary>
/// Stable render-target lease minted by a <see cref="Context"/>.
/// </summary>
public sealed class SchemeRenderTargetLease : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal SchemeRenderTargetLease(nint handle) => Handle = handle;

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.SchemeRenderTargetLeaseDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}
