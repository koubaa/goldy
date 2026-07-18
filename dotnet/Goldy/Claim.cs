using Goldy.Native;

namespace Goldy;

/// <summary>
/// One submission's claim extracted from a <see cref="Transaction"/>.
/// </summary>
public sealed class Claim : IDisposable
{
    internal nint Handle;
    private bool _disposed;
    private bool _settled;

    internal Claim(nint handle) => Handle = handle;

    public void Consume()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = NativeMethods.ClaimConsume(Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Claim consume");
        _settled = true;
        Handle = nint.Zero;
        _disposed = true;
    }

    public void Discard()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = NativeMethods.ClaimDiscard(Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Claim discard");
        _settled = true;
        Handle = nint.Zero;
        _disposed = true;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        if (!_settled)
            NativeMethods.ClaimDiscard(Handle);
        Handle = nint.Zero;
        _disposed = true;
    }
}
