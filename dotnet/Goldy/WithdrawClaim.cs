using Goldy.Native;

namespace Goldy;

/// <summary>
/// Linear claim for one submission's memory withdrawal.
/// Settle with <see cref="Consume"/> or <see cref="Discard"/>.
/// </summary>
public sealed class WithdrawClaim : IDisposable
{
    internal nint Handle;
    private bool _disposed;
    private bool _settled;

    internal WithdrawClaim(nint handle) => Handle = handle;

    /// <summary>
    /// Wait for the submission, read staging into CPU bytes.
    /// Takes ownership of this claim (do not dispose afterward).
    /// </summary>
    public WithdrawBytes Consume()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var bytes = NativeMethods.WithdrawClaimConsume(Handle);
        _settled = true;
        Handle = nint.Zero;
        _disposed = true;
        if (bytes == nint.Zero)
            throw GoldyException.FromLastError("WithdrawClaim consume");
        return new WithdrawBytes(bytes);
    }

    public void Discard()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = NativeMethods.WithdrawClaimDiscard(Handle);
        _settled = true;
        Handle = nint.Zero;
        _disposed = true;
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("WithdrawClaim discard");
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        if (!_settled)
            NativeMethods.WithdrawClaimDestroy(Handle);
        Handle = nint.Zero;
        _disposed = true;
    }
}
