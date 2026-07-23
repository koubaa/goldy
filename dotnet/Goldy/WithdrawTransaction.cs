using Goldy.Native;

namespace Goldy;

/// <summary>
/// Stable withdraw relationship recorded in one <see cref="Scheme"/>.
/// </summary>
public sealed class WithdrawTransaction : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal WithdrawTransaction(nint handle) => Handle = handle;

    public ulong ByteSize
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.WithdrawTransactionByteSize(Handle);
        }
    }

    public WithdrawClaim Claim(SchemeSubmission submission)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(submission);
        var claim = NativeMethods.WithdrawTransactionClaim(Handle, submission.Handle);
        if (claim == nint.Zero)
            throw GoldyException.FromLastError("WithdrawTransaction claim");
        return new WithdrawClaim(claim);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        NativeMethods.WithdrawTransactionDestroy(Handle);
        Handle = nint.Zero;
        _disposed = true;
    }
}
