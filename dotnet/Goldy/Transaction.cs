using Goldy.Native;

namespace Goldy;

/// <summary>
/// Erased exchange transaction recorded in a scheme.
/// </summary>
public sealed class Transaction : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal Transaction(nint handle) => Handle = handle;

    public uint BindingId
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.TransactionBindingId(Handle);
        }
    }

    public ulong Generation
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.TransactionGeneration(Handle);
        }
    }

    public Claim Claim(SchemeSubmission submission)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(submission);
        var claim = NativeMethods.TransactionClaim(Handle, submission.Handle);
        if (claim == nint.Zero)
            throw GoldyException.FromLastError("Transaction claim");
        return new Claim(claim);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.TransactionDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}
