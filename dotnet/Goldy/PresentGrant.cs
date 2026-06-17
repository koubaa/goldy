using Goldy.Native;

namespace Goldy;

/// <summary>
/// Present easement grant recorded once via <see cref="Scheme.GrantPresent"/>.
/// </summary>
public sealed class PresentGrant : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal PresentGrant(nint handle) => Handle = handle;

    public void Consume(SchemeSubmission submission)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(submission);
        var result = NativeMethods.PresentGrantConsume(Handle, submission.Handle);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("PresentGrant consume");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.PresentGrantDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}
