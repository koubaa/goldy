namespace Goldy;

/// <summary>
/// Per-submission identity returned by <see cref="Scheme.Submit"/>.
/// </summary>
public sealed class SchemeSubmission : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal SchemeSubmission(nint handle) => Handle = handle;

    /// <summary>True when this submission's GPU work has retired.</summary>
    public bool IsSettled
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Native.NativeMethods.SchemeSubmissionIsSettled(Handle);
        }
    }

    /// <summary>Block until this submission's GPU work has completed.</summary>
    public void WaitUntilSettled()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = Native.NativeMethods.SchemeSubmissionWaitUntilSettled(Handle);
        if (result != Native.GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme submission wait until settled");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Native.NativeMethods.SchemeSubmissionDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}
