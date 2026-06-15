namespace Goldy;

/// <summary>
/// Per-submission identity returned by <see cref="Scheme.Submit"/>.
/// </summary>
public sealed class SchemeSubmission : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal SchemeSubmission(nint handle) => Handle = handle;

    /// <summary>Timeline value for this submission (debug only).</summary>
    public ulong TimelineValue
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Native.NativeMethods.SchemeSubmissionTimelineValue(Handle);
        }
    }

    /// <summary>Block until this submission's GPU work has completed.</summary>
    public void Wait(Context ctx)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ctx.ThrowIfDisposed();
        var result = Native.NativeMethods.SchemeSubmissionWait(ctx.Handle, Handle);
        if (result != Native.GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme submission wait");
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
