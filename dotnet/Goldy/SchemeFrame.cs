namespace Goldy;

/// <summary>
/// Per-submission identity returned by <see cref="Scheme.Submit"/>.
/// </summary>
public readonly struct SchemeFrame
{
    internal SchemeFrame(ulong timelineValue) => TimelineValue = timelineValue;

    /// <summary>Timeline value for this submission.</summary>
    public ulong TimelineValue { get; }

    /// <summary>Block until this submission's GPU work has completed.</summary>
    public void Wait(Context ctx)
    {
        ctx.ThrowIfDisposed();
        var result = Native.NativeMethods.ContextWaitUntil(ctx.Handle, TimelineValue);
        if (result != Native.GoldyResult.Ok)
            throw GoldyException.FromLastError("Scheme frame wait");
    }
}
