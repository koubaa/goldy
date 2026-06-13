using Goldy.Native;

namespace Goldy;

/// <summary>
/// GPU submission context — one per retained <see cref="Scheme"/>.
/// </summary>
public sealed class Context : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal Context(nint handle)
    {
        Handle = handle;
    }

    /// <summary>
    /// Create a context bound to a device.
    /// </summary>
    public static Context Create(Device device)
    {
        device.ThrowIfDisposed();
        var handle = NativeMethods.ContextCreate(device.Handle);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("Context creation");
        return new Context(handle);
    }

    internal void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>
    /// Block until the GPU has completed all work scheduled up to <paramref name="timelineValue"/>.
    /// </summary>
    public void WaitUntil(ulong timelineValue)
    {
        ThrowIfDisposed();
        var result = NativeMethods.ContextWaitUntil(Handle, timelineValue);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("Context wait_until");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.ContextDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}
