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

    public void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>
    /// Mint a render-target lease on this context (the lessor).
    /// </summary>
    public SchemeRenderTargetLease LeaseRenderTarget(
        uint width,
        uint height,
        TextureFormat format,
        DepthFormat? depthFormat = null)
    {
        ThrowIfDisposed();
        var hasDepth = depthFormat.HasValue;
        var depth = depthFormat ?? default;
        var lease = NativeMethods.ContextLeaseRenderTarget(Handle, width, height, format, hasDepth, depth);
        if (lease == nint.Zero)
            throw GoldyException.FromLastError("Context lease_render_target");
        return new SchemeRenderTargetLease(lease);
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
