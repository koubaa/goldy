using Goldy.Native;

namespace Goldy;

/// <summary>
/// Pool of OS swapchain drawables for retained scheme present.
/// </summary>
public sealed class SwapchainPool : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    private SwapchainPool(nint handle) => Handle = handle;

    public static SwapchainPool CreateWin32(Context ctx, nint hwnd, uint depth = 3)
    {
        ctx.ThrowIfDisposed();
        var handle = NativeMethods.SwapchainPoolCreateWin32(ctx.Handle, hwnd, depth);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("SwapchainPool creation");
        return new SwapchainPool(handle);
    }

    public static SwapchainPool CreateAppKit(Context ctx, nint nsView, uint depth = 3)
    {
        ctx.ThrowIfDisposed();
        var handle = NativeMethods.SwapchainPoolCreateAppKit(ctx.Handle, nsView, depth);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("SwapchainPool creation");
        return new SwapchainPool(handle);
    }

    public static SwapchainPool CreateWayland(Context ctx, nint display, nint surface, uint depth = 3)
    {
        ctx.ThrowIfDisposed();
        var handle = NativeMethods.SwapchainPoolCreateWayland(ctx.Handle, display, surface, depth);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("SwapchainPool creation");
        return new SwapchainPool(handle);
    }

    public uint Width => NativeMethods.SwapchainPoolWidth(Handle);

    public uint Height => NativeMethods.SwapchainPoolHeight(Handle);

    public TextureFormat Format => NativeMethods.SwapchainPoolFormat(Handle);

    public void Resize(uint width, uint height)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (width == 0 || height == 0)
            return;
        var result = NativeMethods.SwapchainPoolResize(Handle, width, height);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("SwapchainPool resize");
    }

    public PresentLease Lease()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var lease = NativeMethods.SwapchainPoolLease(Handle);
        if (lease == nint.Zero)
            throw GoldyException.FromLastError("SwapchainPool lease");
        return new PresentLease(lease);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.SwapchainPoolDestroy(Handle);
            _disposed = true;
        }
    }
}
