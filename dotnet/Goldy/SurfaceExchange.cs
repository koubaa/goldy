using Goldy.Native;

namespace Goldy;

/// <summary>
/// Window-surface exchange for present-on-scheme.
/// </summary>
public sealed class SurfaceExchange : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    private SurfaceExchange(nint handle) => Handle = handle;

    public static SurfaceExchange CreateWin32(Context ctx, nint hwnd, uint depth = 3)
    {
        ctx.ThrowIfDisposed();
        var handle = NativeMethods.SurfaceExchangeCreateWin32(ctx.Handle, hwnd, depth);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("SurfaceExchange creation");
        return new SurfaceExchange(handle);
    }

    public static SurfaceExchange CreateAppKit(Context ctx, nint nsView, uint depth = 3)
    {
        ctx.ThrowIfDisposed();
        var handle = NativeMethods.SurfaceExchangeCreateAppKit(ctx.Handle, nsView, depth);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("SurfaceExchange creation");
        return new SurfaceExchange(handle);
    }

    public static SurfaceExchange CreateWayland(Context ctx, nint display, nint surface, uint depth = 3)
    {
        ctx.ThrowIfDisposed();
        var handle = NativeMethods.SurfaceExchangeCreateWayland(ctx.Handle, display, surface, depth);
        if (handle == nint.Zero)
            throw GoldyException.FromLastError("SurfaceExchange creation");
        return new SurfaceExchange(handle);
    }

    public uint Width => NativeMethods.SurfaceExchangeWidth(Handle);

    public uint Height => NativeMethods.SurfaceExchangeHeight(Handle);

    public TextureFormat Format => NativeMethods.SurfaceExchangeFormat(Handle);

    public ulong Generation => NativeMethods.SurfaceExchangeGeneration(Handle);

    public void Resize(uint width, uint height)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (width == 0 || height == 0)
            return;
        var result = NativeMethods.SurfaceExchangeResize(Handle, width, height);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("SurfaceExchange resize");
    }

    public PresentLease Lease()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var lease = NativeMethods.SurfaceExchangeLease(Handle);
        if (lease == nint.Zero)
            throw GoldyException.FromLastError("SurfaceExchange lease");
        return new PresentLease(lease);
    }

    public Transaction BindRenderTarget(Scheme scheme, SchemeRenderTargetLease source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(scheme);
        ArgumentNullException.ThrowIfNull(source);
        var tx = NativeMethods.SurfaceExchangeBindRenderTarget(Handle, scheme.Handle, source.Handle);
        if (tx == nint.Zero)
            throw GoldyException.FromLastError("SurfaceExchange bind_render_target");
        return new Transaction(tx);
    }

    public Transaction Bind(Scheme scheme, Texture source)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(scheme);
        ArgumentNullException.ThrowIfNull(source);
        var tx = NativeMethods.SurfaceExchangeBind(Handle, scheme.Handle, source.Handle);
        if (tx == nint.Zero)
            throw GoldyException.FromLastError("SurfaceExchange bind");
        return new Transaction(tx);
    }

    public (PresentLease Lease, Transaction Transaction) BindDestination(Scheme scheme)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(scheme);
        var result = NativeMethods.SurfaceExchangeBindDestination(Handle, scheme.Handle, out var lease, out var transaction);
        if (result != GoldyResult.Ok)
            throw GoldyException.FromLastError("SurfaceExchange bind_destination");
        if (lease == nint.Zero || transaction == nint.Zero)
            throw GoldyException.FromLastError("SurfaceExchange bind_destination");
        return (new PresentLease(lease), new Transaction(transaction));
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.SurfaceExchangeDestroy(Handle);
            _disposed = true;
        }
    }
}
