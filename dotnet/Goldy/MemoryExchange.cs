using Goldy.Native;

namespace Goldy;

/// <summary>
/// CPU→GPU memory exchange (deposits / uploads).
/// </summary>
public sealed class MemoryExchange : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    public MemoryExchange(Context ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        ctx.ThrowIfDisposed();
        Handle = NativeMethods.MemoryExchangeCreate(ctx.Handle);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("MemoryExchange creation");
    }

    public DepositTransaction BindDeposit(Scheme scheme, DepositTarget target)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(scheme);
        var native = target.Raw;
        var tx = NativeMethods.MemoryExchangeBindDeposit(Handle, scheme.Handle, native);
        if (tx == nint.Zero)
            throw GoldyException.FromLastError("MemoryExchange bind_deposit");
        return new DepositTransaction(tx);
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        NativeMethods.MemoryExchangeDestroy(Handle);
        Handle = nint.Zero;
        _disposed = true;
    }
}
