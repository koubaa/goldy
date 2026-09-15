using Goldy.Native;

namespace Goldy;

/// <summary>
/// CPU↔GPU memory exchange: withdrawals (readback) and deposits (upload).
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

    public WithdrawTransaction BindWithdraw(Scheme scheme, Parcel parcel)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(scheme);
        ArgumentNullException.ThrowIfNull(parcel);
        var tx = NativeMethods.MemoryExchangeBindWithdraw(Handle, scheme.Handle, parcel.Handle);
        if (tx == nint.Zero)
            throw GoldyException.FromLastError("MemoryExchange bind_withdraw");
        return new WithdrawTransaction(tx);
    }

    public WithdrawTransaction BindWithdrawTexture(Scheme scheme, Texture texture)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(scheme);
        ArgumentNullException.ThrowIfNull(texture);
        var tx = NativeMethods.MemoryExchangeBindWithdrawTexture(Handle, scheme.Handle, texture.Handle);
        if (tx == nint.Zero)
            throw GoldyException.FromLastError("MemoryExchange bind_withdraw_texture");
        return new WithdrawTransaction(tx);
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
