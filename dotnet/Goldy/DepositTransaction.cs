using Goldy.Native;

namespace Goldy;

/// <summary>
/// Stable deposit relationship recorded in one <see cref="Scheme"/>.
/// Write staging bytes before <see cref="Scheme.Submit"/> (`Write` or <c>deposit &lt;&lt; data</c>);
/// submit claims the occurrence internally.
/// </summary>
public sealed class DepositTransaction : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal DepositTransaction(nint handle) => Handle = handle;

    public ulong Capacity
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.DepositTransactionCapacity(Handle);
        }
    }

    public uint Id
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.DepositTransactionId(Handle);
        }
    }

    public void Write(ReadOnlySpan<byte> data, ulong offset = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        unsafe
        {
            fixed (byte* p = data)
            {
                var result = NativeMethods.DepositTransactionWrite(
                    Handle, offset, (nint)p, (nuint)data.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("DepositTransaction write");
            }
        }
    }

    /// <summary>
    /// Tender bytes for this submission (<c>deposit &lt;&lt; data</c>). Offset 0; use
    /// <see cref="Write"/> for partial fills.
    /// </summary>
    public static DepositTransaction operator <<(DepositTransaction deposit, byte[] data)
    {
        ArgumentNullException.ThrowIfNull(deposit);
        ArgumentNullException.ThrowIfNull(data);
        deposit.Write(data);
        return deposit;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        NativeMethods.DepositTransactionDestroy(Handle);
        Handle = nint.Zero;
        _disposed = true;
    }
}
