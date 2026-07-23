using Goldy.Native;

namespace Goldy;

/// <summary>
/// Stable deposit relationship recorded in one <see cref="Scheme"/>.
/// Write staging bytes before <see cref="Scheme.Submit"/>; no claim afterward.
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

    public void Write(Scheme scheme, ReadOnlySpan<byte> data, ulong offset = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(scheme);
        unsafe
        {
            fixed (byte* p = data)
            {
                var result = NativeMethods.DepositTransactionWrite(
                    Handle, scheme.Handle, offset, (nint)p, (nuint)data.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("DepositTransaction write");
            }
        }
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
