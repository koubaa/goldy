using Goldy.Native;

namespace Goldy;

/// <summary>
/// CPU-readable bytes from a consumed <see cref="WithdrawClaim"/>.
/// Dropping recycles staging.
/// </summary>
public sealed class WithdrawBytes : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal WithdrawBytes(nint handle) => Handle = handle;

    public int Length
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return checked((int)NativeMethods.WithdrawBytesLen(Handle));
        }
    }

    public byte this[int index]
    {
        get
        {
            var span = AsSpan();
            return span[index];
        }
    }

    public unsafe ReadOnlySpan<byte> AsSpan()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var len = Length;
        if (len == 0)
            return ReadOnlySpan<byte>.Empty;
        var data = NativeMethods.WithdrawBytesData(Handle);
        if (data == nint.Zero)
            throw GoldyException.FromLastError("WithdrawBytes data");
        return new ReadOnlySpan<byte>((void*)data, len);
    }

    public byte[] ToArray()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var output = new byte[Length];
        unsafe
        {
            fixed (byte* p = output)
            {
                var result = NativeMethods.WithdrawBytesCopy(Handle, (nint)p, (nuint)output.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("WithdrawBytes copy");
            }
        }
        return output;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        NativeMethods.WithdrawBytesDestroy(Handle);
        Handle = nint.Zero;
        _disposed = true;
    }
}
