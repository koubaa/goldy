using Goldy.Native;

namespace Goldy;

/// <summary>
/// Host-claimed parcel bytes after a submission (<c>goldy_scheme_submission_take</c>).
/// Dropping releases the host claim.
/// </summary>
public sealed class HostView : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal HostView(nint handle) => Handle = handle;

    public int Length
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return checked((int)NativeMethods.HostViewLen(Handle));
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
        var data = NativeMethods.HostViewData(Handle);
        if (data == nint.Zero)
            throw GoldyException.FromLastError("HostView data");
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
                var result = NativeMethods.HostViewCopy(Handle, (nint)p, (nuint)output.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("HostView copy");
            }
        }
        return output;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        NativeMethods.HostViewDestroy(Handle);
        Handle = nint.Zero;
        _disposed = true;
    }
}
