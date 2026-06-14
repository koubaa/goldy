namespace Goldy;

/// <summary>
/// Read easement grant recorded once via <see cref="Scheme.GrantRead"/>.
/// </summary>
public sealed class ReadGrant : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal ReadGrant(nint handle) => Handle = handle;

    /// <summary>Logical byte size of readable data for this grant.</summary>
    public ulong ByteSize
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return Native.NativeMethods.ReadGrantByteSize(Handle);
        }
    }

    /// <summary>
    /// Readable bytes for <paramref name="frame"/>'s submission.
    /// </summary>
    public byte[] Read(SchemeFrame frame)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(frame);

        var output = new byte[ByteSize];
        unsafe
        {
            fixed (byte* p = output)
            {
                var result = Native.NativeMethods.ReadGrantRead(Handle, frame.Handle, (nint)p, (nuint)output.Length);
                if (result != Native.GoldyResult.Ok)
                    throw GoldyException.FromLastError("ReadGrant read");
            }
        }
        return output;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Native.NativeMethods.ReadGrantDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}
