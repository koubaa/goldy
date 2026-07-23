using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// Acquired retained GPU buffer (possibly partitioned into bindable units).
/// </summary>
public sealed class Buffer : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    internal Buffer(nint handle)
    {
        Handle = handle;
    }

    /// <summary>
    /// Total committed byte size of this buffer.
    /// </summary>
    public ulong ByteSize
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.BufferByteSize(Handle);
        }
    }

    /// <summary>
    /// Number of independently bindable units in this buffer.
    /// </summary>
    public uint UnitCount
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.BufferUnitCount(Handle);
        }
    }

    /// <summary>
    /// Byte size of one buffer unit.
    /// </summary>
    public ulong UnitByteSize(uint unit)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return NativeMethods.BufferUnitByteSize(Handle, unit);
    }

    /// <summary>
    /// Borrow one bindable unit as an owned parcel handle.
    /// </summary>
    public Parcel Field(uint unit)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var parcel = NativeMethods.BufferField(Handle, unit);
        if (parcel == nint.Zero)
            throw GoldyException.FromLastError("Buffer field");
        return new Parcel(parcel, UnitByteSize(unit));
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.BufferDestroy(Handle);
            _disposed = true;
        }
    }
}
