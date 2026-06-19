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
    /// Bindless resource slot index for one buffer unit.
    /// </summary>
    public uint UnitResourceIndex(uint unit, ResourceAccess access)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var idx = NativeMethods.BufferUnitResourceIndex(Handle, unit, access);
        if (idx == uint.MaxValue)
            throw GoldyException.FromLastError("Buffer unit_resource_index");
        return idx;
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

    /// <summary>
    /// Read one buffer unit back to CPU memory.
    /// </summary>
    public byte[] UnitReadToCpu(uint unit, Device device)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        device.ThrowIfDisposed();
        var size = UnitByteSize(unit);
        var output = new byte[size];
        unsafe
        {
            fixed (byte* p = output)
            {
                var result = NativeMethods.BufferUnitReadToCpu(
                    Handle, unit, device.Handle, (nint)p, (nuint)output.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("Buffer unit_read_to_cpu");
            }
        }
        return output;
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
