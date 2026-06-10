using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// Opaque retained GPU parcel (buffer or texture).
/// </summary>
public sealed class Parcel : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    internal Parcel(nint handle, ulong byteSize)
    {
        Handle = handle;
        ByteSize = byteSize;
    }

    /// <summary>
    /// Approximate committed byte size of this parcel.
    /// </summary>
    public ulong ByteSize { get; }

    /// <summary>
    /// Bindless resource slot index for shader binding.
    /// </summary>
    public uint ResourceIndex(ResourceAccess access)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var idx = NativeMethods.ParcelResourceIndex(Handle, access);
        if (idx == uint.MaxValue)
            throw GoldyException.FromLastError("Parcel resource_index");
        return idx;
    }

    /// <summary>
    /// Read buffer parcel contents back to CPU memory from offset 0.
    /// </summary>
    public byte[] ReadToCpu(Device device)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        device.ThrowIfDisposed();

        var output = new byte[ByteSize];
        unsafe
        {
            fixed (byte* p = output)
            {
                var result = NativeMethods.ParcelReadToCpu(Handle, device.Handle, (nint)p, (nuint)output.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("Parcel read_to_cpu");
            }
        }
        return output;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.ParcelDestroy(Handle);
            _disposed = true;
        }
    }
}
