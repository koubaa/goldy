using System.Runtime.InteropServices;
using Goldy.Native;

namespace Goldy;

/// <summary>
/// Deed-governed pool for retained GPU parcels.
/// </summary>
public sealed class RetainedPool : IDisposable
{
    internal readonly nint Handle;
    private bool _disposed;

    public RetainedPool(Device device)
    {
        device.ThrowIfDisposed();
        Handle = NativeMethods.RetainedPoolCreate(device.Handle);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("RetainedPool creation");
    }

    /// <summary>
    /// Acquire an uninitialized retained buffer parcel.
    /// </summary>
    public Parcel AcquireBuffer(ulong size, BufferKind access, uint elementStride = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        unsafe
        {
            var parcel = NativeMethods.RetainedPoolAcquireBuffer(
                Handle, size, access, elementStride, nint.Zero, 0);
            if (parcel == nint.Zero)
                throw GoldyException.FromLastError("RetainedPool acquire_buffer");
            return new Parcel(parcel, size);
        }
    }

    /// <summary>
    /// Acquire a retained buffer parcel initialized with raw bytes.
    /// </summary>
    public Parcel AcquireBuffer(ReadOnlySpan<byte> data, BufferKind access, uint elementStride = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        unsafe
        {
            fixed (byte* ptr = data)
            {
                var parcel = NativeMethods.RetainedPoolAcquireBuffer(
                    Handle, (ulong)data.Length, access, elementStride, (nint)ptr, (nuint)data.Length);
                if (parcel == nint.Zero)
                    throw GoldyException.FromLastError("RetainedPool acquire_buffer");
                return new Parcel(parcel, (ulong)data.Length);
            }
        }
    }

    /// <summary>
    /// Acquire a retained buffer parcel initialized with typed data.
    /// </summary>
    public Parcel AcquireBuffer<T>(ReadOnlySpan<T> data, BufferKind access) where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(data);
        var stride = (uint)Marshal.SizeOf<T>();
        return AcquireBuffer(bytes, access, stride);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.RetainedPoolDestroy(Handle);
            _disposed = true;
        }
    }
}
