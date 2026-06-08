using Goldy.Native;

namespace Goldy;

/// <summary>
/// Bump allocator over a single GPU buffer.
/// </summary>
public sealed class BufferPool : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    public BufferPool(Device device, ulong capacity)
    {
        ObjectDisposedException.ThrowIf(device is null, nameof(device));
        Handle = NativeMethods.BufferPoolCreate(device.Handle, capacity);
        if (Handle == nint.Zero)
            throw GoldyException.FromLastError("BufferPool creation");
    }

    public BufferView AllocU32(ulong count)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var view = NativeMethods.BufferPoolAllocU32(Handle, count);
        if (view == nint.Zero)
            throw GoldyException.FromLastError("BufferPool alloc_u32");
        return new BufferView(view);
    }

    public void WriteBacking(ulong byteOffset, ReadOnlySpan<byte> data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        unsafe
        {
            fixed (byte* p = data)
            {
                var result = NativeMethods.BufferPoolWriteBacking(Handle, byteOffset, (nint)p, (nuint)data.Length);
                if (result != GoldyResult.Ok)
                    throw GoldyException.FromLastError("BufferPool write_backing");
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.BufferPoolDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}

/// <summary>
/// Sub-range view into a buffer pool allocation.
/// </summary>
public sealed class BufferView : IDisposable
{
    internal nint Handle;
    private bool _disposed;

    internal BufferView(nint handle) => Handle = handle;

    public ulong Offset
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return NativeMethods.BufferViewOffset(Handle);
        }
    }

    public uint ResourceIndex(ResourceAccess access)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var idx = NativeMethods.BufferViewResourceIndex(Handle, access);
        if (idx == uint.MaxValue)
            throw GoldyException.FromLastError("BufferView resource_index");
        return idx;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            NativeMethods.BufferViewDestroy(Handle);
            Handle = nint.Zero;
            _disposed = true;
        }
    }
}
